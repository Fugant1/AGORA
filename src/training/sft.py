"""Stage-1 Multitask Cue-Weighted SFT (exact mathematical parity with ser_sft.ipynb)."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CUE_COLUMNS,
    CUE_LABELS,
    CUES,
    GLOBAL_CUES,
    ModelConfig,
    SFTConfig,
)
from src.data.dataset import SERDataset
from src.data.partitions import distribution_audit, load_or_create_partitions, validate_partitions
from src.models.qwen_omni import generation_eos_ids, load_thinker_model
from src.prompts.templates import (
    build_bridge_generation_prompt,
    format_sft_target,
    inventory_text,
    messages_for,
)
from src.utils.audio import load_audio
from src.utils.io import atomic_csv, load_manifest, stage_clean_data


def inverse_frequency_cue_weights(frame: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """Fit tempered inverse-frequency cue weights strictly on `sft_train`.

    alpha=0.0 -> unweighted (all weights = 1.0).
    alpha=1.0 -> full inverse-frequency N_sft / (K * count).
    0 < alpha < 1 -> mean-1 normalized n_c^(-alpha) interpolation.
    """
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"CUE_WEIGHT_ALPHA must be in [0.0, 1.0], got {alpha!r}")
    if frame.empty:
        raise ValueError("Cannot weight an empty SFT partition")
    rows = []
    n_total = float(len(frame))
    for cue in CUES:
        labels = CUE_LABELS[cue]
        values = frame[CUE_COLUMNS[cue]]
        if not values.isin(labels).all():
            raise ValueError(f"Missing or invalid SFT labels for {cue}")
        counts = values.value_counts().reindex(labels, fill_value=0)
        if (counts == 0).any():
            raise ValueError(f"{cue}: every cue class needs SFT examples; enlarge the training partition")
        if alpha == 0.0:
            weights = {label: 1.0 for label in labels}
        else:
            c_arr = counts.to_numpy(dtype=np.float64)
            denom = float(np.sum(np.power(c_arr, 1.0 - alpha)))
            w_arr = n_total * np.power(c_arr, -alpha) / denom
            weights = {label: float(w) for label, w in zip(labels, w_arr)}
        for label, count in counts.items():
            rows.append(dict(cue=cue, label=label, count=int(count), alpha=alpha, weight=weights[label]))
    return pd.DataFrame(rows)


def inventory_cue_spans(
    tokenizer: Any,
    row: Dict[str, Any],
    cue_weights: Dict[Tuple[str, str], float],
) -> Tuple[List[int], List[Tuple[str, List[int], float]]]:
    target = row["sft_gt"]
    prefix = "<think>\n[Acoustic Inventory]\n"
    expected = prefix + inventory_text(row) + "\n[Prosodic Reasoning]\n"
    if not target.startswith(expected):
        raise ValueError("SFT inventory differs from the saved cue targets")
    if not tokenizer.is_fast:
        raise ValueError("Cue-span loss requires a fast tokenizer with character offsets")
    encoded = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    spans = []
    seen = set()
    cursor = len(prefix)
    for cue in CUES:
        label = row[CUE_COLUMNS[cue]]
        start = cursor + len(f"- {cue}: ")
        end = start + len(label)
        positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > start]
        if not positions:
            raise ValueError(f"No target tokens found for {cue}")
        if seen.intersection(positions):
            raise ValueError(f"Overlapping target token span for {cue}")
        seen.update(positions)
        spans.append((cue, positions, float(cue_weights[(cue, label)])))
        cursor = end + 1
    return encoded["input_ids"], spans


def build_assistant_token_weights(
    n_total: int,
    cue_spans: List[Tuple[str, List[int], float]],
    equal_multitask: bool = True,
    cue_task_share: float = 0.25,
) -> List[float]:
    """Place `cue_task_share` strictly on `span_pos[0]` ('low'/'moderate'/'high') and `1-cue_task_share` on other tokens."""
    if n_total <= 0:
        raise ValueError(f"Expected positive supervised assistant token count, got {n_total}")
    if not equal_multitask:
        weights = [1.0] * n_total
        for _, span_pos, w_cue in cue_spans:
            for j in span_pos:
                weights[j] = float(w_cue)
        return weights
    cue_task_share = float(cue_task_share)
    if not math.isfinite(cue_task_share) or cue_task_share <= 0.0 or cue_task_share >= 1.0:
        raise ValueError(f"CUE_TASK_SHARE must be in (0.0, 1.0), got {cue_task_share!r}")
    if not cue_spans:
        raise ValueError("Expected at least one acoustic cue span")
    decision_indices = {span_pos[0] for _, span_pos, _ in cue_spans}
    if len(decision_indices) != len(cue_spans) or any(j < 0 or j >= n_total for j in decision_indices):
        raise ValueError("Invalid or overlapping cue decision token indices")
    n_other = n_total - len(decision_indices)
    if n_other <= 0:
        raise ValueError("Multi-task SFT requires non-inventory reasoning/emotion assistant tokens")
    other_w = float(n_total) * (1.0 - cue_task_share) / float(n_other)
    weights = [other_w] * n_total
    n_cues = float(len(cue_spans))
    for _, span_pos, w_cue in cue_spans:
        weights[span_pos[0]] = float(n_total) * cue_task_share * float(w_cue) / n_cues
    return weights


def inventory_token_weights(
    tokenizer: Any,
    row: Dict[str, Any],
    cue_weights: Dict[Tuple[str, str], float],
) -> Tuple[List[int], List[float]]:
    target_ids, spans = inventory_cue_spans(tokenizer, row, cue_weights)
    return target_ids, build_assistant_token_weights(len(target_ids), spans, equal_multitask=False)


def mask_answer_labels(full: Dict[str, Any], prompt: Dict[str, Any]) -> Any:
    labels = full["input_ids"].clone()
    for i in range(len(labels)):
        prompt_ids = prompt["input_ids"][i][prompt["attention_mask"][i].bool()]
        full_positions = full["attention_mask"][i].nonzero(as_tuple=True)[0]
        full_ids = full["input_ids"][i][full_positions]
        n = len(prompt_ids)
        if n >= len(full_ids) or not (full_ids[:n] == prompt_ids).all():
            raise ValueError("Chat template prompt is not an exact prefix of the target sequence")
        labels[i, full_positions[:n]] = -100
    labels[full["attention_mask"] == 0] = -100
    return labels


class SERCollator:
    def __init__(
        self,
        processor: Any,
        cue_weights: Dict[Tuple[str, str], float],
        equal_multitask: bool = True,
        cue_task_share: float = 0.25,
    ) -> None:
        self.processor = processor
        self.cue_weights = cue_weights
        self.equal_multitask = bool(equal_multitask)
        self.cue_task_share = float(cue_task_share)

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        audios = [load_audio(item["audio_path"]) for item in items]
        prompts, targets = [], []
        for item in items:
            messages = messages_for(item)
            prompts.append(
                self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            targets.append(
                self.processor.apply_chat_template(
                    messages + [{"role": "assistant", "content": item["sft_gt"]}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        kwargs = dict(audio=audios, sampling_rate=16000, padding=True, return_tensors="pt")
        full = self.processor(text=targets, **kwargs)
        prompt = self.processor(text=prompts, **kwargs)
        labels = mask_answer_labels(full, prompt)
        token_weights = torch.zeros_like(labels, dtype=torch.float32)
        for i, item in enumerate(items):
            target_ids, cue_spans = inventory_cue_spans(self.processor.tokenizer, item, self.cue_weights)
            positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(positions) == 0 or int(positions[0].item()) == 0:
                raise ValueError("Expected masked prompt prefix before supervised assistant tokens")
            if len(positions) < len(target_ids) or labels[i, positions[: len(target_ids)]].tolist() != target_ids:
                raise ValueError("Assistant token offsets do not align with the multimodal target")
            weights = build_assistant_token_weights(
                len(positions),
                cue_spans,
                equal_multitask=self.equal_multitask,
                cue_task_share=self.cue_task_share,
            )
            token_weights[i, positions] = torch.tensor(weights, dtype=torch.float32)
        full["labels"] = labels
        full["token_weights"] = token_weights
        return full


def weighted_sft_loss(logits: Any, labels: Any, token_weights: Any) -> Any:
    import torch

    if token_weights.shape != labels.shape:
        raise ValueError("Expected one loss weight per target token")
    targets = labels[:, 1:]
    active = targets != -100
    counts = active.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Every example needs supervised assistant tokens")
    weights = token_weights[:, 1:].to(device=logits.device, dtype=torch.float32)[active]
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Supervised tokens need finite positive loss weights")
    rows, positions = active.nonzero(as_tuple=True)
    token_losses = torch.nn.functional.cross_entropy(
        logits[rows, positions].float(), targets[active], reduction="none"
    )
    response_losses = token_losses.new_zeros(len(labels)).scatter_add(0, rows, token_losses * weights) / counts
    return response_losses.mean()


def generate_completions(
    model: Any,
    processor: Any,
    rows: List[Dict[str, Any]],
    *,
    compute_dtype: Any,
    max_new_tokens: int = 512,
    teacher: bool = False,
) -> List[str]:
    import torch

    was_training = model.training
    padding_side = processor.tokenizer.padding_side
    eos = generation_eos_ids(processor.tokenizer)
    try:
        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        if hasattr(model.config, "get_text_config"):
            model.config.get_text_config().use_cache = True
        processor.tokenizer.padding_side = "left"
        texts = [
            processor.apply_chat_template(messages_for(row, teacher), tokenize=False, add_generation_prompt=True)
            for row in rows
        ]
        inputs = processor(
            text=texts,
            audio=[load_audio(row["audio_path"]) for row in rows],
            sampling_rate=16000,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        token_limit = int(max_new_tokens)
        for attempt in range(2 if teacher else 1):
            with torch.inference_mode(), torch.autocast("cuda", dtype=compute_dtype):
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=eos,
                    max_new_tokens=token_limit,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            tokens = generated[:, inputs["input_ids"].shape[1] :]
            if not teacher:
                break
            truncated = [
                row["audio_path"]
                for row, sequence in zip(rows, tokens)
                if len(sequence) >= token_limit and not any(int(t) in eos for t in sequence)
            ]
            if not truncated:
                break
            if attempt == 1:
                raise ValueError(f"Teacher explanation still incomplete after {token_limit} tokens: {truncated}")
            token_limit *= 2
        return processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    finally:
        model.config.use_cache = False
        if hasattr(model.config, "get_text_config"):
            model.config.get_text_config().use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.train(was_training)
        processor.tokenizer.padding_side = padding_side


def generate_sft_train(
    model: Any,
    processor: Any,
    frame: pd.DataFrame,
    output_path: Path,
    *,
    compute_dtype: Any,
    batch_size: int = 16,
    max_new_tokens: int = 512,
    data_dir: Optional[Path] = None,
    drive_data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    df = frame.copy().reset_index(drop=True)
    for column in ("reasoning", "sft_gt", "bridge_prompt"):
        if column not in df:
            df[column] = ""
        df[column] = df[column].fillna("").astype(object)
    pending = []
    for idx, row in df.iterrows():
        if row.reasoning:
            expected = format_sft_target(row, row.reasoning)
            accepted_prompts = (
                build_bridge_generation_prompt(row),
                build_bridge_generation_prompt(row, concise=False),
            )
            if row.sft_gt != expected or row.bridge_prompt not in accepted_prompts:
                raise ValueError("Cached SFT target/prompt mismatch: inspect cache or use a new split_dir")
        else:
            if row.sft_gt:
                raise ValueError("Cached target has no reasoning; inspect saved SFT CSV")
            pending.append(idx)

    for start in range(0, len(pending), batch_size):
        indices = pending[start : start + batch_size]
        rows = df.loc[indices].to_dict("records")
        explanations = generate_completions(
            model, processor, rows, compute_dtype=compute_dtype, max_new_tokens=max_new_tokens, teacher=True
        )
        if len(explanations) != len(rows):
            raise ValueError("Teacher returned an unexpected number of completions")
        targets = [format_sft_target(row, text) for row, text in zip(rows, explanations)]
        for idx, row, text, target in zip(indices, rows, explanations, targets):
            df.at[idx, "reasoning"] = text.strip()
            df.at[idx, "sft_gt"] = target
            df.at[idx, "bridge_prompt"] = build_bridge_generation_prompt(row)
        atomic_csv(df, output_path, data_dir=data_dir, drive_data_dir=drive_data_dir)
    return df


def build_cue_weighted_sft_trainer(
    model: Any,
    training_args: Any,
    train_dataset: Any,
    data_collator: SERCollator,
) -> Any:
    """Instantiate CueWeightedSFTTrainer (lazy transformers.Trainer import for CPU testability)."""
    from transformers import Trainer

    class CueWeightedSFTTrainer(Trainer):
        loss_is_scaled_for_ga = False  # compute_loss returns a microbatch mean.

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            labels = inputs.pop("labels")
            token_weights = inputs.pop("token_weights")
            outputs = model(**inputs, use_cache=False)
            loss = weighted_sft_loss(outputs.logits, labels, token_weights)
            return (loss, outputs) if return_outputs else loss

    return CueWeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )


def run_sft_training(
    sft_cfg: SFTConfig,
    model_cfg: ModelConfig,
) -> Dict[str, Any]:
    """End-to-end Stage-1 Multitask Cue-Weighted SFT matching ser_sft.ipynb."""
    import torch
    from transformers import TrainingArguments

    from src.training.evaluation import evaluate_sft_validation

    if sft_cfg.drive_data_dir is not None and Path(sft_cfg.drive_data_dir).resolve() != Path(sft_cfg.data_dir).resolve():
        stage_clean_data(sft_cfg.drive_data_dir, sft_cfg.data_dir)

    clean_dir = Path(sft_cfg.data_dir) / "emo_tag_clean"
    train_csv = clean_dir / "clean_train_set_with_cues.csv"
    val_csv = clean_dir / "clean_val_set_with_cues.csv"
    quartiles_json = clean_dir / "gender_acoustic_quartiles.json"

    clean_train = load_manifest(train_csv, "train", sft_cfg.data_dir, require_transcripts=True)
    val_set = load_manifest(val_csv, "val", sft_cfg.data_dir)
    if set(clean_train.audio_path) & set(val_set.audio_path):
        raise ValueError("Training/validation audio overlap")
    quartiles = json.loads(quartiles_json.read_text(encoding="utf-8"))

    sft_train, grpo_train, split_manifest = load_or_create_partitions(
        clean_train,
        val_set,
        split_dir=sft_cfg.split_dir,
        train_csv=train_csv,
        quartiles_json=quartiles_json,
        data_dir=sft_cfg.data_dir,
        drive_data_dir=sft_cfg.drive_data_dir,
        seed=sft_cfg.seed,
        prompt_version=sft_cfg.prompt_version,
        model_id=model_cfg.model_id,
        partition_mode=sft_cfg.partition_mode,
        neutral_target_fraction=sft_cfg.neutral_target_fraction,
    )
    balanced_train = pd.concat(
        [sft_train[clean_train.columns], grpo_train[clean_train.columns]], ignore_index=True
    )
    audit = distribution_audit(
        dict(clean_train=clean_train, balanced=balanced_train, sft=sft_train, grpo=grpo_train)
    )
    atomic_csv(audit, sft_cfg.split_dir / "distribution_audit.csv", sft_cfg.data_dir, sft_cfg.drive_data_dir)

    # Load base model first (without LoRA) if teacher explanations are pending
    needs_teacher = (
        "reasoning" not in sft_train.columns
        or sft_train["reasoning"].fillna("").astype(str).str.strip().eq("").any()
    )
    if needs_teacher:
        base_model, processor, compute_dtype, _ = load_thinker_model(
            model_cfg, seed=sft_cfg.seed, padding_side="right", attach_lora=False
        )
        sft_train = generate_sft_train(
            base_model,
            processor,
            sft_train,
            sft_cfg.sft_csv,
            compute_dtype=compute_dtype,
            batch_size=sft_cfg.generation_batch_size,
            max_new_tokens=sft_cfg.max_new_tokens,
            data_dir=sft_cfg.data_dir,
            drive_data_dir=sft_cfg.drive_data_dir,
        )
        validate_partitions(
            sft_train, grpo_train, clean_train, val_set,
            require_uniform_quotas=(sft_cfg.partition_mode == "uniform"),
        )
        del base_model
        torch.cuda.empty_cache()

    model, processor, compute_dtype, _ = load_thinker_model(
        model_cfg, seed=sft_cfg.seed, padding_side="right", use_sft_model=False, attach_lora=True
    )
    cue_loss_weights = inverse_frequency_cue_weights(sft_train, alpha=sft_cfg.cue_weight_alpha)
    cue_weight_lookup = cue_loss_weights.set_index(["cue", "label"]).weight.to_dict()

    output_dir = sft_cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cue_loss_weights.to_csv(output_dir / "cue_loss_weights.csv", index=False)
    (output_dir / "experiment.json").write_text(
        json.dumps(
            dict(
                model=model_cfg.model_id,
                seed=sft_cfg.seed,
                epochs=sft_cfg.epochs,
                split_manifest=split_manifest,
                prompt_version=sft_cfg.prompt_version,
                gender_input="effective_gender",
                cues=list(CUES),
                cue_columns=dict(CUE_COLUMNS),
                global_cues=sorted(GLOBAL_CUES),
                val_sha256=hashlib.sha256(val_csv.read_bytes()).hexdigest(),
                quartiles=quartiles,
                loss=dict(
                    name=(
                        "equal_multitask_cue_and_reasoning_ce"
                        if sft_cfg.equal_multitask_cue_loss
                        else (
                            "unweighted_assistant_ce"
                            if sft_cfg.cue_weight_alpha == 0.0
                            else "tempered_inverse_frequency_inventory_token_ce"
                        )
                    ),
                    equal_multitask_cue_loss=bool(sft_cfg.equal_multitask_cue_loss),
                    cue_weight_alpha=float(sft_cfg.cue_weight_alpha),
                    cue_task_share=float(sft_cfg.cue_task_share) if sft_cfg.equal_multitask_cue_loss else None,
                    weights=cue_loss_weights.to_dict("records"),
                ),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    before_metrics = None
    if sft_cfg.evaluate_before:
        before_metrics = evaluate_sft_validation(
            model, processor, val_set, "before", output_dir,
            compute_dtype=compute_dtype, batch_size=sft_cfg.eval_batch_size,
            max_new_tokens=sft_cfg.max_new_tokens,
            data_dir=sft_cfg.data_dir, drive_data_dir=sft_cfg.drive_data_dir,
        )

    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=sft_cfg.epochs,
        per_device_train_batch_size=sft_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
        learning_rate=sft_cfg.learning_rate,
        lr_scheduler_type=sft_cfg.lr_scheduler_type,
        weight_decay=sft_cfg.weight_decay,
        max_grad_norm=sft_cfg.max_grad_norm,
        logging_steps=sft_cfg.logging_steps,
        save_strategy="epoch",
        save_total_limit=sft_cfg.save_total_limit,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        seed=sft_cfg.seed,
    )
    trainer = build_cue_weighted_sft_trainer(
        model=model,
        training_args=training_args,
        train_dataset=SERDataset(sft_train),
        data_collator=SERCollator(
            processor,
            cue_weight_lookup,
            equal_multitask=sft_cfg.equal_multitask_cue_loss,
            cue_task_share=sft_cfg.cue_task_share,
        ),
    )
    trainer.train()
    trainer.save_model(str(sft_cfg.final_adapter_dir))
    processor.save_pretrained(str(sft_cfg.final_adapter_dir))

    after_metrics = evaluate_sft_validation(
        model, processor, val_set, "after", output_dir,
        compute_dtype=compute_dtype, batch_size=sft_cfg.eval_batch_size,
        max_new_tokens=sft_cfg.max_new_tokens,
        data_dir=sft_cfg.data_dir, drive_data_dir=sft_cfg.drive_data_dir,
    )
    return {
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "adapter_dir": sft_cfg.final_adapter_dir,
        "output_dir": output_dir,
    }
