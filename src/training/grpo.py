"""Stage-2 Multimodal Hybrid Cue-Split AudioGRPOTrainer with colocated 24 GB vLLM rollouts.

100% mathematically identical to ser_grpo_v3.ipynb:
  - Rollouts: generated via VLLMRolloutEngine (with live LoRA sync & sleep mode) or HF fallback,
    producing exact completion_token_ids tensors.
  - Rewards & Credit Assignment: exact R1 (z-score) + R2 (per-cue centered, chain-rule span-sum L) + R3.
  - Loss & KL: exact completion_logps teacher-forcing on Qwen2_5OmniThinkerForConditionalGeneration,
    ratio = exp(logp - logp.detach()), L/n-scaled CUE_KL_ONLY against frozen sft_reference adapter.
"""

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score

from src.config import (
    CUE_COLUMNS,
    CUES,
    EMOTIONS,
    GLOBAL_CUES,
    GRPOConfig as AgoraGRPOConfig,
    ModelConfig,
    RewardConfig,
    VALID_EMOTIONS,
    VLLMConfig,
)
from src.data.dataset import prepare_grpo_dataset
from src.models.qwen_omni import generation_eos_ids, load_thinker_model, reference_policy
from src.models.vllm_engine import VLLMRolloutEngine
from src.prompts.templates import PERSONA
from src.rewards.composite import GRPORewardCalibrator, validate_grpo_reward_mathematics
from src.rewards.parser import parse_completion
from src.training.callbacks import build_adapter_update_monitor
from src.training.credit_assignment import (
    GRPO_ADVANTAGE_EPS,
    R2_CUE_NAMES,
    build_cue_token_advantages,
    build_span_split_token_advantages,
    check_mask_and_loss_math,
    completion_boundary_mask,
    completion_logps,
    compute_span_split_advantages,
    cue_group_advantages,
    fixed_completion_loss,
    group_standardize,
    reduce_completion_loss,
    validate_credit_assignment,
)
from src.utils.audio import load_audio
from src.utils.io import load_manifest, stage_clean_data, validate_grpo_partition


def pad_token_id_lists(token_id_lists: List[List[int]], pad_token_id: int, device: Any) -> Any:
    """Right-pad variable-length vLLM completion token ID lists into a rectangular [B*G, T_max] tensor."""
    import torch

    max_len = max((len(seq) for seq in token_id_lists), default=1)
    max_len = max(1, max_len)
    padded = torch.full((len(token_id_lists), max_len), int(pad_token_id), dtype=torch.long, device=device)
    for i, seq in enumerate(token_id_lists):
        if seq:
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return padded


def build_audio_grpo_trainer(
    model: Any,
    processor: Any,
    *,
    training_args: Any,
    reward_calibrator: GRPORewardCalibrator,
    train_dataset: Any,
    compute_dtype: Any,
    reference_adapter_name: Optional[str] = "sft_reference",
    use_split_advantage: bool = True,
    use_split_r2: bool = True,
    cue_kl_only: bool = True,
    vllm_engine: Optional[VLLMRolloutEngine] = None,
) -> Any:
    """Build AudioGRPOTrainer (subclassing trl.GRPOTrainer) with vLLM rollout support and exact V4 math."""
    import torch
    from trl import GRPOTrainer

    class AudioGRPOTrainer(GRPOTrainer):
        loss_is_scaled_for_ga = True

        def __init__(self, *args, **kwargs):
            kwargs.setdefault("processing_class", processor.tokenizer)
            super().__init__(*args, **kwargs)
            self.processor = processor
            self.compute_dtype = compute_dtype
            self.reward_calibrator = reward_calibrator
            self.vllm_engine = vllm_engine
            if self.num_iterations != 1:
                raise ValueError("This on-policy loss requires num_iterations=1.")
            self.use_split_advantage = bool(use_split_advantage)
            self.use_split_r2 = bool(use_split_r2)
            self.cue_kl_only = bool(cue_kl_only)
            if self.cue_kl_only and not self.use_split_r2:
                raise ValueError("cue_kl_only=True requires use_split_r2=True.")
            validate_credit_assignment(self.args.loss_type, self.use_split_advantage, self.use_split_r2)
            self.add_callback(build_adapter_update_monitor(reference_adapter_name))
            self.reference_adapter_name = reference_adapter_name
            if reference_adapter_name is not None:
                self.model.set_requires_grad(reference_adapter_name, False)

        def _compute_advantages(self, rewards_per_func: List[torch.Tensor]) -> torch.Tensor:
            g = self.args.num_generations
            loss_type = self.args.loss_type
            if loss_type == "hybrid":
                raise ValueError("Hybrid requires component traces and cue-local advantages, not a scalar reward.")
            weights = getattr(self.args, "reward_weights", None) or [1.0] * len(rewards_per_func)
            total_advantages = None
            for r, w in zip(rewards_per_func, weights):
                r_grouped = r.view(-1, g)
                mean = r_grouped.mean(dim=-1, keepdim=True)
                if loss_type == "grpo":
                    adv_std = r_grouped.std(dim=-1, keepdim=True)
                    adv_k = torch.nan_to_num((r_grouped - mean) / (adv_std + 1e-8), nan=0.0)
                else:
                    adv_k = torch.nan_to_num(r_grouped - mean, nan=0.0)
                weighted_adv = w * adv_k
                total_advantages = weighted_adv if total_advantages is None else (total_advantages + weighted_adv)
            return total_advantages.view(-1)

        def _generate_and_score_completions(self, inputs: Dict[str, Any]):
            """Generate per-row gender-conditioned rollouts (vLLM or HF), score R1/R2/R3, and split token credit."""
            g = self.args.num_generations
            device = self.accelerator.device

            if isinstance(inputs, list):
                inputs = {k: [item[k] for item in inputs] for k in inputs[0].keys()}

            if len(inputs["audio_path"]) % g:
                raise ValueError("The local rollout batch must contain complete GRPO groups.")
            for start in range(0, len(inputs["audio_path"]), g):
                for key in ("audio_path", "label", "effective_gender", "prompt"):
                    group = inputs[key][start : start + g]
                    if any(value != group[0] for value in group):
                        raise ValueError(f"Misaligned GRPO group for {key} at row {start}.")

            formatted_prompts = []
            for prompt_text, audio_p in zip(inputs["prompt"], inputs["audio_path"]):
                messages = [
                    {"role": "system", "content": PERSONA},
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": audio_p},
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                ]
                formatted_prompts.append(
                    self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                )

            # Load each unique group audio once, then replicate for processor inputs
            unique_audio_paths = inputs["audio_path"][::g]
            unique_prompts = formatted_prompts[::g]
            unique_audios = [load_audio(a) for a in unique_audio_paths]
            raw_audios = [wav for wav in unique_audios for _ in range(g)]

            orig_padding_side = self.processor.tokenizer.padding_side
            self.processor.tokenizer.padding_side = "left"
            processed = self.processor(
                text=formatted_prompts,
                audio=raw_audios,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).to(device)
            expanded_inputs = dict(processed)
            del processed, raw_audios

            prompt_input_ids = expanded_inputs["input_ids"]
            prompt_len = prompt_input_ids.shape[1]
            replicated_kwargs = {k: v for k, v in inputs.items() if k != "prompt"}

            unwrapped_model = self.accelerator.unwrap_model(self.model)
            eos_ids = generation_eos_ids(self.processor.tokenizer)
            pad_id = self.processor.tokenizer.pad_token_id

            if self.vllm_engine is not None:
                # Fast 24 GB-safe vLLM rollout path: encodes each unique audio ONCE (n=G) with live LoRA sync
                try:
                    token_id_lists, _ = self.vllm_engine.generate_rollouts(
                        unwrapped_model,
                        unique_prompts,
                        unique_audios,
                        num_generations=g,
                        temperature=float(self.args.temperature),
                        max_tokens=int(self.args.max_completion_length),
                        stop_token_ids=eos_ids,
                        adapter_name="default",
                    )
                finally:
                    self.processor.tokenizer.padding_side = orig_padding_side
                completion_ids = pad_token_id_lists(token_id_lists, pad_id, device)
                generated_ids = torch.cat([prompt_input_ids, completion_ids], dim=1)
            else:
                # Exact HuggingFace fallback matching ser_grpo_v3.ipynb
                unwrapped_model.eval()
                if hasattr(unwrapped_model, "gradient_checkpointing_disable"):
                    unwrapped_model.gradient_checkpointing_disable()
                unwrapped_model.config.use_cache = True
                if hasattr(unwrapped_model.config, "get_text_config"):
                    unwrapped_model.config.get_text_config().use_cache = True
                try:
                    with torch.no_grad(), torch.autocast("cuda", dtype=self.compute_dtype):
                        generated_ids = unwrapped_model.generate(
                            **expanded_inputs,
                            use_cache=True,
                            max_new_tokens=self.args.max_completion_length,
                            do_sample=True,
                            temperature=self.args.temperature,
                            pad_token_id=pad_id,
                            eos_token_id=eos_ids,
                        )
                finally:
                    self.processor.tokenizer.padding_side = orig_padding_side
                    if hasattr(unwrapped_model, "gradient_checkpointing_enable"):
                        unwrapped_model.gradient_checkpointing_enable()
                    unwrapped_model.config.use_cache = False
                    if hasattr(unwrapped_model.config, "get_text_config"):
                        unwrapped_model.config.get_text_config().use_cache = False
                    unwrapped_model.train()
                completion_ids = generated_ids[:, prompt_len:]

            completion_mask = completion_boundary_mask(completion_ids, eos_ids, pad_id)
            completions = self.processor.tokenizer.batch_decode(completion_ids, skip_special_tokens=False)
            expanded_prompts = list(inputs["prompt"])
            total_rollouts = len(completions)

            parsed_completions = [parse_completion(c) for c in completions]
            replicated_kwargs["parsed_completions"] = parsed_completions

            parsed_predictions = [
                p["answer_norm"] if p["is_valid_emotion"] else "invalid" for p in parsed_completions
            ]
            counts = Counter(parsed_predictions)
            dist_metrics = {f"dist/{emo}": counts.get(emo, 0) / total_rollouts for emo in EMOTIONS}
            invalid_count = counts.get("invalid", 0) + sum(
                counts.get(k, 0) for k in counts if k not in VALID_EMOTIONS and k != "invalid"
            )
            dist_metrics["dist/invalid_format"] = invalid_count / total_rollouts
            pred_entropy = 0.0
            for emo in EMOTIONS:
                p = dist_metrics[f"dist/{emo}"]
                if p > 0.0:
                    pred_entropy -= p * math.log2(p)
            dist_metrics["metrics/prediction_entropy"] = pred_entropy

            perf_metrics = {}
            raw_labels = replicated_kwargs.get("label", [])
            if raw_labels and len(raw_labels) == len(parsed_predictions):
                targets = [str(lbl).lower().strip() for lbl in raw_labels]
                perf_metrics["metrics/acc_step"] = sum(
                    1 for p, t in zip(parsed_predictions, targets) if p == t
                ) / max(len(targets), 1)
                if not hasattr(self, "_f1_window_preds"):
                    self._f1_window_preds = []
                    self._f1_window_targets = []
                self._f1_window_preds.extend(parsed_predictions)
                self._f1_window_targets.extend(targets)
                if len(self._f1_window_preds) > 160:
                    self._f1_window_preds = self._f1_window_preds[-160:]
                    self._f1_window_targets = self._f1_window_targets[-160:]
                perf_metrics["metrics/f1_macro_running"] = float(
                    f1_score(
                        self._f1_window_targets,
                        self._f1_window_preds,
                        labels=list(EMOTIONS),
                        average="macro",
                        zero_division=0,
                    )
                )
                perf_metrics["metrics/f1_macro_step"] = float(
                    f1_score(targets, parsed_predictions, labels=list(EMOTIONS), average="macro", zero_division=0)
                )

            rewards_per_func = []
            reward_trace = []
            for r_func in self.reward_funcs:
                scores = r_func(
                    prompts=expanded_prompts,
                    completions=completions,
                    _reward_trace_sink=reward_trace,
                    **replicated_kwargs,
                )
                assert len(scores) == total_rollouts
                rewards_per_func.append(torch.tensor(scores, dtype=torch.float32, device=device))
            assert len(reward_trace) == total_rollouts

            rw = self.reward_calibrator.config.reward_weights
            span_adv_dict = compute_span_split_advantages(
                reward_trace, g, reward_weights=rw, device=device, loss_type=self.args.loss_type
            )
            advantages = (
                span_adv_dict["adv_total"]
                if self.args.loss_type == "hybrid"
                else self._compute_advantages(rewards_per_func)
            )
            token_advantages, acoustic_span_mask, emotion_span_mask, span_has_boundary = (
                build_span_split_token_advantages(
                    completion_ids,
                    completion_mask,
                    span_adv_dict["adv_acoustic"],
                    span_adv_dict["adv_emotion"],
                    adv_total=advantages,
                    tokenizer=self.processor.tokenizer,
                    use_split_advantage=self.use_split_advantage,
                )
            )

            cue_advantages, cue_token_counts, cue_mask_status = {}, {}, []
            cue_kl_scale = torch.zeros_like(token_advantages)
            if self.use_split_r2:
                cue_advantages = cue_group_advantages(reward_trace, g, rw["r2"], self.args.loss_type)
                (
                    token_advantages,
                    cue_kl_scale,
                    acoustic_span_mask,
                    emotion_span_mask,
                    span_has_boundary,
                    cue_token_counts,
                    cue_mask_status,
                ) = build_cue_token_advantages(
                    completion_ids,
                    completion_mask,
                    cue_advantages,
                    span_adv_dict["adv_r3"],
                    span_adv_dict["adv_emotion"],
                    self.processor.tokenizer,
                )

            if self.accelerator.is_main_process:
                trace_path = Path(self.args.output_dir) / "grpo_groups.jsonl"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                labels = [str(x).lower().strip() for x in replicated_kwargs["label"]]
                speakers = replicated_kwargs.get("speaker", [])
                eff_genders = replicated_kwargs.get("effective_gender", [])

                step_now = int(self.state.global_step)
                if getattr(self, "_last_logged_group_step", None) != step_now:
                    self._last_logged_group_step = step_now
                    self._step_group_offset = 0

                group_rows = []
                for start in range(0, total_rollouts, g):
                    end = start + g
                    group_preds = parsed_predictions[start:end]
                    group_labels = labels[start:end]
                    group_rewards = reward_trace[start:end]
                    target = group_labels[0]
                    correct = [pred == target for pred in group_preds]
                    totals = [r["total"] for r in group_rewards]
                    r1_values = [r["r1"] for r in group_rewards]
                    r2_values = [r["r2"] for r in group_rewards]
                    best_idx = int(np.argmax(totals))
                    has_correct = any(correct)
                    mixed = has_correct and (not all(correct))
                    reward_margin = (
                        min(t for t, ok in zip(totals, correct) if ok)
                        - max(t for t, ok in zip(totals, correct) if not ok)
                    ) if mixed else None

                    row = {
                        "step_before_update": step_now,
                        "group_index": self._step_group_offset + (start // g),
                        "speaker": str(speakers[start]) if len(speakers) == total_rollouts else None,
                        "effective_gender": str(eff_genders[start]) if len(eff_genders) == total_rollouts else None,
                        "target": target,
                        "audio_path": str(replicated_kwargs["audio_path"][start]),
                        "formatted_prompt": formatted_prompts[start],
                        "completion_token_ids": [
                            ids[mask.bool()].detach().cpu().tolist()
                            for ids, mask in zip(completion_ids[start:end], completion_mask[start:end])
                        ],
                        "reward_weights": dict(rw),
                        "reward_mode": self.reward_calibrator.mode,
                        "inverse_frequency_alpha": (
                            self.reward_calibrator.alpha
                            if self.reward_calibrator.mode == "inverse_frequency"
                            else None
                        ),
                        "use_distance_based_payoff": bool(self.reward_calibrator.config.use_distance_based_payoff),
                        "acoustic_reward_mode": self.reward_calibrator.acoustic_mode,
                        "acoustic_normalization_scale": self.reward_calibrator.normalization_scale,
                        "use_split_advantage": bool(self.use_split_advantage),
                        "use_split_r2": self.use_split_r2,
                        "cue_kl_only": bool(self.cue_kl_only),
                        "rollout_backend": "vllm" if self.vllm_engine is not None else "hf",
                        "loss_type": self.args.loss_type,
                        "advantage_version": "cue_hybrid_v1" if self.args.loss_type == "hybrid" else "cue_grpo_v1",
                        "emotion_advantage_normalization": (
                            "group_std" if self.args.loss_type in ("grpo", "hybrid") else "center"
                        ),
                        "r2_advantage_normalization": "group_std" if self.args.loss_type == "grpo" else "center",
                        "loss_reduction": (
                            "fixed_max_length" if self.args.loss_type == "dr_grpo" else "mean_of_completion_means"
                        ),
                        "cue_token_reduction": "chain_rule_span_sum" if self.use_split_r2 else None,
                        "cue_advantage_scale": float(rw["r2"]) / len(R2_CUE_NAMES) if self.use_split_r2 else None,
                        "scalar_advantage_is_diagnostic": self.args.loss_type == "hybrid",
                        "advantage_std_correction": 1,
                        "advantage_eps": GRPO_ADVANTAGE_EPS,
                        "r2_cues": [r["r2_cues"] for r in group_rewards],
                        "advantages_cues": {c: v[start:end].tolist() for c, v in cue_advantages.items()},
                        "cue_value_tokens": {c: v[start:end] for c, v in cue_token_counts.items()},
                        "cue_mask_status": cue_mask_status[start:end],
                        "token_advantages": [
                            v[m.bool()].detach().cpu().tolist()
                            for v, m in zip(token_advantages[start:end], completion_mask[start:end])
                        ],
                        "reference_adapter": self.reference_adapter_name,
                        "learning_rate": self.args.learning_rate,
                        "num_generations": g,
                        "max_completion_length": self.args.max_completion_length,
                        "predictions": group_preds,
                        "correct": correct,
                        "r1": r1_values,
                        "r2": r2_values,
                        "r3": [r["r3"] for r in group_rewards],
                        "total_rewards": totals,
                        "advantages": [float(x) for x in advantages[start:end].detach().cpu().tolist()],
                        "advantages_acoustic": [
                            float(x) for x in span_adv_dict["adv_acoustic"][start:end].detach().cpu().tolist()
                        ],
                        "advantages_emotion": [
                            float(x) for x in span_adv_dict["adv_emotion"][start:end].detach().cpu().tolist()
                        ],
                        "advantages_r1": [float(x) for x in span_adv_dict["adv_r1"][start:end].detach().cpu().tolist()],
                        "advantages_r2": [float(x) for x in span_adv_dict["adv_r2"][start:end].detach().cpu().tolist()],
                        "advantages_r3": [float(x) for x in span_adv_dict["adv_r3"][start:end].detach().cpu().tolist()],
                        "acoustic_span_tokens": [
                            int(x) for x in acoustic_span_mask[start:end].sum(dim=1).detach().cpu().tolist()
                        ],
                        "emotion_span_tokens": [
                            int(x) for x in emotion_span_mask[start:end].sum(dim=1).detach().cpu().tolist()
                        ],
                        "unique_emotions": len(set(group_preds)),
                        "all_neutral": all(p == "neutral" for p in group_preds),
                        "neutral_fraction": group_preds.count("neutral") / g,
                        "has_correct": has_correct,
                        "mixed_correct_wrong": mixed,
                        "best_reward_correct": correct[best_idx],
                        "correct_wrong_margin": reward_margin,
                        "r1_std": float(np.std(r1_values)),
                        "r2_std": float(np.std(r2_values)),
                        "reward_std": float(np.std(totals)),
                        "all_same_prediction": len(set(group_preds)) == 1,
                        "completions": completions[start:end],
                    }
                    group_rows.append(row)

                with trace_path.open("a", encoding="utf-8") as f:
                    for row in group_rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._step_group_offset += len(group_rows)

                live_metrics = {
                    "reward/R1_accuracy": float(np.mean([r["r1"] for r in reward_trace])),
                    "reward/R2_acoustic": float(np.mean([r["r2"] for r in reward_trace])),
                    "reward/R3_format": float(np.mean([r["r3"] for r in reward_trace])),
                    "reward/total_weighted": float(np.mean([r["total"] for r in reward_trace])),
                    "debug/group_all_neutral": float(np.mean([r["all_neutral"] for r in group_rows])),
                    "debug/group_has_correct": float(np.mean([r["has_correct"] for r in group_rows])),
                    "debug/group_mixed": float(np.mean([r["mixed_correct_wrong"] for r in group_rows])),
                    "debug/group_unique_emotions": float(np.mean([r["unique_emotions"] for r in group_rows])),
                    "debug/R1_mean": float(np.mean([r["r1"] for r in reward_trace])),
                    "debug/R2_mean": float(np.mean([r["r2"] for r in reward_trace])),
                    "debug/R1_group_std": float(np.mean([r["r1_std"] for r in group_rows])),
                    "debug/R2_group_std": float(np.mean([r["r2_std"] for r in group_rows])),
                    **dist_metrics,
                    **perf_metrics,
                }
                if self.use_split_r2:
                    for c in R2_CUE_NAMES:
                        live_metrics[f"cue/{c}/zero_advantage_group_fraction"] = float(
                            np.mean(np.max(np.abs(cue_advantages[c].reshape(-1, g)), axis=1) < 1e-12)
                        )
                        live_metrics[f"cue/{c}/unmapped_fraction"] = float(
                            np.mean([r[c] != "ok" for r in cue_mask_status])
                        )
                self.log(live_metrics)

            output_dict = {
                "generated_ids": generated_ids,
                "completion_ids": completion_ids,
                "advantages": advantages,
                "token_advantages": token_advantages,
                "cue_kl_scale": cue_kl_scale,
                "acoustic_span_mask": acoustic_span_mask,
                "emotion_span_mask": emotion_span_mask,
                "prompt_input_ids": prompt_input_ids,
                "prompt_attention_mask": expanded_inputs["attention_mask"],
                "completion_mask": completion_mask,
            }
            for k, v in expanded_inputs.items():
                if k not in ("input_ids", "attention_mask") and isinstance(v, torch.Tensor):
                    output_dict[k] = v

            del generated_ids, expanded_inputs
            return output_dict

        def compute_loss(
            self,
            model,
            inputs: Dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: int = None,
            **kwargs,
        ):
            unwrapped_model = self.accelerator.unwrap_model(model)
            per_token_logps = completion_logps(model, inputs)
            mask = inputs["completion_mask"].float()
            ratio = torch.exp(per_token_logps - per_token_logps.detach())
            token_advantages = inputs.get("token_advantages")
            if token_advantages is None:
                if self.args.loss_type == "hybrid":
                    raise ValueError("Hybrid loss requires cue-specific token_advantages.")
                token_advantages = inputs["advantages"]
            if token_advantages.ndim == 1:
                token_advantages = token_advantages.unsqueeze(-1)
            policy_loss = -ratio * token_advantages

            if self.args.beta != 0.0:
                with torch.no_grad():
                    with reference_policy(unwrapped_model, self.reference_adapter_name):
                        ref_per_token_logps = completion_logps(unwrapped_model, inputs)
                log_ratio = ref_per_token_logps - per_token_logps
                kl = torch.exp(log_ratio) - log_ratio - 1.0
                if self.cue_kl_only:
                    cue_kl_scale = inputs.get("cue_kl_scale")
                    if cue_kl_scale is None:
                        raise ValueError("cue_kl_only=True requires cue_kl_scale in inputs.")
                    kl_weights = cue_kl_scale
                else:
                    kl_weights = torch.ones_like(kl)
                loss_terms = policy_loss + self.args.beta * kl_weights * kl
            else:
                kl = None
                loss_terms = policy_loss

            accumulation = self.current_gradient_accumulation_steps if model.training else 1
            total_loss = reduce_completion_loss(
                loss_terms, mask, self.args.loss_type, self.args.max_completion_length, accumulation
            )

            mode = "train" if model.training else "eval"
            self._metrics[mode]["diagnostic/policy_loss_fixed"].append(
                float(fixed_completion_loss(policy_loss.detach(), mask, self.args.max_completion_length))
            )
            if kl is not None:
                valid_tokens = mask.sum().clamp(min=1)
                self._metrics[mode]["diagnostic/kl_per_token"].append(
                    float((kl.detach() * mask).sum() / valid_tokens)
                )
                if "cue_kl_scale" in inputs:
                    per_seq_len = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                    cue_span_weights = (inputs["cue_kl_scale"].float() * mask) / per_seq_len
                    denom_cues = max(1.0, float(mask.shape[0] * len(R2_CUE_NAMES)))
                    self._metrics[mode]["diagnostic/kl_cue_values"].append(
                        float((kl.detach() * cue_span_weights).sum() / denom_cues)
                    )
                if "acoustic_span_mask" in inputs and "emotion_span_mask" in inputs:
                    ac_mask = inputs["acoustic_span_mask"].float()
                    em_mask = inputs["emotion_span_mask"].float()
                    self._metrics[mode]["diagnostic/kl_acoustic_span"].append(
                        float((kl.detach() * ac_mask).sum() / ac_mask.sum().clamp(min=1))
                    )
                    self._metrics[mode]["diagnostic/kl_emotion_span"].append(
                        float((kl.detach() * em_mask).sum() / em_mask.sum().clamp(min=1))
                    )
            return (total_loss, {"loss": total_loss}) if return_outputs else total_loss

    return AudioGRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[reward_calibrator],
        train_dataset=train_dataset,
    )


def run_grpo_training(
    grpo_cfg: AgoraGRPOConfig,
    model_cfg: ModelConfig,
    reward_cfg: RewardConfig,
    vllm_cfg: VLLMConfig,
) -> Dict[str, Any]:
    """End-to-end Stage-2 GRPO v3 training with 24 GB-safe colocated vLLM rollouts and validation."""
    import torch
    from trl import GRPOConfig as TRLGRPOConfig

    from src.training.evaluation import evaluate_grpo_validation, plot_grpo_training_curves

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    check_mask_and_loss_math()
    validate_grpo_reward_mathematics(
        use_zero_mean=reward_cfg.use_zero_mean,
        use_map=reward_cfg.use_map,
        use_inverse_frequency=reward_cfg.use_inverse_frequency,
        use_distance_based_payoff=reward_cfg.use_distance_based_payoff,
        inverse_frequency_alpha=reward_cfg.inverse_frequency_alpha,
    )

    if grpo_cfg.drive_data_dir is not None and Path(grpo_cfg.drive_data_dir).resolve() != Path(grpo_cfg.data_dir).resolve():
        stage_clean_data(grpo_cfg.drive_data_dir, grpo_cfg.data_dir)

    clean_dir = Path(grpo_cfg.data_dir) / "emo_tag_clean"
    train_csv = clean_dir / "clean_train_set_with_cues.csv"
    val_csv = clean_dir / "clean_val_set_with_cues.csv"
    quartiles_json = clean_dir / "gender_acoustic_quartiles.json"

    for req in (train_csv, val_csv, quartiles_json, grpo_cfg.grpo_csv, grpo_cfg.split_meta):
        if not Path(req).is_file():
            raise FileNotFoundError(f"Required preprocessing/split file not found: {req}")

    clean_train = load_manifest(train_csv, "train", grpo_cfg.data_dir)
    grpo_train = load_manifest(grpo_cfg.grpo_csv, "train", grpo_cfg.data_dir)
    val_set = load_manifest(val_csv, "val", grpo_cfg.data_dir)
    sft_check = load_manifest(grpo_cfg.sft_csv, "train", grpo_cfg.data_dir) if grpo_cfg.sft_csv.is_file() else None
    split_manifest = json.loads(grpo_cfg.split_meta.read_text(encoding="utf-8"))
    quartiles = json.loads(quartiles_json.read_text(encoding="utf-8"))
    validate_grpo_partition(
        grpo_train,
        val_set,
        sft_df=sft_check,
        clean_train_df=clean_train,
        require_uniform_quotas=(grpo_cfg.partition_mode == "uniform"),
    )

    calibrator = GRPORewardCalibrator(grpo_train, val_set, config=reward_cfg)
    ser_train_dataset = prepare_grpo_dataset(grpo_train)

    gen_batch_size = grpo_cfg.per_device_train_batch_size * grpo_cfg.steps_per_generation
    if gen_batch_size % grpo_cfg.num_generations != 0:
        raise ValueError("per_device_train_batch_size * steps_per_generation must be divisible by num_generations.")
    if (grpo_cfg.per_device_train_batch_size * grpo_cfg.gradient_accumulation_steps) % grpo_cfg.num_generations != 0:
        raise ValueError("per_device_train_batch_size * gradient_accumulation_steps must be divisible by num_generations.")
    groups_per_step = (grpo_cfg.per_device_train_batch_size * grpo_cfg.gradient_accumulation_steps) // grpo_cfg.num_generations
    expected_steps = math.ceil(len(ser_train_dataset) * grpo_cfg.num_train_epochs / groups_per_step)

    # 1. Initialize vLLM rollout engine FIRST (and immediately put to sleep on 24 GB VRAM)
    vllm_engine: Optional[VLLMRolloutEngine] = None
    if vllm_cfg.enabled:
        vllm_engine = VLLMRolloutEngine(model_cfg.model_id, vllm_cfg, seed=grpo_cfg.seed)
        vllm_engine.initialize()

    # 2. Load 4-bit QLoRA training Thinker + frozen sft_reference adapter
    model, processor, compute_dtype, active_ref = load_thinker_model(
        model_cfg,
        seed=grpo_cfg.seed,
        padding_side="left",
        sft_adapter_path=grpo_cfg.sft_adapter_path if grpo_cfg.use_sft_model else None,
        use_sft_model=grpo_cfg.use_sft_model,
        attach_lora=True,
        reference_adapter_name="sft_reference" if grpo_cfg.use_sft_model else None,
    )

    grpo_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_meta = dict(
        model=model_cfg.model_id,
        use_sft_model=grpo_cfg.use_sft_model,
        sft_adapter_path=str(grpo_cfg.sft_adapter_path) if grpo_cfg.use_sft_model else None,
        reference_adapter=active_ref,
        rollout_backend="vllm" if vllm_engine is not None else "hf",
        seed=grpo_cfg.seed,
        num_train_epochs=grpo_cfg.num_train_epochs,
        per_device_train_batch_size=grpo_cfg.per_device_train_batch_size,
        steps_per_generation=grpo_cfg.steps_per_generation,
        gradient_accumulation_steps=grpo_cfg.gradient_accumulation_steps,
        num_generations=grpo_cfg.num_generations,
        expected_optimizer_steps=expected_steps,
        temperature=grpo_cfg.temperature,
        learning_rate=grpo_cfg.learning_rate,
        warmup_steps=grpo_cfg.warmup_steps,
        lr_scheduler_type=grpo_cfg.lr_scheduler_type,
        max_grad_norm=grpo_cfg.max_grad_norm,
        beta=grpo_cfg.beta,
        max_completion_length=grpo_cfg.max_completion_length,
        loss_type=grpo_cfg.loss_type,
        use_split_advantage=grpo_cfg.use_split_advantage,
        use_split_r2=grpo_cfg.use_split_r2,
        cue_kl_only=grpo_cfg.cue_kl_only,
        reward_mode=calibrator.mode,
        r1_inverse_frequency_alpha=calibrator.r1_alpha if calibrator.mode == "inverse_frequency" else None,
        inverse_frequency_alpha=calibrator.alpha if calibrator.mode == "inverse_frequency" else None,
        acoustic_reward_mode=calibrator.acoustic_mode,
        use_distance_based_payoff=reward_cfg.use_distance_based_payoff,
        reward_weights=dict(reward_cfg.reward_weights),
        prompt_version=grpo_cfg.prompt_version,
        gender_input="effective_gender",
        cues=list(CUES),
        cue_columns=dict(CUE_COLUMNS),
        global_cues=sorted(GLOBAL_CUES),
        grpo_rows=len(grpo_train),
        val_rows=len(val_set),
        grpo_sha256=hashlib.sha256(grpo_cfg.grpo_csv.read_bytes()).hexdigest(),
        val_sha256=hashlib.sha256(val_csv.read_bytes()).hexdigest(),
        split_manifest=split_manifest,
        quartiles=quartiles,
    )
    (grpo_cfg.output_dir / "experiment.json").write_text(json.dumps(experiment_meta, indent=2), encoding="utf-8")

    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = TRLGRPOConfig(
        output_dir=str(grpo_cfg.output_dir),
        remove_unused_columns=False,
        num_train_epochs=grpo_cfg.num_train_epochs,
        beta=grpo_cfg.beta,
        loss_type=grpo_cfg.loss_type,
        num_iterations=1,
        save_strategy="steps",
        save_steps=grpo_cfg.save_steps,
        save_total_limit=grpo_cfg.save_total_limit,
        learning_rate=grpo_cfg.learning_rate,
        lr_scheduler_type=grpo_cfg.lr_scheduler_type,
        warmup_steps=grpo_cfg.warmup_steps,
        max_grad_norm=grpo_cfg.max_grad_norm,
        per_device_train_batch_size=grpo_cfg.per_device_train_batch_size,
        steps_per_generation=grpo_cfg.steps_per_generation,
        gradient_accumulation_steps=grpo_cfg.gradient_accumulation_steps,
        multi_objective_aggregation="normalize_then_sum",
        num_generations=grpo_cfg.num_generations,
        temperature=grpo_cfg.temperature,
        max_completion_length=grpo_cfg.max_completion_length,
        logging_steps=grpo_cfg.logging_steps,
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="none",
        seed=grpo_cfg.seed,
    )

    trainer = build_audio_grpo_trainer(
        model=model,
        processor=processor,
        training_args=training_args,
        reward_calibrator=calibrator,
        train_dataset=ser_train_dataset,
        compute_dtype=compute_dtype,
        reference_adapter_name=active_ref,
        use_split_advantage=grpo_cfg.use_split_advantage,
        use_split_r2=grpo_cfg.use_split_r2,
        cue_kl_only=grpo_cfg.cue_kl_only,
        vllm_engine=vllm_engine,
    )

    try:
        if grpo_cfg.resume_checkpoint_dir:
            trainer.train(resume_from_checkpoint=grpo_cfg.resume_checkpoint_dir)
        else:
            trainer.train()

        grpo_cfg.final_adapter_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        unwrapped_model.save_pretrained(str(grpo_cfg.final_adapter_dir))
        processor.save_pretrained(str(grpo_cfg.final_adapter_dir))

        val_vllm = vllm_engine if (vllm_engine is not None and vllm_cfg.use_for_validation) else None
        after_predictions, after_metrics = evaluate_grpo_validation(
            unwrapped_model,
            processor,
            val_set,
            reward_calibrator=calibrator,
            compute_dtype=compute_dtype,
            stage="after",
            output_dir=grpo_cfg.output_dir,
            adapter_dir=grpo_cfg.final_adapter_dir,
            trainer_obj=trainer,
            train_df=grpo_train,
            quartiles=quartiles,
            model_id=model_cfg.model_id,
            sft_adapter_path=grpo_cfg.sft_adapter_path if grpo_cfg.use_sft_model else None,
            reference_adapter=active_ref,
            prompt_version=grpo_cfg.prompt_version,
            batch_size=grpo_cfg.eval_batch_size,
            max_new_tokens=grpo_cfg.eval_max_new_tokens,
            vllm_engine=val_vllm,
            data_dir=grpo_cfg.data_dir,
            drive_data_dir=grpo_cfg.drive_data_dir,
            plot=True,
        )
        plot_grpo_training_curves(
            trainer.state.log_history,
            save_path=grpo_cfg.output_dir / "grpo_training_curves.png",
        )
    finally:
        if vllm_engine is not None:
            vllm_engine.cleanup()

    return {
        "after_predictions": after_predictions,
        "after_metrics": after_metrics,
        "adapter_dir": grpo_cfg.final_adapter_dir,
        "output_dir": grpo_cfg.output_dir,
    }
