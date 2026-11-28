"""Post-training greedy validation, gender breakdown, GRPO model-health report, and plots (exact parity with V4 notebooks)."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm.auto import tqdm

from src.config import CUE_COLUMNS, CUE_LABELS, CUES, DEFAULT_REWARD_WEIGHTS, EMOTIONS, PROMPT_VERSION
from src.models.qwen_omni import generation_eos_ids
from src.models.vllm_engine import VLLMRolloutEngine
from src.prompts.templates import messages_for, student_instruction
from src.rewards.acoustic import (
    DISTANCE_EXTREME_ON_MODERATE_COST,
    DISTANCE_MIN_OPPOSITE_COST,
    DISTANCE_MODERATE_HIT,
    DISTANCE_ORDINAL_MARGIN,
    R2_CUE_COUNT,
    R2_NORM_SCALE,
)
from src.rewards.composite import GRPORewardCalibrator
from src.rewards.parser import parse_completion, parse_completion_strict
from src.training.sft import generate_completions
from src.utils.audio import load_audio
from src.utils.io import atomic_csv

INVENTORY_DIMENSIONS: Dict[str, Dict[str, Any]] = {
    cue: {"title": cue, "options": list(CUE_LABELS[cue])} for cue in CUES
}


def label_metrics(truth: Any, predicted: Any, labels: Any) -> Dict[str, Any]:
    if not len(truth):
        return dict(n=0, accuracy=None, macro_f1=None, weighted_f1=None, macro_recall=None, invalid_rate=None)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=list(labels), zero_division=0
    )
    total_support = int(support.sum())
    weighted_f1 = float(np.sum(f1 * support) / total_support) if total_support > 0 else 0.0
    return dict(
        n=len(truth),
        accuracy=float(accuracy_score(truth, predicted)),
        macro_f1=float(f1.mean()),
        weighted_f1=weighted_f1,
        macro_recall=float(recall.mean()),
        invalid_rate=float(np.mean(np.array(predicted) == "invalid")),
        per_class={
            label: dict(
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1=float(f1[i]),
                support=int(support[i]),
            )
            for i, label in enumerate(labels)
        },
    )


def gender_emotion_metrics(predictions: pd.DataFrame) -> Dict[str, Any]:
    groups = predictions.gender.where(predictions.gender.isin(["F", "M"]), "unknown")
    by_gender = {}
    for gender in ("F", "M", "unknown"):
        group = predictions.loc[groups == gender]
        by_gender[gender] = label_metrics(group.emotion, group.predicted_emotion, EMOTIONS)
    gap = {}
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        female, male = by_gender["F"][metric], by_gender["M"][metric]
        gap[metric] = female - male if female is not None and male is not None else None
    return dict(group_column="gender", groups=by_gender, female_minus_male=gap)


def score_validation(predictions: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "emotion": label_metrics(predictions.emotion, predictions.predicted_emotion, EMOTIONS),
        "emotion_by_gender": gender_emotion_metrics(predictions),
        "cues": {},
    }
    for cue in CUES:
        col = CUE_COLUMNS[cue]
        valid = predictions[col].notna()
        metrics = label_metrics(
            predictions.loc[valid, col], predictions.loc[valid, f"predicted_{cue}"], CUE_LABELS[cue]
        )
        metrics.update(total=len(predictions), missing_targets=int((~valid).sum()))
        report["cues"][cue] = metrics
    if "r_unified_reward" in predictions.columns:
        report["grpo_rewards"] = {
            "mean_r_unified": float(predictions["r_unified_reward"].mean()),
            "mean_r1": float(predictions["r1_reward"].mean()),
            "mean_r2": float(predictions["r2_acoustic_reward"].mean()),
            "std_r2": float(predictions["r2_acoustic_reward"].std(ddof=0)),
            "mean_r3": float(predictions["r3_format_reward"].mean()),
            "mean_exact_match_ratio": float(predictions["acoustic_exact_match_ratio"].mean()),
            "perfect_inventory_rate": float((predictions["acoustic_exact_match_ratio"] == 1.0).mean()),
            "valid_format_rate": float((predictions["valid_slots_count"] == len(CUES)).mean()),
        }
    return report


def evaluate_sft_validation(
    model: Any,
    processor: Any,
    frame: pd.DataFrame,
    stage: str,
    output_dir: Path,
    *,
    compute_dtype: Any,
    batch_size: int = 16,
    max_new_tokens: int = 512,
    data_dir: Optional[Path] = None,
    drive_data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with tqdm(total=len(frame), desc=f"SFT Validation ({stage}, batch={batch_size})", unit="example") as progress:
        for start in range(0, len(frame), batch_size):
            rows = frame.iloc[start : start + batch_size].to_dict("records")
            texts = generate_completions(
                model, processor, rows, compute_dtype=compute_dtype, max_new_tokens=max_new_tokens, teacher=False
            )
            for row, text in zip(rows, texts):
                emotion, inventory = parse_completion_strict(text)
                record = dict(row, completion=text, predicted_emotion=emotion)
                record.update({f"predicted_{cue}": inventory[cue] for cue in CUES})
                records.append(record)
            progress.update(len(rows))
    predictions = pd.DataFrame(records)
    metrics = score_validation(predictions)
    atomic_csv(predictions, output_dir / f"{stage}_validation_predictions.csv", data_dir, drive_data_dir)
    (output_dir / f"{stage}_validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    gender_report = metrics["emotion_by_gender"]
    summary = pd.DataFrame(
        [
            dict(gender=gender, **{key: values[key] for key in ("n", "accuracy", "macro_f1", "weighted_f1")})
            for gender, values in gender_report["groups"].items()
        ]
    )
    atomic_csv(summary, output_dir / f"{stage}_emotion_by_gender.csv", data_dir, drive_data_dir)
    return metrics


def _health_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _health_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_health_json(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _health_hash(value: Any) -> str:
    payload = json.dumps(_health_json(value), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _health_table(headers: List[str], rows: List[List[Any]]) -> str:
    def cell(v: Any) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}" if math.isfinite(v) else "N/A"
        return str(v).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(map(cell, headers)) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(cell, row)) + " |" for row in rows],
        ]
    )


def _health_classification(truth: Any, pred: Any, categories: Any) -> Dict[str, Any]:
    cats = list(categories)
    pairs = [(t, p if p in cats else "invalid") for t, p in zip(truth, pred) if t in cats]
    n = len(pairs)
    cm = {t: {p: 0 for p in cats + ["invalid"]} for t in cats}
    for t, p in pairs:
        cm[t][p] += 1
    rows, recalls, f1s, supports = [], [], [], []
    for c in cats:
        support = sum(cm[c].values())
        predicted = sum(cm[t][c] for t in cats)
        tp = cm[c][c]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else None
        f1 = 2 * tp / (support + predicted) if (support + predicted) else 0.0
        if support:
            recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        rows.append(
            [
                c,
                support,
                predicted,
                support / n if n else None,
                predicted / n if n else None,
                (predicted - support) / n if n else None,
                precision,
                recall,
                f1,
            ]
        )
    tc, pc = Counter(t for t, _ in pairs), Counter(p for _, p in pairs)
    accuracy = sum(t == p for t, p in pairs) / n if n else None
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / n) if n else None
    majority = max(tc.values()) / n if n else None
    independent = sum(tc[c] * pc[c] for c in cats) / n**2 if n else None
    return dict(
        n=n,
        accuracy=accuracy,
        macro_recall=float(np.mean(recalls)) if recalls else None,
        macro_f1=float(np.mean(f1s)) if n else None,
        weighted_f1=weighted_f1,
        majority=majority,
        independent=independent,
        invalid=pc["invalid"] / n if n else None,
        tv=sum(abs(tc[c] - pc[c]) for c in cats + ["invalid"]) / (2 * n) if n else None,
        rows=rows,
        confusion=cm,
    )


def _health_read_groups(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    groups, notes = [], []
    path = Path(path)
    if not path.exists():
        return [], [f"Rollout log unavailable: {path}."]
    discarded = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            g = json.loads(line)
            size = len(g["predictions"])
            if size < 2 or not isinstance(g["target"], str):
                raise ValueError("missing target or group smaller than two")
            for key in ("r1", "r2", "r3"):
                if len(g[key]) != size or not np.isfinite(np.asarray(g[key], dtype=float)).all():
                    raise ValueError(f"invalid {key} array")
            step = int(g["step_before_update"])
            if groups and step < int(groups[-1]["step_before_update"]):
                discarded += len(groups)
                groups = []
            groups.append(g)
        except (ValueError, TypeError, KeyError) as exc:
            notes.append(f"Skipped rollout line {line_no}: {exc}")
    if discarded:
        notes.append(f"Log steps restarted: excluded {discarded} earlier groups; using latest monotonic segment.")
    return groups, notes


def build_grpo_health_report(
    predictions_df: pd.DataFrame,
    groups: List[Dict[str, Any]],
    *,
    dimensions: Optional[Dict[str, Dict[str, Any]]] = None,
    emotion_categories: Any = EMOTIONS,
    reward_weights: Optional[Dict[str, float]] = None,
    cue_priors: Optional[Dict[str, Dict[str, float]]] = None,
    emotion_priors: Optional[Dict[str, float]] = None,
    cue_matrices: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    emotion_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    train_df: Optional[pd.DataFrame] = None,
    quantiles: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    log_notes: Optional[List[str]] = None,
    log_history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    if predictions_df.empty:
        raise ValueError("Cannot report model health on an empty validation result.")
    df = predictions_df.copy()
    dimensions = dimensions or INVENTORY_DIMENSIONS
    cats = list(emotion_categories)
    weights = {k: float((reward_weights or DEFAULT_REWARD_WEIGHTS)[k]) for k in ("r1", "r2", "r3")}
    cue_priors = cue_priors or {}
    emotion_priors = emotion_priors or {}
    cue_matrices = cue_matrices or {}
    notes = list(log_notes or [])
    findings: List[str] = []

    sections = [
        "# Model health and reward sensitivity (GRPO v3)",
        "All rates are fractions (0.25 = 25%). Validation uses greedy decoding; rollout statistics "
        "use sampled training completions at different updates.",
    ]

    em = _health_classification(df["emotion"], df["predicted_emotion"], cats)
    if em["accuracy"] is not None and em["majority"] is not None and em["accuracy"] <= em["majority"]:
        findings.append("Emotion accuracy does not exceed the validation majority-class baseline.")
    if em["invalid"]:
        findings.append(f"Invalid emotion predictions: {em['invalid']:.2%}.")

    def classification_section(title: str, stats: Dict[str, Any], categories: List[str]) -> None:
        headers = ["Class", "GT count", "Pred count", "GT rate", "Pred rate", "Pred-GT", "Precision", "Recall", "F1"]
        sections.extend(
            [
                f"## {title}",
                _health_table(headers, stats["rows"]),
                f"Scorable N={stats['n']}; accuracy={stats['accuracy']}; weighted F1={stats['weighted_f1']}; "
                f"macro F1={stats['macro_f1']}; macro recall (UAR)={stats['macro_recall']}; "
                f"invalid rate={stats['invalid']}; marginal TV distance={stats['tv']}.",
                _health_table(
                    ["GT / prediction", *categories, "invalid"],
                    [[t, *stats["confusion"][t].values()] for t in categories],
                ),
            ]
        )

    classification_section("Validation emotion classification", em, cats)
    gender_report = gender_emotion_metrics(df) if "gender" in df.columns else None
    if gender_report is not None:
        g_rows = [
            [g, v["n"], v["accuracy"], v["macro_f1"], v["weighted_f1"], v["macro_recall"]]
            for g, v in gender_report["groups"].items()
        ]
        sections.extend(
            [
                "## Emotion prediction by annotated speaker gender",
                _health_table(["Gender", "N", "Accuracy", "Macro F1", "Weighted F1", "Macro Recall"], g_rows),
                f"Female minus male gap: {json.dumps(_health_json(gender_report['female_minus_male']))}.",
            ]
        )

    acoustic, cue_rows, total_hits, total_valid = {}, [], 0, 0
    for cue, info in dimensions.items():
        options = info["options"]
        gt_col = CUE_COLUMNS[cue]
        st = _health_classification(df[gt_col], df[f"predicted_{cue}"], options)
        acoustic[cue] = st
        classification_section(f"Acoustic cue: {cue}", st, options)
        cue_rows.append(
            [cue, st["n"], len(df) - st["n"], st["accuracy"], st["macro_f1"], st["macro_recall"], st["majority"], st["independent"]]
        )
        total_valid += st["n"]
        total_hits += sum(st["confusion"][c][c] for c in options)

    micro_acc = total_hits / total_valid if total_valid else None
    reward_cols = ["r1_reward", "r2_acoustic_reward", "r3_format_reward", "r_unified_reward"]
    expected_total = sum(weights[k] * df[col] for k, col in zip(("r1", "r2", "r3"), reward_cols))
    residual = float(np.max(np.abs(expected_total - df["r_unified_reward"])))

    baseline_rows = []
    if all(c in cue_matrices and cue_matrices[c] for c in dimensions):
        distance_mode = (metadata or {}).get("use_distance_based_payoff", False)
        affine_scale = (
            max([1.0] + [abs(v) for m in cue_matrices.values() for row in m.values() for v in row.values()])
            if distance_mode
            else 1.0
        )
        is_if_mode = (not distance_mode) and ((metadata or {}).get("reward_mode") == "inverse_frequency")
        cue_scores, cue_val_arrays = {}, {}
        max_vec = np.zeros(len(df), dtype=float)
        for cue in dimensions:
            truth = df[CUE_COLUMNS[cue]]
            max_vec += np.array([cue_matrices[cue][t][t] if t in cue_matrices[cue] else 0.0 for t in truth], dtype=float)
            cue_scores[cue], cue_val_arrays[cue] = {}, {}
            for pred in cue_matrices[cue]:
                vals = np.array([cue_matrices[cue][t][pred] if t in cue_matrices[cue] else 0.0 for t in truth], dtype=float)
                cue_scores[cue][pred] = (float(np.mean(vals)), int(sum(t == pred for t in truth)))
                cue_val_arrays[cue][pred] = vals
        inventories = []
        for options in product(*(list(cue_scores[c]) for c in dimensions)):
            hits = sum(cue_scores[c][p][1] for c, p in zip(dimensions, options))
            if is_if_mode:
                num_vec = sum(cue_val_arrays[c][p] for c, p in zip(dimensions, options))
                mean_r2 = float(np.mean(np.divide(num_vec, max_vec, out=np.zeros_like(num_vec), where=max_vec > 0)))
            else:
                raw_sum = sum(cue_scores[c][p][0] for c, p in zip(dimensions, options))
                mean_r2 = 0.5 + raw_sum / (float(R2_NORM_SCALE) * affine_scale * int(R2_CUE_COUNT))
            inventories.append((options, mean_r2, hits / total_valid if total_valid else None))
        best_reward = max(inventories, key=lambda r: r[1])
        best_accuracy = max(inventories, key=lambda r: r[2] if r[2] is not None else -1)
        baseline_rows = [
            ["Observed model", df["r2_acoustic_reward"].mean(), micro_acc],
            [f"Best constant by validation accuracy: {best_accuracy[0]}", best_accuracy[1], best_accuracy[2]],
            [f"Best constant by validation reward: {best_reward[0]}", best_reward[1], best_reward[2]],
        ]

    meta = dict(metadata or {})
    meta.update(
        validation_rows=len(df),
        validation_identity_sha256=_health_hash(
            df[["audio_path", "emotion"]].sort_values(["audio_path", "emotion"]).to_dict("records")
        ),
        quantiles_sha256=_health_hash(quantiles or {}),
        priors_sha256=_health_hash({"emotion": emotion_priors, **cue_priors}),
        reward_weights=weights,
        rollout_groups=len(groups),
    )
    sections.insert(2, "## Observed concerns\n\n" + "\n".join("- " + x for x in (findings or ["No structural alerts."])))
    return "\n\n".join(sections) + "\n", _health_json(
        dict(
            metadata=meta,
            findings=findings,
            notes=notes,
            emotion=em,
            emotion_by_gender=gender_report,
            acoustics=acoustic,
            acoustic_micro_accuracy=micro_acc,
            reward_reconstruction_error=residual,
            baseline_rows=baseline_rows,
        )
    )


def write_grpo_health_report(
    predictions_df: pd.DataFrame,
    *,
    reward_calibrator: GRPORewardCalibrator,
    output_dir: Path,
    adapter_dir: Optional[Path] = None,
    trainer: Any = None,
    train_df: Optional[pd.DataFrame] = None,
    quartiles: Optional[Dict[str, Any]] = None,
    model_id: Optional[str] = None,
    sft_adapter_path: Optional[Path] = None,
    reference_adapter: Optional[str] = None,
    prompt_version: str = PROMPT_VERSION,
    groups_path: Optional[Path] = None,
    evaluation_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    output_dir = Path(output_dir)
    groups_path = Path(groups_path) if groups_path is not None else output_dir / "grpo_groups.jsonl"
    groups, notes = _health_read_groups(groups_path)
    args = getattr(trainer, "args", None)
    state = getattr(trainer, "state", None)
    meta = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        optimizer_step=getattr(state, "global_step", None),
        model_id=model_id,
        sft_adapter=str(sft_adapter_path) if sft_adapter_path else None,
        reference_adapter=reference_adapter,
        prompt_version=prompt_version,
        reward_mode=reward_calibrator.mode,
        inverse_frequency_alpha=(
            reward_calibrator.alpha if reward_calibrator.mode == "inverse_frequency" else None
        ),
        use_distance_based_payoff=bool(reward_calibrator.config.use_distance_based_payoff),
        acoustic_reward_mode=reward_calibrator.acoustic_mode,
        acoustic_normalization_scale=reward_calibrator.normalization_scale,
        distance_payoff_parameters=dict(
            moderate_hit=DISTANCE_MODERATE_HIT,
            extreme_on_moderate_cost=DISTANCE_EXTREME_ON_MODERATE_COST,
            minimum_opposite_cost=DISTANCE_MIN_OPPOSITE_COST,
            ordinal_margin=DISTANCE_ORDINAL_MARGIN,
        ),
        use_split_r2=getattr(trainer, "use_split_r2", True),
        use_split_advantage=getattr(trainer, "use_split_advantage", True),
        cue_kl_only=getattr(trainer, "cue_kl_only", True),
        training_args=args.to_dict() if args is not None and hasattr(args, "to_dict") else {},
        evaluation=evaluation_metadata or {},
        rollout_log=str(groups_path),
        rollout_log_sha256=hashlib.sha256(groups_path.read_bytes()).hexdigest() if groups_path.exists() else None,
    )
    text, summary = build_grpo_health_report(
        predictions_df,
        groups,
        dimensions=INVENTORY_DIMENSIONS,
        emotion_categories=EMOTIONS,
        reward_weights=reward_calibrator.config.reward_weights,
        cue_priors=reward_calibrator.cue_priors,
        emotion_priors=reward_calibrator.emotion_priors,
        cue_matrices=reward_calibrator.cue_matrices,
        emotion_matrix=reward_calibrator.emotion_matrix,
        train_df=train_df,
        quantiles=quartiles,
        metadata=meta,
        log_notes=notes,
        log_history=getattr(state, "log_history", []),
    )
    artifacts = {
        "model_health_report.md": text,
        "model_health_metrics.json": json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        "validation_predictions.jsonl": "\n".join(
            json.dumps(_health_json(r), ensure_ascii=False, allow_nan=False)
            for r in predictions_df.to_dict("records")
        ) + "\n",
        "evaluation_calibration.json": json.dumps(
            _health_json(
                dict(
                    quartiles=quartiles,
                    cue_priors=reward_calibrator.cue_priors,
                    emotion_priors=reward_calibrator.emotion_priors,
                    cue_matrices=reward_calibrator.cue_matrices,
                    emotion_matrix=reward_calibrator.emotion_matrix,
                    use_distance_based_payoff=meta["use_distance_based_payoff"],
                    acoustic_reward_mode=meta["acoustic_reward_mode"],
                    inverse_frequency_alpha=meta["inverse_frequency_alpha"],
                    acoustic_normalization_scale=meta["acoustic_normalization_scale"],
                    use_split_r2=meta["use_split_r2"],
                    use_split_advantage=meta["use_split_advantage"],
                    distance_payoff_parameters=meta["distance_payoff_parameters"],
                )
            ),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
    }
    directories = list(dict.fromkeys([output_dir, Path(adapter_dir) if adapter_dir else output_dir]))
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in artifacts.items():
            path = directory / name
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)
    return output_dir / "model_health_report.md"


def generate_validation_completions(
    model: Any,
    processor: Any,
    rows: List[Dict[str, Any]],
    *,
    compute_dtype: Any,
    max_new_tokens: int = 512,
    vllm_engine: Optional[VLLMRolloutEngine] = None,
) -> List[str]:
    """Greedy batched generation on Thinker (via vLLM if available, else exact HF generate)."""
    eos = generation_eos_ids(processor.tokenizer)
    if vllm_engine is not None:
        prompts = [
            processor.apply_chat_template(messages_for(row), tokenize=False, add_generation_prompt=True)
            for row in rows
        ]
        audios = [load_audio(row["audio_path"]) for row in rows]
        return vllm_engine.generate_greedy(
            model, prompts, audios, max_tokens=max_new_tokens, stop_token_ids=eos, adapter_name="default"
        )
    return generate_completions(
        model, processor, rows, compute_dtype=compute_dtype, max_new_tokens=max_new_tokens, teacher=False
    )


def evaluate_grpo_validation(
    model: Any,
    processor: Any,
    frame: pd.DataFrame,
    *,
    reward_calibrator: GRPORewardCalibrator,
    compute_dtype: Any,
    stage: str = "after",
    output_dir: Path,
    adapter_dir: Optional[Path] = None,
    trainer_obj: Any = None,
    train_df: Optional[pd.DataFrame] = None,
    quartiles: Optional[Dict[str, Any]] = None,
    model_id: Optional[str] = None,
    sft_adapter_path: Optional[Path] = None,
    reference_adapter: Optional[str] = None,
    prompt_version: str = PROMPT_VERSION,
    batch_size: int = 16,
    max_new_tokens: int = 512,
    vllm_engine: Optional[VLLMRolloutEngine] = None,
    data_dir: Optional[Path] = None,
    drive_data_dir: Optional[Path] = None,
    plot: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Post-training greedy validation producing exact ser_grpo_v3.ipynb metrics, health report, and plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with tqdm(total=len(frame), desc=f"Validation ({stage}, batch={batch_size})", unit="example") as progress:
        for start in range(0, len(frame), batch_size):
            rows = frame.iloc[start : start + batch_size].to_dict("records")
            texts = generate_validation_completions(
                model,
                processor,
                rows,
                compute_dtype=compute_dtype,
                max_new_tokens=max_new_tokens,
                vllm_engine=vllm_engine,
            )
            batch_parsed = [parse_completion(t) for t in texts]
            batch_prompts = [student_instruction(r) for r in rows]
            batch_trace: List[Dict[str, Any]] = []
            reward_kwargs = {
                "label": [str(r["emotion"]).lower().strip() for r in rows],
                "audio_path": [str(r["audio_path"]) for r in rows],
                "parsed_completions": batch_parsed,
                "_reward_trace_sink": batch_trace,
                **{CUE_COLUMNS[c]: [r.get(CUE_COLUMNS[c]) for r in rows] for c in CUES},
            }
            batch_totals = reward_calibrator(batch_prompts, texts, **reward_kwargs)
            for row, text, parsed, tr, total_r in zip(rows, texts, batch_parsed, batch_trace, batch_totals):
                emotion, strict_inv = parse_completion_strict(text)
                record = dict(
                    row,
                    completion=text,
                    predicted_emotion=emotion,
                    r_unified_reward=float(total_r),
                    r1_reward=float(tr["r1"]),
                    r2_acoustic_reward=float(tr["r2"]),
                    r3_format_reward=float(tr["r3"]),
                    acoustic_exact_match_ratio=float(tr["r2_gate"]),
                    valid_slots_count=int(parsed["valid_slots_count"]),
                )
                for cue in CUES:
                    record[f"predicted_{cue}"] = strict_inv[cue]
                    record[f"match_{cue}"] = bool(tr["r2_cues"][cue]["correct"])
                    record[f"raw_score_{cue}"] = float(tr["r2_cues"][cue]["raw"])
                records.append(record)
            progress.update(len(rows))

    predictions = pd.DataFrame(records)
    metrics = score_validation(predictions)
    atomic_csv(predictions, output_dir / f"{stage}_validation_predictions.csv", data_dir, drive_data_dir)
    (output_dir / f"{stage}_validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    gender_report = metrics["emotion_by_gender"]
    summary = pd.DataFrame(
        [
            dict(gender=gender, **{key: values[key] for key in ("n", "accuracy", "macro_f1", "weighted_f1")})
            for gender, values in gender_report["groups"].items()
        ]
    )
    atomic_csv(summary, output_dir / f"{stage}_emotion_by_gender.csv", data_dir, drive_data_dir)

    print(
        classification_report(
            predictions["emotion"], predictions["predicted_emotion"], labels=list(EMOTIONS), zero_division=0, digits=4
        )
    )
    write_grpo_health_report(
        predictions,
        reward_calibrator=reward_calibrator,
        output_dir=output_dir,
        adapter_dir=adapter_dir,
        trainer=trainer_obj,
        train_df=train_df,
        quartiles=quartiles,
        model_id=model_id,
        sft_adapter_path=sft_adapter_path,
        reference_adapter=reference_adapter,
        prompt_version=prompt_version,
        evaluation_metadata={
            "stage": stage,
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "backend": "vllm" if vllm_engine is not None else "hf",
            "prompt_version": prompt_version,
        },
    )

    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm_labels = list(EMOTIONS) + ["invalid"]
        cm = confusion_matrix(predictions["emotion"], predictions["predicted_emotion"], labels=cm_labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels, vmin=0, vmax=1)
        plt.title("GRPO v3 Normalized Confusion Matrix (MELD Val)", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.tight_layout()
        plt.savefig(output_dir / f"{stage}_confusion_matrix.png", dpi=200, bbox_inches="tight")
        plt.close()

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, cue in zip(axes.flatten(), CUES):
            opts = list(CUE_LABELS[cue])
            gt_col = CUE_COLUMNS[cue]
            pred_counts = predictions[f"predicted_{cue}"].value_counts()
            gt_counts = predictions[gt_col].value_counts()
            n_gt = max(int(gt_counts.sum()), 1)
            p_vals = [pred_counts.get(o, 0) / max(len(predictions), 1) * 100.0 for o in opts]
            t_vals = [gt_counts.get(o, 0) / n_gt * 100.0 for o in opts]
            x = np.arange(len(opts))
            ax.bar(x - 0.18, p_vals, 0.35, label="Model (Pred)", color="#2b5c8f", alpha=0.85)
            ax.bar(x + 0.18, t_vals, 0.35, label="Ground Truth", color="#e27c38", alpha=0.85)
            cm_cue = metrics["cues"][cue]
            ax.set_title(f"{cue} | Acc={cm_cue['accuracy']:.1%} | F1={cm_cue['macro_f1']:.1%} | UAR={cm_cue['macro_recall']:.1%}")
            ax.set_xticks(x)
            ax.set_xticklabels(opts, rotation=12, ha="right", fontsize=9)
            ax.set_ylabel("Percentage (%)")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.legend(fontsize=8)
        plt.suptitle("GRPO v3 Acoustic Inventory Distributions (Model vs Ground Truth)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / f"{stage}_acoustic_distributions.png", dpi=200, bbox_inches="tight")
        plt.close()

    return predictions, metrics


def plot_grpo_training_curves(log_history: List[Dict[str, Any]], save_path: Optional[Path] = None) -> None:
    if not log_history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df_logs = pd.DataFrame(log_history)
    if "step" not in df_logs.columns and "global_step" in df_logs.columns:
        df_logs["step"] = df_logs["global_step"]
    idx_series = pd.Series(df_logs.index, index=df_logs.index)
    df_logs["step"] = df_logs["step"].ffill().fillna(idx_series) if "step" in df_logs.columns else idx_series
    df_step = df_logs.groupby("step", as_index=False).mean(numeric_only=True)

    def smooth(series: pd.Series, alpha: float = 0.35) -> pd.Series:
        s = series.dropna()
        return s.ewm(alpha=alpha).mean() if len(s) > 1 else s

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("GRPO v3 Mid-Train Metrics — Qwen2.5-Omni-7B (MELD)", fontsize=15, fontweight="bold")
    for col, label, color in [
        ("reward/R1_accuracy", "R1: Emotion", "#1f77b4"),
        ("reward/R2_acoustic", "R2: 4-Cue Acoustic", "#2ca02c"),
        ("reward/R3_format", "R3: Format", "#ff7f0e"),
        ("reward/total_weighted", "Total Weighted Reward", "#9467bd"),
    ]:
        if col in df_step.columns:
            valid = df_step.dropna(subset=[col])
            if len(valid):
                axes[0, 0].plot(valid["step"], valid[col], alpha=0.25, color=color, linestyle="--")
                axes[0, 0].plot(valid["step"], smooth(valid[col]), label=label, color=color, linewidth=2.2)
    axes[0, 0].set_title("1. Rewards Evolution")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    for col, label, color in [
        ("metrics/f1_macro_running", "Running Macro-F1", "#d62728"),
        ("metrics/acc_step", "Step Accuracy", "#1f77b4"),
    ]:
        if col in df_step.columns:
            valid = df_step.dropna(subset=[col])
            if len(valid):
                axes[0, 1].plot(valid["step"], smooth(valid[col]), label=label, color=color, linewidth=2.0)
    axes[0, 1].set_title("2. Classification Performance")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    for col, label, color in [
        ("metrics/prediction_entropy", "Prediction Entropy (bits)", "#9467bd"),
        ("diagnostic/kl_cue_values", "Cue-Value KL", "#2ca02c"),
    ]:
        if col in df_step.columns:
            valid = df_step.dropna(subset=[col])
            if len(valid):
                axes[1, 0].plot(valid["step"], smooth(valid[col]), label=label, color=color, linewidth=2.0)
    axes[1, 0].set_title("3. Policy Entropy & Cue KL")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    for emo in EMOTIONS:
        col = f"dist/{emo}"
        if col in df_step.columns:
            valid = df_step.dropna(subset=[col])
            if len(valid):
                axes[1, 1].plot(valid["step"], smooth(valid[col]), label=emo, linewidth=1.6)
    axes[1, 1].set_title("4. Emotion Distribution Dynamics")
    axes[1, 1].legend(fontsize=8, ncol=2)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
