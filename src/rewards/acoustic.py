"""R2: Empirical 4-Cue Ordinal Distance-Based, Zero-Mean, MAP, or IF Payoffs (exact match to ser_grpo_v3.ipynb)."""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import CUE_COLUMNS, CUE_LABELS, CUES
from src.rewards.accuracy import (
    build_payoff_matrix,
    compute_hit_miss,
    compute_zero_mean_hit_miss,
    prove_payoff_matrix,
    resolve_inverse_frequency_alpha,
    resolve_reward_mode,
)
from src.rewards.parser import parse_completion

R2_NORM_SCALE: float = 2.0
R2_CUE_COUNT: int = len(CUES)

DISTANCE_MODERATE_HIT = 0.55
DISTANCE_EXTREME_ON_MODERATE_COST = 0.18
DISTANCE_MIN_OPPOSITE_COST = 0.56
DISTANCE_ORDINAL_MARGIN = 0.01


def resolve_acoustic_reward_mode(
    use_zero_mean: Optional[bool] = None,
    use_map: Optional[bool] = None,
    use_inverse_frequency: Optional[bool] = None,
    use_distance_based_payoff: Optional[bool] = None,
) -> str:
    base = resolve_reward_mode(use_zero_mean, use_map, use_inverse_frequency)
    distance = False if use_distance_based_payoff is None else use_distance_based_payoff
    if not isinstance(distance, (bool, np.bool_)):
        raise ValueError("use_distance_based_payoff must be a bool.")
    return "distance_based" if distance else base


def distance_based_payoff_matrix(
    cue_priors: Dict[str, float],
    ordered_labels: Tuple[str, str, str] = ("low", "moderate", "high"),
) -> Dict[str, Dict[str, float]]:
    if len(ordered_labels) != 3:
        raise ValueError("Ordinal distance payoffs require exactly 3 ordered labels.")
    if set(cue_priors) != set(ordered_labels):
        raise ValueError(f"Distance payoffs require empirical priors for {ordered_labels}, got {tuple(cue_priors)}.")
    validated = compute_zero_mean_hit_miss(cue_priors)
    low_lbl, mod_lbl, high_lbl = ordered_labels
    low, moderate, high = validated[low_lbl]["prior"], validated[mod_lbl]["prior"], validated[high_lbl]["prior"]
    a, b = DISTANCE_MODERATE_HIT, DISTANCE_EXTREME_ON_MODERATE_COST
    adjacent = a * moderate / (low + high)
    opposite = max(DISTANCE_MIN_OPPOSITE_COST, adjacent + DISTANCE_ORDINAL_MARGIN)
    hit_low = (b * moderate + opposite * high) / low
    hit_high = (opposite * low + b * moderate) / high
    return {
        low_lbl: {low_lbl: hit_low, mod_lbl: -adjacent, high_lbl: -opposite},
        mod_lbl: {low_lbl: -b, mod_lbl: a, high_lbl: -b},
        high_lbl: {low_lbl: -opposite, mod_lbl: -adjacent, high_lbl: hit_high},
    }


def prove_distance_payoff_matrix(
    priors: Dict[str, float],
    matrix: Dict[str, Dict[str, float]],
    ordered_labels: Tuple[str, str, str],
) -> Dict[str, Any]:
    low_lbl, mod_lbl, high_lbl = ordered_labels
    probabilities = {c: info["prior"] for c, info in compute_zero_mean_hit_miss(priors).items()}
    means = {p: math.fsum(probabilities[t] * matrix[t][p] for t in ordered_labels) for p in ordered_labels}
    bound = max(1.0, max(abs(v) for row in matrix.values() for v in row.values()))
    centered = all(abs(v) <= 1e-12 * bound for v in means.values())
    hits_win = all(
        matrix[t][t] > max(v for p, v in row.items() if p != t)
        for t, row in matrix.items()
    )
    ordered = (
        matrix[low_lbl][low_lbl] > matrix[low_lbl][mod_lbl] > matrix[low_lbl][high_lbl]
        and matrix[high_lbl][high_lbl] > matrix[high_lbl][mod_lbl] > matrix[high_lbl][low_lbl]
    )
    if not (centered and hits_win and ordered):
        raise ArithmeticError("Distance payoff failed zero-mean or ordinal ordering checks.")
    return {
        "mode": "distance_based",
        "zero_mean_satisfied": centered,
        "prior_random_zero_mean_satisfied": centered,
        "column_blind_means": means,
        "ordinal_satisfied": ordered,
        "hits_beat_misses": hits_win,
        "unit_contrast_satisfied": False,
        "map_satisfied": False,
        "inverse_frequency_satisfied": False,
    }


def get_ordinal_cue_payoff_matrix(
    cue_priors: Dict[str, float],
    cue_name: str,
    *,
    use_zero_mean: Optional[bool] = None,
    use_map: Optional[bool] = None,
    use_inverse_frequency: Optional[bool] = None,
    use_distance_based_payoff: Optional[bool] = None,
    inverse_frequency_alpha: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    if cue_name not in CUE_LABELS:
        raise ValueError(f"Unknown acoustic cue: {cue_name!r}")
    ordered_labels = CUE_LABELS[cue_name]
    if (
        resolve_acoustic_reward_mode(
            use_zero_mean, use_map, use_inverse_frequency, use_distance_based_payoff
        )
        == "distance_based"
    ):
        return distance_based_payoff_matrix(cue_priors, ordered_labels)
    return build_payoff_matrix(
        cue_priors,
        ordered_labels,
        use_zero_mean=use_zero_mean,
        use_map=use_map,
        use_inverse_frequency=use_inverse_frequency,
        inverse_frequency_alpha=inverse_frequency_alpha,
    )


def get_acoustic_reward_scale(
    priors_dict: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    use_distance_based_payoff: Optional[bool] = None,
) -> float:
    distance = False if use_distance_based_payoff is None else use_distance_based_payoff
    if not isinstance(distance, (bool, np.bool_)):
        raise ValueError("use_distance_based_payoff must be a bool.")
    if not distance:
        return 1.0
    if not priors_dict:
        raise ValueError("Empirical calibration is required for distance normalization.")
    matrices = [distance_based_payoff_matrix(priors_dict[cue], CUE_LABELS[cue]) for cue in CUES]
    return max([1.0] + [abs(v) for m in matrices for row in m.values() for v in row.values()])


def normalize_acoustic_reward(
    raw_scores: List[float],
    max_possible_scores: Optional[List[float]] = None,
    *,
    use_zero_mean: Optional[bool] = None,
    use_map: Optional[bool] = None,
    use_inverse_frequency: Optional[bool] = None,
    use_distance_based_payoff: Optional[bool] = None,
    priors_dict: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """Normalize per-cue acoustic scores to [0, 1] across the 4 ordinal inventory slots."""
    if len(raw_scores) > R2_CUE_COUNT:
        raise ValueError("More acoustic scores than inventory slots.")
    mode = resolve_acoustic_reward_mode(
        use_zero_mean, use_map, use_inverse_frequency, use_distance_based_payoff
    )
    if mode == "distance_based":
        scale = get_acoustic_reward_scale(priors_dict, use_distance_based_payoff=True)
        result = 0.5 + math.fsum(raw_scores) / (R2_NORM_SCALE * scale * R2_CUE_COUNT)
        if not -1e-12 <= result <= 1.0 + 1e-12:
            raise ValueError("Distance scores exceed the frozen calibration bounds.")
        return float(max(0.0, min(1.0, result)))
    if mode == "inverse_frequency":
        if max_possible_scores is None:
            raise ValueError("max_possible_scores is required in inverse_frequency mode.")
        if len(max_possible_scores) != len(raw_scores):
            raise ValueError("raw_scores and max_possible_scores must have identical length.")
        max_sum = math.fsum(max_possible_scores)
        if max_sum <= 0.0:
            return 0.0
        return float(max(0.0, min(1.0, math.fsum(raw_scores) / max_sum)))
    return float(max(0.0, min(1.0, 0.5 + math.fsum(raw_scores) / (R2_NORM_SCALE * R2_CUE_COUNT))))


def compute_acoustic_cue_priors(
    train_df: pd.DataFrame,
    return_counts: bool = False,
    *,
    use_zero_mean: Optional[bool] = None,
    use_map: Optional[bool] = None,
    use_inverse_frequency: Optional[bool] = None,
    inverse_frequency_alpha: Optional[float] = None,
) -> Any:
    """Count authoritative precomputed cue labels directly on grpo_train_set; never refit or smooth."""
    resolve_reward_mode(use_zero_mean, use_map, use_inverse_frequency)
    counts: Dict[str, Dict[str, int]] = {}
    priors: Dict[str, Dict[str, float]] = {}
    for cue in CUES:
        col = CUE_COLUMNS[cue]
        if col not in train_df.columns:
            raise ValueError(f"grpo_train_set is missing required cue column: {col}")
        series = train_df[col].dropna().astype(str).str.strip().str.lower()
        unknown = set(series) - set(CUE_LABELS[cue])
        if unknown:
            raise ValueError(f"Unknown {cue} labels in GRPO calibration: {sorted(unknown)}")
        cat_counts = {lbl: int((series == lbl).sum()) for lbl in CUE_LABELS[cue]}
        total = sum(cat_counts.values())
        counts[cue] = cat_counts
        priors[cue] = {lbl: (cnt / total if total else 0.0) for lbl, cnt in cat_counts.items()}
        try:
            compute_hit_miss(
                priors[cue],
                use_zero_mean=use_zero_mean,
                use_map=use_map,
                use_inverse_frequency=use_inverse_frequency,
                inverse_frequency_alpha=inverse_frequency_alpha,
            )
        except ValueError as exc:
            raise ValueError(f"Cannot calibrate acoustic cue {cue!r}: {exc}") from exc
    return (priors, counts) if return_counts else priors


def calibrate_acoustic_payoffs(
    grpo_train_df: pd.DataFrame,
    validation_df: Optional[pd.DataFrame] = None,
    *,
    use_zero_mean: Optional[bool] = None,
    use_map: Optional[bool] = None,
    use_inverse_frequency: Optional[bool] = None,
    use_distance_based_payoff: Optional[bool] = None,
    inverse_frequency_alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """Derive 4-cue ordinal payoff matrices strictly from grpo_train_set's saved cue columns."""
    mode = resolve_acoustic_reward_mode(
        use_zero_mean, use_map, use_inverse_frequency, use_distance_based_payoff
    )
    if grpo_train_df is None or grpo_train_df.empty:
        raise ValueError("Non-empty grpo_train_set is required for acoustic calibration.")
    if (
        validation_df is not None
        and "audio_path" in validation_df.columns
        and "audio_path" in grpo_train_df.columns
    ):
        overlap = set(grpo_train_df["audio_path"].astype(str)) & set(validation_df["audio_path"].astype(str))
        if overlap:
            raise ValueError(f"grpo_train_set overlaps validation audio paths: {len(overlap)}")

    alpha = resolve_inverse_frequency_alpha(inverse_frequency_alpha)
    priors, counts = compute_acoustic_cue_priors(
        grpo_train_df,
        return_counts=True,
        use_zero_mean=use_zero_mean,
        use_map=use_map,
        use_inverse_frequency=use_inverse_frequency,
        inverse_frequency_alpha=alpha,
    )

    if mode == "distance_based":
        matrices = {cue: distance_based_payoff_matrix(priors[cue], CUE_LABELS[cue]) for cue in CUES}
        proofs = {cue: prove_distance_payoff_matrix(priors[cue], matrices[cue], CUE_LABELS[cue]) for cue in CUES}
        payoffs, rows = {}, []
        for cue in CUES:
            matrix = matrices[cue]
            payoffs[cue] = {}
            for pred in CUE_LABELS[cue]:
                misses = [matrix[t][pred] for t in CUE_LABELS[cue] if t != pred]
                info = {
                    "mode": "distance_based",
                    "prior": priors[cue][pred],
                    "hit": matrix[pred][pred],
                    "min_miss": min(misses),
                    "max_miss": max(misses),
                    "e_blind": proofs[cue]["column_blind_means"][pred],
                }
                payoffs[cue][pred] = info
                rows.append(
                    {
                        "Cue": cue,
                        "Category": pred,
                        "Count": counts[cue][pred],
                        "Prior": info["prior"],
                        "Hit": info["hit"],
                        "Min miss": info["min_miss"],
                        "Max miss": info["max_miss"],
                        "E[blind]": info["e_blind"],
                        "Mode": "distance_based",
                    }
                )
        return {
            "mode": "distance_based",
            "use_distance_based_payoff": True,
            "normalization_scale": get_acoustic_reward_scale(priors, use_distance_based_payoff=True),
            "priors": priors,
            "counts": counts,
            "matrices": matrices,
            "proofs": proofs,
            "payoffs": payoffs,
            "summary": pd.DataFrame(rows).set_index(["Cue", "Category"]),
            "grpo_rows": len(grpo_train_df),
        }

    payoffs = {
        cue: compute_hit_miss(
            priors[cue],
            use_zero_mean=use_zero_mean,
            use_map=use_map,
            use_inverse_frequency=use_inverse_frequency,
            inverse_frequency_alpha=alpha,
        )
        for cue in CUES
    }
    matrices = {
        cue: get_ordinal_cue_payoff_matrix(
            priors[cue],
            cue,
            use_zero_mean=use_zero_mean,
            use_map=use_map,
            use_inverse_frequency=use_inverse_frequency,
            use_distance_based_payoff=use_distance_based_payoff,
            inverse_frequency_alpha=alpha,
        )
        for cue in CUES
    }
    proofs = {
        cue: prove_payoff_matrix(
            priors[cue],
            matrices[cue],
            use_zero_mean=use_zero_mean,
            use_map=use_map,
            use_inverse_frequency=use_inverse_frequency,
            inverse_frequency_alpha=alpha,
        )
        for cue in CUES
    }
    rows = []
    for cue in CUES:
        for category, p in payoffs[cue].items():
            rows.append(
                {
                    "Cue": cue,
                    "Category": category,
                    "Mode": p["mode"],
                    "Count": counts[cue][category],
                    "Prior": p["prior"],
                    "Prior_Baseline": p["prior_baseline"],
                    "Hit": p["hit"],
                    "Miss": p["miss"],
                    "Delta": p["delta"],
                    "E[blind]": p["e_blind"],
                    "E[prior_random]": p["e_prior_random"],
                    "Zero_Mean": p["zero_mean_consistent"],
                    "MAP_Consistent": p["map_consistent"],
                }
            )
    return {
        "mode": mode,
        "inverse_frequency_alpha": alpha if mode == "inverse_frequency" else None,
        "use_distance_based_payoff": False,
        "normalization_scale": 1.0,
        "priors": priors,
        "counts": counts,
        "matrices": matrices,
        "proofs": proofs,
        "payoffs": payoffs,
        "summary": pd.DataFrame(rows).set_index(["Cue", "Category"]),
        "grpo_rows": len(grpo_train_df),
    }


def score_cue_payoff(
    truth_label: str,
    pred_label: Optional[str],
    cue_name: str,
    priors_dict: Dict[str, Dict[str, float]],
    *,
    use_zero_mean: Optional[bool] = None,
    use_map: Optional[bool] = None,
    use_inverse_frequency: Optional[bool] = None,
    use_distance_based_payoff: Optional[bool] = None,
    inverse_frequency_alpha: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Return (raw payoff, truth hit payoff, exact match) for one cue under the active mode."""
    mode = resolve_acoustic_reward_mode(
        use_zero_mean, use_map, use_inverse_frequency, use_distance_based_payoff
    )
    if cue_name not in priors_dict:
        raise ValueError(f"No empirical calibration for cue {cue_name!r}.")
    matrix = get_ordinal_cue_payoff_matrix(
        priors_dict[cue_name],
        cue_name,
        use_zero_mean=use_zero_mean,
        use_map=use_map,
        use_inverse_frequency=use_inverse_frequency,
        use_distance_based_payoff=use_distance_based_payoff,
        inverse_frequency_alpha=inverse_frequency_alpha,
    )
    truth = str(truth_label).lower().strip()
    if truth not in matrix:
        raise ValueError(f"Unknown truth label for {cue_name}: {truth_label!r}")
    pred = str(pred_label).lower().strip() if pred_label is not None else None
    hit_reward = float(matrix[truth][truth])
    if pred not in matrix[truth]:
        if mode == "distance_based":
            return -get_acoustic_reward_scale(priors_dict, use_distance_based_payoff=True), hit_reward, 0.0
        return (0.0 if mode == "inverse_frequency" else -1.0), hit_reward, 0.0
    return float(matrix[truth][pred]), hit_reward, float(pred == truth)


def _clean_gt_label(val: Any, cue: str) -> Optional[str]:
    if val is None or pd.isna(val):
        return None
    s = str(val).strip().lower()
    return s if s in CUE_LABELS[cue] else None


def reward_acoustic_inventory(
    prompts: List[str],
    completions: List[str],
    *,
    priors_dict: Dict[str, Dict[str, float]],
    audio_lookup: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    return_gate_scores: bool = False,
    **kwargs: Any,
) -> Any:
    """Compute 4-cue R2 in [0, 1] directly against precomputed cue columns."""
    use_distance_based_payoff = kwargs.get("use_distance_based_payoff")
    use_zero_mean = kwargs.get("use_zero_mean")
    use_map = kwargs.get("use_map")
    use_inverse_frequency = kwargs.get("use_inverse_frequency")
    inverse_frequency_alpha = kwargs.get("inverse_frequency_alpha")
    resolve_acoustic_reward_mode(use_zero_mean, use_map, use_inverse_frequency, use_distance_based_payoff)

    if not priors_dict:
        raise ValueError("Empirical cue priors_dict is required before scoring R2.")

    cue_trace_sink = kwargs.get("_cue_trace_sink")
    audio_paths = kwargs.get("audio_path", [None] * len(completions))
    parsed_list = kwargs.get("parsed_completions")
    rewards: List[float] = []
    gate_rewards: List[float] = []

    for idx, comp in enumerate(completions):
        parsed_inv = (
            parsed_list[idx]["inventory"]
            if (parsed_list is not None and idx < len(parsed_list))
            else parse_completion(comp)["inventory"]
        )
        audio_p = audio_paths[idx] if idx < len(audio_paths) else None
        lookup_data = audio_lookup.get(audio_p, {}) if (audio_p and audio_lookup) else {}

        cue_details = {
            c: {"raw": 0.0, "truth": None, "pred": None, "valid_gt": False, "correct": False}
            for c in CUES
        }
        achieved_scores: List[float] = []
        max_possible_scores: List[float] = []
        unweighted_hits = 0.0
        total_cues = 0

        for cue in CUES:
            col = CUE_COLUMNS[cue]
            raw_gt = None
            if col in kwargs and idx < len(kwargs[col]):
                raw_gt = kwargs[col][idx]
            elif cue in kwargs and idx < len(kwargs[cue]):
                raw_gt = kwargs[cue][idx]
            elif col in lookup_data:
                raw_gt = lookup_data[col]
            truth_lbl = _clean_gt_label(raw_gt, cue)
            if truth_lbl is None:
                continue

            pred_lbl = parsed_inv.get(cue)
            sc, mx, hit = score_cue_payoff(
                truth_lbl,
                pred_lbl,
                cue,
                priors_dict,
                use_zero_mean=use_zero_mean,
                use_map=use_map,
                use_inverse_frequency=use_inverse_frequency,
                use_distance_based_payoff=use_distance_based_payoff,
                inverse_frequency_alpha=inverse_frequency_alpha,
            )
            cue_details[cue] = {
                "raw": float(sc),
                "truth": truth_lbl,
                "pred": pred_lbl,
                "valid_gt": True,
                "correct": bool(hit),
            }
            achieved_scores.append(sc)
            max_possible_scores.append(mx)
            unweighted_hits += hit
            total_cues += 1

        rewards.append(
            normalize_acoustic_reward(
                achieved_scores,
                max_possible_scores,
                priors_dict=priors_dict,
                use_zero_mean=use_zero_mean,
                use_map=use_map,
                use_inverse_frequency=use_inverse_frequency,
                use_distance_based_payoff=use_distance_based_payoff,
            )
        )
        gate_rewards.append(float(unweighted_hits / total_cues) if total_cues else 0.0)
        if cue_trace_sink is not None:
            cue_trace_sink.append(cue_details)

    if return_gate_scores:
        return rewards, gate_rewards
    return rewards
