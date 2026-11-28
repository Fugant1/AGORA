"""R1: Empirical Zero-Mean, Prior-Centered MAP, or Inverse-Frequency Emotion Reward (exact match to ser_grpo_v3.ipynb)."""

from collections import Counter
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import EMOTIONS, VALID_EMOTIONS
from src.rewards.parser import parse_completion

DEFAULT_USE_ZERO_MEAN = False
DEFAULT_USE_MAP = False
DEFAULT_USE_INVERSE_FREQUENCY = True
DEFAULT_INVERSE_FREQUENCY_ALPHA = 1.0


def resolve_inverse_frequency_alpha(inverse_frequency_alpha: Optional[float] = None) -> float:
    raw = DEFAULT_INVERSE_FREQUENCY_ALPHA if inverse_frequency_alpha is None else inverse_frequency_alpha
    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, float, np.number)):
        raise ValueError(f"INVERSE_FREQUENCY_ALPHA must be a finite float in [0.0, 1.0], got {raw!r}")
    alpha = float(raw)
    if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"INVERSE_FREQUENCY_ALPHA must be in [0.0, 1.0], got {alpha!r}")
    return alpha


def resolve_reward_mode(
    use_zero_mean: Optional[bool] = None,
    use_map: Optional[bool] = None,
    use_inverse_frequency: Optional[bool] = None,
) -> str:
    """Resolve and validate the mutually exclusive reward mode ('zero_mean', 'map', or 'inverse_frequency')."""
    if use_zero_mean is None and use_map is None and use_inverse_frequency is None:
        zm = DEFAULT_USE_ZERO_MEAN
        mp = DEFAULT_USE_MAP
        ifreq = DEFAULT_USE_INVERSE_FREQUENCY
    elif use_zero_mean is not None and use_map is not None:
        zm = use_zero_mean
        mp = use_map
        ifreq = False if use_inverse_frequency is None else use_inverse_frequency
    else:
        raise ValueError(
            "Pass both use_zero_mean and use_map explicitly (with optional use_inverse_frequency), "
            "or omit all flags to use defaults."
        )
    if (
        not isinstance(zm, (bool, np.bool_))
        or not isinstance(mp, (bool, np.bool_))
        or not isinstance(ifreq, (bool, np.bool_))
        or sum([bool(zm), bool(mp), bool(ifreq)]) != 1
    ):
        raise ValueError(
            f"Mutually exclusive reward flags violated (use_zero_mean={zm!r}, use_map={mp!r}, "
            f"use_inverse_frequency={ifreq!r}): exactly one must be True."
        )
    if bool(zm):
        return "zero_mean"
    if bool(mp):
        return "map"
    return "inverse_frequency"


def compute_hit_miss(
    priors: Dict[str, float],
    *,
    use_zero_mean: Optional[bool] = None,
    use_map: Optional[bool] = None,
    use_inverse_frequency: Optional[bool] = None,
    inverse_frequency_alpha: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-category Hit/Miss payoffs under Zero-Mean, MAP, or Inverse-Frequency mode."""
    mode = resolve_reward_mode(use_zero_mean, use_map, use_inverse_frequency)
    alpha = resolve_inverse_frequency_alpha(inverse_frequency_alpha)
    if len(priors) < 2:
        raise ValueError("Calibration requires at least two observed categories.")

    values = {k: float(v) for k, v in priors.items()}
    invalid = [k for k, v in values.items() if not math.isfinite(v) or v <= 0.0]
    if invalid:
        raise ValueError(f"Cannot calibrate unobserved or invalid categories: {invalid}")
    total = math.fsum(values.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Calibration frequencies must have a positive finite sum.")
    probabilities = {k: v / total for k, v in values.items()}
    if any(p >= 1.0 for p in probabilities.values()):
        raise ValueError("Calibration frequencies are numerically degenerate.")

    prior_baseline = math.fsum(p * p for p in probabilities.values())
    if not (0.0 < prior_baseline < 1.0):
        raise ArithmeticError(f"Invalid prior squared baseline: {prior_baseline}")
    is_uniform = (max(probabilities.values()) - min(probabilities.values())) <= 1e-12
    inv_freq_norm = math.fsum(p ** (1.0 - alpha) for p in probabilities.values())

    raw_payoffs: Dict[str, Dict[str, Any]] = {}
    for k, p in probabilities.items():
        if mode == "zero_mean":
            hit, miss = 1.0 - p, -p
        elif mode == "map":
            hit, miss = 1.0 - prior_baseline, -prior_baseline
        else:
            hit, miss = (p ** (-alpha)) / inv_freq_norm, 0.0

        delta = hit - miss
        e_blind = p * hit + (1.0 - p) * miss
        if mode in ("zero_mean", "map"):
            if abs(delta - 1.0) > 1e-12:
                raise ArithmeticError(f"Unit-contrast slope failed for {k!r}: delta={delta}")
            if not (0.0 < hit <= 1.0 and -1.0 <= miss < 0.0):
                raise ArithmeticError(f"Payoff bounds violated for {k!r}: hit={hit}, miss={miss}")
            if mode == "zero_mean" and abs(e_blind) > 1e-12:
                raise ArithmeticError(f"Zero-Mean failed for {k!r}: E[blind]={e_blind}")
            if mode == "map" and abs(e_blind - (p - prior_baseline)) > 1e-12:
                raise ArithmeticError(f"MAP blind expectation failed for {k!r}: {e_blind}")
        else:
            expected_blind = (p ** (1.0 - alpha)) / inv_freq_norm
            if not (hit > 0.0 and miss == 0.0):
                raise ArithmeticError(f"Inverse-frequency bounds violated for {k!r}: hit={hit}, miss={miss}")
            if abs(e_blind - expected_blind) > 1e-12:
                raise ArithmeticError(f"Inverse-frequency blind expectation failed for {k!r}")

        raw_payoffs[k] = {
            "mode": mode,
            "inverse_frequency_alpha": alpha if mode == "inverse_frequency" else None,
            "prior": p,
            "prior_baseline": prior_baseline,
            "hit": hit,
            "miss": miss,
            "delta": delta,
            "e_blind": e_blind,
            "zero_mean_consistent": abs(e_blind) <= 1e-12,
            "map_consistent": (mode == "map") or is_uniform,
            "inverse_frequency_consistent": mode == "inverse_frequency",
        }

    e_prior_random = math.fsum(probabilities[k] * raw_payoffs[k]["e_blind"] for k in probabilities)
    if mode in ("zero_mean", "map"):
        if abs(e_prior_random) > 1e-12:
            raise ArithmeticError(f"Prior-random zero-mean baseline failed: {e_prior_random}")
    else:
        expected_prior_random = math.fsum(p ** (2.0 - alpha) for p in probabilities.values()) / inv_freq_norm
        oracle_mean = math.fsum(probabilities[k] * raw_payoffs[k]["hit"] for k in probabilities)
        if abs(e_prior_random - expected_prior_random) > 1e-12 or abs(oracle_mean - 1.0) > 1e-12:
            raise ArithmeticError("Inverse-frequency baseline verification failed.")
    for info in raw_payoffs.values():
        info["e_prior_random"] = e_prior_random

    if mode == "map":
        hits = [info["hit"] for info in raw_payoffs.values()]
        misses = [info["miss"] for info in raw_payoffs.values()]
        if (max(hits) - min(hits) > 1e-12) or (max(misses) - min(misses) > 1e-12):
            raise ArithmeticError("MAP mode requires identical Hit and Miss across all categories.")
    return raw_payoffs


def compute_zero_mean_hit_miss(priors: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    return compute_hit_miss(priors, use_zero_mean=True, use_map=False, use_inverse_frequency=False)


def compute_map_hit_miss(priors: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    return compute_hit_miss(priors, use_zero_mean=False, use_map=True, use_inverse_frequency=False)


def compute_inverse_frequency_hit_miss(
    priors: Dict[str, float],
    inverse_frequency_alpha: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    return compute_hit_miss(
        priors,
        use_zero_mean=False,
        use_map=False,
        use_inverse_frequency=True,
        inverse_frequency_alpha=inverse_frequency_alpha,
    )


def prove_payoff_matrix(
    priors: Dict[str, float],
    matrix: Dict[str, Dict[str, float]],
    *,
    use_zero_mean: Optional[bool] = None,
    use_map: Optional[bool] = None,
    use_inverse_frequency: Optional[bool] = None,
    inverse_frequency_alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """Verify active-mode invariants ('zero_mean', 'map', or 'inverse_frequency') on a KxK payoff matrix."""
    mode = resolve_reward_mode(use_zero_mean, use_map, use_inverse_frequency)
    alpha = resolve_inverse_frequency_alpha(inverse_frequency_alpha)
    categories = tuple(matrix.keys())
    n_classes = len(categories)
    if n_classes < 2:
        raise ValueError("Payoff matrix must have at least two categories.")
    total = math.fsum(float(priors[c]) for c in categories)
    probs = {c: float(priors[c]) / total for c in categories}

    column_blind_means: Dict[str, float] = {}
    column_contrasts: Dict[str, float] = {}
    hits: List[float] = []
    misses: List[float] = []
    for pred in categories:
        e_blind = math.fsum(probs[truth] * float(matrix[truth][pred]) for truth in categories)
        column_blind_means[pred] = e_blind
        hit_val = float(matrix[pred][pred])
        miss_vals = [float(matrix[truth][pred]) for truth in categories if truth != pred]
        if max(miss_vals) - min(miss_vals) > 1e-12:
            raise ArithmeticError(f"Matrix column {pred!r} has unequal miss penalties: {miss_vals}")
        contrast = hit_val - miss_vals[0]
        if mode in ("zero_mean", "map") and abs(contrast - 1.0) > 1e-12:
            raise ArithmeticError(f"Matrix column {pred!r} failed Unit-Contrast proof: delta={contrast}")
        column_contrasts[pred] = contrast
        hits.append(hit_val)
        misses.append(miss_vals[0])

    e_prior_random = math.fsum(probs[c] * column_blind_means[c] for c in categories)
    prior_random_zero_mean_satisfied = abs(e_prior_random) <= 1e-12
    if mode in ("zero_mean", "map") and not prior_random_zero_mean_satisfied:
        raise ArithmeticError(f"Matrix failed prior-random zero baseline: {e_prior_random}")

    unit_contrast_satisfied = all(abs(c - 1.0) <= 1e-12 for c in column_contrasts.values())
    zero_mean_satisfied = all(abs(v) <= 1e-12 for v in column_blind_means.values())
    common_hit_miss = (max(hits) - min(hits) <= 1e-12) and (max(misses) - min(misses) <= 1e-12)
    inv_freq_norm = math.fsum(probs[c] ** (1.0 - alpha) for c in categories)
    inverse_frequency_satisfied = (
        all(abs(m) <= 1e-12 for m in misses)
        and all(abs(float(matrix[c][c]) - ((probs[c] ** (-alpha)) / inv_freq_norm)) <= 1e-12 for c in categories)
        and all(abs(column_blind_means[c] - ((probs[c] ** (1.0 - alpha)) / inv_freq_norm)) <= 1e-12 for c in categories)
        and abs(math.fsum(column_blind_means.values()) - 1.0) <= 1e-12
    )

    a = np.array([[float(matrix[t][c]) for c in categories] for t in categories], dtype=float)
    p_vec = np.array([probs[c] for c in categories], dtype=float)
    rng = np.random.default_rng(0)
    probes = np.vstack([rng.dirichlet(np.ones(n_classes), size=256), p_vec])
    scores = probes @ a
    map_idx = probes.argmax(axis=1)
    max_map_regret = float(np.max(scores.max(axis=1) - scores[np.arange(len(probes)), map_idx]))
    map_satisfied = bool(common_hit_miss and max_map_regret <= 1e-12)

    if mode == "zero_mean" and not zero_mean_satisfied:
        raise ArithmeticError(f"Matrix failed Zero-Mean invariant: {column_blind_means}")
    if mode == "map" and not map_satisfied:
        raise ArithmeticError(f"Matrix failed universal MAP invariant: max_regret={max_map_regret}")
    if mode == "inverse_frequency" and not inverse_frequency_satisfied:
        raise ArithmeticError(f"Matrix failed Inverse-Frequency invariant: {column_blind_means}")

    return {
        "mode": mode,
        "zero_mean_satisfied": zero_mean_satisfied,
        "prior_random_zero_mean_satisfied": prior_random_zero_mean_satisfied,
        "unit_contrast_satisfied": unit_contrast_satisfied,
        "map_satisfied": map_satisfied,
        "inverse_frequency_satisfied": inverse_frequency_satisfied,
        "max_map_regret": max_map_regret,
        "column_blind_means": column_blind_means,
        "column_contrasts": column_contrasts,
    }


def build_payoff_matrix(
    priors: Dict[str, float],
    categories: Optional[Tuple[str, ...]] = None,
    *,
    use_zero_mean: Optional[bool] = None,
    use_map: Optional[bool] = None,
    use_inverse_frequency: Optional[bool] = None,
    inverse_frequency_alpha: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    cats = tuple(priors.keys()) if categories is None else tuple(categories)
    if set(priors) != set(cats):
        raise ValueError(f"Expected calibration for all categories: {cats}")
    payoffs = compute_hit_miss(
        priors,
        use_zero_mean=use_zero_mean,
        use_map=use_map,
        use_inverse_frequency=use_inverse_frequency,
        inverse_frequency_alpha=inverse_frequency_alpha,
    )
    matrix = {
        truth: {pred: float(payoffs[pred]["hit" if truth == pred else "miss"]) for pred in cats}
        for truth in cats
    }
    prove_payoff_matrix(
        priors,
        matrix,
        use_zero_mean=use_zero_mean,
        use_map=use_map,
        use_inverse_frequency=use_inverse_frequency,
        inverse_frequency_alpha=inverse_frequency_alpha,
    )
    return matrix


def calibrate_emotion_payoffs(
    train_df: Any,
    return_matrix: bool = False,
    *,
    use_zero_mean: Optional[bool] = None,
    use_map: Optional[bool] = None,
    use_inverse_frequency: Optional[bool] = None,
    inverse_frequency_alpha: Optional[float] = None,
) -> Any:
    """Fit R1 to the actual GRPO emotion distribution under the active reward mode."""
    mode = resolve_reward_mode(use_zero_mean, use_map, use_inverse_frequency)
    labels = [str(e).lower().strip() for e in train_df["emotion"]]
    unknown = set(labels) - set(VALID_EMOTIONS)
    if unknown:
        raise ValueError(f"Unknown emotion targets in GRPO calibration: {sorted(unknown)}")
    counts = Counter(labels)
    total = len(labels)
    ordered_emotions = tuple(EMOTIONS)
    priors = {e: counts[e] / total if total else 0.0 for e in ordered_emotions}
    payoffs = compute_hit_miss(
        priors,
        use_zero_mean=use_zero_mean,
        use_map=use_map,
        use_inverse_frequency=use_inverse_frequency,
        inverse_frequency_alpha=inverse_frequency_alpha,
    )
    matrix = build_payoff_matrix(
        priors,
        ordered_emotions,
        use_zero_mean=use_zero_mean,
        use_map=use_map,
        use_inverse_frequency=use_inverse_frequency,
        inverse_frequency_alpha=inverse_frequency_alpha,
    )
    proof = prove_payoff_matrix(
        priors,
        matrix,
        use_zero_mean=use_zero_mean,
        use_map=use_map,
        use_inverse_frequency=use_inverse_frequency,
        inverse_frequency_alpha=inverse_frequency_alpha,
    )
    if mode == "zero_mean" and not proof["zero_mean_satisfied"]:
        raise ArithmeticError("R1 emotion payoff matrix failed Zero-Mean verification.")
    if mode == "map" and not proof["map_satisfied"]:
        raise ArithmeticError("R1 emotion payoff matrix failed MAP verification.")
    if mode == "inverse_frequency" and not proof["inverse_frequency_satisfied"]:
        raise ArithmeticError("R1 emotion payoff matrix failed Inverse-Frequency verification.")
    return (priors, payoffs, matrix) if return_matrix else (priors, payoffs)


def reward_label_accuracy_weighted(
    prompts: List[str],
    completions: List[str],
    label: List[str],
    *,
    emotion_priors: Dict[str, float],
    parsed_completions: Optional[List[Dict[str, Any]]] = None,
    use_zero_mean: Optional[bool] = None,
    use_map: Optional[bool] = None,
    use_inverse_frequency: Optional[bool] = None,
    inverse_frequency_alpha: Optional[float] = None,
    **kwargs: Any,
) -> List[float]:
    """Dynamic Zero-Mean, Prior-Centered MAP, or Inverse-Frequency R1 calibrated on grpo_train_set."""
    mode = resolve_reward_mode(use_zero_mean, use_map, use_inverse_frequency)
    if len(completions) != len(label):
        raise ValueError("Each completion must have exactly one emotion target.")
    if set(emotion_priors) != set(VALID_EMOTIONS):
        raise ValueError("Calibrated emotion_priors covering all 7 emotions are required before scoring R1.")
    matrix = build_payoff_matrix(
        emotion_priors,
        tuple(EMOTIONS),
        use_zero_mean=use_zero_mean,
        use_map=use_map,
        use_inverse_frequency=use_inverse_frequency,
        inverse_frequency_alpha=inverse_frequency_alpha,
    )
    invalid_penalty = 0.0 if mode == "inverse_frequency" else -1.0
    rewards: List[float] = []
    for idx, (comp, target) in enumerate(zip(completions, label)):
        parsed = parsed_completions[idx] if parsed_completions is not None else parse_completion(comp)
        pred = parsed["answer_norm"]
        target_norm = str(target).lower().strip()
        if target_norm not in matrix:
            raise ValueError(f"Unknown emotion target: {target!r}")
        rewards.append(float(matrix[target_norm][pred]) if pred in matrix[target_norm] else invalid_penalty)
    return rewards
