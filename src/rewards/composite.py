"""Composite GRPO reward manager and mathematical self-check (exact match to ser_grpo_v3.ipynb)."""

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import CUE_COLUMNS, CUE_LABELS, CUES, DEFAULT_REWARD_WEIGHTS, EMOTIONS, RewardConfig
from src.rewards.accuracy import (
    calibrate_emotion_payoffs,
    prove_payoff_matrix,
    resolve_inverse_frequency_alpha,
    resolve_reward_mode,
    reward_label_accuracy_weighted,
)
from src.rewards.acoustic import (
    calibrate_acoustic_payoffs,
    resolve_acoustic_reward_mode,
    reward_acoustic_inventory,
    score_cue_payoff,
)
from src.rewards.format import reward_format
from src.rewards.parser import parse_completion


class GRPORewardCalibrator:
    """Holds empirical R1/R2 calibrations derived strictly from `grpo_train_set` and exposes TRL reward callable."""

    def __init__(
        self,
        grpo_train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        config: Optional[RewardConfig] = None,
    ) -> None:
        self.config = config or RewardConfig()
        self.mode = resolve_reward_mode(
            self.config.use_zero_mean,
            self.config.use_map,
            self.config.use_inverse_frequency,
        )
        self.r1_alpha = resolve_inverse_frequency_alpha(self.config.r1_inverse_frequency_alpha)
        self.alpha = resolve_inverse_frequency_alpha(self.config.inverse_frequency_alpha)
        self.acoustic_mode = resolve_acoustic_reward_mode(
            self.config.use_zero_mean,
            self.config.use_map,
            self.config.use_inverse_frequency,
            self.config.use_distance_based_payoff,
        )

        self.emotion_priors, self.emotion_payoffs, self.emotion_matrix = calibrate_emotion_payoffs(
            grpo_train_df,
            return_matrix=True,
            use_zero_mean=self.config.use_zero_mean,
            use_map=self.config.use_map,
            use_inverse_frequency=self.config.use_inverse_frequency,
            inverse_frequency_alpha=self.r1_alpha,
        )
        self.emotion_proof = prove_payoff_matrix(
            self.emotion_priors,
            self.emotion_matrix,
            use_zero_mean=self.config.use_zero_mean,
            use_map=self.config.use_map,
            use_inverse_frequency=self.config.use_inverse_frequency,
            inverse_frequency_alpha=self.r1_alpha,
        )

        self.acoustic_calibration = calibrate_acoustic_payoffs(
            grpo_train_df,
            val_df,
            use_zero_mean=self.config.use_zero_mean,
            use_map=self.config.use_map,
            use_inverse_frequency=self.config.use_inverse_frequency,
            use_distance_based_payoff=self.config.use_distance_based_payoff,
            inverse_frequency_alpha=self.alpha,
        )
        self.cue_priors = self.acoustic_calibration["priors"]
        self.cue_payoffs = self.acoustic_calibration["payoffs"]
        self.cue_matrices = self.acoustic_calibration["matrices"]
        self.cue_proofs = self.acoustic_calibration["proofs"]
        self.normalization_scale = float(self.acoustic_calibration["normalization_scale"])

        lookup_sources = [
            df for df in (grpo_train_df, val_df) if df is not None and "audio_path" in df.columns
        ]
        if lookup_sources:
            lookup_df = pd.concat(lookup_sources, ignore_index=True).drop_duplicates("audio_path")
            cue_cols = [CUE_COLUMNS[c] for c in CUES]
            self.audio_lookup = lookup_df.set_index("audio_path")[cue_cols].to_dict(orient="index")
        else:
            self.audio_lookup = {}

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        w_r1: Optional[float] = None,
        w_r2: Optional[float] = None,
        w_r3: Optional[float] = None,
        **kwargs: Any,
    ) -> List[float]:
        return unified_gated_grpo_reward(
            prompts,
            completions,
            w_r1=self.config.reward_weights["r1"] if w_r1 is None else w_r1,
            w_r2=self.config.reward_weights["r2"] if w_r2 is None else w_r2,
            w_r3=self.config.reward_weights["r3"] if w_r3 is None else w_r3,
            emotion_priors=kwargs.pop("emotion_priors", self.emotion_priors),
            priors_dict=kwargs.pop("priors_dict", self.cue_priors),
            audio_lookup=kwargs.pop("audio_lookup", self.audio_lookup),
            use_zero_mean=kwargs.pop("use_zero_mean", self.config.use_zero_mean),
            use_map=kwargs.pop("use_map", self.config.use_map),
            use_inverse_frequency=kwargs.pop("use_inverse_frequency", self.config.use_inverse_frequency),
            r1_inverse_frequency_alpha=kwargs.pop("r1_inverse_frequency_alpha", self.r1_alpha),
            inverse_frequency_alpha=kwargs.pop("inverse_frequency_alpha", self.alpha),
            use_distance_based_payoff=kwargs.pop(
                "use_distance_based_payoff", self.config.use_distance_based_payoff
            ),
            **kwargs,
        )


def unified_gated_grpo_reward(
    prompts: List[str],
    completions: List[str],
    w_r1: Optional[float] = None,
    w_r2: Optional[float] = None,
    w_r3: Optional[float] = None,
    **kwargs: Any,
) -> List[float]:
    mode = resolve_reward_mode(
        kwargs.get("use_zero_mean"), kwargs.get("use_map"), kwargs.get("use_inverse_frequency")
    )
    w_r1 = DEFAULT_REWARD_WEIGHTS["r1"] if w_r1 is None else float(w_r1)
    w_r2 = DEFAULT_REWARD_WEIGHTS["r2"] if w_r2 is None else float(w_r2)
    w_r3 = DEFAULT_REWARD_WEIGHTS["r3"] if w_r3 is None else float(w_r3)
    trace_sink = kwargs.pop("_reward_trace_sink", None)

    if "parsed_completions" not in kwargs:
        kwargs["parsed_completions"] = [parse_completion(c) for c in completions]

    r1_alpha = kwargs.pop("r1_inverse_frequency_alpha", kwargs.get("inverse_frequency_alpha"))
    r1_kwargs = dict(kwargs, inverse_frequency_alpha=r1_alpha)
    r1_scores = reward_label_accuracy_weighted(prompts, completions, **r1_kwargs)
    cue_trace: List[Dict[str, Any]] = []
    r2_scores, r2_gate_scores = reward_acoustic_inventory(
        prompts, completions, return_gate_scores=True, _cue_trace_sink=cue_trace, **kwargs
    )
    r3_scores = reward_format(prompts, completions, **kwargs)

    # Record per-example R1 inverse-frequency hit weight so compute_span_split_advantages
    # can apply class weighting OUTSIDE group_standardize (preventing cancellation on imbalanced splits).
    r1_class_weights: List[float] = [1.0] * len(completions)
    if mode == "inverse_frequency" and "emotion_priors" in kwargs and "label" in kwargs:
        from src.rewards.accuracy import compute_hit_miss

        payoffs = compute_hit_miss(
            kwargs["emotion_priors"],
            use_zero_mean=kwargs.get("use_zero_mean"),
            use_map=kwargs.get("use_map"),
            use_inverse_frequency=kwargs.get("use_inverse_frequency"),
            inverse_frequency_alpha=r1_alpha,
        )
        r1_class_weights = [
            float(payoffs[str(lbl).lower().strip()]["hit"]) for lbl in kwargs["label"]
        ]

    assert len(r1_scores) == len(completions)
    assert len(r2_scores) == len(completions)
    assert len(r3_scores) == len(completions)

    unified_scores: List[float] = []
    for idx, (r1, r2, r3) in enumerate(zip(r1_scores, r2_scores, r3_scores)):
        r1_val = float(r1)
        r2_val = max(0.0, min(1.0, float(r2)))
        r3_val = max(0.0, min(1.0, float(r3)))
        total = w_r1 * r1_val + w_r2 * r2_val + w_r3 * r3_val
        unified_scores.append(float(total))
        if trace_sink is not None:
            trace_sink.append(
                {
                    "r2_cues": cue_trace[idx],
                    "r1": r1_val,
                    "r1_class_weight": float(r1_class_weights[idx]),
                    "r2": r2_val,
                    "r3": r3_val,
                    "r2_gate": float(r2_gate_scores[idx]),
                    "total": float(total),
                }
            )
    return unified_scores


def build_sample_completion(
    speaking_rate: str = "moderate",
    pitch_level: str = "moderate",
    vocal_energy: str = "moderate",
    timbre: str = "moderate (modal)",
    emotion: str = "neutral",
) -> str:
    return (
        "<think>\n[Acoustic Inventory]\n"
        f"- speaking_rate: {speaking_rate}\n"
        f"- pitch_level: {pitch_level}\n"
        f"- vocal_energy: {vocal_energy}\n"
        f"- timbre: {timbre}\n"
        "[Prosodic Reasoning]\n"
        "1. The voice shows steady prosody across rate, pitch, energy, and timbre.\n"
        "2. The delivery aligns with the spoken words without elevated arousal.\n"
        f"3. Taken together, these audible cues point to {emotion} rather than a competing state.\n"
        f"</think>\n<answer>{emotion}</answer>"
    )


def validate_grpo_reward_mathematics(
    use_zero_mean: Optional[bool] = None,
    use_map: Optional[bool] = None,
    use_inverse_frequency: Optional[bool] = None,
    use_distance_based_payoff: Optional[bool] = None,
    inverse_frequency_alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """Verify active-mode R1 and 4-cue ordinal R2 proofs on an imbalanced synthetic GRPO split."""
    mode = resolve_reward_mode(use_zero_mean, use_map, use_inverse_frequency)
    alpha = resolve_inverse_frequency_alpha(inverse_frequency_alpha)
    acoustic_mode = resolve_acoustic_reward_mode(
        use_zero_mean, use_map, use_inverse_frequency, use_distance_based_payoff
    )
    zm_flag = mode == "zero_mean"
    map_flag = mode == "map"
    ifreq_flag = mode == "inverse_frequency"

    cue_counts = {
        "speaking_rate": (4, 9, 7),
        "pitch_level": (8, 5, 7),
        "vocal_energy": (3, 11, 6),
        "timbre": (5, 10, 5),
    }
    cue_series = {}
    for cue, (n_l, n_m, n_h) in cue_counts.items():
        l_lbl, m_lbl, h_lbl = CUE_LABELS[cue]
        cue_series[CUE_COLUMNS[cue]] = [l_lbl] * n_l + [m_lbl] * n_m + [h_lbl] * n_h
    labels = [
        e
        for e, n in {
            "neutral": 7,
            "joy": 4,
            "anger": 3,
            "sadness": 2,
            "fear": 2,
            "disgust": 1,
            "surprise": 1,
        }.items()
        for _ in range(n)
    ]
    grpo = pd.DataFrame({"audio_path": [f"grpo_{i}.wav" for i in range(len(labels))], "emotion": labels, **cue_series})

    calibration = calibrate_acoustic_payoffs(
        grpo,
        use_zero_mean=zm_flag,
        use_map=map_flag,
        use_inverse_frequency=ifreq_flag,
        use_distance_based_payoff=use_distance_based_payoff,
        inverse_frequency_alpha=alpha,
    )
    emotion_priors, _, emotion_matrix = calibrate_emotion_payoffs(
        grpo,
        return_matrix=True,
        use_zero_mean=zm_flag,
        use_map=map_flag,
        use_inverse_frequency=ifreq_flag,
        inverse_frequency_alpha=alpha,
    )
    r1_proof = prove_payoff_matrix(
        emotion_priors,
        emotion_matrix,
        use_zero_mean=zm_flag,
        use_map=map_flag,
        use_inverse_frequency=ifreq_flag,
        inverse_frequency_alpha=alpha,
    )
    if mode in ("zero_mean", "map"):
        assert r1_proof["unit_contrast_satisfied"] and r1_proof["prior_random_zero_mean_satisfied"]
        assert r1_proof["zero_mean_satisfied"] is (mode == "zero_mean")
        assert r1_proof["map_satisfied"] is (mode == "map")
    else:
        assert r1_proof["inverse_frequency_satisfied"] is True

    kwargs = {
        "label": labels,
        "audio_path": grpo["audio_path"].tolist(),
        "emotion_priors": emotion_priors,
        "priors_dict": calibration["priors"],
        "use_distance_based_payoff": use_distance_based_payoff,
        "use_zero_mean": zm_flag,
        "use_map": map_flag,
        "use_inverse_frequency": ifreq_flag,
        "inverse_frequency_alpha": alpha,
        **{CUE_COLUMNS[c]: grpo[CUE_COLUMNS[c]].tolist() for c in CUES},
    }
    prompts = ["p"] * len(grpo)
    r1_blind_means = {}
    r1_inv_norm = math.fsum(q ** (1.0 - alpha) for q in emotion_priors.values())
    for emotion in EMOTIONS:
        completions = [build_sample_completion(emotion=emotion)] * len(grpo)
        mean = float(np.mean(reward_label_accuracy_weighted(prompts, completions, **kwargs)))
        if mode == "zero_mean":
            assert abs(mean) < 1e-12
        elif mode == "inverse_frequency":
            assert abs(mean - ((emotion_priors[emotion] ** (1.0 - alpha)) / r1_inv_norm)) < 1e-12
        r1_blind_means[emotion] = mean

    cue_blind_raw_means = {}
    for cue, priors in calibration["priors"].items():
        cue_blind_raw_means[cue] = {}
        cue_inv_norm = math.fsum(q ** (1.0 - alpha) for q in priors.values())
        for pred in CUE_LABELS[cue]:
            expected = sum(
                p
                * score_cue_payoff(
                    truth,
                    pred,
                    cue,
                    calibration["priors"],
                    use_zero_mean=zm_flag,
                    use_map=map_flag,
                    use_inverse_frequency=ifreq_flag,
                    use_distance_based_payoff=use_distance_based_payoff,
                    inverse_frequency_alpha=alpha,
                )[0]
                for truth, p in priors.items()
            )
            if acoustic_mode in ("zero_mean", "distance_based"):
                assert abs(expected) < 1e-12
            elif acoustic_mode == "inverse_frequency":
                assert abs(expected - ((priors[pred] ** (1.0 - alpha)) / cue_inv_norm)) < 1e-12
            cue_blind_raw_means[cue][pred] = expected

    return {
        "mode": mode,
        "acoustic_mode": acoustic_mode,
        "r1_blind_means": r1_blind_means,
        "r1_matrix_proof": r1_proof,
        "cue_blind_raw_means": cue_blind_raw_means,
        "r2_matrix_proofs": calibration["proofs"],
    }
