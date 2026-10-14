"""R2: Acoustic Inventory physical grounding reward.

Evaluates whether the discrete acoustic cues generated during Chain-of-Thought
reasoning accurately ground to empirical physical eGeMAPS functionals using
speaker-calibrated quartiles for the top speakers, including composite Voice Quality
(Jitter, Shimmer, HNR, Hammarberg index).
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import pandas as pd

from src.rewards.parser import extract_acoustic_inventory, parse_completion
from src.utils.logging import get_logger

logger = get_logger("rewards.acoustic")

# Mapping of the 4 continuous acoustic descriptors in the prompt to eGeMAPSv02 features
CUE_FEATURE_MAP: Dict[str, str] = {
    "pitch_height": "F0semitoneFrom27.5Hz_sma3nz_amean",
    "pitch_dynamics": "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "vocal_energy": "loudness_sma3_percentile50.0",
    "speaking_rate": "VoicedSegmentsPerSec",
}

# Physical features used for composite Voice Quality grounding
VOICE_QUALITY_FEATURE_MAP: Dict[str, str] = {
    "jitter": "jitterLocal_sma3nz_amean",
    "shimmer": "shimmerLocaldB_sma3nz_amean",
    "hnr": "HNRdBACF_sma3nz_amean",
    "hammarberg": "hammarbergIndexV_sma3nz_amean",
}

# Comprehensive dictionary of all acoustic features
ALL_ACOUSTIC_FEATURES: Dict[str, str] = {
    **CUE_FEATURE_MAP,
    **VOICE_QUALITY_FEATURE_MAP,
}

# Normalization of continuous cue prompt vocabulary to the 3 ordinal tiers (low, moderate, high)
PRED_TIER_MAP: Dict[str, Dict[str, str]] = {
    "pitch_height": {
        "low": "low",
        "moderate": "moderate",
        "elevated": "high",
        "high": "high",
    },
    "pitch_dynamics": {
        "narrow": "low",
        "low": "low",
        "moderate": "moderate",
        "wide": "high",
        "high": "high",
    },
    "vocal_energy": {
        "quiet": "low",
        "low": "low",
        "moderate": "moderate",
        "loud": "high",
        "high": "high",
    },
    "speaking_rate": {
        "slow": "low",
        "low": "low",
        "moderate": "moderate",
        "fast": "high",
        "high": "high",
    },
}

# Canonical normalization map for Voice Quality choices
VOICE_QUALITY_CHOICES: Dict[str, str] = {
    "pressed/tense": "pressed/tense",
    "pressed": "pressed/tense",
    "tense": "pressed/tense",
    "breathy": "breathy",
    "harsh/creaky": "harsh/creaky",
    "harsh": "harsh/creaky",
    "creaky": "harsh/creaky",
    "modal/normal": "modal/normal",
    "modal": "modal/normal",
    "normal": "modal/normal",
}


def compute_top_speaker_acoustic_quantiles(
    df: Any,
    top_n: int = 6,
    cues_map: Dict[str, str] = ALL_ACOUSTIC_FEATURES,
    filter_zeroes: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Computes the 4 quantiles (quartiles: Q25 and Q75) for the top N most frequent speakers.

    When filter_zeroes is True, excludes 0.0 values (unvoiced/corrupted frames)
    from quantile calculation to ensure accurate physical grounding.

    Quantile divisions:
      - 1st quantile (x <= Q25): Low
      - 2nd & 3rd quantiles (Q25 < x <= Q75): Moderate
      - 4th quantile (x > Q75): High

    Includes '__GLOBAL__' as fallback for speakers outside top N.
    """
    spk_counts = df["speaker"].value_counts()
    top_speakers = spk_counts.head(top_n).index.tolist()

    quantiles_dict: Dict[str, Dict[str, Dict[str, float]]] = {}

    strictly_positive_cues = {
        "F0semitoneFrom27.5Hz_sma3nz_amean",
        "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
        "loudness_sma3_percentile50.0",
        "VoicedSegmentsPerSec",
        "jitterLocal_sma3nz_amean",
        "shimmerLocaldB_sma3nz_amean",
    }

    for spk in top_speakers:
        sub = df[df["speaker"] == spk]
        quantiles_dict[spk] = {}
        for cue_name, col_name in cues_map.items():
            if col_name in sub.columns:
                series = sub[col_name].dropna()
                if filter_zeroes:
                    series = series[series != 0.0]
                    if col_name in strictly_positive_cues:
                        series = series[series > 0.0]
                if len(series) > 0:
                    quantiles_dict[spk][cue_name] = {
                        "q25": float(series.quantile(0.25)),
                        "q75": float(series.quantile(0.75)),
                    }

    # Global fallback for speakers with few utterances or out-of-top-N
    quantiles_dict["__GLOBAL__"] = {}
    for cue_name, col_name in cues_map.items():
        if col_name in df.columns:
            series = df[col_name].dropna()
            if filter_zeroes:
                series = series[series != 0.0]
                if col_name in strictly_positive_cues:
                    series = series[series > 0.0]
            if len(series) > 0:
                quantiles_dict["__GLOBAL__"][cue_name] = {
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                }

    logger.info("Computed acoustic quartiles for top %d speakers (filter_zeroes=%s): %s", len(top_speakers), filter_zeroes, top_speakers)
    return quantiles_dict


def build_acoustic_audio_lookup(
    df: Any,
    features_map: Dict[str, str] = ALL_ACOUSTIC_FEATURES,
) -> Dict[str, Dict[str, Any]]:
    """Builds O(1) audio_path indexed lookup dictionary for rapid rollout evaluation."""
    if "audio_path" not in df.columns:
        return {}
    cols_to_keep = [c for c in list(features_map.values()) + ["speaker"] if c in df.columns]
    return df.set_index("audio_path")[cols_to_keep].to_dict(orient="index")


def score_acoustic_piece(
    truth_tier: str,
    pred_tier: Optional[str],
    cue_name: Optional[str] = None,
    weights_dict: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """Computes the reward score for an individual continuous acoustic dimension.

    Calibrated via square-root smoothed inverse frequency:
      - Match: calibrated weight w_c(truth_tier) in (0.0, 1.0].
      - Mismatch / Missing / None: 0.0.
    """
    if pred_tier is None or not truth_tier:
        return 0.0

    t = str(truth_tier).lower().strip()
    p = str(pred_tier).lower().strip()

    if p != t:
        return 0.0

    if weights_dict and cue_name and cue_name in weights_dict and t in weights_dict[cue_name]:
        return float(weights_dict[cue_name][t])

    # Smooth theoretical fallback for quartiles (25% low, 50% moderate, 25% high):
    # sqrt(0.25 / 0.50) = 0.7071 for moderate; 1.0 for low/high
    if t == "moderate":
        return 0.7071
    return 1.0


def determine_voice_quality_ground_truth(
    jitter_val: float,
    shimmer_val: float,
    hnr_val: float,
    hammarberg_val: float,
    quantiles: Dict[str, Dict[str, float]],
) -> str:
    """Determines physical Voice Quality ground truth from speaker-calibrated quantiles
    of Jitter, Shimmer, HNR, and Hammarberg index.

    Acoustic Phonetics Criteria:
      - 'harsh/creaky': Elevated cycle-to-cycle perturbation / aperiodicity (high Jitter/Shimmer, low HNR).
      - 'breathy': Incomplete glottal closure, steep spectral roll-off (high Hammarberg) or aspiration noise (low HNR).
      - 'pressed/tense': Vocal fold hyperadduction, flat spectral tilt with boosted higher harmonics (low Hammarberg).
      - 'modal/normal': Balanced, regular vocal fold vibration (median/moderate acoustics).
    """
    def _tier(cue_name: str, val: float) -> str:
        q = quantiles.get(cue_name, {})
        q25 = q.get("q25", float("-inf"))
        q75 = q.get("q75", float("inf"))
        if val <= q25:
            return "low"
        elif val <= q75:
            return "moderate"
        else:
            return "high"

    j_tier = _tier("jitter", jitter_val)
    s_tier = _tier("shimmer", shimmer_val)
    h_tier = _tier("hnr", hnr_val)
    tilt_tier = _tier("hammarberg", hammarberg_val)

    # 1. Harsh / Creaky: High perturbation / cycle-to-cycle instability
    if (j_tier == "high" and s_tier == "high") or \
       ((j_tier == "high" or s_tier == "high") and h_tier == "low"):
        return "harsh/creaky"

    # 2. Breathy: Steep spectral tilt (fundamental dominates) or aspiration noise
    if tilt_tier == "high" and j_tier != "high":
        return "breathy"
    if tilt_tier == "moderate" and h_tier == "low" and j_tier != "high" and s_tier != "high":
        return "breathy"

    # 3. Pressed / Tense: Hyperadduction (flat tilt, boosted 2-5 kHz band) with periodic vibration
    if tilt_tier == "low" and j_tier != "high" and s_tier != "high":
        return "pressed/tense"

    # Isolated elevated perturbation leans harsh / creaky
    if j_tier == "high" or s_tier == "high":
        return "harsh/creaky"

    # 4. Modal / Normal: Balanced phonation
    return "modal/normal"


def score_voice_quality_piece(
    truth_quality: str,
    pred_quality: Optional[str],
    weights_dict: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """Computes the reward score for the Voice Quality dimension.

    Calibrated via square-root smoothed inverse frequency:
      - Match: calibrated weight w_vq(truth_quality) in (0.0, 1.0].
      - Mismatch / Missing / None: 0.0.
    """
    if pred_quality is None or not truth_quality:
        return 0.0

    t = str(truth_quality).lower().strip()
    p = str(pred_quality).lower().strip()

    if p != t:
        return 0.0

    if weights_dict and "voice_quality" in weights_dict and t in weights_dict["voice_quality"]:
        return float(weights_dict["voice_quality"][t])

    # Fallback when weights are not calibrated
    if t == "modal/normal":
        return 0.7071
    return 1.0


def compute_acoustic_cue_weights(
    train_df: pd.DataFrame,
    quantiles_dict: Dict[str, Dict[str, Dict[str, float]]],
    power: float = 0.5,
    cues_map: Dict[str, str] = CUE_FEATURE_MAP,
    vq_map: Dict[str, str] = VOICE_QUALITY_FEATURE_MAP,
) -> Dict[str, Dict[str, float]]:
    """Computes calibrated weights via square-root smoothed inverse frequency (power=0.5)
    for each category across all 5 acoustic cues on the GRPO training set.

    Formula:
      w_c(k) = (min_j f_c(j) / f_c(k)) ** 0.5

    Guarantees:
      1. Every acoustic cue is calibrated to the empirical distribution in the dataset.
      2. The maximum score per cue is exactly 1.0 (scale [0.0, 1.0] per cue).
      3. Normalized reward is strictly in [0.0, 1.0].
      4. Eliminates arbitrary 'moderate' penalties and blind extreme bonuses.
    """
    counts: Dict[str, Dict[str, int]] = {
        cue: {"low": 0, "moderate": 0, "high": 0} for cue in cues_map
    }
    counts["voice_quality"] = {
        "modal/normal": 0,
        "pressed/tense": 0,
        "breathy": 0,
        "harsh/creaky": 0,
    }

    for _, row in train_df.iterrows():
        spk = row.get("speaker", "__GLOBAL__")
        spk_q = quantiles_dict.get(spk, quantiles_dict.get("__GLOBAL__", {}))

        for cue_name, feat_col in cues_map.items():
            if feat_col in row and not pd.isna(row[feat_col]) and cue_name in spk_q:
                val = float(row[feat_col])
                q25 = spk_q[cue_name]["q25"]
                q75 = spk_q[cue_name]["q75"]
                if val <= q25:
                    counts[cue_name]["low"] += 1
                elif val <= q75:
                    counts[cue_name]["moderate"] += 1
                else:
                    counts[cue_name]["high"] += 1

        vq_vals = {}
        for sub_cue, col_name in vq_map.items():
            if col_name in row and not pd.isna(row[col_name]):
                vq_vals[sub_cue] = float(row[col_name])

        if len(vq_vals) == len(vq_map) and all(k in spk_q for k in vq_vals):
            vq_gt = determine_voice_quality_ground_truth(
                jitter_val=vq_vals["jitter"],
                shimmer_val=vq_vals["shimmer"],
                hnr_val=vq_vals["hnr"],
                hammarberg_val=vq_vals["hammarberg"],
                quantiles=spk_q,
            )
            if vq_gt in counts["voice_quality"]:
                counts["voice_quality"][vq_gt] += 1

    weights_dict: Dict[str, Dict[str, float]] = {}

    for cue_name, cat_counts in counts.items():
        total_valid = sum(cat_counts.values())
        weights_dict[cue_name] = {}

        if total_valid == 0:
            for cat in cat_counts:
                weights_dict[cue_name][cat] = 0.7071 if cat in ("moderate", "modal/normal") else 1.0
            continue

        num_cats = len(cat_counts)
        freqs = {
            cat: (cnt + 1) / (total_valid + num_cats)
            for cat, cnt in cat_counts.items()
        }

        raw_inv = {cat: (1.0 / f) ** power for cat, f in freqs.items()}
        max_inv = max(raw_inv.values())

        for cat, inv_val in raw_inv.items():
            weights_dict[cue_name][cat] = round(float(inv_val / max_inv), 4)

    return weights_dict


class AcousticInventoryReward:
    """R2 Reward function evaluating Acoustic Inventory grounding against speaker quartiles."""

    def __init__(
        self,
        quantiles_ref: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        weights_ref: Optional[Dict[str, Dict[str, float]]] = None,
        audio_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
        normalize: bool = False,
    ):
        self.quantiles_ref = quantiles_ref or {}
        self.weights_ref = weights_ref or {}
        self.audio_lookup = audio_lookup or {}
        self.normalize = normalize

    def set_quantiles(self, quantiles_ref: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        """Update active speaker quantiles reference."""
        self.quantiles_ref = quantiles_ref

    def set_weights(self, weights_ref: Dict[str, Dict[str, float]]) -> None:
        """Update calibrated acoustic cue weights reference."""
        self.weights_ref = weights_ref

    def set_audio_lookup(self, audio_lookup: Dict[str, Dict[str, Any]]) -> None:
        """Update audio_path -> feature cache lookup."""
        self.audio_lookup = audio_lookup

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs: Any,
    ) -> List[float]:
        """Compute R2 acoustic inventory rewards for a batch of GRPO rollouts.

        Returns:
            List of float rewards per completion.
        """
        rewards: List[float] = []

        speakers = kwargs.get("speaker", [None] * len(completions))
        audio_paths = kwargs.get("audio_path", [None] * len(completions))
        parsed_list = kwargs.get("parsed_completions")
        active_weights = kwargs.get("acoustic_cue_weights", self.weights_ref)

        for idx, comp in enumerate(completions):
            if parsed_list is not None and idx < len(parsed_list):
                parsed_inv = parsed_list[idx]["inventory"]
            else:
                parsed_inv = parse_completion(comp)["inventory"]

            spk = speakers[idx] if idx < len(speakers) else None
            audio_p = audio_paths[idx] if idx < len(audio_paths) else None

            lookup_data = self.audio_lookup.get(audio_p, {}) if audio_p else {}
            if spk is None:
                spk = lookup_data.get("speaker", "__GLOBAL__")

            spk_q = self.quantiles_ref.get(
                spk, self.quantiles_ref.get("__GLOBAL__", {})
            )

            cue_scores: Dict[str, float] = {}

            # 1. Evaluate 4 continuous physical acoustic dimensions
            for cue_name, feat_col in CUE_FEATURE_MAP.items():
                val = None
                if feat_col in kwargs and idx < len(kwargs[feat_col]):
                    val = kwargs[feat_col][idx]
                elif feat_col in lookup_data:
                    val = lookup_data[feat_col]

                if val is None or cue_name not in spk_q:
                    continue

                q25 = spk_q[cue_name]["q25"]
                q75 = spk_q[cue_name]["q75"]

                # Physical truth quartile classification
                if val <= q25:
                    truth_tier = "low"
                elif val <= q75:
                    truth_tier = "moderate"
                else:
                    truth_tier = "high"

                # Parse and map model prediction
                raw_pred = parsed_inv.get(cue_name)
                pred_tier = None
                if raw_pred:
                    raw_clean = str(raw_pred).lower().strip()
                    pred_tier = PRED_TIER_MAP.get(cue_name, {}).get(raw_clean)

                cue_scores[cue_name] = score_acoustic_piece(
                    truth_tier, pred_tier, cue_name=cue_name, weights_dict=active_weights
                )

            # 2. Evaluate composite Voice Quality (Jitter, Shimmer, HNR, Hammarberg)
            vq_vals = {}
            for sub_cue, col_name in VOICE_QUALITY_FEATURE_MAP.items():
                v = None
                if col_name in kwargs and idx < len(kwargs[col_name]):
                    v = kwargs[col_name][idx]
                elif col_name in lookup_data:
                    v = lookup_data[col_name]
                if v is not None:
                    vq_vals[sub_cue] = float(v)

            if len(vq_vals) == len(VOICE_QUALITY_FEATURE_MAP):
                truth_vq = determine_voice_quality_ground_truth(
                    jitter_val=vq_vals["jitter"],
                    shimmer_val=vq_vals["shimmer"],
                    hnr_val=vq_vals["hnr"],
                    hammarberg_val=vq_vals["hammarberg"],
                    quantiles=spk_q,
                )
                raw_vq_pred = parsed_inv.get("voice_quality")
                pred_vq = None
                if raw_vq_pred:
                    raw_vq_clean = str(raw_vq_pred).lower().strip()
                    pred_vq = VOICE_QUALITY_CHOICES.get(raw_vq_clean)

                cue_scores["voice_quality"] = score_voice_quality_piece(
                    truth_vq, pred_vq, weights_dict=active_weights
                )

            if cue_scores:
                total = sum(cue_scores.values())
                score = (total / len(cue_scores)) if self.normalize else total
                rewards.append(round(float(score), 3))
            else:
                rewards.append(0.0)

        return rewards


def reward_acoustic_inventory(
    prompts: List[str],
    completions: List[str],
    quantiles_ref: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    weights_ref: Optional[Dict[str, Dict[str, float]]] = None,
    audio_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    normalize: bool = False,
    **kwargs: Any,
) -> List[float]:
    """Functional wrapper for AcousticInventoryReward."""
    rewarder = AcousticInventoryReward(
        quantiles_ref=quantiles_ref,
        weights_ref=weights_ref,
        audio_lookup=audio_lookup,
        normalize=normalize,
    )
    return rewarder(prompts, completions, **kwargs)
