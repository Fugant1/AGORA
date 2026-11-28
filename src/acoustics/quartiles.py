"""Gender-specific acoustic quartile calibration and classification (exact match to data_processing.ipynb)."""

from typing import Dict, List, Optional

import pandas as pd

from src.config import ACOUSTIC_CUES, TIMBRE_LABEL_MAP


def compute_gender_acoustic_quartiles(
    train_df: pd.DataFrame,
    cues_map: Dict[str, str] = ACOUSTIC_CUES,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute Q25/Q50/Q75 strictly on clean_train_set separately for 'F' and 'M', plus '__GLOBAL__' and F0 midpoint."""
    if "gender" not in train_df.columns:
        raise ValueError("Column 'gender' ('F'/'M') not found in training DataFrame.")

    quartiles: Dict[str, Dict[str, Dict[str, float]]] = {"F": {}, "M": {}, "__GLOBAL__": {}}
    for gender_key in ("F", "M"):
        sub = train_df.loc[train_df["gender"] == gender_key]
        if len(sub) == 0:
            raise ValueError(f"No samples found for gender={gender_key!r} in training DataFrame.")
        for cue_alias, col_name in cues_map.items():
            src_col = col_name if col_name in sub.columns else cue_alias
            s = pd.to_numeric(sub[src_col], errors="coerce").dropna()
            s = s[s != 0.0]
            quartiles[gender_key][cue_alias] = {
                "q25": float(s.quantile(0.25)),
                "q50": float(s.quantile(0.50)),
                "q75": float(s.quantile(0.75)),
                "count": int(len(s)),
            }

    for cue_alias, col_name in cues_map.items():
        src_col = col_name if col_name in train_df.columns else cue_alias
        s = pd.to_numeric(train_df[src_col], errors="coerce").dropna()
        s = s[s != 0.0]
        quartiles["__GLOBAL__"][cue_alias] = {
            "q25": float(s.quantile(0.25)),
            "q50": float(s.quantile(0.50)),
            "q75": float(s.quantile(0.75)),
            "count": int(len(s)),
        }

    f0_m = quartiles["M"]["pitch_level"]["q50"]
    f0_f = quartiles["F"]["pitch_level"]["q50"]
    quartiles["__META__"] = {"f0_gender_boundary_st": float(0.5 * (f0_m + f0_f))}
    return quartiles


def classify_value_by_quartile(
    val: float,
    q25: float,
    q75: float,
    cue_alias: str,
) -> Optional[str]:
    """Classify numeric feature into <Q25 ('low'), >Q75 ('high'), else ('moderate'), with timbre mapping."""
    if val is None or pd.isna(val) or float(val) == 0.0:
        return None
    v = float(val)
    if v < q25:
        tier = "low"
    elif v > q75:
        tier = "high"
    else:
        tier = "moderate"
    return TIMBRE_LABEL_MAP[tier] if cue_alias == "timbre" else tier


def apply_gender_quartile_classification(
    df: pd.DataFrame,
    quartiles: Dict[str, Dict[str, Dict[str, float]]],
    split_name: str = "split",
    cues_map: Dict[str, str] = ACOUSTIC_CUES,
) -> pd.DataFrame:
    """Classify each acoustic cue into gender-relative quartile tiers using train-fitted quartiles."""
    out = df.copy().reset_index(drop=True)
    f0_boundary = quartiles.get("__META__", {}).get("f0_gender_boundary_st", 36.0)
    f0_col = cues_map["pitch_level"] if cues_map["pitch_level"] in out.columns else "pitch_level"

    resolved_genders: List[str] = []
    for _, row in out.iterrows():
        g = row.get("gender")
        if g in ("F", "M"):
            resolved_genders.append(g)
        else:
            f0_val = row.get(f0_col, float("nan"))
            if pd.notna(f0_val) and float(f0_val) > 0.0:
                resolved_genders.append("F" if float(f0_val) >= f0_boundary else "M")
            else:
                resolved_genders.append("__GLOBAL__")
    out["effective_gender"] = resolved_genders

    for cue_alias, col_name in cues_map.items():
        src_col = col_name if col_name in out.columns else cue_alias
        out[cue_alias] = pd.to_numeric(out[src_col], errors="coerce")
        labels: List[Optional[str]] = []
        for idx, row in out.iterrows():
            g_key = out.at[idx, "effective_gender"]
            q_info = quartiles.get(g_key, quartiles["__GLOBAL__"]).get(cue_alias)
            labels.append(
                classify_value_by_quartile(
                    row[cue_alias], q25=q_info["q25"], q75=q_info["q75"], cue_alias=cue_alias
                )
            )
        out[f"{cue_alias}_cue"] = labels
    return out
