"""openSMILE eGeMAPSv02 extraction and training integrity filtering (exact match to data_processing.ipynb)."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from src.config import ACOUSTIC_CUES, ACOUSTIC_FEATURE_COLS

_SMILE_EXTRACTOR = None


def _get_smile_extractor():
    global _SMILE_EXTRACTOR
    if _SMILE_EXTRACTOR is None:
        import opensmile

        _SMILE_EXTRACTOR = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _SMILE_EXTRACTOR


def _extract_egemaps_for_row(task: Tuple[int, str]) -> Tuple[int, Dict[str, float], Optional[str]]:
    idx, audio_path = task
    if not audio_path or not Path(audio_path).exists():
        return idx, {col: float("nan") for col in ACOUSTIC_FEATURE_COLS}, f"Missing file: {audio_path}"
    try:
        feats_df = _get_smile_extractor().process_file(str(audio_path))
        row_feats = {
            col: float(feats_df[col].iloc[0]) if col in feats_df.columns else float("nan")
            for col in ACOUSTIC_FEATURE_COLS
        }
        return idx, row_feats, None
    except Exception as exc:
        return idx, {col: float("nan") for col in ACOUSTIC_FEATURE_COLS}, str(exc)


def extract_egemaps_cues(
    df: pd.DataFrame,
    split_name: str,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Extract the 5 ACOUSTIC_CUES (eGeMAPSv02 Functionals) in parallel for a split DataFrame."""
    out = df.copy().reset_index(drop=True)
    if "duration" not in out.columns or out["duration"].isna().any():
        if "time" in out.columns:
            out["duration"] = pd.to_numeric(out["time"], errors="coerce")
        else:
            out["duration"] = [
                float(sf.info(str(p)).duration) if p and Path(str(p)).exists() else float("nan")
                for p in out["audio_path"]
            ]

    tasks = [(idx, str(row["audio_path"])) for idx, row in out.iterrows()]
    extracted_rows: List[Optional[Dict[str, float]]] = [None] * len(out)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_extract_egemaps_for_row, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"eGeMAPSv02 [{split_name}]"):
            idx, feat_dict, _ = fut.result()
            extracted_rows[idx] = feat_dict

    feats_table = pd.DataFrame(extracted_rows)
    for col in ACOUSTIC_FEATURE_COLS:
        out[col] = feats_table[col].values
    for alias, col in ACOUSTIC_CUES.items():
        out[alias] = out[col]
    return out


def filter_training_set_by_duration_and_nonzero_cues(
    df: pd.DataFrame,
    split_name: str = "train",
    min_dur: float = 0.5,
    max_dur: float = 20.0,
    delete_rejected_wavs_in_data_dir: bool = True,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Remove training utterances outside [min_dur, max_dur] or with any zero/NaN eGeMAPSv02 feature."""
    dur_series = pd.to_numeric(df["duration"], errors="coerce")
    bad_dur = dur_series.isna() | (dur_series < min_dur) | (dur_series > max_dur)

    bad_cues = pd.Series(False, index=df.index)
    for _, col in ACOUSTIC_CUES.items():
        s = pd.to_numeric(df[col], errors="coerce")
        bad_cues = bad_cues | s.isna() | (s == 0.0)

    reject_mask = bad_dur | bad_cues
    rejected_df = df.loc[reject_mask]
    filtered_df = df.loc[~reject_mask].reset_index(drop=True)

    if delete_rejected_wavs_in_data_dir and data_dir is not None:
        data_root = Path(data_dir).resolve()
        for wav_str in rejected_df["audio_path"].dropna().unique():
            wav_p = Path(str(wav_str)).resolve()
            if wav_p.exists() and wav_p.is_relative_to(data_root):
                try:
                    wav_p.unlink()
                except OSError:
                    pass
    return filtered_df
