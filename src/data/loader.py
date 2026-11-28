"""MELD split loading and parallel 16 kHz mono WAV standardization (exact match to data_processing.ipynb)."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from src.config import TARGET_CHANNELS, TARGET_SR
from src.utils.audio import resample_audio

SPLIT_CONFIG: Dict[str, Tuple[str, str]] = {
    "train": ("preprocessed_train_splits", "train"),
    "val": ("preprocessed_dev_splits_complete", "dev"),
    "test": ("preprocessed_output_repeated_splits_test", "test"),
}


def all_clean_outputs_cached(data_dir: Path) -> bool:
    """Return True when EMO-TAG, eGeMAPSv02, and quartile artifacts already exist locally."""
    clean_dir = Path(data_dir) / "emo_tag_clean"
    required = (
        "clean_train_set_with_cues.csv",
        "clean_val_set_with_cues.csv",
        "train_egemaps.csv",
        "val_egemaps.csv",
        "gender_acoustic_quartiles.json",
    )
    return all((clean_dir / name).is_file() for name in required)


def extract_tar(archive_path: Path, destination: Path) -> None:
    """Extract a .tar archive to destination."""
    archive_path = Path(archive_path)
    destination = Path(destination)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    destination.mkdir(parents=True, exist_ok=True)
    start = time.time()
    subprocess.run(["tar", "-xf", str(archive_path), "-C", str(destination)], check=True)
    print(f"Extraction of {archive_path.name} completed in {time.time() - start:.1f}s")


def find_file(root: Path, pattern: str) -> Path:
    matches = list(Path(root).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find '{pattern}' under {root}")
    return matches[0]


def resolve_split_dir(destination: Path, split: str) -> Tuple[Path, str]:
    destination = Path(destination)
    if split in SPLIT_CONFIG:
        folder_name, csv_prefix = SPLIT_CONFIG[split]
        candidate = destination / folder_name
        if candidate.exists():
            return candidate, csv_prefix
    prefix = "dev" if split in ("val", "dev") else split
    candidates = [
        destination / f"preprocessed_{prefix}_splits",
        destination / f"preprocessed_{prefix}_splits_complete",
        destination / f"preprocessed_output_repeated_splits_{prefix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, prefix
    raise FileNotFoundError(f"Split directory for '{split}' not found under {destination}")


def load_meld_split(split_dir: Path, split: str = "train") -> pd.DataFrame:
    """Load a MELD split and resolve audio paths against {split}_preprocessed.csv and {split}_sent_emo.csv."""
    split_dir = Path(split_dir)
    audio_files = list(split_dir.rglob("*.wav"))
    preprocessed_csv = find_file(split_dir, f"{split}_preprocessed.csv")
    emotion_csv = find_file(split_dir, f"{split}_sent_emo.csv")

    df = pd.read_csv(preprocessed_csv)
    df_emo = pd.read_csv(emotion_csv)

    audio_map = {path.name: str(path.resolve()) for path in audio_files}
    df["audio_path"] = df["audio_path"].astype(str).map(lambda x: Path(x).name).map(audio_map)

    df["dialogue"] = df["dialogue"].astype(int)
    df["utterance"] = df["utterance"].astype(int)
    df_emo["Dialogue_ID"] = df_emo["Dialogue_ID"].astype(int)
    df_emo["Utterance_ID"] = df_emo["Utterance_ID"].astype(int)

    df = df.merge(
        df_emo[["Dialogue_ID", "Utterance_ID", "Emotion", "Utterance", "Speaker"]],
        left_on=["dialogue", "utterance"],
        right_on=["Dialogue_ID", "Utterance_ID"],
        how="left",
    )
    df = df.drop(columns=["Dialogue_ID", "Utterance_ID"]).rename(
        columns={"Utterance": "text", "Emotion": "emotion", "Speaker": "speaker"}
    )
    if "time" in df.columns and "duration" not in df.columns:
        df["duration"] = pd.to_numeric(df["time"], errors="coerce")
    return df


def load_meld(
    destination: Path,
    split: str = "train",
    archive_path: Optional[Path] = None,
) -> pd.DataFrame:
    destination = Path(destination)
    if not destination.exists():
        if archive_path is None:
            raise FileNotFoundError(f"MELD directory not found and no archive_path provided: {destination}")
        extract_tar(archive_path, destination)
    split_dir, csv_prefix = resolve_split_dir(destination, split)
    return load_meld_split(split_dir, csv_prefix)


def _process_single_audio(
    task: Tuple[int, str, Path, int, int, bool]
) -> Tuple[int, Optional[str], float, Optional[str]]:
    idx, src_path, dst_path, target_sr, target_channels, overwrite = task
    if not src_path or not Path(src_path).exists():
        return idx, None, float("nan"), f"Missing source: {src_path}"
    try:
        if dst_path.exists() and not overwrite:
            info = sf.info(str(dst_path))
            if info.samplerate == target_sr and info.channels == target_channels:
                return idx, str(dst_path.resolve()), float(info.duration), None

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        src_info = sf.info(str(src_path))
        if (
            src_info.samplerate == target_sr
            and src_info.channels == target_channels
            and src_info.subtype == "PCM_16"
            and Path(src_path).suffix.lower() == ".wav"
        ):
            shutil.copyfile(src_path, dst_path)
            return idx, str(dst_path.resolve()), float(src_info.duration), None

        wav, sr = sf.read(str(src_path), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1) if wav.shape[1] > 1 else wav[:, 0]
        if sr != target_sr:
            wav = resample_audio(wav, sr, target_sr)
        wav = np.clip(wav, -1.0, 1.0)
        sf.write(str(dst_path), wav, target_sr, subtype="PCM_16")
        return idx, str(dst_path.resolve()), float(len(wav) / target_sr), None
    except Exception as exc:
        return idx, None, float("nan"), str(exc)


def export_split_16k_mono(
    df: Optional[pd.DataFrame],
    split_name: str,
    data_dir: Path,
    max_workers: int = 8,
    overwrite: bool = False,
    target_sr: int = TARGET_SR,
    target_channels: int = TARGET_CHANNELS,
) -> Optional[pd.DataFrame]:
    """Standardize all audio in a split to 16 kHz mono PCM_16 WAV in parallel."""
    data_dir = Path(data_dir)
    csv_path = data_dir / f"raw_{split_name}_set.csv"
    if not overwrite and df is None and all_clean_outputs_cached(data_dir):
        return None
    if csv_path.exists() and not overwrite:
        cached_df = pd.read_csv(csv_path)
        valid_paths = cached_df["audio_path"].dropna().astype(str).tolist()
        if len(cached_df) > 0 and len(valid_paths) == len(cached_df):
            spot_check = valid_paths[:5] + valid_paths[-5:]
            if all(Path(p).exists() for p in spot_check) or all_clean_outputs_cached(data_dir):
                return cached_df
    if df is None:
        raise ValueError(f"Cannot export {split_name}: DataFrame is None and no cache found.")

    out_df = df.copy().reset_index(drop=True)
    split_audio_dir = data_dir / split_name
    split_audio_dir.mkdir(parents=True, exist_ok=True)
    existing_files = set() if overwrite else {p.name for p in split_audio_dir.glob("*.wav")}

    new_paths = [None] * len(out_df)
    durations = [float("nan")] * len(out_df)
    tasks = []
    for idx, row in out_df.iterrows():
        fname = f"dia{int(row['dialogue'])}_utt{int(row['utterance'])}.wav"
        dst_path = split_audio_dir / fname
        if fname in existing_files:
            new_paths[idx] = str(dst_path.resolve())
            dur = row.get("duration", row.get("time", float("nan")))
            durations[idx] = float(dur) if pd.notna(dur) else float("nan")
        else:
            tasks.append((idx, str(row["audio_path"]), dst_path, target_sr, target_channels, overwrite))

    errors = []
    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_single_audio, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Exporting {split_name} (16k mono)"):
                idx, out_path, dur, err = fut.result()
                new_paths[idx] = out_path
                durations[idx] = dur
                if err is not None:
                    errors.append((idx, err))

    out_df["source_audio_path"] = out_df["audio_path"]
    out_df["audio_path"] = new_paths
    out_df["duration"] = durations
    valid_mask = out_df["audio_path"].notna() & out_df["emotion"].notna()
    out_df = out_df.loc[valid_mask].reset_index(drop=True)
    out_df.to_csv(csv_path, index=False)
    return out_df


def verify_split(
    df: Optional[pd.DataFrame],
    name: str,
    target_sr: int = TARGET_SR,
    target_channels: int = TARGET_CHANNELS,
    sample_check: int = 100,
) -> None:
    if df is None or len(df) == 0:
        return
    assert df["audio_path"].isna().sum() == 0, f"{name}: NaN audio_path found"
    assert df["emotion"].isna().sum() == 0, f"{name}: NaN emotion found"
    bad_sr = bad_ch = 0
    for p in df["audio_path"].head(sample_check).tolist():
        info = sf.info(p)
        if info.samplerate != target_sr:
            bad_sr += 1
        if info.channels != target_channels:
            bad_ch += 1
    if bad_sr or bad_ch:
        raise ValueError(f"{name}: audio verification failed (bad_sr={bad_sr}, bad_ch={bad_ch})")
