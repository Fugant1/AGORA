"""Fail-closed manifest loading, path resolution, staging, and atomic CSV I/O."""

import shutil
import tarfile
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.config import CUE_COLUMNS, CUE_LABELS, CUES, EMOTIONS


def atomic_csv(
    df: pd.DataFrame,
    path: Union[str, Path],
    data_dir: Optional[Path] = None,
    drive_data_dir: Optional[Path] = None,
) -> None:
    """Write a DataFrame atomically to CSV, optionally rewriting local temp paths to persistent paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.partial")
    saved = df.copy()
    if (
        "audio_path" in saved
        and data_dir is not None
        and drive_data_dir is not None
        and Path(data_dir).resolve() != Path(drive_data_dir).resolve()
    ):
        local_root = Path(data_dir).resolve()
        drive_root = Path(drive_data_dir)
        saved["audio_path"] = saved["audio_path"].map(
            lambda p: str(drive_root / Path(p).resolve().relative_to(local_root))
            if Path(p).resolve().is_relative_to(local_root)
            else str(p)
        )
    try:
        saved.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_audio_path(value: object, split: str, data_dir: Union[str, Path]) -> str:
    """Resolve a cleaned audio path relative to `data_dir / 'emo_tag_clean'` with traversal checks."""
    if pd.isna(value):
        raise ValueError("Missing audio_path")
    path = Path(str(value))
    parts = path.parts
    if "emo_tag_clean" in parts:
        relative = Path(*parts[parts.index("emo_tag_clean") + 1 :])
    elif not path.is_absolute():
        relative = path
    else:
        raise ValueError(f"Expected a cleaned audio path: {path}")
    if not relative.parts or relative.parts[0] != split or ".." in relative.parts:
        raise ValueError(f"Unexpected {split} path: {path}")
    data_dir = Path(data_dir)
    resolved = (data_dir / "emo_tag_clean" / relative).resolve()
    if not resolved.is_relative_to((data_dir / "emo_tag_clean" / split).resolve()):
        raise ValueError(f"Audio escapes split directory: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return str(resolved)


def load_manifest(
    csv_path: Union[str, Path],
    split: str,
    data_dir: Union[str, Path],
    require_transcripts: bool = False,
) -> pd.DataFrame:
    """Load and strictly validate a cleaned MELD manifest CSV (exact parity with V4 notebooks)."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    required = {
        "audio_path",
        "effective_gender",
        "gender",
        "emotion",
        "text",
        "speaker",
        "duration",
        "dialogue",
        "utterance",
    } | {CUE_COLUMNS[cue] for cue in CUES}
    if required - set(df):
        raise ValueError(f"{csv_path}: missing columns {required - set(df)}")
    if df.empty:
        raise ValueError(f"{csv_path}: empty split")
    if not df.effective_gender.isin(["F", "M", "__GLOBAL__"]).all():
        raise ValueError(f"{csv_path}: missing or invalid effective_gender; rerun preprocessing")
    df["audio_path"] = df.audio_path.map(lambda p: resolve_audio_path(p, split, data_dir))
    if df.audio_path.duplicated().any():
        raise ValueError(f"{csv_path}: duplicate audio")
    for cue in CUES:
        labels = df[CUE_COLUMNS[cue]]
        unknown = set(labels.dropna()) - set(CUE_LABELS[cue])
        if unknown:
            raise ValueError(f"{csv_path}: unknown {cue} labels {unknown}")
        if split == "train" and labels.isna().any():
            raise ValueError(f"{csv_path}: missing training labels for {cue}; check preprocessing")
    if not df.emotion.isin(EMOTIONS).all():
        raise ValueError(f"{csv_path}: missing or unknown emotion labels")
    if df[["dialogue", "utterance"]].duplicated().any():
        raise ValueError(f"{csv_path}: duplicate utterance IDs")
    if split == "train":
        if not df.gender.isin(["F", "M"]).all() or not df.gender.equals(df.effective_gender):
            raise ValueError("Training gender must match preprocessing effective_gender (F/M)")
        if require_transcripts and (
            df.text.isna().any() or df.text.astype(str).str.strip().eq("").any()
        ):
            raise ValueError("Training transcripts are required for explanation generation")
    return df


def validate_grpo_partition(
    grpo_df: pd.DataFrame,
    val_df: pd.DataFrame,
    sft_df: Optional[pd.DataFrame] = None,
    clean_train_df: Optional[pd.DataFrame] = None,
    *,
    require_uniform_quotas: bool = False,
) -> pd.Series:
    """Fail-closed provenance and zero-leakage checks for the saved GRPO split."""
    if grpo_df.empty or grpo_df.audio_path.duplicated().any():
        raise ValueError("GRPO: empty or duplicate audio")
    if set(grpo_df.audio_path) & set(val_df.audio_path):
        raise ValueError("GRPO/validation audio overlap detected")
    if sft_df is not None and (set(grpo_df.audio_path) & set(sft_df.audio_path)):
        raise ValueError("SFT/GRPO audio overlap detected")
    if clean_train_df is not None:
        source = clean_train_df.set_index("audio_path")
        if not set(grpo_df.audio_path) <= set(source.index):
            raise ValueError("GRPO: audio outside saved clean training source")
        try:
            pd.testing.assert_frame_equal(
                grpo_df[clean_train_df.columns].reset_index(drop=True),
                source.loc[grpo_df.audio_path].reset_index()[clean_train_df.columns],
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-10,
            )
        except AssertionError as exc:
            raise ValueError("GRPO: cached metadata/labels differ from preprocessing") from exc
    counts = grpo_df.groupby(["emotion", "effective_gender"]).size().reindex(
        pd.MultiIndex.from_product([EMOTIONS, ("F", "M")]), fill_value=0
    )
    if counts.min() == 0:
        raise ValueError(f"GRPO: missing one or more emotion/gender strata: {counts.to_dict()}")
    if require_uniform_quotas and counts.nunique() != 1:
        raise ValueError(f"GRPO: emotion/gender quotas are not balanced: {counts.to_dict()}")
    return counts


def stage_clean_data(drive_dir: Union[str, Path], local_dir: Union[str, Path]) -> None:
    """Persist a missing tar archive in drive_dir, then extract a local working copy."""
    drive_dir, local_dir = Path(drive_dir), Path(local_dir)
    if local_dir.resolve().is_relative_to(drive_dir.resolve()):
        raise ValueError("Local working directory must be outside Drive data storage")
    metadata = (
        "clean_train_set_with_cues.csv",
        "clean_val_set_with_cues.csv",
        "gender_acoustic_quartiles.json",
    )
    archive = drive_dir / "meld_16k_mono_clean.tar"
    clean_source = drive_dir / "emo_tag_clean"
    if not archive.is_file():
        for name in metadata:
            if not (clean_source / name).is_file():
                raise FileNotFoundError(f"Missing preprocessing export: {clean_source / name}")
        for split in ("train", "val"):
            frame = pd.read_csv(clean_source / f"clean_{split}_set_with_cues.csv")
            if frame.empty:
                raise ValueError(f"Cannot archive an empty {split} manifest")
            for path in frame.audio_path:
                resolve_audio_path(path, split, drive_dir)
        partial = archive.with_suffix(".tar.partial")
        try:
            with tarfile.open(partial, "w") as bundle:
                bundle.add(clean_source, arcname="meld_16k_mono/emo_tag_clean")
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)

    local_dir.mkdir(parents=True, exist_ok=True)
    local_archive = local_dir.parent / archive.name
    partial = local_archive.with_suffix(".tar.partial")
    try:
        shutil.copyfile(archive, partial)
        partial.replace(local_archive)
    finally:
        partial.unlink(missing_ok=True)

    extracted_audio = set()
    with tarfile.open(local_archive, "r:") as bundle:
        for member in bundle:
            parts = Path(member.name).parts
            if (
                not member.isfile()
                or ".." in parts
                or len(parts) not in (3, 4)
                or parts[:2] != ("meld_16k_mono", "emo_tag_clean")
            ):
                continue
            is_audio = (
                len(parts) == 4
                and parts[2] in ("train", "val", "test")
                and Path(parts[3]).suffix == ".wav"
            )
            is_metadata = len(parts) == 3 and Path(parts[2]).suffix in (".csv", ".json")
            if not (is_audio or is_metadata):
                continue
            target = local_dir.joinpath(*parts[1:])
            if not target.resolve().is_relative_to(local_dir.resolve()):
                raise ValueError(f"Archive target escapes local working directory: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".partial")
            try:
                with bundle.extractfile(member) as source, partial.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
            if is_audio:
                extracted_audio.add(str(target.resolve()))

    for name in metadata:
        source = clean_source / name
        target = local_dir / "emo_tag_clean" / name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if not target.is_file():
            raise FileNotFoundError(f"Missing metadata in export/archive: {name}")
    for split in ("train", "val"):
        frame = pd.read_csv(local_dir / "emo_tag_clean" / f"clean_{split}_set_with_cues.csv")
        for path in frame.audio_path:
            resolved = resolve_audio_path(path, split, local_dir)
            if resolved not in extracted_audio:
                raise ValueError(f"Archive is missing a manifest audio: {resolved}")
