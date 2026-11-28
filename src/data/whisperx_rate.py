"""Batched WhisperX true syllable & word speaking-rate tracking (CMUdict + g2p_en; exact match to data_processing.ipynb)."""

from concurrent.futures import ThreadPoolExecutor
import gc
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import TARGET_SR
from src.data.emo_tag import tokenize_emo_tag
from src.utils.audio import load_wav_16k_mono
from src.utils.io import atomic_csv

WHISPERX_RATE_BATCH_SIZE = 32
WHISPERX_RATE_SILENCE_PAD_SEC = 0.50
WHISPERX_RATE_CHECKPOINT_EVERY_BATCHES = 25

WHISPERX_RATE_COLUMNS = (
    "whisperx_word_count",
    "whisperx_syllable_count",
    "whisperx_speech_duration",
    "whisperx_pause_duration",
    "whisperx_utterance_span",
    "whisperx_syllable_rate",
    "whisperx_syllable_rate_cue",
    "whisperx_word_rate",
    "whisperx_word_rate_cue",
)

_NUMERIC_RATE_COLS = (
    "whisperx_word_count",
    "whisperx_syllable_count",
    "whisperx_speech_duration",
    "whisperx_pause_duration",
    "whisperx_utterance_span",
    "whisperx_syllable_rate",
    "whisperx_word_rate",
)


def count_arpabet_syllables(phonemes: Sequence[str]) -> int:
    """Count syllables from an ARPAbet phoneme sequence via vowel stress markers (0, 1, 2)."""
    return sum(1 for p in phonemes if isinstance(p, str) and len(p) > 0 and p[-1].isdigit())


class PhoneticSyllableCounter:
    """English phonetic syllable counter backed by CMUdict + g2p_en fallback with memoization."""

    def __init__(
        self,
        cmu_dict: Optional[Dict[str, List[List[str]]]] = None,
        g2p_fn: Any = None,
    ) -> None:
        self._cmu_dict = cmu_dict
        self._g2p_fn = g2p_fn
        self._cache: Dict[str, int] = {}
        self.cmu_hits: int = 0
        self.g2p_hits: int = 0

    def _ensure_lexicon(self) -> Dict[str, List[List[str]]]:
        if self._cmu_dict is None:
            try:
                import cmudict

                self._cmu_dict = cmudict.dict()
            except ImportError:
                import nltk

                nltk.download("cmudict", quiet=True)
                from nltk.corpus import cmudict as nltk_cmudict

                self._cmu_dict = nltk_cmudict.dict()
        return self._cmu_dict

    def _ensure_g2p(self) -> Any:
        if self._g2p_fn is None:
            import nltk

            for resource in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "cmudict"):
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass
            from g2p_en import G2p

            self._g2p_fn = G2p()
        return self._g2p_fn

    def count_word(self, word: str) -> int:
        if not isinstance(word, str):
            return 0
        cleaned = word.replace("\u2019", "'").replace("\u2018", "'").strip().lower()
        cleaned = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", cleaned)
        if not cleaned:
            return 0
        if cleaned in self._cache:
            return self._cache[cleaned]

        cmu = self._ensure_lexicon()
        prons = cmu.get(cleaned)
        if prons:
            syl = max(1, count_arpabet_syllables(prons[0]))
            self._cache[cleaned] = syl
            self.cmu_hits += 1
            return syl

        if "-" in cleaned:
            parts = [p for p in cleaned.split("-") if p]
            if len(parts) > 1:
                syl = sum(self.count_word(p) for p in parts)
                self._cache[cleaned] = syl
                return syl

        g2p = self._ensure_g2p()
        syl = max(1, count_arpabet_syllables(g2p(cleaned)))
        self._cache[cleaned] = syl
        self.g2p_hits += 1
        return syl


def compute_speech_rate_metrics_from_words(
    word_segments: Sequence[Dict[str, Any]],
    audio_duration: float,
    syllable_counter: PhoneticSyllableCounter,
    reference_text: str = "",
) -> Dict[str, float]:
    """Compute true syllable/word rates from WhisperX word timestamps + inter-word pause duration."""
    dur = (
        float(audio_duration)
        if (audio_duration is not None and np.isfinite(audio_duration) and audio_duration > 0.0)
        else float("nan")
    )
    valid_words: List[Tuple[float, float, int]] = []
    for w in word_segments or []:
        if not isinstance(w, dict):
            continue
        syl = syllable_counter.count_word(str(w.get("word", "")).strip())
        if syl <= 0:
            continue
        start, end = w.get("start"), w.get("end")
        if start is None or end is None:
            continue
        try:
            s_val, e_val = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(s_val) and math.isfinite(e_val)):
            continue
        if math.isfinite(dur):
            s_val = max(0.0, min(dur, s_val))
            e_val = max(0.0, min(dur, e_val))
        if e_val > s_val:
            valid_words.append((s_val, e_val, syl))

    if valid_words:
        valid_words.sort(key=lambda item: (item[0], item[1]))
        word_count = len(valid_words)
        syllable_count = sum(item[2] for item in valid_words)
        merged: List[List[float]] = [[valid_words[0][0], valid_words[0][1]]]
        for s_val, e_val, _ in valid_words[1:]:
            if s_val <= merged[-1][1]:
                if e_val > merged[-1][1]:
                    merged[-1][1] = e_val
            else:
                merged.append([s_val, e_val])
        speech_duration = float(sum(end - start for start, end in merged))
        pause_duration = float(
            sum(max(0.0, merged[k][0] - merged[k - 1][1]) for k in range(1, len(merged)))
        )
        utterance_span = speech_duration + pause_duration
        if utterance_span <= 1e-4 and math.isfinite(dur) and dur > 0.0:
            utterance_span = dur
            speech_duration = dur
            pause_duration = 0.0
    else:
        ref_tokens = tokenize_emo_tag(reference_text)
        word_count = len(ref_tokens)
        syllable_count = sum(syllable_counter.count_word(tok) for tok in ref_tokens)
        if not (math.isfinite(dur) and dur > 0.0) or word_count == 0:
            return {
                "whisperx_word_count": 0,
                "whisperx_syllable_count": 0,
                "whisperx_speech_duration": float("nan"),
                "whisperx_pause_duration": float("nan"),
                "whisperx_utterance_span": float("nan"),
                "whisperx_syllable_rate": float("nan"),
                "whisperx_word_rate": float("nan"),
            }
        speech_duration = dur
        pause_duration = 0.0
        utterance_span = dur

    return {
        "whisperx_word_count": int(word_count),
        "whisperx_syllable_count": int(syllable_count),
        "whisperx_speech_duration": round(float(speech_duration), 6),
        "whisperx_pause_duration": round(float(pause_duration), 6),
        "whisperx_utterance_span": round(float(utterance_span), 6),
        "whisperx_syllable_rate": round(float(syllable_count / utterance_span), 6),
        "whisperx_word_rate": round(float(word_count / utterance_span), 6),
    }


def fit_global_quartile_thresholds(series: pd.Series) -> Dict[str, float]:
    """Fit global (gender-agnostic) Q25/Q50/Q75 thresholds strictly on positive finite training values."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    valid = valid[valid > 0.0]
    if valid.empty:
        raise ValueError("Cannot fit global quartiles: no positive finite values found.")
    return {
        "q25": float(valid.quantile(0.25)),
        "q50": float(valid.quantile(0.50)),
        "q75": float(valid.quantile(0.75)),
        "count": int(len(valid)),
    }


def apply_global_quartiles(series: pd.Series, thresholds: Dict[str, float]) -> pd.Series:
    """Classify positive numeric values into low/moderate/high using pre-fitted training Q25/Q75 cutoffs."""
    q25 = float(thresholds["q25"])
    q75 = float(thresholds["q75"])
    numeric = pd.to_numeric(series, errors="coerce")

    def _label(val: Any) -> Optional[str]:
        if val is None or pd.isna(val):
            return None
        v = float(val)
        if not math.isfinite(v) or v <= 0.0:
            return None
        if v < q25:
            return "low"
        if v > q75:
            return "high"
        return "moderate"

    return numeric.map(_label)


def classify_global_quartiles(series: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    thresholds = fit_global_quartile_thresholds(series)
    return apply_global_quartiles(series, thresholds), thresholds


def _resolve_clean_split_wav(raw_path: Any, clean_dir: Path, split: str) -> Path:
    p = Path(str(raw_path))
    if p.is_file():
        return p
    rebased = Path(clean_dir) / split / p.name
    if rebased.is_file():
        return rebased
    raise FileNotFoundError(f"Cleaned {split} WAV not found locally: {raw_path} (also checked {rebased})")


def _transcribe_whisperx_batch(
    asr_model: Any,
    batch_wavs: Sequence[np.ndarray],
    batch_size: int = WHISPERX_RATE_BATCH_SIZE,
    language: str = "en",
) -> List[str]:
    if not batch_wavs:
        return []
    try:
        if hasattr(asr_model, "_use_interleaved_context"):
            asr_model._use_interleaved_context = False
        if hasattr(asr_model, "previous_batch_context_tokens"):
            asr_model.previous_batch_context_tokens = [[]] * len(batch_wavs)
        batch_inputs = [{"inputs": w} for w in batch_wavs]
        raw_outputs = list(asr_model(batch_inputs, batch_size=min(batch_size, len(batch_wavs))))
        if len(raw_outputs) == len(batch_wavs):
            return [
                str(out.get("text", "")).strip() if isinstance(out, dict) else str(out).strip()
                for out in raw_outputs
            ]
    except Exception:
        pass
    texts: List[str] = []
    for wav in batch_wavs:
        res = asr_model.transcribe(wav, batch_size=batch_size, language=language)
        segs = res.get("segments", []) if isinstance(res, dict) else []
        texts.append(" ".join(str(s.get("text", "")).strip() for s in segs).strip())
    return texts


def _align_whisperx_batch(
    batch_wavs: Sequence[np.ndarray],
    batch_durs: Sequence[float],
    batch_texts: Sequence[str],
    batch_ref_texts: Sequence[str],
    align_model: Any,
    align_metadata: Dict[str, Any],
    device: str,
    align_fn: Any = None,
    sr: int = TARGET_SR,
    pad_sec: float = WHISPERX_RATE_SILENCE_PAD_SEC,
) -> List[List[Dict[str, Any]]]:
    if align_fn is None:
        import whisperx

        align_fn = whisperx.align

    n_items = len(batch_wavs)
    results: List[List[Dict[str, Any]]] = [[] for _ in range(n_items)]
    pad_samples = int(round(pad_sec * sr))
    pad_zeros = np.zeros(pad_samples, dtype=np.float32)

    pieces: List[np.ndarray] = []
    batch_segments: List[Dict[str, Any]] = []
    active_map: List[Tuple[int, float, float]] = []
    cursor_samples = 0
    for i, (wav, dur, asr_txt, ref_txt) in enumerate(zip(batch_wavs, batch_durs, batch_texts, batch_ref_texts)):
        txt = asr_txt.strip() if isinstance(asr_txt, str) and asr_txt.strip() else str(ref_txt or "").strip()
        start_sec = cursor_samples / float(sr)
        end_sec = start_sec + dur
        pieces.append(wav)
        if i < n_items - 1:
            pieces.append(pad_zeros)
        cursor_samples += len(wav) + (pad_samples if i < n_items - 1 else 0)
        if txt:
            batch_segments.append({"text": txt, "start": start_sec, "end": end_sec})
            active_map.append((i, start_sec, dur))

    batch_ok = False
    if batch_segments:
        combined_wav = np.concatenate(pieces, axis=0)
        try:
            aligned = align_fn(
                batch_segments, align_model, align_metadata, combined_wav, device, return_char_alignments=False
            )
            aligned_segs = aligned.get("segments", []) if isinstance(aligned, dict) else []
            if len(aligned_segs) == len(active_map):
                for seg, (item_idx, offset_sec, dur) in zip(aligned_segs, active_map):
                    shifted: List[Dict[str, Any]] = []
                    for w in seg.get("words", []) or []:
                        if not isinstance(w, dict) or w.get("start") is None or w.get("end") is None:
                            continue
                        s_rel = max(0.0, min(dur, float(w["start"]) - offset_sec))
                        e_rel = max(0.0, min(dur, float(w["end"]) - offset_sec))
                        if e_rel > s_rel:
                            shifted.append({"word": w.get("word", ""), "start": s_rel, "end": e_rel})
                    results[item_idx] = shifted
                batch_ok = True
        except Exception:
            batch_ok = False

    for i, (wav, dur, asr_txt, ref_txt) in enumerate(zip(batch_wavs, batch_durs, batch_texts, batch_ref_texts)):
        if batch_ok and results[i]:
            continue
        candidates = []
        if isinstance(asr_txt, str) and asr_txt.strip():
            candidates.append(asr_txt.strip())
        if isinstance(ref_txt, str) and ref_txt.strip() and ref_txt.strip() not in candidates:
            candidates.append(ref_txt.strip())
        for cand_txt in candidates:
            try:
                single = align_fn(
                    [{"text": cand_txt, "start": 0.0, "end": dur}],
                    align_model,
                    align_metadata,
                    wav,
                    device,
                    return_char_alignments=False,
                )
                words = [
                    {
                        "word": w.get("word", ""),
                        "start": max(0.0, min(dur, float(w["start"]))),
                        "end": max(0.0, min(dur, float(w["end"]))),
                    }
                    for w in (single.get("word_segments", []) if isinstance(single, dict) else [])
                    if isinstance(w, dict)
                    and w.get("start") is not None
                    and w.get("end") is not None
                    and float(w["end"]) > float(w["start"])
                ]
                if words:
                    results[i] = words
                    break
            except Exception:
                continue
    return results


def _run_whisperx_rate_pass(
    df: pd.DataFrame,
    split: str,
    clean_dir: Path,
    partial_ckpt: Path,
    batch_size: int,
    overwrite: bool,
    whisper_model_name: str,
    asr_model: Any,
    align_model: Any,
    align_metadata: Optional[Dict[str, Any]],
    device: Optional[str],
    align_fn: Any,
    syllable_counter: Optional[PhoneticSyllableCounter],
    max_workers: int = 8,
    checkpoint_every: int = WHISPERX_RATE_CHECKPOINT_EVERY_BATCHES,
) -> Dict[int, Dict[str, Any]]:
    done_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if partial_ckpt.is_file() and not overwrite:
        try:
            for _, prow in pd.read_csv(partial_ckpt).iterrows():
                if pd.notna(prow.get("whisperx_word_count")):
                    key = (int(prow["dialogue"]), int(prow["utterance"]))
                    done_by_key[key] = {col: prow[col] for col in _NUMERIC_RATE_COLS if col in prow}
        except Exception:
            done_by_key = {}

    pending_indices: List[int] = []
    results_map: Dict[int, Dict[str, Any]] = {}
    for idx, row in df.iterrows():
        key = (int(row["dialogue"]), int(row["utterance"]))
        if key in done_by_key:
            results_map[idx] = done_by_key[key]
        else:
            pending_indices.append(idx)

    loaded_locally = False
    if pending_indices:
        if syllable_counter is None:
            syllable_counter = PhoneticSyllableCounter()
            syllable_counter._ensure_lexicon()
        if asr_model is None or align_model is None or align_metadata is None:
            import torch
            import whisperx

            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            compute_type = "float16" if device == "cuda" else "int8"
            asr_model = whisperx.load_model(
                whisper_model_name, device=device, compute_type=compute_type, language="en"
            )
            align_model, align_metadata = whisperx.load_align_model(language_code="en", device=device)
            try:
                asr_model.transcribe(np.zeros(TARGET_SR, dtype=np.float32), batch_size=batch_size, language="en")
            except Exception:
                pass
            loaded_locally = True

        total_batches = math.ceil(len(pending_indices) / float(batch_size))
        with tqdm(total=len(pending_indices), desc=f"WhisperX {split} rate (batch={batch_size})", unit="audio") as prog:
            for batch_num, start in enumerate(range(0, len(pending_indices), batch_size), start=1):
                batch_idx = pending_indices[start : start + batch_size]
                batch_paths = [_resolve_clean_split_wav(df.at[i, "audio_path"], clean_dir, split) for i in batch_idx]
                batch_ref_texts = [str(df.at[i, "text"]) if pd.notna(df.at[i, "text"]) else "" for i in batch_idx]
                with ThreadPoolExecutor(max_workers=min(max_workers, len(batch_paths))) as pool:
                    loaded_audio = list(pool.map(load_wav_16k_mono, batch_paths))
                batch_wavs = [item[0] for item in loaded_audio]
                batch_durs = [item[1] for item in loaded_audio]
                batch_asr_texts = _transcribe_whisperx_batch(asr_model, batch_wavs, batch_size=batch_size)
                batch_words = _align_whisperx_batch(
                    batch_wavs, batch_durs, batch_asr_texts, batch_ref_texts,
                    align_model, align_metadata, device or "cpu", align_fn=align_fn,
                )
                for idx, words, dur, ref_txt in zip(batch_idx, batch_words, batch_durs, batch_ref_texts):
                    results_map[idx] = compute_speech_rate_metrics_from_words(
                        words, audio_duration=dur, syllable_counter=syllable_counter, reference_text=ref_txt
                    )
                prog.set_postfix(batch=f"{batch_num}/{total_batches}")
                prog.update(len(batch_idx))
                if checkpoint_every > 0 and (batch_num % checkpoint_every == 0):
                    ckpt_rows = [
                        {"dialogue": int(df.at[i, "dialogue"]), "utterance": int(df.at[i, "utterance"]), **results_map[i]}
                        for i in sorted(results_map)
                    ]
                    atomic_csv(pd.DataFrame(ckpt_rows), partial_ckpt)

        if loaded_locally:
            del asr_model, align_model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    return results_map


def add_whisperx_speech_rates_to_clean_train(
    train_csv_path: Path,
    drive_dir: Optional[Path] = None,
    batch_size: int = WHISPERX_RATE_BATCH_SIZE,
    overwrite: bool = False,
    whisper_model_name: str = "large-v3",
    asr_model: Any = None,
    align_model: Any = None,
    align_metadata: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    align_fn: Any = None,
    syllable_counter: Optional[PhoneticSyllableCounter] = None,
) -> pd.DataFrame:
    """Compute WhisperX true syllable/word rates and global 25/50/25 quartiles on clean_train_set."""
    train_csv_path = Path(train_csv_path)
    df = pd.read_csv(train_csv_path)
    if df.empty:
        raise ValueError(f"Cleaned training CSV is empty: {train_csv_path}")
    if all(col in df.columns for col in WHISPERX_RATE_COLUMNS) and not overwrite:
        complete = (
            pd.to_numeric(df["whisperx_syllable_rate"], errors="coerce").notna().all()
            and pd.to_numeric(df["whisperx_word_rate"], errors="coerce").notna().all()
            and df["whisperx_syllable_rate_cue"].isin(["low", "moderate", "high"]).all()
            and df["whisperx_word_rate_cue"].isin(["low", "moderate", "high"]).all()
        )
        if complete:
            return df

    partial_ckpt = train_csv_path.parent / "whisperx_rate_train.partial.csv"
    results_map = _run_whisperx_rate_pass(
        df, "train", train_csv_path.parent, partial_ckpt, batch_size, overwrite,
        whisper_model_name, asr_model, align_model, align_metadata, device, align_fn, syllable_counter,
    )
    for col in _NUMERIC_RATE_COLS:
        df[col] = [results_map[i][col] for i in range(len(df))]
    df["whisperx_syllable_rate_cue"], _ = classify_global_quartiles(df["whisperx_syllable_rate"])
    df["whisperx_word_rate_cue"], _ = classify_global_quartiles(df["whisperx_word_rate"])
    atomic_csv(df, train_csv_path)
    if partial_ckpt.exists():
        partial_ckpt.unlink()
    if drive_dir is not None:
        atomic_csv(df, Path(drive_dir) / "emo_tag_clean" / train_csv_path.name)
    return df


def add_whisperx_speech_rates_to_clean_val(
    val_csv_path: Path,
    train_csv_path: Path,
    split: str = "val",
    drive_dir: Optional[Path] = None,
    batch_size: int = WHISPERX_RATE_BATCH_SIZE,
    overwrite: bool = False,
    whisper_model_name: str = "large-v3",
    asr_model: Any = None,
    align_model: Any = None,
    align_metadata: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    align_fn: Any = None,
    syllable_counter: Optional[PhoneticSyllableCounter] = None,
) -> pd.DataFrame:
    """Compute WhisperX rates on val/test and classify strictly via train-fitted global Q25/Q75 cutoffs."""
    val_csv_path, train_csv_path = Path(val_csv_path), Path(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    train_df = pd.read_csv(train_csv_path)
    if val_df.empty:
        raise ValueError(f"Cleaned {split} CSV is empty: {val_csv_path}")

    if all(col in val_df.columns for col in WHISPERX_RATE_COLUMNS) and not overwrite:
        syl_num = pd.to_numeric(val_df["whisperx_syllable_rate"], errors="coerce")
        wrd_num = pd.to_numeric(val_df["whisperx_word_rate"], errors="coerce")
        valid_syl = syl_num.notna() & (syl_num > 0.0)
        valid_wrd = wrd_num.notna() & (wrd_num > 0.0)
        if (
            pd.to_numeric(val_df["whisperx_word_count"], errors="coerce").notna().all()
            and valid_syl.any()
            and valid_wrd.any()
            and val_df.loc[valid_syl, "whisperx_syllable_rate_cue"].isin(["low", "moderate", "high"]).all()
            and val_df.loc[valid_wrd, "whisperx_word_rate_cue"].isin(["low", "moderate", "high"]).all()
        ):
            return val_df

    syl_q = fit_global_quartile_thresholds(train_df["whisperx_syllable_rate"])
    wrd_q = fit_global_quartile_thresholds(train_df["whisperx_word_rate"])
    partial_ckpt = val_csv_path.parent / f"whisperx_rate_{split}.partial.csv"
    results_map = _run_whisperx_rate_pass(
        val_df, split, val_csv_path.parent, partial_ckpt, batch_size, overwrite,
        whisper_model_name, asr_model, align_model, align_metadata, device, align_fn, syllable_counter,
    )
    for col in _NUMERIC_RATE_COLS:
        val_df[col] = [results_map[i][col] for i in range(len(val_df))]
    val_df["whisperx_syllable_rate_cue"] = apply_global_quartiles(val_df["whisperx_syllable_rate"], syl_q)
    val_df["whisperx_word_rate_cue"] = apply_global_quartiles(val_df["whisperx_word_rate"], wrd_q)
    atomic_csv(val_df, val_csv_path)
    if partial_ckpt.exists():
        partial_ckpt.unlink()
    if drive_dir is not None:
        atomic_csv(val_df, Path(drive_dir) / "emo_tag_clean" / val_csv_path.name)
    return val_df
