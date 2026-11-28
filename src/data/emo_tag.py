"""EMO-TAG 5-step single-utterance audio cleaner (Mohamed et al., 2026; exact match to data_processing.ipynb)."""

from dataclasses import dataclass
import difflib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from src.config import TARGET_SR
from src.utils.audio import resample_audio

EMO_TAG_SR = TARGET_SR
PAUSE_MAX_RMS_DBFS = -35.0
PAUSE_MAX_PEAK_DBFS = -25.0
BOUNDARY_MARGIN_SEC = 0.05
MAX_CONSECUTIVE_INSERTIONS = 2
MAX_REPLACEMENTS = 3
MIN_SPEECH_DURATION_SEC = 0.15


@dataclass
class IntervalMetrics:
    start_sec: float
    end_sec: float
    is_speech: bool
    rms_dbfs: float
    peak_dbfs: float
    frame_energy_var: float
    voiced_frame_ratio: float


def normalize_emo_tag_transcript(text: str) -> str:
    """Appendix A.2: lowercase, remove punctuation except apostrophes, and normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\u2019", "'").replace("\u2018", "'").lower()
    text = re.sub(r"[^a-z0-9'\s]+", " ", text)
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_emo_tag(text: str) -> List[str]:
    norm = normalize_emo_tag_transcript(text)
    return norm.split() if norm else []


def compute_interval_acoustic_metrics(
    wav: np.ndarray,
    sr: int = EMO_TAG_SR,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    is_speech: bool = False,
) -> IntervalMetrics:
    """Compute RMS (dBFS), peak (dBFS), frame-level energy variation, and voiced frame ratio."""
    eps = 1e-10
    if end_sec is None:
        end_sec = len(wav) / float(sr)
    if len(wav) == 0:
        return IntervalMetrics(start_sec, end_sec, is_speech, -120.0, -120.0, 0.0, 0.0)

    rms = float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))
    peak = float(np.max(np.abs(wav)))
    rms_dbfs = float(20.0 * np.log10(max(rms, eps)))
    peak_dbfs = float(20.0 * np.log10(max(peak, eps)))

    win_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    if len(wav) < win_len:
        frames = wav[None, :]
    else:
        n_frames = 1 + (len(wav) - win_len) // hop_len
        frames = np.lib.stride_tricks.sliding_window_view(wav, win_len)[::hop_len][:n_frames]

    frame_rms_db = 20.0 * np.log10(np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1)) + eps)
    frame_energy_var = float(np.var(frame_rms_db)) if len(frame_rms_db) > 1 else 0.0
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    voiced_mask = (frame_rms_db > -45.0) & (zcr < 0.25)
    voiced_frame_ratio = float(np.mean(voiced_mask)) if len(voiced_mask) > 0 else 0.0

    return IntervalMetrics(
        start_sec=start_sec,
        end_sec=end_sec,
        is_speech=is_speech,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        frame_energy_var=frame_energy_var,
        voiced_frame_ratio=voiced_frame_ratio,
    )


def is_natural_pause_or_sigh(
    metrics: IntervalMetrics,
    max_rms_dbfs: float = PAUSE_MAX_RMS_DBFS,
    max_peak_dbfs: float = PAUSE_MAX_PEAK_DBFS,
) -> bool:
    """Retain intervening non-speech intervals as natural pauses/sighs iff RMS <= -35 dBFS and Peak <= -25 dBFS."""
    return (metrics.rms_dbfs <= max_rms_dbfs) and (metrics.peak_dbfs <= max_peak_dbfs)


def rvad_frame_decisions(wav: np.ndarray, sr: int = EMO_TAG_SR) -> np.ndarray:
    """Run rVAD (Tan et al., 2020) frame-level voice activity detection (10 ms frame shift)."""
    if len(wav) < int(0.025 * sr):
        return np.zeros(0, dtype=np.int32)

    try:
        import speechproc

        winlen, ovrlen = 0.025, 0.010
        nftt = 512
        ftThres, vadThres = 0.5, 0.4
        nfr10 = int(np.floor((len(wav) - (winlen - ovrlen) * sr) / (ovrlen * sr)))
        if nfr10 > 0:
            ft, flen, fsh10, nfr10 = speechproc.sflux(wav, sr, winlen, ovrlen, nftt)
            pv01 = np.zeros(nfr10)
            pv01[np.less_equal(ft, ftThres)] = 1
            pvblk = speechproc.pitchblockdetect(pv01, pv01.copy(), nfr10, 1)
            fdata = speechproc.highpass(wav, sr)
            noise_samp, _, n_noise_samp = speechproc.snre_highenergy(
                fdata, nfr10, flen, fsh10, np.round(flen * 0.5), pv01, pvblk
            )
            for j in range(n_noise_samp):
                fdata[range(int(noise_samp[j, 0]), int(noise_samp[j, 1]) + 1)] = 0
            vad_seg = speechproc.snre_vad(
                fdata, nfr10, flen, fsh10, np.round(flen * 0.5), pv01, pvblk, vadThres
            )
            return np.asarray(vad_seg, dtype=np.int32)
    except Exception:
        pass

    win_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    n_frames = 1 + (len(wav) - win_len) // hop_len
    if n_frames <= 0:
        return np.zeros(0, dtype=np.int32)

    hp_wav = np.append(wav[0], wav[1:] - 0.97 * wav[:-1])
    frames = np.lib.stride_tricks.sliding_window_view(hp_wav, win_len)[::hop_len][:n_frames]
    window = np.hamming(win_len).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window[None, :], n=512, axis=1)) + 1e-10

    geo_mean = np.exp(np.mean(np.log(spec), axis=1))
    arith_mean = np.mean(spec, axis=1)
    flatness = geo_mean / arith_mean

    raw_frames = np.lib.stride_tricks.sliding_window_view(wav, win_len)[::hop_len][:n_frames]
    rms_db = 20.0 * np.log10(np.sqrt(np.mean(np.square(raw_frames, dtype=np.float64), axis=1)) + 1e-10)

    noise_floor_db = float(np.percentile(rms_db, 15))
    snr_db = rms_db - max(noise_floor_db, -65.0)
    speech_frames = (rms_db > -45.0) & ((snr_db > 6.0) | ((flatness < 0.45) & (rms_db > -38.0)))

    decisions = speech_frames.astype(np.int32)
    idx_ones = np.where(decisions == 1)[0]
    if len(idx_ones) > 1:
        for a, b in zip(idx_ones[:-1], idx_ones[1:]):
            if 1 < (b - a) <= 15:
                decisions[a:b] = 1
    i = 0
    while i < len(decisions):
        if decisions[i] == 1:
            j = i
            while j < len(decisions) and decisions[j] == 1:
                j += 1
            if (j - i) < 8:
                decisions[i:j] = 0
            i = j
        else:
            i += 1
    return decisions


def segment_audio_with_rvad(wav: np.ndarray, sr: int = EMO_TAG_SR) -> List[IntervalMetrics]:
    """Partition audio into speech and non-speech regions using rVAD."""
    total_dur = len(wav) / float(sr)
    decisions = rvad_frame_decisions(wav, sr=sr)
    if len(decisions) == 0 or not np.any(decisions == 1):
        return []

    hop_sec = 0.010
    intervals: List[IntervalMetrics] = []
    n = len(decisions)
    i = 0
    while i < n:
        state = bool(decisions[i] == 1)
        j = i
        while j < n and (decisions[j] == 1) == state:
            j += 1
        start_sec = round(i * hop_sec, 4)
        end_sec = round(min(total_dur, j * hop_sec if j < n else total_dur), 4)
        seg_wav = wav[int(round(start_sec * sr)) : int(round(end_sec * sr))]
        if len(seg_wav) > 0:
            intervals.append(
                compute_interval_acoustic_metrics(
                    seg_wav, sr=sr, start_sec=start_sec, end_sec=end_sec, is_speech=state
                )
            )
        i = j
    return intervals


def compute_word_edit_diagnostics(
    hyp_tokens: Sequence[str],
    ref_tokens: Sequence[str],
) -> Dict[str, Any]:
    """Compare WhisperX transcript tokens against ground-truth reference tokens."""
    ref_set = set(ref_tokens)
    overlap_count = sum(1 for tok in hyp_tokens if tok in ref_set)

    matcher = difflib.SequenceMatcher(a=list(ref_tokens), b=list(hyp_tokens), autojunk=False)
    max_consec_ins = 0
    total_replacements = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            max_consec_ins = max(max_consec_ins, j2 - j1)
        elif tag == "replace":
            ref_len = i2 - i1
            hyp_len = j2 - j1
            total_replacements += min(ref_len, hyp_len)
            max_consec_ins = max(max_consec_ins, max(0, hyp_len - ref_len))

    reject_insertions = max_consec_ins >= MAX_CONSECUTIVE_INSERTIONS
    reject_replacements = total_replacements >= MAX_REPLACEMENTS
    return {
        "token_overlap_count": int(overlap_count),
        "max_consecutive_insertions": int(max_consec_ins),
        "total_replacements": int(total_replacements),
        "reject_insertions": bool(reject_insertions),
        "reject_replacements": bool(reject_replacements),
        "passes_edit_gate": bool(not reject_insertions and not reject_replacements and overlap_count > 0),
    }


class EmoTagWhisperXCleaner:
    """Full EMO-TAG Single-Utterance Audio Cleaner (rVAD + two-pass WhisperX + laughter filter)."""

    def __init__(
        self,
        whisper_model_name: str = "large-v3",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: str = "en",
    ) -> None:
        self.whisper_model_name = whisper_model_name
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.asr_model = None
        self.align_model = None
        self.align_metadata = None

    def _lazy_init_whisperx(self) -> None:
        if self.asr_model is not None and self.align_model is not None:
            return
        import torch
        import whisperx

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"

        self.asr_model = whisperx.load_model(
            self.whisper_model_name,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
        )
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=self.language,
            device=self.device,
        )

    def transcribe_and_align(
        self,
        wav: np.ndarray,
        sr: int = EMO_TAG_SR,
        reference_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._lazy_init_whisperx()
        import whisperx

        audio_f32 = wav.astype(np.float32)
        asr_res = self.asr_model.transcribe(audio_f32, batch_size=1, language=self.language)
        segments = asr_res.get("segments", [])
        asr_text = " ".join(seg.get("text", "").strip() for seg in segments).strip()

        align_input = segments
        if not align_input and reference_text:
            align_input = [{"text": reference_text, "start": 0.0, "end": float(len(audio_f32) / sr)}]

        word_segments: List[Dict[str, Any]] = []
        if align_input:
            try:
                aligned = whisperx.align(
                    align_input,
                    self.align_model,
                    self.align_metadata,
                    audio_f32,
                    self.device,
                    return_char_alignments=False,
                )
                word_segments = [
                    w
                    for w in aligned.get("word_segments", [])
                    if isinstance(w, dict) and w.get("start") is not None and w.get("end") is not None
                ]
            except Exception:
                word_segments = []

        return {
            "asr_text": asr_text,
            "asr_tokens": tokenize_emo_tag(asr_text),
            "word_segments": word_segments,
        }

    def clean_utterance(
        self,
        src_audio_path: str,
        reference_transcript: str,
        dst_audio_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        ref_tokens = tokenize_emo_tag(reference_transcript)
        if not ref_tokens:
            return {"retained": False, "reject_reason": "empty_reference_transcript"}
        if not src_audio_path or not Path(src_audio_path).exists():
            return {"retained": False, "reject_reason": "missing_audio_file"}

        wav, sr = sf.read(str(src_audio_path), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1) if wav.shape[1] > 1 else wav[:, 0]
        if sr != EMO_TAG_SR:
            wav = resample_audio(wav, sr, EMO_TAG_SR)
            sr = EMO_TAG_SR

        orig_duration = float(len(wav) / sr)
        if orig_duration < MIN_SPEECH_DURATION_SEC:
            return {"retained": False, "reject_reason": "empty_or_too_short_raw", "orig_duration": orig_duration}

        intervals = segment_audio_with_rvad(wav, sr=sr)
        speech_indices = [idx for idx, iv in enumerate(intervals) if iv.is_speech]
        if not speech_indices:
            return {"retained": False, "reject_reason": "empty_rvad_no_speech", "orig_duration": orig_duration}

        matched_speech_indices: List[int] = []
        unrelated_speech_removed = 0
        ref_set = set(ref_tokens)
        for idx in speech_indices:
            iv = intervals[idx]
            seg_wav = wav[int(round(iv.start_sec * sr)) : int(round(iv.end_sec * sr))]
            if len(seg_wav) < int(MIN_SPEECH_DURATION_SEC * sr):
                continue
            seg_res = self.transcribe_and_align(seg_wav, sr=sr)
            if any(tok in ref_set for tok in seg_res["asr_tokens"]):
                matched_speech_indices.append(idx)
            else:
                unrelated_speech_removed += 1

        if not matched_speech_indices:
            return {
                "retained": False,
                "reject_reason": "unrelated_speech_no_token_overlap",
                "orig_duration": orig_duration,
                "unrelated_speech_removed": unrelated_speech_removed,
            }

        first_sp, last_sp = matched_speech_indices[0], matched_speech_indices[-1]
        matched_set = set(matched_speech_indices)
        retained_indices: List[int] = []
        laughter_intervals_removed = 0
        laughter_seconds_removed = 0.0
        retained_pauses_count = 0

        for idx, iv in enumerate(intervals):
            if idx < first_sp or idx > last_sp:
                if not iv.is_speech and not is_natural_pause_or_sigh(iv):
                    laughter_intervals_removed += 1
                    laughter_seconds_removed += iv.end_sec - iv.start_sec
                continue
            if idx in matched_set:
                retained_indices.append(idx)
            elif not iv.is_speech:
                if is_natural_pause_or_sigh(iv):
                    retained_indices.append(idx)
                    retained_pauses_count += 1
                else:
                    laughter_intervals_removed += 1
                    laughter_seconds_removed += iv.end_sec - iv.start_sec

        is_contiguous = retained_indices == list(range(first_sp, last_sp + 1))
        if is_contiguous:
            stage4_wav = wav[
                int(round(intervals[first_sp].start_sec * sr)) : int(round(intervals[last_sp].end_sec * sr))
            ]
        else:
            pieces = [
                wav[int(round(intervals[i].start_sec * sr)) : int(round(intervals[i].end_sec * sr))]
                for i in retained_indices
            ]
            stage4_wav = np.concatenate(pieces, axis=0) if pieces else np.zeros(0, dtype=np.float32)

        if len(stage4_wav) < int(MIN_SPEECH_DURATION_SEC * sr):
            return {"retained": False, "reject_reason": "empty_after_laughter_removal", "orig_duration": orig_duration}

        pass2_before = self.transcribe_and_align(stage4_wav, sr=sr, reference_text=reference_transcript)
        final_wav = stage4_wav
        word_segs = pass2_before["word_segments"]
        if word_segs:
            crop_start_sec = max(0.0, float(word_segs[0]["start"]) - BOUNDARY_MARGIN_SEC)
            crop_end_sec = min(float(len(stage4_wav) / sr), float(word_segs[-1]["end"]) + BOUNDARY_MARGIN_SEC)
            if crop_end_sec - crop_start_sec >= MIN_SPEECH_DURATION_SEC:
                candidate_crop = stage4_wav[int(round(crop_start_sec * sr)) : int(round(crop_end_sec * sr))]
                pass2_after = self.transcribe_and_align(candidate_crop, sr=sr)
                if pass2_after["asr_tokens"] == pass2_before["asr_tokens"] and len(pass2_after["asr_tokens"]) > 0:
                    final_wav = candidate_crop
                    pass2_final = pass2_after
                else:
                    pass2_final = pass2_before
            else:
                pass2_final = pass2_before
        else:
            pass2_final = pass2_before

        final_metrics = compute_interval_acoustic_metrics(final_wav, sr=sr, is_speech=True)
        if final_metrics.rms_dbfs <= -50.0 or len(final_wav) < int(MIN_SPEECH_DURATION_SEC * sr):
            return {"retained": False, "reject_reason": "inaudible_after_trim", "orig_duration": orig_duration}

        edit_diag = compute_word_edit_diagnostics(pass2_final["asr_tokens"], ref_tokens)
        if not edit_diag["passes_edit_gate"]:
            reason = (
                "consec_insertions>=2"
                if edit_diag["reject_insertions"]
                else ("replacements>=3" if edit_diag["reject_replacements"] else "zero_final_overlap")
            )
            return {
                "retained": False,
                "reject_reason": reason,
                "orig_duration": orig_duration,
                "clean_duration": float(len(final_wav) / sr),
                "asr_transcript": " ".join(pass2_final["asr_tokens"]),
                "ref_transcript": " ".join(ref_tokens),
                **edit_diag,
            }

        clean_duration = float(len(final_wav) / sr)
        out_path_str = None
        if dst_audio_path is not None:
            dst_audio_path = Path(dst_audio_path)
            dst_audio_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst_audio_path), np.clip(final_wav, -1.0, 1.0), sr, subtype="PCM_16")
            out_path_str = str(dst_audio_path.resolve())

        return {
            "retained": True,
            "reject_reason": "retained",
            "clean_audio_path": out_path_str,
            "orig_duration": orig_duration,
            "clean_duration": clean_duration,
            "laughter_intervals_removed": int(laughter_intervals_removed),
            "laughter_seconds_removed": round(float(laughter_seconds_removed), 4),
            "unrelated_speech_removed": int(unrelated_speech_removed),
            "retained_pauses_count": int(retained_pauses_count),
            "was_concatenated": bool(not is_contiguous),
            "asr_transcript": " ".join(pass2_final["asr_tokens"]),
            "ref_transcript": " ".join(ref_tokens),
            **edit_diag,
        }


def run_emo_tag_cleaning_on_split(
    df: Optional[pd.DataFrame],
    split_name: str,
    cleaner: EmoTagWhisperXCleaner,
    clean_root: Path,
    overwrite: bool = False,
    checkpoint_every: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run EMO-TAG 5-step cleaning on a MELD split with resumable checkpoints."""
    clean_root = Path(clean_root)
    split_out_dir = clean_root / split_name
    split_out_dir.mkdir(parents=True, exist_ok=True)
    clean_csv_path = clean_root / f"clean_{split_name}_set.csv"
    audit_csv_path = clean_root / f"emo_tag_audit_{split_name}.csv"
    partial_audit_path = clean_root / f"emo_tag_audit_{split_name}.partial.csv"
    cues_csv_path = clean_root / f"clean_{split_name}_set_with_cues.csv"

    if clean_csv_path.exists() and audit_csv_path.exists() and not overwrite:
        return pd.read_csv(clean_csv_path), pd.read_csv(audit_csv_path)
    if not overwrite and clean_csv_path.exists():
        return pd.read_csv(clean_csv_path), pd.DataFrame()
    if not overwrite and cues_csv_path.exists():
        return pd.read_csv(cues_csv_path), pd.DataFrame()
    if df is None:
        return pd.DataFrame(), pd.DataFrame()

    done_lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if partial_audit_path.exists() and not overwrite:
        try:
            for _, prow in pd.read_csv(partial_audit_path).iterrows():
                key = (int(prow["dialogue"]), int(prow["utterance"]))
                if not bool(prow.get("retained", False)) or (
                    pd.notna(prow.get("clean_audio_path")) and Path(str(prow["clean_audio_path"])).exists()
                ):
                    done_lookup[key] = prow.to_dict()
        except Exception:
            done_lookup = {}

    audit_records: List[Dict[str, Any]] = []
    retained_rows: List[Dict[str, Any]] = []
    for step_idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"EMO-TAG [{split_name}]"), 1):
        dia_id, utt_id = int(row["dialogue"]), int(row["utterance"])
        key = (dia_id, utt_id)
        dst_wav = split_out_dir / f"dia{dia_id}_utt{utt_id}.wav"
        if key in done_lookup:
            audit_row = done_lookup[key]
            res = audit_row
        else:
            res = cleaner.clean_utterance(
                src_audio_path=str(row["audio_path"]),
                reference_transcript=str(row.get("text", "")),
                dst_audio_path=dst_wav,
            )
            audit_row = {
                "dialogue": dia_id,
                "utterance": utt_id,
                "speaker": row.get("speaker"),
                "emotion": row.get("emotion"),
                "raw_audio_path": row.get("audio_path"),
                **res,
            }
        audit_records.append(audit_row)
        if bool(res.get("retained", False)):
            clean_row = row.to_dict()
            clean_row["raw_audio_path"] = row.get("audio_path")
            clean_row["audio_path"] = res["clean_audio_path"]
            clean_row["orig_duration"] = res["orig_duration"]
            clean_row["duration"] = res["clean_duration"]
            clean_row["laughter_intervals_removed"] = res["laughter_intervals_removed"]
            clean_row["laughter_seconds_removed"] = res["laughter_seconds_removed"]
            clean_row["was_concatenated"] = res["was_concatenated"]
            retained_rows.append(clean_row)
        if checkpoint_every > 0 and (step_idx % checkpoint_every == 0):
            pd.DataFrame(audit_records).to_csv(partial_audit_path, index=False)

    clean_df = pd.DataFrame(retained_rows)
    audit_df = pd.DataFrame(audit_records)
    clean_df.to_csv(clean_csv_path, index=False)
    audit_df.to_csv(audit_csv_path, index=False)
    if partial_audit_path.exists():
        partial_audit_path.unlink()
    return clean_df, audit_df
