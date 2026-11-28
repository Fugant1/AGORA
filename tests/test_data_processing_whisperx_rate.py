"""Tests for data_processing.ipynb syntax, Run-All idempotency, and Section 10 batched WhisperX + CMUdict/g2p_en speech-rate tracking."""
import ast
import json
import math
import shutil
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

NOTEBOOK = Path(__file__).resolve().parents[1] / "data_processing.ipynb"

# Authentic CMU Pronouncing Dictionary v0.7b ARPAbet entries for unit testing
SAMPLE_CMUDICT: Dict[str, List[List[str]]] = {
    "hello": [["HH", "AH0", "L", "OW1"], ["HH", "EH0", "L", "OW1"]],
    "my": [["M", "AY1"]],
    "people": [["P", "IY1", "P", "AH0", "L"]],
    "didn't": [["D", "IH1", "D", "AH0", "N", "T"]],
    "don't": [["D", "OW1", "N", "T"]],
    "phoebe": [["F", "IY1", "B", "IY0"]],
    "joey": [["JH", "OW1", "IY0"]],
    "rachel": [["R", "EY1", "CH", "AH0", "L"]],
    "okay": [["OW2", "K", "EY1"]],
    "ok": [["OW2", "K", "EY1"]],
    "you're": [["Y", "UH1", "R"]],
    "no": [["N", "OW1"]],
    "hey": [["HH", "EY1"]],
    "how": [["HH", "AW1"]],
    "are": [["AA1", "R"]],
    "you": [["Y", "UW1"]],
    "doing": [["D", "UW1", "IH0", "NG"]],
    "today": [["T", "AH0", "D", "EY1"]],
    "world": [["W", "ER1", "L", "D"]],
    "friend": [["F", "R", "EH1", "N", "D"]],
    "go": [["G", "OW1"]],
}


def _clean_source(cell_source: List[str]) -> str:
    return "".join(
        line for line in cell_source if not line.lstrip().startswith(("!", "%"))
    )


def test_all_notebook_cells_parse_cleanly_and_have_no_ad_hoc_exception_tables():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 16
    full_code = []
    for idx, cell in enumerate(code_cells):
        src = _clean_source(cell["source"])
        ast.parse(src, filename=f"data_processing.ipynb#code_cell_{idx}")
        full_code.append(src)
    combined = "\n".join(full_code)
    assert "_SYLLABLE_EXCEPTIONS" not in combined
    assert "_DIGIT_SYLLABLES" not in combined
    assert "PhoneticSyllableCounter" in combined
    assert "averaged_perceptron_tagger_eng" in combined


@pytest.fixture
def env(tmp_path):
    """Load helper functions and classes from data_processing.ipynb without running top-level I/O."""
    import gc
    import re

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    scope: Dict[str, Any] = {
        "dataclass": dataclass,
        "Path": Path,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "shutil": shutil,
        "np": np,
        "pd": pd,
        "sf": sf,
        "re": re,
        "gc": gc,
        "math": math,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Sequence": Sequence,
        "Tuple": Tuple,
        "DATA_DIR": tmp_path / "local" / "meld_16k_mono",
        "DRIVE_DATA_DIR": tmp_path / "drive" / "meld_16k_mono",
        "DRIVE_ARCHIVE": tmp_path / "MELD.tar",
        "MELD_DIR": tmp_path / "prep_meld",
        "CLEAN_DATA_DIR": tmp_path / "local" / "meld_16k_mono" / "emo_tag_clean",
        "EMO_TAG_SR": 16000,
        "PAUSE_MAX_RMS_DBFS": -35.0,
        "PAUSE_MAX_PEAK_DBFS": -25.0,
        "MIN_TRAIN_DURATION_SEC": 0.5,
        "MAX_TRAIN_DURATION_SEC": 20.0,
        "ACOUSTIC_CUES": {},
        "TARGET_SR": 16000,
        "TARGET_CHANNELS": 1,
        "MAX_WORKERS": 4,
        "OVERWRITE_AUDIO": False,
        "tqdm": None,
    }
    wanted_names = {
        "EMO_TAG_SR",
        "WHISPERX_RATE_BATCH_SIZE",
        "WHISPERX_RATE_SILENCE_PAD_SEC",
        "WHISPERX_RATE_CHECKPOINT_EVERY_BATCHES",
        "OVERWRITE_WHISPERX_RATE",
        "OVERWRITE_WHISPERX_RATE_VAL",
        "WHISPERX_RATE_COLUMNS",
    }
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        if cell.get("id") not in ("KYeTZExdNJk-", "YPuivv9SNJlC", "wxRateCode10", "wxRateValCode11"):
            continue
        tree = ast.parse(_clean_source(cell["source"]))
        nodes = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                nodes.append(node)
            elif isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id in wanted_names for t in node.targets):
                    nodes.append(node)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id in wanted_names:
                    nodes.append(node)
        if nodes:
            exec(compile(ast.Module(body=nodes, type_ignores=[]), f"data_processing.ipynb#{cell.get('id')}", "exec"), scope)
    return scope


def test_phonetic_syllable_counter_cmudict_and_g2p_fallback(env):
    g2p_calls: List[str] = []

    def fake_g2p(token: str) -> List[str]:
        g2p_calls.append(token)
        if token == "200":
            return ["T", "UW1", " ", "HH", "AH1", "N", "D", "R", "AH0", "D"]  # 3 syllables
        return ["G", "UW1", "F", "B", "AO2", "L"]  # 2 syllables

    counter = env["PhoneticSyllableCounter"](cmu_dict=SAMPLE_CMUDICT, g2p_fn=fake_g2p)

    # Primary CMUdict hits (never touch g2p_en)
    assert counter.count_word("didn't") == 2
    assert counter.count_word("don't") == 1
    assert counter.count_word("people!") == 2
    assert counter.count_word("Phoebe") == 2
    assert counter.count_word("Joey") == 2
    assert counter.count_word("OK") == 2
    assert counter.count_word("no-no") == 2  # hyphenated compound decomposed into CMUdict parts
    assert counter.count_word("...") == 0
    assert g2p_calls == []

    # OOV token & numeral trigger g2p_en fallback once and are memoized
    assert counter.count_word("200") == 3
    assert counter.count_word("200") == 3
    assert counter.count_word("goofball") == 2
    assert g2p_calls == ["200", "goofball"]
    assert counter.cmu_hits == 7
    assert counter.g2p_hits == 2


def test_speech_rate_metrics_from_word_timestamps_and_pauses(env):
    counter = env["PhoneticSyllableCounter"](cmu_dict=SAMPLE_CMUDICT, g2p_fn=lambda w: ["AH0"])
    fn = env["compute_speech_rate_metrics_from_words"]

    # 3 words across [0.10, 0.50], [0.80, 1.20], [1.50, 2.10] inside a 2.50s WAV
    # Active speech = 0.40 + 0.40 + 0.60 = 1.40s
    # Inter-word pauses = 0.30 + 0.30 = 0.60s
    # Total spoken span (speech + pause) = 2.00s
    words = [
        {"word": "Hello", "start": 0.10, "end": 0.50},   # 2 syl (CMUdict)
        {"word": "my", "start": 0.80, "end": 1.20},      # 1 syl (CMUdict)
        {"word": "people!", "start": 1.50, "end": 2.10}, # 2 syl (CMUdict)
    ]
    res = fn(words, audio_duration=2.50, syllable_counter=counter, reference_text="Hello my people!")
    assert res["whisperx_word_count"] == 3
    assert res["whisperx_syllable_count"] == 5
    assert res["whisperx_speech_duration"] == pytest.approx(1.40, abs=1e-5)
    assert res["whisperx_pause_duration"] == pytest.approx(0.60, abs=1e-5)
    assert res["whisperx_utterance_span"] == pytest.approx(2.00, abs=1e-5)
    assert res["whisperx_syllable_rate"] == pytest.approx(2.50, abs=1e-5)
    assert res["whisperx_word_rate"] == pytest.approx(1.50, abs=1e-5)

    # Fallback when alignment returns no timestamped words
    fb = fn([], audio_duration=2.0, syllable_counter=counter, reference_text="Okay don't go")
    assert fb["whisperx_word_count"] == 3
    assert fb["whisperx_syllable_count"] == 4
    assert fb["whisperx_utterance_span"] == pytest.approx(2.0)
    assert fb["whisperx_syllable_rate"] == pytest.approx(2.0)


def test_global_quartiles_are_gender_agnostic_25_50_25(env):
    classify = env["classify_global_quartiles"]
    vals = pd.Series(np.arange(1.0, 21.0))
    labels, stats = classify(vals)
    assert stats["count"] == 20
    assert (labels == "low").sum() == 5
    assert (labels == "moderate").sum() == 10
    assert (labels == "high").sum() == 5
    assert set(labels.unique()) == {"low", "moderate", "high"}

    # Identical values across F and M receive identical global labels
    mixed = pd.Series([1.0, 1.0, 5.0, 5.0, 10.0, 10.0, 20.0, 20.0])
    mixed_labels, _ = classify(mixed)
    assert mixed_labels.iloc[0] == mixed_labels.iloc[1] == "low"
    assert mixed_labels.iloc[2] == mixed_labels.iloc[3] == "moderate"
    assert mixed_labels.iloc[6] == mixed_labels.iloc[7] == "high"


def test_batched_16_whisperx_tracking_preserves_existing_data_and_skips_on_rerun(env, tmp_path):
    local_dir = tmp_path / "local" / "meld_16k_mono"
    clean_dir = local_dir / "emo_tag_clean"
    train_wav_dir = clean_dir / "train"
    drive_dir = tmp_path / "drive" / "meld_16k_mono"
    drive_clean_dir = drive_dir / "emo_tag_clean"
    train_wav_dir.mkdir(parents=True)
    drive_clean_dir.mkdir(parents=True)

    # Register pre-existing files that must NEVER be rewritten
    quartiles_file = clean_dir / "gender_acoustic_quartiles.json"
    quartiles_file.write_text('{"F": {}, "M": {}}', encoding="utf-8")
    val_file = clean_dir / "clean_val_set_with_cues.csv"
    val_file.write_text("audio_path,emotion\na.wav,joy\n", encoding="utf-8")
    drive_tar = drive_dir / "meld_16k_mono_clean.tar"
    drive_tar.write_bytes(b"existing-tar-bytes")
    drive_quartiles = drive_clean_dir / "gender_acoustic_quartiles.json"
    drive_quartiles.write_text('{"F": {}, "M": {}}', encoding="utf-8")

    # Create 35 cleaned training WAVs (tests batches of [16, 16, 3])
    n_rows = 35
    rows = []
    for i in range(n_rows):
        dur = 1.0 + (i % 5) * 0.4
        wav = np.zeros(int(round(dur * 16000)), dtype=np.float32)
        wav_path = train_wav_dir / f"dia{i}_utt0.wav"
        sf.write(str(wav_path), wav, 16000, subtype="PCM_16")
        n_words = 2 + (i % 6)
        text = " ".join(["hello"] * n_words)
        rows.append(
            {
                "audio_path": str(wav_path),
                "dialogue": i,
                "utterance": 0,
                "emotion": "joy" if i % 2 == 0 else "neutral",
                "text": text,
                "speaker": "Rachel" if i % 2 == 0 else "Ross",
                "duration": dur,
                "gender": "F" if i % 2 == 0 else "M",
                "effective_gender": "F" if i % 2 == 0 else "M",
                "speaking_rate": 3.0 + i * 0.01,
                "speaking_rate_cue": "moderate",
            }
        )
    orig_df = pd.DataFrame(rows)
    train_csv = clean_dir / "clean_train_set_with_cues.csv"
    orig_df.to_csv(train_csv, index=False)

    bars = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.updates = []
            self.postfixes = []
            self.closed = False
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def update(self, n):
            self.updates.append(n)

        def set_postfix(self, **kwargs):
            self.postfixes.append(kwargs)

    env["tqdm"] = FakeTqdm
    asr_batch_sizes = []

    class FakeAsrModel:
        _use_interleaved_context = True
        previous_batch_context_tokens = [[1, 2, 3]]

        def __call__(self, batch_inputs, batch_size=16):
            assert self._use_interleaved_context is False
            assert self.previous_batch_context_tokens == [[]] * len(batch_inputs)
            asr_batch_sizes.append(len(batch_inputs))
            return [{"text": f"hello world {'friend ' * (idx % 4)}".strip()} for idx, _ in enumerate(batch_inputs)]

    align_calls = []

    def fake_align(segments, align_model, align_metadata, audio, device, return_char_alignments=False):
        align_calls.append(len(segments))
        out_segs = []
        for seg in segments:
            tokens = seg["text"].split()
            t0 = float(seg["start"])
            t1 = float(seg["end"])
            span = max(0.2, t1 - t0)
            step = span / (len(tokens) + 1)
            words = []
            for k, tok in enumerate(tokens):
                ws = t0 + k * step
                we = ws + step * 0.7
                words.append({"word": tok, "start": ws, "end": we})
            out_segs.append({"text": seg["text"], "start": t0, "end": t1, "words": words})
        return {"segments": out_segs, "word_segments": [w for s in out_segs for w in s["words"]]}

    counter = env["PhoneticSyllableCounter"](cmu_dict=SAMPLE_CMUDICT, g2p_fn=lambda w: ["AH0"])

    # 1. save_local_pipeline_to_drive never overwrites existing Drive tar or existing Drive metadata
    env["save_local_pipeline_to_drive"](local_dir=local_dir, drive_dir=drive_dir, overwrite=False)
    assert drive_tar.read_bytes() == b"existing-tar-bytes"
    assert drive_quartiles.read_text(encoding="utf-8") == '{"F": {}, "M": {}}'

    # 2. Run batched WhisperX + CMUdict speech-rate tracking (16 audios/batch)
    updated_df = env["add_whisperx_speech_rates_to_clean_train"](
        train_csv_path=train_csv,
        drive_dir=drive_dir,
        batch_size=16,
        overwrite=False,
        asr_model=FakeAsrModel(),
        align_model=object(),
        align_metadata={"language": "en"},
        device="cpu",
        align_fn=fake_align,
        syllable_counter=counter,
    )

    assert asr_batch_sizes == [16, 16, 3]
    assert align_calls == [16, 16, 3]
    assert bars[0].kwargs["total"] == 35
    assert "{remaining}" in bars[0].kwargs["bar_format"]
    assert bars[0].updates == [16, 16, 3]
    assert bars[0].closed is True

    for col in env["WHISPERX_RATE_COLUMNS"]:
        assert col in updated_df.columns
    pd.testing.assert_frame_equal(updated_df[orig_df.columns], orig_df)

    assert (updated_df["whisperx_pause_duration"] > 0.0).all()
    assert np.allclose(
        updated_df["whisperx_speech_duration"] + updated_df["whisperx_pause_duration"],
        updated_df["whisperx_utterance_span"],
        atol=1e-5,
    )
    assert set(updated_df["whisperx_syllable_rate_cue"].unique()) == {"low", "moderate", "high"}
    assert set(updated_df["whisperx_word_rate_cue"].unique()) == {"low", "moderate", "high"}

    drive_train_csv = drive_clean_dir / "clean_train_set_with_cues.csv"
    assert drive_train_csv.is_file()
    pd.testing.assert_frame_equal(pd.read_csv(drive_train_csv), pd.read_csv(train_csv))
    assert drive_tar.read_bytes() == b"existing-tar-bytes"
    assert quartiles_file.read_text(encoding="utf-8") == '{"F": {}, "M": {}}'
    assert val_file.read_text(encoding="utf-8") == "audio_path,emotion\na.wav,joy\n"
    assert not (clean_dir / "whisperx_rate_train.partial.csv").exists()

    # 3. Re-running skips completely with 0 WhisperX calls
    asr_batch_sizes.clear()
    align_calls.clear()
    second_df = env["add_whisperx_speech_rates_to_clean_train"](
        train_csv_path=train_csv,
        drive_dir=drive_dir,
        batch_size=16,
        overwrite=False,
        asr_model=FakeAsrModel(),
        align_model=object(),
        align_metadata={"language": "en"},
        device="cpu",
        align_fn=fake_align,
        syllable_counter=counter,
    )
    assert asr_batch_sizes == []
    assert align_calls == []
    pd.testing.assert_frame_equal(second_df, updated_df)


def test_val_whisperx_uses_train_fitted_global_quartiles_without_leakage_and_skips_on_rerun(env, tmp_path):
    local_dir = tmp_path / "local" / "meld_16k_mono"
    clean_dir = local_dir / "emo_tag_clean"
    val_wav_dir = clean_dir / "val"
    drive_dir = tmp_path / "drive" / "meld_16k_mono"
    drive_clean_dir = drive_dir / "emo_tag_clean"
    val_wav_dir.mkdir(parents=True)
    drive_clean_dir.mkdir(parents=True)

    # Pre-existing train CSV with high global rates (Q25=10.75, Q75=18.25) and protected files
    train_csv = clean_dir / "clean_train_set_with_cues.csv"
    train_df = pd.DataFrame(
        {
            "audio_path": [f"/clean/train/dia{i}_utt0.wav" for i in range(20)],
            "dialogue": list(range(20)),
            "utterance": [0] * 20,
            "whisperx_syllable_rate": np.linspace(8.0, 22.0, 20),
            "whisperx_word_rate": np.linspace(4.0, 12.0, 20),
            "whisperx_syllable_rate_cue": ["moderate"] * 20,
            "whisperx_word_rate_cue": ["moderate"] * 20,
        }
    )
    train_df.to_csv(train_csv, index=False)
    train_bytes_before = train_csv.read_bytes()

    quartiles_file = clean_dir / "gender_acoustic_quartiles.json"
    quartiles_file.write_text('{"F": {}, "M": {}}', encoding="utf-8")
    drive_tar = drive_dir / "meld_16k_mono_clean.tar"
    drive_tar.write_bytes(b"existing-tar-bytes")

    # Create 19 val WAVs (tests batches of [16, 3]) whose rates will all fall below train Q25 (~11.5 syl/s)
    val_rows = []
    for i in range(19):
        dur = 2.0 + (i % 3) * 0.5
        wav_path = val_wav_dir / f"dia{i}_utt0.wav"
        sf.write(str(wav_path), np.zeros(int(round(dur * 16000)), dtype=np.float32), 16000, subtype="PCM_16")
        val_rows.append(
            {
                "audio_path": f"/old/runtime/emo_tag_clean/val/{wav_path.name}",
                "dialogue": i,
                "utterance": 0,
                "emotion": "neutral",
                "text": "hello world",
                "speaker": "Rachel" if i % 2 == 0 else "Ross",
                "duration": dur,
                "gender": "F" if i % 2 == 0 else "M",
                "effective_gender": "F" if i % 2 == 0 else "M",
                "speaking_rate": 3.0,
                "speaking_rate_cue": "high",
            }
        )
    orig_val = pd.DataFrame(val_rows)
    val_csv = clean_dir / "clean_val_set_with_cues.csv"
    orig_val.to_csv(val_csv, index=False)

    bars = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.updates = []
            self.closed = False
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def update(self, n):
            self.updates.append(n)

        def set_postfix(self, **kwargs):
            pass

    env["tqdm"] = FakeTqdm
    asr_batches = []

    class FakeAsrModel:
        def __call__(self, batch_inputs, batch_size=16):
            asr_batches.append(len(batch_inputs))
            return [{"text": "hello world"} for _ in batch_inputs]

    def fake_align(segments, align_model, align_metadata, audio, device, return_char_alignments=False):
        out_segs = []
        for seg in segments:
            t0 = float(seg["start"])
            t1 = float(seg["end"])
            # 3 syllables ("hello"=2, "world"=1) across ~1.5s => ~2.0 syl/s (strictly < train Q25)
            words = [
                {"word": "hello", "start": t0 + 0.1, "end": t0 + 0.6},
                {"word": "world", "start": t0 + 1.0, "end": t0 + 1.6},
            ]
            out_segs.append({"text": seg["text"], "start": t0, "end": t1, "words": words})
        return {"segments": out_segs, "word_segments": [w for s in out_segs for w in s["words"]]}

    counter = env["PhoneticSyllableCounter"](cmu_dict=SAMPLE_CMUDICT, g2p_fn=lambda w: ["AH0"])

    # Missing WhisperX columns in train CSV must fail early
    bad_train = tmp_path / "train_no_wx.csv"
    pd.DataFrame([{"audio_path": "a.wav"}]).to_csv(bad_train, index=False)
    with pytest.raises(ValueError, match="missing 'whisperx_syllable_rate'"):
        env["add_whisperx_speech_rates_to_clean_val"](
            val_csv_path=val_csv, train_csv_path=bad_train, drive_dir=drive_dir
        )

    updated_val = env["add_whisperx_speech_rates_to_clean_val"](
        val_csv_path=val_csv,
        train_csv_path=train_csv,
        drive_dir=drive_dir,
        batch_size=16,
        overwrite=False,
        asr_model=FakeAsrModel(),
        align_model=object(),
        align_metadata={"language": "en"},
        device="cpu",
        align_fn=fake_align,
        syllable_counter=counter,
    )

    assert asr_batches == [16, 3]
    assert bars[0].kwargs["total"] == 19
    assert "{remaining}" in bars[0].kwargs["bar_format"]
    assert bars[0].updates == [16, 3]
    assert bars[0].closed is True
    for col in env["WHISPERX_RATE_COLUMNS"]:
        assert col in updated_val.columns
    pd.testing.assert_frame_equal(updated_val[orig_val.columns], orig_val)

    # Zero leakage check: all val rates (~2.0 syl/s, ~1.33 words/s) are below train Q25, so ALL must be "low"
    assert (updated_val["whisperx_syllable_rate_cue"] == "low").all()
    assert (updated_val["whisperx_word_rate_cue"] == "low").all()

    drive_val_csv = drive_clean_dir / "clean_val_set_with_cues.csv"
    assert drive_val_csv.is_file()
    pd.testing.assert_frame_equal(pd.read_csv(drive_val_csv), pd.read_csv(val_csv))
    assert train_csv.read_bytes() == train_bytes_before
    assert drive_tar.read_bytes() == b"existing-tar-bytes"
    assert quartiles_file.read_text(encoding="utf-8") == '{"F": {}, "M": {}}'
    assert not (clean_dir / "whisperx_rate_val.partial.csv").exists()

    # Re-run skips completely with 0 WhisperX calls
    asr_batches.clear()
    second_val = env["add_whisperx_speech_rates_to_clean_val"](
        val_csv_path=val_csv,
        train_csv_path=train_csv,
        drive_dir=drive_dir,
        batch_size=16,
        overwrite=False,
        asr_model=FakeAsrModel(),
        align_model=object(),
        align_metadata={"language": "en"},
        device="cpu",
        align_fn=fake_align,
        syllable_counter=counter,
    )
    assert asr_batches == []
    pd.testing.assert_frame_equal(second_val, updated_val)


def test_full_run_all_simulation_skips_completed_stages_and_runs_only_whisperx_rate(tmp_path, monkeypatch):
    """Simulate Colab 'Run All' on a pre-populated workspace where Stages 1-9 are already cached."""
    local_dir = tmp_path / "agora_work" / "meld_16k_mono"
    clean_dir = local_dir / "emo_tag_clean"
    train_wav_dir = clean_dir / "train"
    val_wav_dir = clean_dir / "val"
    train_wav_dir.mkdir(parents=True)
    val_wav_dir.mkdir(parents=True)

    rows = []
    for i in range(18):
        dur = 1.2 + (i % 3) * 0.5
        wav_path = train_wav_dir / f"dia{i}_utt0.wav"
        sf.write(str(wav_path), np.zeros(int(round(dur * 16000)), dtype=np.float32), 16000, subtype="PCM_16")
        rows.append(
            {
                "audio_path": str(wav_path),
                "dialogue": i,
                "utterance": 0,
                "emotion": "joy",
                "text": "Hey how are you doing today",
                "speaker": "Rachel" if i % 2 == 0 else "Ross",
                "duration": dur,
                "gender": "F" if i % 2 == 0 else "M",
                "effective_gender": "F" if i % 2 == 0 else "M",
                "F0semitoneFrom27.5Hz_sma3nz_percentile50.0": 38.0 if i % 2 == 0 else 31.0,
                "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2": 4.5,
                "loudness_sma3_percentile50.0": 1.1,
                "VoicedSegmentsPerSec": 3.2,
                "alphaRatioV_sma3nz_amean": -6.0,
                "pitch_level": 38.0 if i % 2 == 0 else 31.0,
                "pitch_variation": 4.5,
                "vocal_energy": 1.1,
                "speaking_rate": 3.2,
                "timbre": -6.0,
                "pitch_level_cue": "moderate",
                "pitch_variation_cue": "moderate",
                "vocal_energy_cue": "moderate",
                "speaking_rate_cue": "moderate",
                "timbre_cue": "moderate (modal)",
            }
        )
    train_df = pd.DataFrame(rows)
    val_rows = []
    for i in range(4):
        dur = 1.5
        wav_path = val_wav_dir / f"dia{i}_utt0.wav"
        sf.write(str(wav_path), np.zeros(int(round(dur * 16000)), dtype=np.float32), 16000, subtype="PCM_16")
        val_row = dict(rows[i], audio_path=str(wav_path), duration=dur)
        val_rows.append(val_row)
    val_df = pd.DataFrame(val_rows)

    train_df.to_csv(clean_dir / "clean_train_set.csv", index=False)
    train_df.to_csv(clean_dir / "train_egemaps.csv", index=False)
    train_df.to_csv(clean_dir / "clean_train_set_with_cues.csv", index=False)
    val_df.to_csv(clean_dir / "clean_val_set.csv", index=False)
    val_df.to_csv(clean_dir / "val_egemaps.csv", index=False)
    val_df.to_csv(clean_dir / "clean_val_set_with_cues.csv", index=False)
    (clean_dir / "gender_acoustic_quartiles.json").write_text('{"F": {}, "M": {}, "__GLOBAL__": {}}', encoding="utf-8")

    asr_batches = []

    class MockAsr:
        def transcribe(self, wav, batch_size=16, language="en"):
            return {"segments": []}

        def __call__(self, batch_inputs, batch_size=16):
            asr_batches.append(len(batch_inputs))
            return [{"text": "hey how are you doing today"} for _ in batch_inputs]

    mock_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))
    mock_cmudict = types.SimpleNamespace(dict=lambda: SAMPLE_CMUDICT)
    mock_whisperx = types.SimpleNamespace(
        load_model=lambda *a, **kw: MockAsr(),
        load_align_model=lambda *a, **kw: (object(), {"language": "en"}),
        align=lambda segments, m, meta, audio, dev, return_char_alignments=False: {
            "segments": [
                {
                    "text": s["text"],
                    "start": s["start"],
                    "end": s["end"],
                    "words": [
                        {"word": w, "start": s["start"] + k * 0.18, "end": s["start"] + k * 0.18 + 0.12}
                        for k, w in enumerate(s["text"].split())
                    ],
                }
                for s in segments
            ]
        },
    )
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "cmudict", mock_cmudict)
    monkeypatch.setitem(sys.modules, "whisperx", mock_whisperx)

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns: Dict[str, Any] = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = _clean_source(cell["source"])
        if not src.strip():
            continue
        if cell.get("id") == "nTVctV94NJk1":
            src = src.replace(
                'DATA_DIR = Path("/home/fuganti/Projects/AGORA/data/meld_16k_mono")',
                f'DATA_DIR = Path({str(local_dir)!r})',
            ).replace(
                'MELD_DIR = Path("/home/fuganti/Projects/AFTA/data/prep_meld")',
                f'MELD_DIR = Path({str(tmp_path / "missing_meld")!r})',
            )
        exec(compile(src, f"data_processing.ipynb#{cell.get('id')}", "exec"), ns)

    assert asr_batches == [16, 2, 4]
    saved_train = pd.read_csv(clean_dir / "clean_train_set_with_cues.csv")
    saved_val = pd.read_csv(clean_dir / "clean_val_set_with_cues.csv")
    assert len(saved_train) == 18
    assert len(saved_val) == 4
    for frame in (saved_train, saved_val):
        for col in ns["WHISPERX_RATE_COLUMNS"]:
            assert col in frame.columns
        # "hey how are you doing today" = 1+1+1+1+2+2 = 8 phonetic syllables in CMUdict
        assert (frame["whisperx_syllable_count"] == 8).all()
        assert (frame["whisperx_word_count"] == 6).all()
        assert frame["whisperx_syllable_rate_cue"].isin(["low", "moderate", "high"]).all()
        assert frame["whisperx_word_rate_cue"].isin(["low", "moderate", "high"]).all()
    assert (clean_dir / "gender_acoustic_quartiles.json").read_text(encoding="utf-8") == '{"F": {}, "M": {}, "__GLOBAL__": {}}'
