"""Comprehensive mathematical parity & vLLM lifecycle tests for src/ vs V4 Colab notebooks."""

import io
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.acoustics.quartiles import (
    apply_gender_quartile_classification,
    classify_value_by_quartile,
    compute_gender_acoustic_quartiles,
)
from src.config import (
    ACOUSTIC_CUES,
    CUE_COLUMNS,
    CUE_LABELS,
    CUES,
    EMOTIONS,
    ExperimentConfig,
    RewardConfig,
    VLLMConfig,
)
from src.data.emo_tag import (
    IntervalMetrics,
    compute_word_edit_diagnostics,
    is_natural_pause_or_sigh,
    normalize_emo_tag_transcript,
)
from src.data.gender import get_lexical_gender
from src.data.partitions import (
    compute_semi_balanced_quotas,
    make_partitions,
    validate_partitions,
)
from src.data.whisperx_rate import (
    PhoneticSyllableCounter,
    apply_global_quartiles,
    compute_speech_rate_metrics_from_words,
    count_arpabet_syllables,
    fit_global_quartile_thresholds,
)
from src.models.vllm_engine import VLLMRolloutEngine
from src.pipeline import build_arg_parser
from src.prompts.templates import (
    build_bridge_generation_prompt,
    format_sft_target,
    student_instruction,
)
from src.rewards.composite import (
    GRPORewardCalibrator,
    unified_gated_grpo_reward,
    validate_grpo_reward_mathematics,
)
from src.rewards.parser import parse_completion, parse_completion_strict
from src.training.credit_assignment import (
    compute_span_split_advantages_np,
    cue_group_advantages,
    cue_span_length_scale,
    group_standardize,
    validate_credit_assignment,
)
from src.training.evaluation import score_validation
from src.training.sft import build_assistant_token_weights, inverse_frequency_cue_weights
from src.utils.io import validate_grpo_partition

ROOT = Path(__file__).resolve().parents[1]
GRPO_ZIP = ROOT / "grpo_v3_hybrid_invfreq_4cue_mt_sft_seed42_cuekl005_v4.zip"
SFT_ZIP = ROOT / "full_sft_gender_wxrate_multitask_alpha05_seed42_v4.zip"


def test_all_reward_modes_pass_mathematical_proofs():
    for kwargs in [
        dict(use_zero_mean=False, use_map=False, use_inverse_frequency=True, use_distance_based_payoff=False, inverse_frequency_alpha=1.0),
        dict(use_zero_mean=False, use_map=False, use_inverse_frequency=True, use_distance_based_payoff=False, inverse_frequency_alpha=0.7),
        dict(use_zero_mean=False, use_map=False, use_inverse_frequency=True, use_distance_based_payoff=False, inverse_frequency_alpha=0.5),
        dict(use_zero_mean=True, use_map=False, use_inverse_frequency=False, use_distance_based_payoff=False),
        dict(use_zero_mean=False, use_map=True, use_inverse_frequency=False, use_distance_based_payoff=False),
        dict(use_zero_mean=False, use_map=False, use_inverse_frequency=True, use_distance_based_payoff=True),
    ]:
        report = validate_grpo_reward_mathematics(**kwargs)
        assert report["r1_matrix_proof"] is not None


@pytest.mark.skipif(not GRPO_ZIP.is_file(), reason="Saved GRPO Colab archive not present")
def test_exact_zero_error_parity_with_colab_grpo_run():
    """Verify src/ reproduces R1, R2, R3, total_rewards, cue advantages, and val metrics to <1e-12 vs Colab."""
    with zipfile.ZipFile(GRPO_ZIP) as zf:
        cal = json.loads(zf.read("evaluation_calibration.json"))
        groups = [
            json.loads(line)
            for line in zf.read("grpo_groups.jsonl").decode("utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        val_df = pd.read_csv(io.BytesIO(zf.read("after_validation_predictions.csv")))
        saved_metrics = json.loads(zf.read("after_validation_metrics.json"))

    # 1. Re-score first 30 rollout groups (240 completions) using src/
    max_r1_err = max_r2_err = max_r3_err = max_adv_err = 0.0
    for g in groups[:30]:
        trace = []
        cue_kwargs = {
            CUE_COLUMNS[c]: [g["r2_cues"][i][c]["truth"] for i in range(len(g["completions"]))]
            for c in CUES
        }
        totals = unified_gated_grpo_reward(
            prompts=["p"] * len(g["completions"]),
            completions=g["completions"],
            w_r1=g["reward_weights"]["r1"],
            w_r2=g["reward_weights"]["r2"],
            w_r3=g["reward_weights"]["r3"],
            label=[g["target"]] * len(g["completions"]),
            emotion_priors=cal["emotion_priors"],
            priors_dict=cal["cue_priors"],
            use_zero_mean=False,
            use_map=False,
            use_inverse_frequency=True,
            inverse_frequency_alpha=1.0,
            use_distance_based_payoff=False,
            _reward_trace_sink=trace,
            **cue_kwargs,
        )
        for i, tr in enumerate(trace):
            max_r1_err = max(max_r1_err, abs(tr["r1"] - g["r1"][i]))
            max_r2_err = max(max_r2_err, abs(tr["r2"] - g["r2"][i]))
            max_r3_err = max(max_r3_err, abs(tr["r3"] - g["r3"][i]))
            assert abs(totals[i] - g["total_rewards"][i]) < 1e-12

        cue_advs = cue_group_advantages(trace, group_size=len(trace), r2_weight=g["reward_weights"]["r2"], loss_type="hybrid")
        for c in CUES:
            max_adv_err = max(max_adv_err, float(np.max(np.abs(cue_advs[c] - np.array(g["advantages_cues"][c])))))

    assert max_r1_err < 1e-12
    assert max_r2_err < 1e-12
    assert max_r3_err < 1e-12
    assert max_adv_err < 1e-12

    # 2. Verify validation scoring matches after_validation_metrics.json identically
    recomputed = score_validation(val_df)
    for k in ("accuracy", "macro_f1", "weighted_f1", "macro_recall", "invalid_rate"):
        assert abs(recomputed["emotion"][k] - saved_metrics["emotion"][k]) < 1e-12
    for c in CUES:
        for k in ("accuracy", "macro_f1", "macro_recall"):
            assert abs(recomputed["cues"][c][k] - saved_metrics["cues"][c][k]) < 1e-12


@pytest.mark.skipif(not SFT_ZIP.is_file(), reason="Saved SFT-MT Colab archive not present")
def test_exact_sft_multitask_weighting_parity():
    with zipfile.ZipFile(SFT_ZIP) as zf:
        saved_w = pd.read_csv(io.BytesIO(zf.read("cue_loss_weights.csv")))

    # Reconstruct exact counts DataFrame and verify inverse_frequency_cue_weights(alpha=0.5)
    n_sft = int(saved_w.loc[saved_w["cue"] == CUES[0], "count"].sum())
    synth = {}
    for cue in CUES:
        sub = saved_w.loc[saved_w["cue"] == cue]
        col = []
        for _, r in sub.iterrows():
            col.extend([r["label"]] * int(r["count"]))
        assert len(col) == n_sft
        synth[CUE_COLUMNS[cue]] = col
    df_synth = pd.DataFrame(synth)
    recomputed_w = inverse_frequency_cue_weights(df_synth, alpha=0.5)
    merged = saved_w.merge(recomputed_w, on=["cue", "label"], suffixes=("_saved", "_src"))
    assert np.max(np.abs(merged["weight_saved"] - merged["weight_src"])) < 1e-12

    # Verify multitask token weight sum conservation (sum(w_i) == N_total when all cue weights == 1.0)
    spans = [
        ("speaking_rate", [5], 1.0),
        ("pitch_level", [10], 1.0),
        ("vocal_energy", [15], 1.0),
        ("timbre", [20, 21, 22, 23], 1.0),
    ]
    weights = build_assistant_token_weights(100, spans, equal_multitask=True, cue_task_share=0.25)
    assert abs(sum(weights) - 100.0) < 1e-9
    # Only span_pos[0] receives the cue decision weight (25 / 4 = 6.25); suffix tokens [21,22,23] get other_w
    assert abs(weights[20] - 6.25) < 1e-9
    assert abs(weights[21] - (75.0 / 96.0)) < 1e-9


def test_data_processing_and_credit_assignment_primitives():
    assert get_lexical_gender("Rachel") == "F"
    assert get_lexical_gender("Joey") == "M"
    assert get_lexical_gender("Waitress") == "F"
    assert get_lexical_gender("All") is None

    assert normalize_emo_tag_transcript("Don’t go, Joey!!!") == "don't go joey"
    assert is_natural_pause_or_sigh(IntervalMetrics(0, 1, False, -36.0, -26.0, 0.0, 0.0)) is True
    assert is_natural_pause_or_sigh(IntervalMetrics(0, 1, False, -30.0, -26.0, 0.0, 0.0)) is False

    diag = compute_word_edit_diagnostics(["hello", "extra1", "extra2", "world"], ["hello", "world"])
    assert diag["reject_insertions"] is True and diag["passes_edit_gate"] is False

    assert count_arpabet_syllables(["B", "AE1", "N", "AE1", "N", "AH0"]) == 3
    counter = PhoneticSyllableCounter(cmu_dict={"hello": [["HH", "AH0", "L", "OW1"]], "world": [["W", "ER1", "L", "D"]]})
    m = compute_speech_rate_metrics_from_words(
        [{"word": "hello", "start": 0.1, "end": 0.5}, {"word": "world", "start": 0.7, "end": 1.1}],
        audio_duration=1.5,
        syllable_counter=counter,
    )
    assert m["whisperx_syllable_count"] == 3
    assert abs(m["whisperx_utterance_span"] - 1.0) < 1e-6
    assert abs(m["whisperx_syllable_rate"] - 3.0) < 1e-6

    # Chain-rule span-sum scaling: 1..8 tokens -> L; >8 tokens -> L * 8 / span_len
    assert cue_span_length_scale(150, 1) == 150.0
    assert cue_span_length_scale(150, 4) == 150.0
    assert cue_span_length_scale(150, 7) == 150.0
    assert cue_span_length_scale(150, 16) == 75.0
    validate_credit_assignment("hybrid", True, True)

    # Prompt parity check
    instr = student_instruction({"effective_gender": "F"})
    assert "Acoustic reference group (effective speaker gender): F (female)." in instr
    assert "- timbre: low (muffled/soft) | moderate (modal) | high (tense/bright)" in instr


def test_vllm_rollout_engine_live_lora_sync_and_sleep_lifecycle(tmp_path, monkeypatch):
    """Verify VLLMRolloutEngine deduplicates audio, increments LoRARequest ID, and always sleeps on exit."""
    events = []

    class FakeCompletionOutput:
        def __init__(self, token_ids, text):
            self.token_ids = token_ids
            self.text = text

    class FakeRequestOutput:
        def __init__(self, outputs):
            self.outputs = outputs

    class FakeLLM:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))

        def sleep(self, level=1):
            events.append(("sleep", level))

        def wake_up(self):
            events.append(("wake_up", None))

        def generate(self, inputs, sampling_params, lora_request=None, use_tqdm=False):
            events.append(("generate", len(inputs), sampling_params.n, lora_request.lora_int_id, lora_request.lora_path))
            return [
                FakeRequestOutput([FakeCompletionOutput([10 + i, 20 + k], f"c_{i}_{k}") for k in range(sampling_params.n)])
                for i in range(len(inputs))
            ]

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FakeLoRARequest:
        def __init__(self, lora_name, lora_int_id, lora_path):
            self.lora_name = lora_name
            self.lora_int_id = lora_int_id
            self.lora_path = lora_path

    class FakePeftModel:
        def save_pretrained(self, save_dir, selected_adapters=None):
            p = Path(save_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "adapter_config.json").write_text(json.dumps({"selected": selected_adapters}))

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    fake_lora = types.ModuleType("vllm.lora")
    fake_lora_req = types.ModuleType("vllm.lora.request")
    fake_lora_req.LoRARequest = FakeLoRARequest
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", fake_lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake_lora_req)

    cfg = VLLMConfig(enabled=True, lora_sync_dir=tmp_path / "lora_sync", enable_sleep_mode=True)
    engine = VLLMRolloutEngine("Qwen/Qwen2.5-Omni-7B", cfg, seed=42)
    engine.initialize()
    assert events[0][0] == "init"
    assert events[0][1]["quantization"] == "bitsandbytes"
    assert events[0][1]["enable_sleep_mode"] is True
    assert events[1] == ("sleep", 1)

    wavs = [np.zeros(1600, dtype=np.float32), np.ones(1600, dtype=np.float32)]
    toks, texts = engine.generate_rollouts(
        FakePeftModel(),
        unique_prompts=["p0", "p1"],
        unique_audios=wavs,
        num_generations=8,
        temperature=1.0,
        max_tokens=256,
        stop_token_ids=[151645],
    )
    assert len(toks) == 16 and len(texts) == 16
    assert events[2] == ("wake_up", None)
    assert events[3][0] == "generate" and events[3][1] == 2 and events[3][2] == 8 and events[3][3] == 1
    assert events[4] == ("sleep", 1)

    # Second call increments lora_int_id to 2
    engine.generate_rollouts(
        FakePeftModel(), ["p0"], [wavs[0]], num_generations=4, temperature=1.0, max_tokens=128, stop_token_ids=[151645]
    )
    assert events[6][3] == 2
    engine.cleanup()


def test_semi_balanced_all_non_neutral_plus_25pct_neutral_partition():
    """Verify semi_balanced keeps 100% of non-neutral (mod 5) + 25% neutral, MILP cue-stratified, exact 60/40."""
    rng = np.random.default_rng(123)
    # Realistic MELD clean train counts (~6,059 clips across 14 emotion x gender strata)
    stratum_counts = {
        ("neutral", "F"): 1390, ("neutral", "M"): 1425,
        ("joy", "F"): 540,      ("joy", "M"): 530,
        ("surprise", "F"): 380, ("surprise", "M"): 360,
        ("anger", "F"): 330,    ("anger", "M"): 353,
        ("sadness", "F"): 215,  ("sadness", "M"): 203,
        ("fear", "F"): 87,      ("fear", "M"): 82,
        ("disgust", "F"): 81,   ("disgust", "M"): 85,
    }
    rows = []
    idx = 0
    for (emo, gen), n in stratum_counts.items():
        for _ in range(n):
            row = {
                "utterance_id": f"utt_{idx:05d}",
                "emotion": emo,
                "effective_gender": gen,
                "audio_path": f"/fake/utt_{idx:05d}.wav",
                "text": f"line {idx}",
            }
            for cue in CUES:
                row[CUE_COLUMNS[cue]] = str(rng.choice(CUE_LABELS[cue], p=[0.25, 0.50, 0.25]))
            rows.append(row)
            idx += 1
    df = pd.DataFrame(rows)
    val_df = pd.DataFrame([{"audio_path": "/fake/val_00000.wav", "emotion": "neutral", "effective_gender": "F"}])

    quotas = compute_semi_balanced_quotas(df, neutral_target_fraction=0.25)
    # 1. Every non-neutral stratum keeps exact floor(N / 5) * 5 (100% mod 5)
    for (emo, gen), n in stratum_counts.items():
        if emo != "neutral":
            assert quotas[(emo, gen)] == (n // 5) * 5

    # 2. Neutral is sized to ~25% of the final pool
    total_pool = sum(quotas.values())
    neutral_pool = quotas[("neutral", "F")] + quotas[("neutral", "M")]
    assert 4200 <= total_pool <= 4450
    assert abs((neutral_pool / total_pool) - 0.25) < 0.005

    # 3. End-to-end MILP partitioning produces exact 60/40 SFT/GRPO with zero overlap
    sft_df, grpo_df, manifest = make_partitions(
        df,
        partition_mode="semi_balanced",
        neutral_target_fraction=0.25,
        seed=42,
    )
    assert manifest["partition_mode"] == "semi_balanced"
    assert len(sft_df) + len(grpo_df) == total_pool
    assert abs((len(sft_df) / total_pool) - 0.60) < 1e-12
    assert abs((len(grpo_df) / total_pool) - 0.40) < 1e-12
    assert abs(((sft_df["emotion"] == "neutral").mean()) - 0.25) < 0.005
    assert abs(((grpo_df["emotion"] == "neutral").mean()) - 0.25) < 0.005

    # 4. Both partition validators pass with require_uniform_quotas=False
    validate_partitions(sft_df, grpo_df, df, val_df, require_uniform_quotas=False)
    validate_grpo_partition(grpo_df, val_df, sft_df=sft_df, clean_train_df=df, require_uniform_quotas=False)

    # 5. Legacy uniform mode still works and passes require_uniform_quotas=True
    sft_u, grpo_u, manifest_u = make_partitions(df, partition_mode="uniform", seed=42)
    assert manifest_u["partition_mode"] == "uniform"
    validate_partitions(sft_u, grpo_u, df, val_df, require_uniform_quotas=True)


def test_imbalanced_r1_class_weighting_preserved_outside_group_std():
    """Verify R1 tempered alpha=0.5 survives group_standardize on imbalanced splits and matches V4 on uniform splits."""
    imbalanced_counts = {
        "neutral": 250,
        "joy": 240,
        "surprise": 170,
        "anger": 160,
        "sadness": 100,
        "fear": 40,
        "disgust": 40,
    }
    imbalanced_emotion_priors = {e: n / 1000.0 for e, n in imbalanced_counts.items()}
    cue_priors = {c: {lbl: p for lbl, p in zip(CUE_LABELS[c], (0.25, 0.50, 0.25))} for c in CUES}

    # Build a 1,000-row synthetic grpo_train_df with exact 25/50/25 cue marginals & imbalanced emotions
    rows = []
    idx = 0
    for emo, n in imbalanced_counts.items():
        for _ in range(n):
            row = {"emotion": emo, "audio_path": f"/fake/g_{idx}.wav"}
            q_idx = 0 if (idx % 4 == 0) else (2 if (idx % 4 == 3) else 1)
            for c in CUES:
                row[CUE_COLUMNS[c]] = CUE_LABELS[c][q_idx]
            rows.append(row)
            idx += 1
    grpo_df = pd.DataFrame(rows)

    # 1. Verify full GRPORewardCalibrator path: R1 uses r1_alpha=0.5, R2 uses alpha=1.0
    cal = GRPORewardCalibrator(
        grpo_df,
        config=RewardConfig(
            use_zero_mean=False,
            use_map=False,
            use_inverse_frequency=True,
            inverse_frequency_alpha=1.0,
            r1_inverse_frequency_alpha=0.5,
            use_distance_based_payoff=False,
        ),
    )
    weighted_mean = sum(imbalanced_emotion_priors[e] * cal.emotion_payoffs[e]["hit"] for e in EMOTIONS)
    assert abs(weighted_mean - 1.0) < 1e-12
    assert abs((cal.emotion_payoffs["disgust"]["hit"] / cal.emotion_payoffs["neutral"]["hit"]) - 2.5) < 1e-9
    # R2 cues remain governed by alpha=1.0 (not r1_alpha=0.5): 25/50/25 -> 4/3, 2/3, 4/3
    assert abs(cal.cue_payoffs["pitch_level"]["low"]["hit"] - (4.0 / 3.0)) < 1e-12
    assert abs(cal.cue_payoffs["pitch_level"]["moderate"]["hit"] - (2.0 / 3.0)) < 1e-12

    # 2. Build two 8-rollout groups with identical 4/8 accuracy & valid format: group 0 = disgust, group 1 = neutral
    def make_completion(emo: str) -> str:
        return (
            "<think>\n[Acoustic Inventory]\n"
            "- speaking_rate: moderate\n- pitch_level: moderate\n- vocal_energy: moderate\n- timbre: moderate\n\n"
            f"[Prosodic Reasoning]\nSounds like {emo}.\n</think>\n<answer>{emo}</answer>"
        )

    comps_disgust = [make_completion("disgust")] * 4 + [make_completion("neutral")] * 4
    comps_neutral = [make_completion("neutral")] * 4 + [make_completion("disgust")] * 4
    all_comps = comps_disgust + comps_neutral
    all_labels = ["disgust"] * 8 + ["neutral"] * 8
    cue_kwargs = {CUE_COLUMNS[c]: ["moderate"] * 16 for c in CUES}

    trace = []
    unified_gated_grpo_reward(
        prompts=["p"] * 16,
        completions=all_comps,
        label=all_labels,
        emotion_priors=imbalanced_emotion_priors,
        priors_dict=cue_priors,
        use_zero_mean=False,
        use_map=False,
        use_inverse_frequency=True,
        inverse_frequency_alpha=1.0,
        r1_inverse_frequency_alpha=0.5,
        use_distance_based_payoff=False,
        _reward_trace_sink=trace,
        **cue_kwargs,
    )
    assert abs((trace[0]["r1_class_weight"] / trace[8]["r1_class_weight"]) - 2.5) < 1e-9

    # 3. compute_span_split_advantages_np preserves exact 2.5x ratio in adv_emotion & adv_r1!
    advs = compute_span_split_advantages_np(trace, num_generations=8, loss_type="hybrid")
    ratio_emotion = abs(advs["adv_emotion"][0]) / abs(advs["adv_emotion"][8])
    ratio_r1 = abs(advs["adv_r1"][0]) / abs(advs["adv_r1"][8])
    assert abs(ratio_emotion - 2.5) < 1e-9
    assert abs(ratio_r1 - 2.5) < 1e-9

    # 4. On uniform 1/7 priors, r1_class_weight == 1.0 identically -> bit-for-bit identical to V4!
    uniform_priors = {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
    trace_u = []
    unified_gated_grpo_reward(
        prompts=["p"] * 16,
        completions=all_comps,
        label=all_labels,
        emotion_priors=uniform_priors,
        priors_dict=cue_priors,
        use_zero_mean=False,
        use_map=False,
        use_inverse_frequency=True,
        inverse_frequency_alpha=1.0,
        r1_inverse_frequency_alpha=0.5,
        use_distance_based_payoff=False,
        _reward_trace_sink=trace_u,
        **cue_kwargs,
    )
    assert all(abs(tr["r1_class_weight"] - 1.0) < 1e-12 for tr in trace_u)
    advs_u = compute_span_split_advantages_np(trace_u, num_generations=8, loss_type="hybrid")
    raw_v4 = group_standardize(
        np.array([1.0 * tr["r1"] + 0.1 * tr["r3"] for tr in trace_u]),
        group_size=8,
    )
    assert np.max(np.abs(advs_u["adv_emotion"] - raw_v4)) < 1e-12


def test_cli_parser_defaults():
    args = build_arg_parser().parse_args(["grpo", "--batch-size", "4", "--grad-accum", "8"])
    assert args.stage == "grpo"
    assert args.batch_size == 4 and args.grad_accum == 8 and args.no_vllm is False
    assert args.partition_mode == "semi_balanced"
    assert abs(args.neutral_fraction - 0.25) < 1e-12
    assert abs(args.r1_alpha - 0.5) < 1e-12
    assert abs(args.alpha - 1.0) < 1e-12
