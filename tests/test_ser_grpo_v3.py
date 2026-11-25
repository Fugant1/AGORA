"""CPU regression suite for ser_grpo_v3.ipynb.

Validates:
- Exact prompt, persona, cue schema, and strict parser parity with ser_sft.ipynb (V4).
- Zero-leakage GRPO partition verification (disjoint from SFT and validation, balanced strata).
- R1 (zero_mean, map, inverse_frequency) and 4-cue ordinal R2 (distance_based including timbre,
  zero_mean, map, inverse_frequency) mathematical proofs and blind baselines.
- R3 4-slot format reward and strict rejection of synonyms/malformed tags.
- Hybrid cue-split credit assignment (w2/4 scaling, L/n compensation for multi-token timbre labels,
  fail-closed token alignment, first-EOS boundary masking, dynamic/fixed loss equivalence).
- Model/trainer config safety (Thinker-only class, no Talker patches/kwargs, 1 epoch, G=8, beta=0.0,
  mid-train metrics logged, post-training-only validation, replay never calls generate()).
- Post-training SFT-compatible validation scoring (accuracy, macro/weighted F1, UAR, gender gap)
  and combined GRPO model health report end-to-end.
"""
import ast
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / "ser_grpo_v3.ipynb"
SFT_PATH = ROOT / "ser_sft.ipynb"


def load_nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code_cells(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def cell_containing(nb: dict, marker: str) -> str:
    for src in code_cells(nb):
        if marker in src:
            return src
    raise KeyError(f"Marker not found in notebook: {marker!r}")


def load_v3_cpu_env() -> dict:
    """Execute pure-CPU cells of ser_grpo_v3.ipynb (config, data helpers, prompt/parser, R1/R2/R3, report)."""
    nb = load_nb(V3_PATH)
    env: dict = {"__builtins__": __builtins__, "display": lambda *a, **k: None}
    markers_in_order = [
        "PROMPT_VERSION =",
        "def stage_clean_data(",
        "def student_instruction(",
        "USE_DISTANCE_BASED_PAYOFF =",
        "def resolve_reward_mode(",
        "def distance_based_payoff_matrix(",
        "def compute_format_reward_from_parsed(",
        "def unified_gated_grpo_reward(",
        "def label_metrics(",
    ]
    for marker in markers_in_order:
        src = cell_containing(nb, marker)
        # Skip live validation call at end of composite cell so tests control execution cleanly
        lines = [ln for ln in src.splitlines() if not ln.startswith("validation_math_report =")]
        exec(compile("\n".join(lines), f"<v3:{marker}>", "exec"), env)
    return env


class FakeArray(np.ndarray):
    def eq(self, value):
        return (self == value).view(FakeArray)

    def ne(self, value):
        return (self != value).view(FakeArray)

    def long(self):
        return self.astype(np.int64).view(FakeArray)

    def float(self):
        return self.astype(np.float32).view(FakeArray)

    def cumsum(self, dim):
        return np.asarray(self).cumsum(axis=dim).view(FakeArray)

    def mean(self, dim=None, keepdim=False, **kwargs):
        return np.asarray(self).mean(axis=dim, keepdims=keepdim, **kwargs).view(FakeArray)

    def sum(self, dim=None, keepdim=False, **kwargs):
        if dim is not None:
            return np.asarray(self).sum(axis=dim, keepdims=keepdim).view(FakeArray)
        return np.asarray(self).sum(**kwargs)

    def clamp(self, min=None, max=None):
        return np.clip(np.asarray(self), min, max).view(FakeArray)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def unsqueeze(self, dim):
        return np.expand_dims(self, axis=dim).view(FakeArray)

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], type) and issubclass(shape[0], np.ndarray):
            return super().view(shape[0])
        return np.asarray(self).reshape(*shape).view(FakeArray)

    @property
    def device(self):
        return None


def make_fake_torch():
    return SimpleNamespace(
        bool=bool,
        float32=np.float32,
        zeros_like=lambda a, dtype=None: np.zeros_like(a, dtype=dtype or a.dtype).view(FakeArray),
        zeros=lambda shape, dtype=bool, device=None: np.zeros(shape, dtype=dtype).view(FakeArray),
        tensor=lambda data, dtype=None, device=None: np.array(data, dtype=dtype).view(FakeArray),
        nan_to_num=lambda a, nan=0.0: np.nan_to_num(a, nan=nan).view(FakeArray),
        cat=lambda xs, dim=0: np.concatenate(xs, axis=dim).view(FakeArray),
    )


def extract_trainer_helpers(env_extra: dict) -> dict:
    nb = load_nb(V3_PATH)
    src = cell_containing(nb, "class AudioGRPOTrainer(")
    tree = ast.parse(src)
    env = dict(env_extra)
    target_defs = {
        "completion_boundary_mask",
        "completion_span_masks",
        "group_standardize",
        "cue_group_advantages",
        "acoustic_value_token_indices",
        "build_cue_token_advantages",
        "compute_span_split_advantages",
        "build_span_split_token_advantages",
        "fixed_completion_loss",
        "dynamic_completion_loss",
        "reduce_completion_loss",
        "validate_credit_assignment",
        "load_replay_groups",
    }
    target_assigns = {"SPAN2_MARKER_SEQUENCES", "R2_CUE_NAMES", "R2_CUE_LABELS", "GRPO_ADVANTAGE_EPS"}
    nodes = [
        n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in target_defs)
        or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in target_assigns for t in n.targets))
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<trainer_helpers>", "exec"), env)
    return env


def test_all_code_cells_parse_without_syntax_warnings():
    nb = load_nb(V3_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            cleaned = "\n".join("" if ln.lstrip().startswith("%") else ln for ln in src.splitlines())
            ast.parse(cleaned, filename=f"ser_grpo_v3.ipynb#cell_{idx}")


def test_exact_prompt_schema_and_path_parity_with_ser_sft():
    nb_sft = load_nb(SFT_PATH)
    nb_v3 = load_nb(V3_PATH)

    def extract_sft_prompt_symbols(nb):
        env = {"__builtins__": __builtins__}
        cfg_src = cell_containing(nb, "CUES =")
        tree = ast.parse("\n".join("" if ln.lstrip().startswith("%") else ln for ln in cfg_src.splitlines()))
        keep_cfg = [
            n for n in tree.body
            if isinstance(n, ast.Assign) and any(
                (isinstance(t, ast.Name) and t.id in ("PROMPT_VERSION", "CUES", "CUE_COLUMNS", "GLOBAL_CUES", "EMOTIONS", "CUE_LABELS", "EVAL_BATCH_SIZE", "MODEL_ID", "SEED"))
                or (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "CUE_LABELS")
                for t in n.targets
            )
        ]
        exec(compile(ast.Module(body=keep_cfg, type_ignores=[]), "<cfg>", "exec"), env)

        p_src = cell_containing(nb, "def student_instruction(")
        p_tree = ast.parse(p_src)
        keep_p = [
            n for n in p_tree.body
            if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PERSONA" for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name in ("gender_context", "inventory_text", "student_instruction"))
        ]
        exec(compile(ast.Module(body=keep_p, type_ignores=[]), "<prompt>", "exec"), env)
        return env

    sft_env = extract_sft_prompt_symbols(nb_sft)
    v3_env = extract_sft_prompt_symbols(nb_v3)

    for key in ("PERSONA", "PROMPT_VERSION", "CUES", "CUE_COLUMNS", "GLOBAL_CUES", "EMOTIONS", "CUE_LABELS", "EVAL_BATCH_SIZE", "MODEL_ID", "SEED"):
        assert v3_env[key] == sft_env[key], f"Mismatch in {key}: v3={v3_env[key]!r} vs sft={sft_env[key]!r}"

    for gender in ("F", "M", "__GLOBAL__"):
        assert v3_env["student_instruction"]({"effective_gender": gender}) == sft_env["student_instruction"]({"effective_gender": gender})

    # Verify V4 split/adapter paths match ser_sft.ipynb V4
    cfg_v3 = cell_containing(nb_v3, "SPLIT_DIR =")
    assert "sft_grpo_gender_wxrate_60_40_v4" in cfg_v3
    assert "full_sft_gender_wxrate_unweighted_seed42_v4" in cfg_v3
    assert "final_sft_adapter" in cfg_v3


def test_strict_parser_and_4cue_format_reward():
    env = load_v3_cpu_env()
    parse = env["parse_completion"]
    parse_strict = env["parse_completion_strict"]
    r3_fn = env["compute_format_reward_from_parsed"]

    good = env["build_sample_completion"](
        speaking_rate="high",
        pitch_level="low",
        vocal_energy="moderate",
        timbre="high (tense/bright)",
        emotion="anger",
    )
    p = parse(good + "<|im_end|><|im_end|>")
    assert p["is_valid_emotion"] is True
    assert p["answer_norm"] == "anger"
    assert p["valid_tag_order"] is True
    assert p["valid_slots_count"] == 4
    assert p["inventory"] == {
        "speaking_rate": "high",
        "pitch_level": "low",
        "vocal_energy": "moderate",
        "timbre": "high (tense/bright)",
    }
    assert r3_fn(p) == pytest.approx(1.0)
    emo, inv = parse_strict(good)
    assert emo == "anger"
    assert inv == p["inventory"]

    # Strict parser rejects synonyms (no LABEL_SYNONYMS fallback)
    synonym = env["build_sample_completion"](emotion="happy")
    ps = parse(synonym)
    assert ps["is_valid_emotion"] is False
    assert ps["answer_norm"] == "invalid"
    assert r3_fn(ps) == pytest.approx(0.80)

    # Strict parser rejects short timbre alias ("tense/bright" instead of "high (tense/bright)")
    bad_timbre = env["build_sample_completion"](timbre="tense/bright", emotion="joy")
    pt = parse(bad_timbre)
    assert pt["inventory"]["timbre"] is None
    assert pt["valid_slots_count"] == 3
    assert r3_fn(pt) == pytest.approx(round(0.20 + 0.35 * (3 / 4.0) + 0.25 + 0.20, 3))

    # Placeholder reasoning gets zero reasoning credit
    placeholder_text = (
        "<think>\n[Acoustic Inventory]\n"
        "- speaking_rate: low\n- pitch_level: low\n- vocal_energy: low\n- timbre: low (muffled/soft)\n"
        "[Prosodic Reasoning]\nYour concise reasoning.\n</think>\n<answer>sadness</answer>"
    )
    pp = parse(placeholder_text)
    assert pp["reasoning"] == ""
    assert r3_fn(pp) == pytest.approx(0.75)


@pytest.mark.parametrize(
    "zm,mp,ifreq,dist",
    [
        (False, True, False, True),   # Default: MAP R1 + Distance 4-cue R2
        (True, False, False, True),   # Zero-Mean R1 + Distance 4-cue R2
        (False, False, True, True),   # Inverse-Frequency R1 + Distance 4-cue R2
        (False, True, False, False),  # MAP R1 + MAP 4-cue R2
        (True, False, False, False),  # Zero-Mean R1 + Zero-Mean 4-cue R2
        (False, False, True, False),  # Inverse-Frequency R1 + Inverse-Frequency 4-cue R2
    ],
)
def test_reward_math_all_modes_4_ordinal_cues_including_timbre(zm, mp, ifreq, dist):
    env = load_v3_cpu_env()
    assert env["R2_CUE_COUNT"] == 4
    report = env["validate_grpo_reward_mathematics"](
        use_zero_mean=zm,
        use_map=mp,
        use_inverse_frequency=ifreq,
        use_distance_based_payoff=dist,
    )
    assert set(report["r2_matrix_proofs"].keys()) == {"speaking_rate", "pitch_level", "vocal_energy", "timbre"}
    if dist:
        # Verify timbre obeys strict ordinal ordering: low > moderate > high when truth is low, and vice versa
        assert report["r2_matrix_proofs"]["timbre"]["ordinal_satisfied"] is True
        assert report["r2_matrix_proofs"]["timbre"]["zero_mean_satisfied"] is True
        for policy_mean in report["r2_blind_means"].values():
            assert policy_mean == pytest.approx(0.5, abs=1e-12)


def test_timbre_ordinal_distance_payoff_and_missing_gt_handling():
    env = load_v3_cpu_env()
    priors = {
        "speaking_rate": {"low": 0.25, "moderate": 0.50, "high": 0.25},
        "pitch_level": {"low": 0.25, "moderate": 0.50, "high": 0.25},
        "vocal_energy": {"low": 0.25, "moderate": 0.50, "high": 0.25},
        "timbre": {"low (muffled/soft)": 0.25, "moderate (modal)": 0.50, "high (tense/bright)": 0.25},
    }
    # Default inverse_frequency R2: valid wrong and invalid predictions BOTH score 0.0 (closes invalid-competitor loophole)
    raw_wrong, _, hit_wrong = env["score_cue_payoff"]("high", "moderate", "speaking_rate", priors)
    raw_invalid, _, hit_inv = env["score_cue_payoff"]("high", None, "speaking_rate", priors)
    raw_hit_high, _, _ = env["score_cue_payoff"]("high", "high", "speaking_rate", priors)
    raw_hit_mod, _, _ = env["score_cue_payoff"]("moderate", "moderate", "speaking_rate", priors)
    assert raw_wrong == 0.0 and raw_invalid == 0.0 and hit_wrong == 0.0 and hit_inv == 0.0
    assert raw_hit_high == pytest.approx(2.0 * raw_hit_mod)  # 2x larger UAR gradient on minority 25% vs 50%

    # Distance-based ablation check (explicit flag)
    m = env["distance_based_payoff_matrix"](priors["timbre"], env["CUE_LABELS"]["timbre"])
    low_l, mod_l, high_l = env["CUE_LABELS"]["timbre"]
    assert m[low_l][low_l] > m[low_l][mod_l] > m[low_l][high_l]
    assert m[high_l][high_l] > m[high_l][mod_l] > m[high_l][low_l]
    assert m[mod_l][mod_l] == pytest.approx(0.55)
    assert m[mod_l][low_l] == pytest.approx(-0.18)

    # Missing GT on 1 cue contributes raw 0.0 and valid_gt=False without crashing
    comp = env["build_sample_completion"](
        speaking_rate="high", pitch_level="low", vocal_energy="moderate", timbre="high (tense/bright)", emotion="joy"
    )
    trace = []
    r2, gate = env["reward_acoustic_inventory"](
        ["p"], [comp], return_gate_scores=True, _cue_trace_sink=trace,
        priors_dict=priors,
        whisperx_syllable_rate_cue=["high"],
        pitch_level_cue=[np.nan],
        vocal_energy_cue=["moderate"],
        timbre_cue=["high (tense/bright)"],
    )
    assert len(r2) == 1 and 0.5 < r2[0] <= 1.0
    assert gate[0] == pytest.approx(1.0)  # 3/3 scorable cues matched
    assert trace[0]["pitch_level"]["valid_gt"] is False
    assert trace[0]["pitch_level"]["raw"] == 0.0
    assert trace[0]["timbre"]["correct"] is True


def test_hybrid_cue_split_token_alignment_including_multitoken_timbre():
    env = load_v3_cpu_env()
    fake_torch = make_fake_torch()
    t_env = extract_trainer_helpers({
        "torch": fake_torch,
        "np": np,
        "re": __import__("re"),
        "CUES": env["CUES"],
        "REWARD_WEIGHTS": env["REWARD_WEIGHTS"],
        "VALID_EMOTIONS": env["VALID_EMOTIONS"],
        "Path": Path,
        "json": json,
        "Counter": Counter,
    })

    # Synthetic tokenizer where timbre's value 'high (tense/bright)' spans 6 distinct tokens
    pieces = [
        "<think>", "\n", "[Acoustic Inventory]", "\n",
        "-", " speaking_rate", ":", " high", "\n",
        "-", " pitch_level", ":", " low", "\n",
        "-", " vocal_energy", ":", " moderate", "\n",
        "-", " timbre", ":", " high", " (", "tense", "/", "bright", ")", "\n",
        "[Prosodic Reasoning]", "\n", "1.", " Clear", " cue", " reasoning.", "\n",
        "</think>", "\n", "<answer>", "anger", "</answer>", "<|im_end|>",
    ]
    id_to_piece = {i + 10: p for i, p in enumerate(pieces)}

    class FakeTokenizer:
        def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            return "".join(id_to_piece[int(i)] for i in ids)

        def encode(self, text, add_special_tokens=False):
            for tid, p in id_to_piece.items():
                if p == text:
                    return [tid]
            return []

    tok = FakeTokenizer()
    ids_0 = [i + 10 for i in range(len(pieces))]
    # Second completion swaps timbre to single-token-different 'low (muffled/soft)' with extra reasoning tokens
    pieces_1 = list(pieces)
    pieces_1[22:28] = [" low", " (", "muffled", "/", "soft", ")"]
    pieces_1.insert(35, " Extra")
    pieces_1.insert(36, " words.")
    next_id = 100
    ids_1 = []
    for p in pieces_1:
        found = [k for k, v in id_to_piece.items() if v == p]
        if found:
            ids_1.append(found[0])
        else:
            id_to_piece[next_id] = p
            ids_1.append(next_id)
            next_id += 1

    # Verify exact multi-token value mapping on all 4 snake_case cues
    pos_0, st_0 = t_env["acoustic_value_token_indices"](ids_0, tok)
    assert all(v == "ok" for v in st_0.values()), st_0
    assert pos_0["speaking_rate"] == [7]
    assert pos_0["pitch_level"] == [12]
    assert pos_0["vocal_energy"] == [17]
    assert pos_0["timbre"] == [22, 23, 24, 25, 26, 27]  # all 6 tokens of 'high (tense/bright)'!

    # Pad to equal length and verify cue_group_advantages (w2 / 4) and L/n compensation
    max_len = max(len(ids_0), len(ids_1))
    eos_id = ids_0[-1]
    pad_0 = ids_0 + [eos_id] * (max_len - len(ids_0))
    pad_1 = ids_1 + [eos_id] * (max_len - len(ids_1))
    comp_ids = fake_torch.tensor([pad_0, pad_1], dtype=np.int64)
    comp_mask = t_env["completion_boundary_mask"](comp_ids, [eos_id], eos_id)
    assert int(comp_mask[0].sum()) == len(ids_0)
    assert int(comp_mask[1].sum()) == len(ids_1)

    trace = [
        {
            "r1": 0.857, "r2": 0.70, "r3": 1.0,
            "r2_cues": {
                "speaking_rate": {"raw": 1.0, "truth": "high", "pred": "high", "valid_gt": True, "correct": True},
                "pitch_level": {"raw": 0.5, "truth": "low", "pred": "low", "valid_gt": True, "correct": True},
                "vocal_energy": {"raw": 0.55, "truth": "moderate", "pred": "moderate", "valid_gt": True, "correct": True},
                "timbre": {"raw": 1.0, "truth": "high (tense/bright)", "pred": "high (tense/bright)", "valid_gt": True, "correct": True},
            },
        },
        {
            "r1": -0.143, "r2": 0.30, "r3": 1.0,
            "r2_cues": {
                "speaking_rate": {"raw": 1.0, "truth": "high", "pred": "high", "valid_gt": True, "correct": True},
                "pitch_level": {"raw": 0.5, "truth": "low", "pred": "low", "valid_gt": True, "correct": True},
                "vocal_energy": {"raw": 0.55, "truth": "moderate", "pred": "moderate", "valid_gt": True, "correct": True},
                "timbre": {"raw": -0.6, "truth": "high (tense/bright)", "pred": "low (muffled/soft)", "valid_gt": True, "correct": False},
            },
        },
    ]
    cue_adv = t_env["cue_group_advantages"](trace, group_size=2, r2_weight=1.0, loss_type="hybrid")
    # Tied cues get exact 0.0; timbre gets (1.0 / 4) * (1.0 - 0.2) = +0.20 / -0.20
    assert cue_adv["speaking_rate"].tolist() == pytest.approx([0.0, 0.0])
    assert cue_adv["timbre"].tolist() == pytest.approx([0.20, -0.20])

    span_adv = t_env["compute_span_split_advantages"](trace, 2, reward_weights={"r1": 1.0, "r2": 1.0, "r3": 0.2}, loss_type="hybrid")
    tok_adv, ac_m, em_m, has_b, counts, statuses = t_env["build_cue_token_advantages"](
        comp_ids, comp_mask, cue_adv, span_adv["adv_r3"], span_adv["adv_emotion"], tok
    )
    assert bool(has_b.all())
    assert counts["timbre"] == [6, 6]
    # Sum of timbre token advantages divided by completion length L equals exact cue advantage (+0.20 / -0.20)!
    assert float(tok_adv[0, pos_0["timbre"]].sum()) / len(ids_0) == pytest.approx(0.20, abs=1e-6)
    pos_1, _ = t_env["acoustic_value_token_indices"](ids_1, tok)
    assert float(tok_adv[1, pos_1["timbre"]].sum()) / len(ids_1) == pytest.approx(-0.20, abs=1e-6)
    # Emotion span receives positive standardized advantage on rollout 0, negative on rollout 1
    assert float(tok_adv[0, 38]) > 0.0 and float(tok_adv[1, 40]) < 0.0


def test_trainer_and_model_config_safety():
    nb = load_nb(V3_PATH)
    full_code = "\n".join(code_cells(nb))

    # Thinker-only class; zero Talker monkey-patches or wrapper kwargs
    assert "Qwen2_5OmniThinkerForConditionalGeneration" in full_code
    assert "Qwen2_5OmniForConditionalGeneration" not in full_code
    assert "DummyTalker" not in full_code
    assert "disable_talker" not in full_code
    assert "return_audio" not in full_code
    assert "thinker_max_new_tokens" not in full_code
    assert "TOP_6_SPEAKERS" not in full_code

    # 1 epoch, G=8, beta=0.0, stabilized batch/LR, parameter-free inverse-frequency R2, hybrid split flags
    cfg_cell = cell_containing(nb, "NUM_TRAIN_EPOCHS =")
    assert "NUM_TRAIN_EPOCHS = 1" in cfg_cell
    assert "NUM_GENERATIONS = 8" in cfg_cell
    assert "PER_DEVICE_TRAIN_BATCH_SIZE = 2" in cfg_cell
    assert "STEPS_PER_GENERATION = 8" in cfg_cell
    assert "GRADIENT_ACCUMULATION_STEPS = 16" in cfg_cell
    assert "MAX_COMPLETION_LENGTH = 256" in cfg_cell
    assert "EVAL_MAX_NEW_TOKENS = 512" in cfg_cell
    assert "LEARNING_RATE = 1e-5" in cfg_cell
    assert "WARMUP_STEPS = 5" in cfg_cell
    assert "LR_SCHEDULER_TYPE = 'cosine'" in cfg_cell
    assert "BETA = 0.0" in cfg_cell
    assert "RESUME_CHECKPOINT_DIR: Optional[str] = None" in cfg_cell

    flags_cell = cell_containing(nb, "USE_DISTANCE_BASED_PAYOFF =")
    assert "USE_ZERO_MEAN = False" in flags_cell
    assert "USE_MAP = False" in flags_cell
    assert "USE_INVERSE_FREQUENCY = True" in flags_cell
    assert "USE_DISTANCE_BASED_PAYOFF = False" in flags_cell

    init_cell = cell_containing(nb, "training_args = GRPOConfig(")
    assert "num_train_epochs=NUM_TRAIN_EPOCHS" in init_cell
    assert "steps_per_generation=STEPS_PER_GENERATION" in init_cell
    assert "generation_batch_size=" not in init_cell  # mutually exclusive with steps_per_generation in TRL
    assert "lr_scheduler_type=LR_SCHEDULER_TYPE" in init_cell
    assert "warmup_steps=WARMUP_STEPS" in init_cell
    assert "beta=BETA" in init_cell
    assert "max_steps=" not in init_cell
    assert "USE_SPLIT_ADVANTAGE = True" in init_cell
    assert "USE_SPLIT_R2 = True" in init_cell
    assert "LOSS_TYPE = 'hybrid'" in init_cell

    # Mid-train metrics logged to trainer.state.log_history + fast beta==0 reference skip merged
    trainer_cell = cell_containing(nb, "class AudioGRPOTrainer(")
    assert "if self.args.beta != 0.0:" in trainer_cell
    assert "fast_compute_loss" not in "\n".join(code_cells(nb))
    for metric_key in (
        "reward/R1_accuracy",
        "reward/R2_acoustic",
        "reward/R3_format",
        "reward/total_weighted",
        "metrics/acc_step",
        "metrics/f1_macro_running",
        "metrics/prediction_entropy",
        "dist/invalid_format",
        "diagnostic/kl_per_token",
    ):
        assert metric_key in trainer_cell, f"Missing mid-train metric {metric_key}"

    # Replay diagnostics never call generate() and always use try/finally
    tree = ast.parse(trainer_cell)
    for fn_name in ("check_one_replay_update", "compare_replay_checkpoints"):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
        assert not any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "generate"
            for n in ast.walk(node)
        )
        assert any(isinstance(n, ast.Try) and n.finalbody for n in ast.walk(node))

    # Validation happens only after training (no evaluate_before toggle)
    assert "EVALUATE_BEFORE" not in full_code


def test_post_training_sft_metrics_and_health_report_end_to_end(tmp_path):
    env = load_v3_cpu_env()

    # Build tiny synthetic balanced GRPO train (28 rows: 2 per emotion x gender) and val (14 rows)
    train_rows = []
    for i, emo in enumerate(env["EMOTIONS"]):
        for g_idx, gender in enumerate(("F", "M")):
            for k in range(2):
                idx = i * 4 + g_idx * 2 + k
                tier = ("low", "moderate", "high")[idx % 3]
                timbre = env["CUE_LABELS"]["timbre"][idx % 3]
                train_rows.append({
                    "audio_path": f"/tmp/train_{idx}.wav",
                    "emotion": emo, "gender": gender, "effective_gender": gender,
                    "speaker": f"spk_{idx % 4}", "duration": 2.5, "dialogue": idx, "utterance": 0, "text": "hi",
                    "whisperx_syllable_rate_cue": tier,
                    "pitch_level_cue": ("moderate", "high", "low")[idx % 3],
                    "vocal_energy_cue": ("high", "low", "moderate")[idx % 3],
                    "timbre_cue": timbre,
                })
    grpo_df = pd.DataFrame(train_rows)

    em_priors, _, em_mat = env["calibrate_emotion_payoffs"](grpo_df, return_matrix=True)
    ac_cal = env["calibrate_acoustic_payoffs"](grpo_df)
    env["EMOTION_PRIORS"] = em_priors
    env["EMOTION_PAYOFF_MATRIX"] = em_mat
    env["ACOUSTIC_CUE_PRIORS"] = ac_cal["priors"]
    env["ACOUSTIC_PAYOFF_MATRICES"] = ac_cal["matrices"]

    # Build synthetic validation predictions (including one missing GT cue and one unknown gender)
    val_records = []
    for idx, emo in enumerate(env["EMOTIONS"]):
        for g in ("F", "M"):
            row_id = idx * 2 + (0 if g == "F" else 1)
            pred_emo = emo if row_id % 3 != 0 else "neutral"
            sr = ("low", "moderate", "high")[row_id % 3]
            pl = ("low", "moderate", "high")[(row_id + 1) % 3]
            ve = ("low", "moderate", "high")[(row_id + 2) % 3]
            tb = env["CUE_LABELS"]["timbre"][row_id % 3]
            comp = env["build_sample_completion"](sr, pl, ve, tb, pred_emo)
            trace = []
            gt_pl = np.nan if row_id == 0 else pl
            totals = env["unified_gated_grpo_reward"](
                ["p"], [comp], label=[emo], audio_path=[f"/tmp/val_{row_id}.wav"],
                whisperx_syllable_rate_cue=[sr], pitch_level_cue=[gt_pl],
                vocal_energy_cue=[ve], timbre_cue=[tb],
                _reward_trace_sink=trace,
            )
            tr = trace[0]
            val_records.append({
                "audio_path": f"/tmp/val_{row_id}.wav",
                "emotion": emo,
                "gender": "unknown" if row_id == 13 else g,
                "effective_gender": "__GLOBAL__" if row_id == 13 else g,
                "whisperx_syllable_rate_cue": sr,
                "pitch_level_cue": gt_pl,
                "vocal_energy_cue": ve,
                "timbre_cue": tb,
                "completion": comp,
                "predicted_emotion": pred_emo,
                "predicted_speaking_rate": sr,
                "predicted_pitch_level": pl,
                "predicted_vocal_energy": ve,
                "predicted_timbre": tb,
                "r_unified_reward": totals[0],
                "r1_reward": tr["r1"],
                "r2_acoustic_reward": tr["r2"],
                "r3_format_reward": tr["r3"],
                "acoustic_exact_match_ratio": tr["r2_gate"],
                "valid_slots_count": 4,
            })
    preds_df = pd.DataFrame(val_records)

    # Verify zero-leakage partition guard & credit-assignment validation
    assert env["validate_grpo_partition"](grpo_df, preds_df).min() == 2
    with pytest.raises(ValueError, match="overlap"):
        env["validate_grpo_partition"](grpo_df, grpo_df.iloc[:1])
    with pytest.raises(ValueError, match="balanced"):
        env["validate_grpo_partition"](grpo_df.iloc[:-1], preds_df)

    # 1. Verify SFT-compatible score_validation structure + weighted_f1 + grpo_rewards
    metrics = env["score_validation"](preds_df)
    assert set(metrics.keys()) == {"emotion", "emotion_by_gender", "cues", "grpo_rewards"}
    assert "weighted_f1" in metrics["emotion"]
    assert set(metrics["emotion_by_gender"]["groups"].keys()) == {"F", "M", "unknown"}
    assert set(metrics["cues"].keys()) == {"speaking_rate", "pitch_level", "vocal_energy", "timbre"}
    assert metrics["cues"]["pitch_level"]["missing_targets"] == 1
    assert metrics["cues"]["pitch_level"]["n"] == 13
    assert metrics["grpo_rewards"]["valid_format_rate"] == pytest.approx(1.0)

    # 2. Verify combined GRPO health report writes cleanly with rollout JSONL
    groups_path = tmp_path / "grpo_groups.jsonl"
    sample_group = {
        "step_before_update": 0, "group_index": 0, "target": "anger",
        "audio_path": "/tmp/train_0.wav", "predictions": ["anger", "neutral"],
        "correct": [True, False], "r1": [6 / 7, -1 / 7], "r2": [0.8, 0.5], "r3": [1.0, 1.0],
        "total_rewards": [6 / 7 + 0.8 + 0.2, -1 / 7 + 0.5 + 0.2],
        "reward_weights": {"r1": 1.0, "r2": 1.0, "r3": 0.2},
        "use_split_r2": True,
        "r2_cues": [
            {c: {"raw": 0.5, "truth": env["CUE_LABELS"][c][1], "pred": env["CUE_LABELS"][c][1], "valid_gt": True, "correct": True} for c in env["CUES"]},
            {c: {"raw": -0.2, "truth": env["CUE_LABELS"][c][1], "pred": env["CUE_LABELS"][c][0], "valid_gt": True, "correct": False} for c in env["CUES"]},
        ],
        "advantages_cues": {c: [0.0875, -0.0875] for c in env["CUES"]},
        "cue_value_tokens": {"speaking_rate": [1, 1], "pitch_level": [1, 1], "vocal_energy": [1, 1], "timbre": [5, 6]},
        "cue_mask_status": [{c: "ok" for c in env["CUES"]}, {c: "ok" for c in env["CUES"]}],
        "completions": [val_records[0]["completion"], val_records[1]["completion"]],
    }
    groups_path.write_text(json.dumps(sample_group) + "\n", encoding="utf-8")

    out_dir = tmp_path / "run_out"
    adapter_dir = tmp_path / "run_adapter"
    report_path = env["write_grpo_health_report"](
        preds_df, output_dir=out_dir, adapter_dir=adapter_dir, train_df=grpo_df, groups_path=groups_path,
    )
    assert report_path.is_file()
    report_md = report_path.read_text(encoding="utf-8")
    assert "## Emotion prediction by annotated speaker gender" in report_md
    assert "## Per-cue R2 training signal" in report_md
    assert "## Constant-inventory baselines (4 cues)" in report_md
    for fname in ("model_health_report.md", "model_health_metrics.json", "validation_predictions.jsonl", "evaluation_calibration.json"):
        assert (out_dir / fname).is_file()
        assert (adapter_dir / fname).is_file()
