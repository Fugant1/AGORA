"""CPU regression checks against the notebook's actual calibration/target code."""
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def cell_with(marker, notebook="ser_sft.ipynb"):
    cells = json.loads((ROOT / notebook).read_text())["cells"]
    return next("".join(c["source"]) for c in cells if marker in "".join(c["source"]))


def exec_nodes(source, names, env):
    nodes = []
    for node in ast.parse(source).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.AnnAssign):
            name = getattr(node.target, "id", None)
        elif isinstance(node, ast.Assign):
            name = getattr(node.targets[0], "id", None)
        if name in names:
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<notebook>", "exec"), env)


@pytest.fixture
def env():
    scope = dict(Any=Any, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple,
                 np=np, pd=pd, re=re, display=lambda *args: None)
    sampling = cell_with("def sample_sft_balanced_504")
    exec(sampling[:sampling.index("grpo_reference_train =")], scope)
    exec(cell_with("def extract_acoustic_inventory"), scope)
    inventory = cell_with("para o prompt do raciocínio SFT")
    exec_nodes(inventory, {"INVENTORY_DIMENSIONS", "evaluate_single_prompt_acoustics",
                          "build_sft_acoustic_inventory", "validate_sft_acoustic_targets"}, scope)
    scope["STRICTLY_POSITIVE_CUES"] = set(scope["STRICTLY_POSITIVE_ACOUSTIC_FEATURES"])
    return scope


def training_data(env):
    rows = []
    # Distinct baselines make accidental global or balanced-subset fitting visible.
    for speaker, offset in [("Joey", 0), ("Rachel", 100)]:
        for i in range(1, 29):
            row = {col: float(offset + i) for col in env["ALL_ACOUSTIC_FEATURES"].values()}
            row.update(speaker=speaker, audio_path=f"{speaker}_{i}.wav",
                       emotion=env["VALID_EMOTIONS"][(i - 1) % 7], duration=2.)
            rows.append(row)
    raw = pd.DataFrame(rows)
    held_out = raw.iloc[[0, 28]].copy()
    # Include duplicated held-out records: all instances must be excluded.
    raw = pd.concat([raw, held_out], ignore_index=True)
    return raw, held_out


def test_full_reference_matches_grpo_and_excludes_validation(env):
    raw, val = training_data(env)
    original = raw.copy(deep=True)
    reference = env["build_acoustic_reference"](raw, val)
    assert len(reference) == 54
    assert not set(reference.audio_path) & set(val.audio_path)
    pd.testing.assert_frame_equal(raw, original)

    sft_quantiles = env["compute_top_speaker_acoustic_quantiles"](reference)
    grpo_env = dict(env)
    exec_nodes(cell_with("def compute_top_speaker_acoustic_quantiles", "ser_grpo_v2.ipynb"),
               {"compute_top_speaker_acoustic_quantiles"}, grpo_env)
    assert sft_quantiles == grpo_env["compute_top_speaker_acoustic_quantiles"](reference)
    assert sft_quantiles["Joey"]["pitch_height"] == {"q25": 8.5, "q75": 21.5}
    assert sft_quantiles["Rachel"]["pitch_height"] == {"q25": 108.5, "q75": 121.5}

    # Later R2 initialization must not overwrite these with a biased GRPO subset.
    env.update(raw_train_set=raw, val_set=val, grpo_train_set=reference.iloc[:8])
    exec(cell_with("def reward_acoustic_inventory"), env)
    assert env["SPEAKER_ACOUSTIC_QUANTILES"] == sft_quantiles
    assert set(raw.audio_path) <= env["AUDIO_ACOUSTIC_LOOKUP"].keys()


def test_sampling_fallback_uses_full_reference(env):
    raw, val = training_data(env)
    grpo = raw.iloc[[1, 2]].copy()
    sampled = env["sample_sft_balanced_504"](
        raw, grpo, val, top_speakers=["Joey", "Rachel"], target_total=14)
    assert sampled.emotion.value_counts().to_dict() == dict.fromkeys(env["VALID_EMOTIONS"], 2)
    assert not set(sampled.audio_path) & (set(val.audio_path) | set(grpo.audio_path))
    quantiles = env["compute_top_speaker_acoustic_quantiles"](env["build_acoustic_reference"](raw, val))
    for _, row in sampled.iterrows():
        inventory = env["build_sft_acoustic_inventory"](
            row.audio_path, row.speaker, row.to_dict(), quantiles_dict=quantiles)
        for cue in env["CUE_FEATURE_MAP"]:
            assert inventory[cue] == env["INVENTORY_DIMENSIONS"][cue]["tier_to_prompt"][row[f"tier_{cue}"]]
        assert inventory["voice_quality"] == row.tier_voice_quality


def test_feature_filtering_and_voice_quality_match_grpo(env):
    raw, val = training_data(env)
    raw.loc[:4, env["CUE_FEATURE_MAP"]["pitch_height"]] = [0., -2., np.nan, 5., 10.]
    raw.loc[:4, env["VOICE_QUALITY_FEATURE_MAP"]["hnr"]] = [-20., -10., 0., np.nan, 10.]
    reference = env["build_acoustic_reference"](raw, val)
    grpo = dict(env)
    exec_nodes(cell_with("def compute_top_speaker_acoustic_quantiles", "ser_grpo_v2.ipynb"),
               {"compute_top_speaker_acoustic_quantiles", "determine_voice_quality_ground_truth"}, grpo)
    assert env["compute_top_speaker_acoustic_quantiles"](reference) == grpo["compute_top_speaker_acoustic_quantiles"](reference)
    q = {cue: {"q25": 1., "q75": 3.} for cue in env["VOICE_QUALITY_FEATURE_MAP"]}
    from itertools import product
    for values in product([1., 2., 4.], repeat=4):
        assert env["determine_voice_quality_ground_truth"](*values, q) == grpo["determine_voice_quality_ground_truth"](*values, q)


def test_cached_targets_rejected_after_boundary_change(env):
    raw, val = training_data(env)
    env["SPEAKER_ACOUSTIC_QUANTILES"] = env["compute_top_speaker_acoustic_quantiles"](
        env["build_acoustic_reference"](raw, val))
    row = raw.iloc[9].to_dict()
    inv = env["build_sft_acoustic_inventory"](row["audio_path"], row["speaker"], row)
    template = ast.literal_eval(ast.parse(cell_with("def generate_sft_train")).body[0].value)
    row["sft_gt"] = template.format(**inv, reasoning="Example explanation.", target_emotion=row["emotion"])
    df = pd.DataFrame([row])
    env["validate_sft_acoustic_targets"](df, "test")
    env["SPEAKER_ACOUSTIC_QUANTILES"]["Joey"]["pitch_height"] = {"q25": 20., "q75": 25.}
    with pytest.raises(ValueError, match="Regenerate their inventories AND reasoning"):
        env["validate_sft_acoustic_targets"](df, "test")


def test_regeneration_uses_stable_inventory_function_after_evaluation(env):
    raw, val = training_data(env)
    env["SPEAKER_ACOUSTIC_QUANTILES"] = env["compute_top_speaker_acoustic_quantiles"](
        env["build_acoustic_reference"](raw, val))
    # The final evaluation cell reuses the old function name with another signature.
    env["evaluate_single_prompt_acoustics"] = lambda completion, **kwargs: {"metrics": 1}
    exec(cell_with("def build_bridge_generation_prompt"), env)
    exec(cell_with("def generate_sft_train"), env)
    env.update(torch=SimpleNamespace(no_grad=nullcontext), tqdm=lambda x, **kw: x)

    class Tokenizer:
        padding_side = "right"
        pad_token_id = 1
        eos_token_id = 2

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, prompts, **kwargs):
            return {"input_ids": np.ones((len(prompts), 3), dtype=int)}

        def batch_decode(self, tokens, **kwargs):
            return ["Example explanation."] * len(tokens)

    model = SimpleNamespace(device="cpu", training=False, eval=lambda: None,
                            generation_config=SimpleNamespace(pad_token_id=1, eos_token_id=2),
                            generate=lambda input_ids, **kw: np.ones((len(input_ids), 5), dtype=int))
    result = env["generate_sft_train"](model, Tokenizer(), raw.iloc[[1, 26]], batch_size=2)
    env["validate_sft_acoustic_targets"](result, "generated")
    assert [x["pitch_height"] for x in result.acoustic_inventory] == ["low", "elevated"]


def test_reference_requires_known_validation_paths(env):
    raw, val = training_data(env)
    with pytest.raises(ValueError, match="audio_path"):
        env["build_acoustic_reference"](raw, val.drop(columns="audio_path"))


def test_training_prefers_fresh_targets_and_calibrates_validation(env, monkeypatch):
    raw, val = training_data(env)
    env["SPEAKER_ACOUSTIC_QUANTILES"] = env["compute_top_speaker_acoustic_quantiles"](
        env["build_acoustic_reference"](raw, val))
    template = ast.literal_eval(ast.parse(cell_with("def generate_sft_train")).body[0].value)
    row = raw.iloc[9].to_dict()
    inv = env["build_sft_acoustic_inventory"](row["audio_path"], row["speaker"], row)
    row["sft_gt"] = template.format(**inv, reasoning="Example explanation.", target_emotion=row["emotion"])
    env.update(sft_train_reasoning_populated_df=pd.DataFrame([row]), val_set=val,
               os=SimpleNamespace(path=SimpleNamespace(exists=lambda p: p.endswith("sft_train_set.csv"))))

    def unexpected_csv(*args, **kwargs):
        raise AssertionError("A stale CSV must not replace newly generated targets")

    monkeypatch.setattr(pd, "read_csv", unexpected_csv)
    training = cell_with("train_csv_path =")
    preparation = training[training.index("train_csv_path ="):training.index("# 2. Configuração do Modelo")]
    exec(preparation, env)
    assert env["sft_train_df"].sft_gt.tolist() == [row["sft_gt"]]
    for target in env["sft_eval_df"].sft_gt:
        inventory = env["extract_acoustic_inventory"](target)
        assert inventory["pitch_height"] == "low"
        assert inventory["voice_quality"] == "pressed/tense"
