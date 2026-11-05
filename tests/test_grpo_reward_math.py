"""Empirical invariants against actual reward code in ser_grpo_v2.ipynb."""
from itertools import product
import numpy as np
import pandas as pd
import pytest
from tests.notebook_helpers import load_reward_environment


def load_notebook_reward_env():
    return load_reward_environment()


@pytest.fixture
def env():
    return load_reward_environment()


def calibration_data(env):
    raw = pd.DataFrame([
        {**{col: float(offset + i) for col in env["ALL_ACOUSTIC_FEATURES"].values()},
         "speaker": spk, "audio_path": f"raw_{spk}_{i}.wav"}
        for spk, offset in (("A", 0), ("B", 100)) for i in range(1, 101)
    ])
    tier_sequences = [[10.] * 4 + [50.] * 9 + [90.] * 7,
                      [10.] * 8 + [50.] * 5 + [90.] * 7,
                      [10.] * 3 + [50.] * 11 + [90.] * 6,
                      [10.] * 7 + [50.] * 9 + [90.] * 4]
    voice_labels = ["modal/normal"] * 8 + ["pressed/tense"] * 3 + ["breathy"] * 5 + ["harsh/creaky"] * 4
    voices = {"modal/normal": (50., 50., 50., 50.), "pressed/tense": (50., 50., 50., 10.),
              "breathy": (50., 50., 50., 90.), "harsh/creaky": (90., 90., 10., 50.)}
    emotions = [e for e, count in {"neutral": 7, "joy": 4, "anger": 3, "fear": 2,
                                  "sadness": 2, "disgust": 1, "surprise": 1}.items() for _ in range(count)]
    rows = []
    for i, emotion in enumerate(emotions):
        offset = 100 if i % 2 else 0
        row = {"speaker": "B" if offset else "A", "audio_path": f"grpo_{i}.wav", "emotion": emotion}
        row.update({col: seq[i] + offset for col, seq in zip(env["CUE_FEATURE_MAP"].values(), tier_sequences)})
        row.update({col: value + offset for col, value in zip(env["VOICE_QUALITY_FEATURE_MAP"].values(), voices[voice_labels[i]])})
        rows.append(row)
    held_out = raw.iloc[:80].copy()
    held_out["audio_path"] = [f"val_{i}.wav" for i in range(len(held_out))]
    held_out[list(env["ALL_ACOUSTIC_FEATURES"].values())] = 1e6
    return pd.concat([raw, held_out], ignore_index=True), pd.DataFrame(rows), held_out


def install_calibration(env, grpo, cal):
    env["SPEAKER_ACOUSTIC_QUANTILES"] = cal["quantiles"]
    env["ACOUSTIC_CUE_PRIORS"] = cal["priors"]
    env["ACOUSTIC_CUE_PAYOFFS"] = cal["payoffs"]
    env["ACOUSTIC_PAYOFF_MATRICES"] = cal["matrices"]
    env["ACOUSTIC_MATRIX_PROOFS"] = cal["proofs"]
    env["EMOTION_PRIORS"], env["EMOTION_PAYOFFS"], env["EMOTION_PAYOFF_MATRIX"] = (
        env["calibrate_emotion_payoffs"](grpo, return_matrix=True)
    )
    return {"speaker": grpo.speaker.tolist(), "label": grpo.emotion.tolist(),
            **{col: grpo[col].tolist() for col in env["ALL_ACOUSTIC_FEATURES"].values()}}


@pytest.fixture
def calibrated(env):
    raw, grpo, val = calibration_data(env)
    cal = env["calibrate_acoustic_payoffs"](grpo, raw, val)
    return env, raw, grpo, val, cal, install_calibration(env, grpo, cal)


@pytest.mark.parametrize("categories", [3, 4, 7])
def test_blind_actions_zero_for_many_empirical_distributions(env, categories):
    rng = np.random.default_rng(42)
    for _ in range(100):
        counts = rng.integers(1, 1000, size=categories)
        probabilities = counts / counts.sum()
        priors_input = dict(zip(map(str, range(categories)), counts))
        payoffs = env["compute_hit_miss"](priors_input)
        assert payoffs == env["compute_zero_mean_hit_miss"](priors_input)
        for i, p in enumerate(payoffs.values()):
            scores = np.full(categories, p["miss"])
            scores[i] = p["hit"]
            assert probabilities @ scores == pytest.approx(0., abs=1e-12)
            assert p["hit"] - p["miss"] == pytest.approx(1., abs=1e-12)
            assert p["delta"] == pytest.approx(1., abs=1e-12)
            assert p["map_consistent"] is True
            assert p["e_map_target"] > 0.0
            assert p["map_margin"] > 0.0
            assert 0 < p["hit"] <= 1 + 1e-12
            assert -1 <= p["miss"] < 0


@pytest.mark.parametrize("priors", [dict(a=0, b=1), dict(a=-1, b=2), dict(a=np.nan, b=1),
                                    dict(a=np.inf, b=1), {}, dict(a=1)])
def test_unobserved_or_invalid_categories_are_rejected(env, priors):
    with pytest.raises(ValueError):
        env["compute_zero_mean_hit_miss"](priors)


def test_raw_quartiles_and_grpo_frequencies_are_separate(calibrated):
    env, raw, grpo, val, cal, _ = calibrated
    original_raw, original_grpo = raw.copy(deep=True), grpo.copy(deep=True)
    again = env["calibrate_acoustic_payoffs"](grpo, raw, val)
    pd.testing.assert_frame_equal(raw, original_raw)
    pd.testing.assert_frame_equal(grpo, original_grpo)
    assert again["reference_rows"] == 200
    assert cal["quantiles"]["A"]["pitch_height"] == {"q25": 25.75, "q75": 75.25}
    assert cal["quantiles"]["B"]["pitch_height"] == {"q25": 125.75, "q75": 175.25}
    assert cal["priors"]["pitch_height"] == {"low": .2, "moderate": .45, "high": .35}
    assert cal["priors"]["pitch_dynamics"] == {"low": .4, "moderate": .25, "high": .35}
    assert cal["counts"]["voice_quality"] == {"modal/normal": 8, "pressed/tense": 3, "breathy": 5, "harsh/creaky": 4}
    np.testing.assert_allclose(cal["summary"]["E[blind]"], 0, atol=1e-12)
    changed = env["calibrate_acoustic_payoffs"](pd.concat([grpo, grpo.iloc[:3]]), raw, val)
    assert changed["quantiles"] == cal["quantiles"]
    assert changed["payoffs"]["pitch_height"] != cal["payoffs"]["pitch_height"]


def test_validation_overlap_and_absent_categories_fail(calibrated):
    env, raw, grpo, val, _, _ = calibrated
    overlap = grpo.copy()
    overlap.loc[0, "audio_path"] = val.audio_path.iloc[0]
    with pytest.raises(ValueError, match="overlaps validation"):
        env["calibrate_acoustic_payoffs"](overlap, raw, val)
    with pytest.raises(ValueError, match="audio_path"):
        env["calibrate_acoustic_payoffs"](grpo, raw, val.drop(columns="audio_path"))
    with pytest.raises(ValueError, match="unobserved"):
        env["calibrate_emotion_payoffs"](grpo[grpo.emotion != "fear"])
    no_high = grpo.copy()
    no_high[env["CUE_FEATURE_MAP"]["pitch_height"]] = 50.
    with pytest.raises(ValueError, match="pitch_height"):
        env["calibrate_acoustic_payoffs"](no_high, raw, val)


def test_notebook_initialization_uses_actual_datasets(calibrated):
    from tests.notebook_helpers import cell_source
    env, raw, grpo, val, cal, _ = calibrated
    env.update(raw_train_set=raw, grpo_train_set=grpo, val_set=val)
    exec(cell_source("def calibrate_emotion_payoffs("), env)
    exec(cell_source("def reward_acoustic_inventory("), env)
    assert env["ACOUSTIC_CUE_PRIORS"] == cal["priors"]
    assert env["SPEAKER_ACOUSTIC_QUANTILES"] == cal["quantiles"]
    assert env["ACOUSTIC_PAYOFF_MATRICES"] == cal["matrices"]
    assert env["EMOTION_PRIORS"]["neutral"] == .35
    assert env["EMOTION_MATRIX_PROOF"]["zero_mean_satisfied"] is True
    assert env["EMOTION_MATRIX_PROOF"]["map_satisfied"] is True


def test_r1_every_emotion_zero_mean_and_valid_misses_equal(calibrated):
    env, _, grpo, _, _, kwargs = calibrated
    reward, build = env["reward_label_accuracy_weighted"], env["build_sample_completion"]
    for emotion in sorted(env["VALID_EMOTIONS"]):
        scores = reward(["p"] * len(grpo), [build(emotion=emotion)] * len(grpo), **kwargs)
        assert np.mean(scores) == pytest.approx(0., abs=1e-12)
        assert all((s > 0) == (t == emotion) for s, t in zip(scores, grpo.emotion))
        hit_scores = [s for s, t in zip(scores, grpo.emotion) if t == emotion]
        miss_scores = [s for s, t in zip(scores, grpo.emotion) if t != emotion]
        np.testing.assert_allclose(miss_scores, miss_scores[0])
        assert hit_scores[0] - miss_scores[0] == pytest.approx(1.0, abs=1e-12)
    wrong = [e for e in env["VALID_EMOTIONS"] if e != "joy"]
    scores = reward(["p"] * len(wrong), [build(emotion=e) for e in wrong], label=["joy"] * len(wrong))
    assert reward(["p"], [build(emotion="invalid")], label=["joy"])[0] <= min(scores)


def test_uncalibrated_scoring_fails(env):
    with pytest.raises(ValueError, match="calibrate_emotion_payoffs"):
        env["reward_label_accuracy_weighted"](["p"], [env["build_sample_completion"]()], label=["joy"])
    with pytest.raises(ValueError, match="calibration"):
        env["score_cue_zero_mean"]("low", "low", "pitch_height")
    with pytest.raises(ValueError, match="calibration"):
        env["reward_acoustic_inventory"](["p"], [env["build_sample_completion"]()])


def test_all_324_constant_inventories_have_equal_r2(calibrated):
    env, _, grpo, _, _, kwargs = calibrated
    options = [("low", "moderate", "elevated"), ("narrow", "moderate", "wide"),
               ("quiet", "moderate", "loud"), ("slow", "moderate", "fast"),
               ("modal/normal", "pressed/tense", "breathy", "harsh/creaky")]
    for inventory in product(*options):
        text = env["build_sample_completion"](*inventory)
        scores = env["reward_acoustic_inventory"](["p"] * len(grpo), [text] * len(grpo), **kwargs)
        assert np.mean(scores) == pytest.approx(.5, abs=1e-12), inventory
        assert all(0 <= score <= 1 for score in scores)


def test_hits_beat_all_misses_and_omission_never_helps(calibrated):
    env, _, _, _, cal, _ = calibrated
    for cue, priors in cal["priors"].items():
        for truth in priors:
            hit, _, exact = env["score_cue_zero_mean"](truth, truth, cue)
            assert exact == 1
            misses = [env["score_cue_zero_mean"](truth, p, cue)[0] for p in priors if p != truth]
            assert hit > max(misses)
            column_misses = [env["score_cue_zero_mean"](other_t, truth, cue)[0] for other_t in priors if other_t != truth]
            assert max(column_misses) == pytest.approx(min(column_misses))
            assert hit - column_misses[0] == pytest.approx(1.0, abs=1e-12)
            assert env["score_cue_zero_mean"](truth, None, cue)[0] <= min(misses)
            assert env["score_cue_zero_mean"](truth, "unknown", cue)[0] <= min(misses)


def test_missing_gt_preserves_baseline_and_train_eval_agreement(calibrated):
    env, raw, grpo, val, _, _ = calibrated
    incomplete = grpo.copy()
    incomplete.loc[0, env["CUE_FEATURE_MAP"]["pitch_height"]] = np.nan
    incomplete.loc[1, env["CUE_FEATURE_MAP"]["pitch_dynamics"]] = 0.
    incomplete.loc[2, env["VOICE_QUALITY_FEATURE_MAP"]["jitter"]] = np.inf
    incomplete.loc[3, list(env["ALL_ACOUSTIC_FEATURES"].values())] = np.nan
    cal = env["calibrate_acoustic_payoffs"](incomplete, raw, val)
    kwargs = install_calibration(env, incomplete, cal)
    for inventory in [("low", "narrow", "quiet", "slow", "modal/normal"),
                      ("moderate", "moderate", "moderate", "moderate", "breathy"),
                      ("elevated", "wide", "loud", "fast", "harsh/creaky")]:
        text = env["build_sample_completion"](*inventory)
        scores = env["reward_acoustic_inventory"](["p"] * len(grpo), [text] * len(grpo), **kwargs)
        assert np.mean(scores) == pytest.approx(.5, abs=1e-12)
        for score, (_, row) in zip(scores, incomplete.iterrows()):
            diagnostic = env["evaluate_single_prompt_acoustics"](text, row.audio_path, row.speaker, row_data=row.to_dict())
            assert diagnostic["r2_acoustic_reward"] == pytest.approx(score)
        assert scores[3] == .5


def test_balanced_decision_rule_and_rare_class_bounds(env):
    f = np.array([.6, .15, .15, .1])
    cats = ("modal/normal", "pressed/tense", "breathy", "harsh/creaky")
    payoff = env["get_voice_quality_payoff_matrix"](dict(zip(cats, f)))
    matrix = np.array([[payoff[t][p] for p in cats] for t in cats])
    likelihood = np.random.default_rng(3).dirichlet([2.] * 4, size=1000)
    posterior = f * likelihood
    posterior /= posterior.sum(axis=1, keepdims=True)
    # Expected reward under posterior q is exactly (q - f), preserving unit slope and MAP prior
    np.testing.assert_allclose(posterior @ matrix, posterior - f, atol=1e-12)
    for target_idx in range(len(cats)):
        q_target = 0.75 * f + 0.25 * np.eye(len(cats))[target_idx]
        assert int((q_target @ matrix).argmax()) == target_idx
    rare = env["compute_zero_mean_hit_miss"](dict(low=1e-9, moderate=.7, high=.3 - 1e-9))
    assert max(x["hit"] for x in rare.values()) == pytest.approx(1.)
    assert all(x["delta"] == pytest.approx(1.) for x in rare.values())


def test_composite_trace_and_internal_validation_preserve_live_calibration(calibrated):
    env, _, grpo, _, cal, kwargs = calibrated
    text, trace = env["build_sample_completion"](), []
    scores = env["unified_gated_grpo_reward"](["p"] * len(grpo), [text] * len(grpo), _reward_trace_sink=trace, **kwargs)
    assert len(trace) == len(grpo)
    for score, components in zip(scores, trace):
        assert score == pytest.approx(sum(env["REWARD_WEIGHTS"][k] * components[k] for k in ("r1", "r2", "r3")))
        assert components["r3"] == 1.
    prior_snapshot = {k: dict(p) for k, p in env["ACOUSTIC_CUE_PRIORS"].items()}
    emotion_snapshot = dict(env["EMOTION_PRIORS"])
    report = env["validate_grpo_reward_mathematics"]()
    expected = .5 * env["REWARD_WEIGHTS"]["r2"] + env["REWARD_WEIGHTS"]["r3"]
    np.testing.assert_allclose(report["r_unified_matrix"].values, expected, atol=1e-12)
    assert env["ACOUSTIC_CUE_PRIORS"] == prior_snapshot == cal["priors"]
    assert env["EMOTION_PRIORS"] == emotion_snapshot
