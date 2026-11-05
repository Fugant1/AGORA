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
        is_uniform = bool((probabilities.max() - probabilities.min()) <= 1e-12)
        priors_input = dict(zip(map(str, range(categories)), counts))
        payoffs = env["compute_hit_miss"](priors_input)
        assert payoffs == env["compute_zero_mean_hit_miss"](priors_input)
        for i, p in enumerate(payoffs.values()):
            scores = np.full(categories, p["miss"])
            scores[i] = p["hit"]
            assert probabilities @ scores == pytest.approx(0., abs=1e-12)
            assert p["hit"] - p["miss"] == pytest.approx(1., abs=1e-12)
            assert p["delta"] == pytest.approx(1., abs=1e-12)
            assert p["zero_mean_consistent"] is True
            assert p["map_consistent"] is is_uniform
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
    assert env["EMOTION_MATRIX_PROOF"]["map_satisfied"] is False

    # Toggling global flags to MAP mode re-calibrates R1 and R2 to universal MAP matrices
    env.update(USE_ZERO_MEAN=False, USE_MAP=True)
    exec(cell_source("def calibrate_emotion_payoffs("), env)
    exec(cell_source("def reward_acoustic_inventory("), env)
    assert env["EMOTION_MATRIX_PROOF"]["map_satisfied"] is True
    assert env["EMOTION_MATRIX_PROOF"]["zero_mean_satisfied"] is False
    assert all(p["map_satisfied"] is True for p in env["ACOUSTIC_MATRIX_PROOFS"].values())
    assert all(p["zero_mean_satisfied"] is False for p in env["ACOUSTIC_MATRIX_PROOFS"].values())


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
    payoff = env["get_voice_quality_payoff_matrix"](dict(zip(cats, f)), use_zero_mean=True, use_map=False)
    matrix = np.array([[payoff[t][p] for p in cats] for t in cats])
    likelihood = np.random.default_rng(3).dirichlet([2.] * 4, size=1000)
    posterior = f * likelihood
    posterior /= posterior.sum(axis=1, keepdims=True)
    # Zero-Mean: expected reward under posterior q is (q - f)
    np.testing.assert_allclose(posterior @ matrix, posterior - f, atol=1e-12)
    for target_idx in range(len(cats)):
        q_target = 0.75 * f + 0.25 * np.eye(len(cats))[target_idx]
        assert int((q_target @ matrix).argmax()) == target_idx

    # MAP mode: expected reward under posterior q is (q - sum(f^2)), matching argmax(q) universally
    map_payoff = env["get_voice_quality_payoff_matrix"](dict(zip(cats, f)), use_zero_mean=False, use_map=True)
    map_matrix = np.array([[map_payoff[t][p] for p in cats] for t in cats])
    prior_baseline = float(np.sum(f ** 2))
    np.testing.assert_allclose(posterior @ map_matrix, posterior - prior_baseline, atol=1e-12)
    np.testing.assert_array_equal((posterior @ map_matrix).argmax(axis=1), posterior.argmax(axis=1))

    rare = env["compute_zero_mean_hit_miss"](dict(low=1e-9, moderate=.7, high=.3 - 1e-9))
    assert max(x["hit"] for x in rare.values()) == pytest.approx(1.)
    assert all(x["delta"] == pytest.approx(1.) for x in rare.values())


def test_mutually_exclusive_flags_reject_invalid_combinations(calibrated):
    env, raw, grpo, val, _, kwargs = calibrated
    text = env["build_sample_completion"]()
    for bad_zm, bad_map in [(True, True), (False, False)]:
        with pytest.raises(ValueError, match="(?i)mutually exclusive|exactly one"):
            env["resolve_reward_mode"](bad_zm, bad_map)
        with pytest.raises(ValueError):
            env["compute_hit_miss"](dict(a=1, b=2), use_zero_mean=bad_zm, use_map=bad_map)
        with pytest.raises(ValueError):
            env["calibrate_emotion_payoffs"](grpo, use_zero_mean=bad_zm, use_map=bad_map)
        with pytest.raises(ValueError):
            env["calibrate_acoustic_payoffs"](grpo, raw, val, use_zero_mean=bad_zm, use_map=bad_map)
        with pytest.raises(ValueError):
            env["reward_label_accuracy_weighted"](
                ["p"] * len(grpo), [text] * len(grpo), use_zero_mean=bad_zm, use_map=bad_map, **kwargs
            )
        with pytest.raises(ValueError):
            env["reward_acoustic_inventory"](
                ["p"] * len(grpo), [text] * len(grpo), use_zero_mean=bad_zm, use_map=bad_map, **kwargs
            )
        with pytest.raises(ValueError):
            env["unified_gated_grpo_reward"](
                ["p"] * len(grpo), [text] * len(grpo), use_zero_mean=bad_zm, use_map=bad_map, **kwargs
            )
    with pytest.raises(ValueError, match="both"):
        env["resolve_reward_mode"](True, None)

    # Global flag mutation to invalid combination must also fail immediately
    for bad_zm, bad_map, bad_if in [
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
        (False, False, False),
    ]:
        env["USE_ZERO_MEAN"] = bad_zm
        env["USE_MAP"] = bad_map
        env["USE_INVERSE_FREQUENCY"] = bad_if
        try:
            with pytest.raises(ValueError, match="(?i)mutually exclusive|exactly one"):
                env["compute_hit_miss"](dict(a=1, b=2))
            with pytest.raises(ValueError, match="(?i)mutually exclusive|exactly one"):
                env["resolve_reward_mode"](bad_zm, bad_map, bad_if)
        finally:
            env["USE_ZERO_MEAN"] = True
            env["USE_MAP"] = False
            env["USE_INVERSE_FREQUENCY"] = False


def test_inverse_frequency_mode_r1_r2_formula_bounds_and_train_eval_agreement(calibrated):
    from tests.notebook_helpers import cell_source

    env, raw, grpo, val, _, kwargs = calibrated
    # Cell 53 guard accepts single-True USE_INVERSE_FREQUENCY and rejects multi-True
    cell53 = cell_source("VALID_EMOTIONS =  {")
    ns_bad = {}
    with pytest.raises(ValueError, match="(?i)mutually exclusive|exactly one"):
        exec(cell53.replace("USE_INVERSE_FREQUENCY = False", "USE_INVERSE_FREQUENCY = True"), ns_bad)

    env["USE_ZERO_MEAN"] = False
    env["USE_MAP"] = False
    env["USE_INVERSE_FREQUENCY"] = True
    try:
        cal_if = env["calibrate_acoustic_payoffs"](grpo, raw, val)
        assert cal_if["mode"] == "inverse_frequency"

        # 1. R1 inverse frequency: neutral -> 1129/(7*169), other hits -> 1129/(7*160), wrong/invalid -> 0.0
        r1_fn = env["reward_label_accuracy_weighted"]
        build = env["build_sample_completion"]
        r1_scores = r1_fn(
            ["p"] * 4,
            [build(emotion="neutral"), build(emotion="anger"), build(emotion="joy"), build(emotion="invalid")],
            label=["neutral", "anger", "sadness", "joy"],
        )
        assert r1_scores[0] == pytest.approx(1129 / (7 * 169), abs=1e-12)
        assert r1_scores[1] == pytest.approx(1129 / (7 * 160), abs=1e-12)
        assert r1_scores[2] == 0.0
        assert r1_scores[3] == 0.0

        # 2. R2 per-cue formula: IF = N_total / (N_classes * count_c) == 1.0 / (N_classes * prior_c) on hit, 0.0 on miss/omission
        for cue, counts in cal_if["counts"].items():
            n_total = sum(counts.values())
            n_classes = len(counts)
            priors = cal_if["priors"][cue]
            for truth, count_t in counts.items():
                expected_if = n_total / (n_classes * count_t)
                assert expected_if == pytest.approx(1.0 / (n_classes * priors[truth]), abs=1e-12)
                sc_hit, mx_hit, exact_hit = env["score_cue_payoff"](truth, truth, cue, cal_if["priors"])
                assert sc_hit == pytest.approx(expected_if, abs=1e-12)
                assert mx_hit == pytest.approx(expected_if, abs=1e-12)
                assert exact_hit == 1.0
                for wrong in counts:
                    if wrong != truth:
                        sc_miss, mx_miss, exact_miss = env["score_cue_payoff"](truth, wrong, cue, cal_if["priors"])
                        assert sc_miss == 0.0
                        assert mx_miss == pytest.approx(expected_if, abs=1e-12)
                        assert exact_miss == 0.0
                assert env["score_cue_payoff"](truth, None, cue, cal_if["priors"])[0] == 0.0
                assert env["score_cue_payoff"](truth, "unknown_val", cue, cal_if["priors"])[0] == 0.0

        # 3. R2 sample normalization: sum(achieved on right cues) / sum(max possible across scorable cues) in [0, 1]
        # Row 0 in calibration_data: values (10, 10, 10, 10) -> all low tiers ('low', 'narrow', 'quiet', 'slow') and 'modal/normal'
        row0 = grpo.iloc[0]
        row0_kwargs = {
            "speaker": [row0.speaker],
            "priors_dict": cal_if["priors"],
            "quantiles_dict": cal_if["quantiles"],
            **{col: [row0[col]] for col in env["ALL_ACOUSTIC_FEATURES"].values()},
        }
        w_ph = 1.0 / (3 * cal_if["priors"]["pitch_height"]["low"])
        w_pd = 1.0 / (3 * cal_if["priors"]["pitch_dynamics"]["low"])
        w_ve = 1.0 / (3 * cal_if["priors"]["vocal_energy"]["low"])
        w_sr = 1.0 / (3 * cal_if["priors"]["speaking_rate"]["low"])
        w_vq = 1.0 / (4 * cal_if["priors"]["voice_quality"]["modal/normal"])
        max_sum = w_ph + w_pd + w_ve + w_sr + w_vq

        all_right = build("low", "narrow", "quiet", "slow", "modal/normal")
        all_wrong = build("elevated", "wide", "loud", "fast", "harsh/creaky")
        partial = build("low", "wide", "quiet", "fast", "harsh/creaky")  # pitch_height + vocal_energy right

        assert env["reward_acoustic_inventory"](["p"], [all_right], **row0_kwargs)[0] == pytest.approx(1.0, abs=1e-12)
        assert env["reward_acoustic_inventory"](["p"], [all_wrong], **row0_kwargs)[0] == pytest.approx(0.0, abs=1e-12)
        assert env["reward_acoustic_inventory"](["p"], ["<think>empty</think><answer>joy</answer>"], **row0_kwargs)[0] == 0.0
        expected_partial = (w_ph + w_ve) / max_sum
        assert 0.0 < expected_partial < 1.0
        assert env["reward_acoustic_inventory"](["p"], [partial], **row0_kwargs)[0] == pytest.approx(expected_partial, abs=1e-12)

        # 4. Missing GT cues + train/eval parity across full dataset
        incomplete = grpo.copy()
        incomplete.loc[0, env["CUE_FEATURE_MAP"]["pitch_height"]] = np.nan
        incomplete.loc[3, list(env["ALL_ACOUSTIC_FEATURES"].values())] = np.nan
        inc_kwargs = {
            "speaker": incomplete.speaker.tolist(),
            "priors_dict": cal_if["priors"],
            "quantiles_dict": cal_if["quantiles"],
            **{col: incomplete[col].tolist() for col in env["ALL_ACOUSTIC_FEATURES"].values()},
        }
        batch_scores = env["reward_acoustic_inventory"](["p"] * len(incomplete), [partial] * len(incomplete), **inc_kwargs)
        assert all(0.0 <= s <= 1.0 for s in batch_scores)
        assert batch_scores[3] == 0.0
        for score, (_, row) in zip(batch_scores, incomplete.iterrows()):
            diag = env["evaluate_single_prompt_acoustics"](
                partial, row.audio_path, row.speaker, row_data=row.to_dict(),
                quantiles_dict=cal_if["quantiles"], priors_dict=cal_if["priors"],
            )
            assert diag["r2_acoustic_reward"] == pytest.approx(score, abs=1e-12)

        report_if = env["validate_grpo_reward_mathematics"]()
        assert report_if["mode"] == "inverse_frequency"
        assert report_if["r1_matrix_proof"]["inverse_frequency_satisfied"] is True
        assert all(p["inverse_frequency_satisfied"] is True for p in report_if["r2_matrix_proofs"].values())
    finally:
        env["USE_ZERO_MEAN"] = True
        env["USE_MAP"] = False
        env["USE_INVERSE_FREQUENCY"] = False


def test_map_mode_r1_r2_priors_and_train_eval_agreement(calibrated):
    env, raw, grpo, val, _, kwargs = calibrated
    env["USE_ZERO_MEAN"] = False
    env["USE_MAP"] = True
    try:
        cal_map = env["calibrate_acoustic_payoffs"](grpo, raw, val)
        emo_priors, emo_payoffs, emo_matrix = env["calibrate_emotion_payoffs"](grpo, return_matrix=True)
        assert cal_map["mode"] == "map"

        # R1 MAP: Hit = 1 - sum(p^2), Miss = -sum(p^2), prior-random expectation = 0
        emo_b = sum(p * p for p in emo_priors.values())
        for e, p_info in emo_payoffs.items():
            assert p_info["hit"] == pytest.approx(1.0 - emo_b, abs=1e-12)
            assert p_info["miss"] == pytest.approx(-emo_b, abs=1e-12)
            scores = env["reward_label_accuracy_weighted"](
                ["p"] * len(grpo), [env["build_sample_completion"](emotion=e)] * len(grpo),
                emotion_priors=emo_priors, **kwargs,
            )
            assert np.mean(scores) == pytest.approx(emo_priors[e] - emo_b, abs=1e-12)
        assert env["reward_label_accuracy_weighted"](
            ["p"], [env["build_sample_completion"](emotion="invalid")],
            emotion_priors=emo_priors, label=["joy"],
        )[0] == -1.0

        # R2 MAP: prior-weighted random cue baseline is 0.0 -> R2 = 0.5, and train/eval match
        sample_text = env["build_sample_completion"]("elevated", "narrow", "loud", "fast", "pressed/tense", "anger")
        r2_scores = env["reward_acoustic_inventory"](
            ["p"] * len(grpo), [sample_text] * len(grpo),
            priors_dict=cal_map["priors"], quantiles_dict=cal_map["quantiles"], **kwargs,
        )
        for score, (_, row) in zip(r2_scores, grpo.iterrows()):
            diag = env["evaluate_single_prompt_acoustics"](
                sample_text, row.audio_path, row.speaker, row_data=row.to_dict(),
                quantiles_dict=cal_map["quantiles"], priors_dict=cal_map["priors"],
            )
            assert diag["r2_acoustic_reward"] == pytest.approx(score, abs=1e-12)

        report_map = env["validate_grpo_reward_mathematics"]()
        assert report_map["mode"] == "map"
        assert report_map["r1_matrix_proof"]["map_satisfied"] is True
        assert report_map["r1_matrix_proof"]["zero_mean_satisfied"] is False
    finally:
        env["USE_ZERO_MEAN"] = True
        env["USE_MAP"] = False


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
