"""Verify distance calibration and the real notebook's scoring/export paths."""
import json
from itertools import product

import numpy as np
import pytest

from tests.notebook_helpers import cell_source, load_reward_environment
from tests.test_grpo_reward_math import calibration_data, install_calibration


@pytest.fixture
def distance():
    env = load_reward_environment(use_zero_mean=False, use_map=True,
                                  use_distance_based_payoff=True)
    raw, train, val = calibration_data(env)
    cal = env['calibrate_acoustic_payoffs'](train, raw, val)
    kwargs = install_calibration(env, train, cal)
    return env, raw, train, val, cal, kwargs


def test_v2_equivalence_and_empirical_ordinal_invariants(distance):
    env, _, _, _, _, _ = distance
    make = env['get_ordinal_cue_payoff_matrix']
    old = make(dict(low=.25, moderate=.5, high=.25))
    np.testing.assert_allclose(list(map(list, [r.values() for r in old.values()])),
                               [[.92, -.55, -.56], [-.18, .55, -.18], [-.56, -.55, .92]])
    supplied = [(266, 555, 308), (244, 598, 287), (241, 602, 286), (297, 571, 261)]
    rng = np.random.default_rng(42)
    for counts in supplied + list(rng.integers(1, 10000, size=(100, 3))):
        p = dict(zip(('low', 'moderate', 'high'), np.array(counts) / sum(counts)))
        matrix = make(p)
        assert matrix['low']['low'] > matrix['low']['moderate'] > matrix['low']['high']
        assert matrix['high']['high'] > matrix['high']['moderate'] > matrix['high']['low']
        assert matrix['moderate']['moderate'] > max(matrix['moderate']['low'], matrix['moderate']['high'])
        for pred in p:
            assert sum(p[t] * matrix[t][pred] for t in p) == pytest.approx(0, abs=1e-12)
    real_priors = {cue: dict(zip(('low', 'moderate', 'high'), counts))
                   for cue, counts in zip(env['CUE_FEATURE_MAP'], supplied)}
    assert env['get_acoustic_reward_scale'](real_priors) == pytest.approx(1.2070794523136523)


@pytest.mark.parametrize('base', ['zero_mean', 'map', 'inverse_frequency'])
def test_distance_override_and_flag_off_preserve_r1_and_old_r2(distance, base):
    env, raw, train, val, _, _ = distance
    flags = dict(use_zero_mean=base == 'zero_mean', use_map=base == 'map',
                 use_inverse_frequency=base == 'inverse_frequency')
    before = env['calibrate_emotion_payoffs'](train, return_matrix=True, **flags)
    enabled = env['calibrate_acoustic_payoffs'](train, raw, val, **flags)
    disabled = env['calibrate_acoustic_payoffs'](train, raw, val, use_distance_based_payoff=False, **flags)
    assert enabled['mode'] == 'distance_based'
    assert disabled['mode'] == base
    assert enabled['counts'] == disabled['counts']
    assert enabled['quantiles'] == disabled['quantiles']
    for cue, priors in disabled['priors'].items():
        assert disabled['matrices'][cue] == env['build_payoff_matrix'](priors, tuple(priors), **flags)
    assert before == env['calibrate_emotion_payoffs'](train, return_matrix=True, **flags)
    result = env['validate_grpo_reward_mathematics'](**flags)
    assert result['acoustic_mode'] == 'distance_based'
    assert env['validate_grpo_reward_mathematics'](use_distance_based_payoff=False, **flags)['acoustic_mode'] == base


def test_constant_inventory_baseline_and_missing_target_train_eval_parity(distance):
    env, raw, train, val, cal, kwargs = distance
    options = [('low', 'moderate', 'elevated'), ('narrow', 'moderate', 'wide'),
               ('quiet', 'moderate', 'loud'), ('slow', 'moderate', 'fast'),
               ('modal/normal', 'pressed/tense', 'breathy', 'harsh/creaky')]
    for inventory in product(*options):
        text = env['build_sample_completion'](*inventory)
        scores = env['reward_acoustic_inventory'](['p'] * len(train), [text] * len(train), **kwargs)
        assert np.mean(scores) == pytest.approx(.5, abs=1e-12)
        assert all(0 <= s <= 1 for s in scores)
    incomplete = train.copy()
    incomplete.loc[0, env['CUE_FEATURE_MAP']['pitch_height']] = np.nan
    incomplete.loc[1, list(env['ALL_ACOUSTIC_FEATURES'].values())] = np.nan
    kwargs = install_calibration(env, incomplete, cal)
    text = env['build_sample_completion'](pitch_height='invalid', voice_quality='invalid')
    scores = env['reward_acoustic_inventory'](['p'] * len(train), [text] * len(train), **kwargs)
    for i, row in incomplete.iterrows():
        evaluation = env['evaluate_single_prompt_acoustics'](text, row.audio_path, row.speaker,
                                                            row_data=row.to_dict())
        assert scores[i] == pytest.approx(evaluation['r2_acoustic_reward'])
        assert all(0 <= s <= 1 for s in evaluation['cue_scores'].values())
    assert scores[1] == .5
    scale = cal['normalization_scale']
    for cue, matrix in cal['matrices'].items():
        for truth in matrix:
            invalid = env['score_cue_payoff'](truth, None, cue)[0]
            assert invalid == -scale
            assert invalid <= min(matrix[truth].values())


def test_export_and_report_use_shared_scale(distance, tmp_path):
    env, _, train, _, cal, _ = distance
    exec(cell_source('def build_grpo_health_report('), env)
    env['VALID_EMOTIONS_LIST'] = sorted(env['VALID_EMOTIONS'])
    records = []
    text = env['build_sample_completion']()
    for _, row in train.iterrows():
        ac = env['evaluate_single_prompt_acoustics'](text, row.audio_path, row.speaker, row_data=row.to_dict())
        record = dict(audio_path=row.audio_path, speaker=row.speaker, truth=row.emotion, pred='neutral',
                      r1_reward=0., r2_acoustic_reward=ac['r2_acoustic_reward'], r3_format_reward=1.,
                      r_unified_reward=ac['r2_acoustic_reward'] + .2, valid_slots_count=5,
                      acoustic_exact_match_ratio=ac['acoustic_exact_match_ratio'])
        for cue in env['INVENTORY_DIMENSIONS']:
            record[f'truth_{cue}'] = ac['cue_truths'][cue]
            record[f'pred_{cue}'] = ac['cue_preds'][cue]
        records.append(record)
    env['write_grpo_health_report'](env['pd'].DataFrame(records), output_dir=tmp_path, train_df=train)
    exported = json.loads((tmp_path / 'evaluation_calibration.json').read_text())
    assert exported['use_distance_based_payoff'] is True
    assert exported['acoustic_reward_mode'] == 'distance_based'
    assert exported['acoustic_normalization_scale'] == cal['normalization_scale']
    report = (tmp_path / 'model_health_report.md').read_text()
    # Every constant inventory must have .5 reward on the calibration population.
    lines = [line for line in report.splitlines() if line.startswith('| Best constant')]
    assert len(lines) == 2
    assert all('| 0.5000 |' in line for line in lines)


def test_invalid_priors_and_flag_rejected(distance):
    env = distance[0]
    with pytest.raises(ValueError, match='unobserved'):
        env['distance_based_payoff_matrix'](dict(low=0, moderate=.5, high=.5))
    with pytest.raises(ValueError, match='bool'):
        env['resolve_acoustic_reward_mode'](use_distance_based_payoff='yes')
