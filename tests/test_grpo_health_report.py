"""Exercise the notebook report without loading a model or requiring Drive."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.notebook_helpers import cell_source, load_reward_environment, notebook_cells
from tests.test_grpo_reward_math import calibration_data, install_calibration


@pytest.fixture
def reporting():
    env = load_reward_environment()
    raw, train, val = calibration_data(env)
    cal = env['calibrate_acoustic_payoffs'](train, raw, val)
    install_calibration(env, train, cal)
    exec(cell_source('def build_grpo_health_report('), env)
    env['VALID_EMOTIONS_LIST'] = sorted(env['VALID_EMOTIONS'])
    records = []
    for i, row in train.iterrows():
        completion = env['build_sample_completion'](emotion='neutral')
        trace = []
        total = env['unified_gated_grpo_reward'](
            ['p'], [completion], label=[row.emotion], speaker=[row.speaker],
            **{col: [row[col]] for col in env['ALL_ACOUSTIC_FEATURES'].values()},
            _reward_trace_sink=trace,
        )[0]
        ac = env['evaluate_single_prompt_acoustics'](
            completion, row.audio_path, row.speaker, row_data=row.to_dict())
        record = dict(audio_path=f'heldout_{i}.wav', speaker=row.speaker,
                      truth=row.emotion, pred='neutral', duration=float(i % 5 + 1),
                      valid_slots_count=5, r1_reward=trace[0]['r1'],
                      r2_acoustic_reward=trace[0]['r2'], r3_format_reward=trace[0]['r3'],
                      r_unified_reward=total, acoustic_exact_match_ratio=ac['acoustic_exact_match_ratio'],
                      completion=completion)
        for cue in env['INVENTORY_DIMENSIONS']:
            record[f'truth_{cue}'] = ac['cue_truths'][cue]
            record[f'pred_{cue}'] = ac['cue_preds'][cue]
        records.append(record)
    return env, train, pd.DataFrame(records)


def build(env, train, results, groups=()):
    return env['build_grpo_health_report'](
        results, list(groups), dimensions=env['INVENTORY_DIMENSIONS'],
        emotion_categories=sorted(env['VALID_EMOTIONS']), reward_weights=env['REWARD_WEIGHTS'],
        cue_priors=env['ACOUSTIC_CUE_PRIORS'], emotion_priors=env['EMOTION_PRIORS'],
        cue_matrices=env['ACOUSTIC_PAYOFF_MATRICES'], emotion_matrix=env['EMOTION_PAYOFF_MATRIX'],
        quantiles=env['SPEAKER_ACOUSTIC_QUANTILES'], train_df=train)


def test_report_reconstructs_metrics_and_does_not_claim_map(reporting):
    env, train, results = reporting
    before = results.copy(deep=True)
    report, metrics = build(env, train, results)
    assert metrics['emotion']['accuracy'] == pytest.approx(7/20)
    assert metrics['reward_reconstruction_error'] < 1e-12
    assert 'MAP counterexample' in report
    assert 'No usable groups available' in report
    assert 'not a proof' in report
    assert 'Confusion counts' in report
    assert any('never predicted' in s for s in metrics['findings'])
    pd.testing.assert_frame_equal(results, before)


def test_missing_gt_and_invalid_predictions_have_explicit_denominators(reporting):
    env, train, results = reporting
    results.loc[0, 'truth_pitch_height'] = None
    results.loc[1, 'pred_pitch_height'] = None
    results.loc[0, 'pred'] = 'invalid'
    _, metrics = build(env, train, results)
    assert metrics['acoustics']['pitch_height']['n'] == 19
    assert metrics['acoustics']['pitch_height']['invalid'] == pytest.approx(1/19)
    assert metrics['emotion']['invalid'] == pytest.approx(1/20)
    json.dumps(metrics, allow_nan=False)


def test_weight_sweep_is_tie_aware_and_respects_span_allocation(reporting):
    env, train, results = reporting
    # Correct answer loses at the original R2 weight; wins when R2 is disabled.
    group = dict(step_before_update=0, predictions=['joy', 'neutral'], target='joy',
                 r1=[1., 0.], r2=[0., 1.], r3=[1., 1.],
                 reward_weights={'r1': 1., 'r2': 2., 'r3': .2}, audio_path='a.wav')
    report, metrics = build(env, train, results, [group])
    sweep = {r[0]: r for r in metrics['sensitivity']}
    assert sweep[0.][1] == 1.
    assert sweep[.5][1] == .5  # scalar tie, not arbitrary first candidate
    assert sweep[1.][1] == 0.
    assert sweep[1.][2] == 1
    assert len({r[-1] for r in sweep.values()}) == 1  # direct emotion advantage unchanged
    assert 'not a retraining experiment' in report


def test_rollout_reader_reports_bad_lines_and_resets(tmp_path, reporting):
    env, _, _ = reporting
    p = tmp_path/'groups.jsonl'
    g = dict(step_before_update=9, target='joy', predictions=['joy','neutral'],
             r1=[1,0], r2=[.5,.5], r3=[1,1])
    p.write_text(json.dumps(g)+'\n{broken\n'+json.dumps({**g,'step_before_update':0})+'\n')
    groups, notes = env['_health_read_groups'](p)
    assert len(groups) == 1 and groups[0]['step_before_update'] == 0
    assert any('Skipped rollout line' in s for s in notes)
    assert any('excluded 1 earlier groups' in s for s in notes)


def test_writer_exports_both_directories_without_generation(tmp_path, reporting):
    env, train, results = reporting
    output, adapter = tmp_path/'checkpoints', tmp_path/'adapter'
    path = env['write_grpo_health_report'](results, output_dir=output, adapter_dir=adapter, train_df=train)
    assert path == output/'model_health_report.md'
    for filename in ('model_health_report.md','model_health_metrics.json',
                     'validation_predictions.jsonl','evaluation_calibration.json'):
        assert (output/filename).read_bytes() == (adapter/filename).read_bytes()
    metrics = json.loads((output/'model_health_metrics.json').read_text())
    assert any('unavailable' in note for note in metrics['notes'])
    assert len((output/'validation_predictions.jsonl').read_text().splitlines()) == len(results)
    assert not list(output.glob('*.tmp'))


def test_combined_evaluation_runs_once_after_training():
    sources = [''.join(c['source']) for c in notebook_cells()]
    training = next(i for i,s in enumerate(sources) if s.startswith('if resume_checkpoint_dir:'))
    report = next(i for i,s in enumerate(sources) if 'def build_grpo_health_report(' in s)
    evaluation = next(i for i,s in enumerate(sources) if 'def evaluate_grpo_validation(' in s)
    assert training < report < evaluation
    assert sum('val_results_df = evaluate_grpo_validation(' in s for s in sources) == 1
    assert 'thinker_max_new_tokens=512' in sources[evaluation]
    assert 'model.train(was_training)' in sources[evaluation]
    assert 'results_df.attrs["health_report_path"]' in sources[evaluation]
