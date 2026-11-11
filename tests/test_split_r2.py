"""CPU checks against notebook code: cue-local credit, GRPO std, and token alignment.

The small NumPy tensor adapter exercises tensor assembly without loading PyTorch,
the audio model, or Drive. It is not an end-to-end GPU training test.
"""
import ast
import copy
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from tests.notebook_helpers import cell_source, load_reward_environment
from tests.test_grpo_reward_math import calibration_data, install_calibration


class Tensor(np.ndarray):
    @property
    def device(self): return None
    def float(self): return self.astype(np.float32).view(Tensor)
    def long(self): return self.astype(np.int64).view(Tensor)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return np.asarray(self)
    def unsqueeze(self, dim): return np.expand_dims(self, dim).view(Tensor)
    def sum(self, dim=None, keepdim=False, **kwargs):
        return np.asarray(np.asarray(self).sum(axis=dim, keepdims=keepdim, **kwargs)).view(Tensor)
    def mean(self, dim=None, keepdim=False, **kwargs):
        return np.asarray(np.asarray(self).mean(axis=dim, keepdims=keepdim, **kwargs)).view(Tensor)
    def clamp(self, min): return np.maximum(self, min).view(Tensor)
    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], type): return super().view(shape[0])
        return np.asarray(self).reshape(*shape).view(Tensor)


def tensor(a, **kwargs): return np.array(a, dtype=kwargs.get('dtype')).view(Tensor)


@pytest.fixture
def code():
    env = dict(np=np, re=re, R2_CUE_NAMES=('pitch_height', 'pitch_dynamics', 'vocal_energy', 'speaking_rate', 'voice_quality'),
               R2_CUE_LABELS=('Pitch Height', 'Pitch Dynamics', 'Vocal Energy', 'Speaking Rate', 'Voice Quality'),
               SPAN2_MARKER_SEQUENCES=(), REWARD_WEIGHTS={'r1': 1., 'r2': 1., 'r3': .2})
    env['torch'] = SimpleNamespace(tensor=tensor, float32=np.float32, bool=bool,
        zeros_like=lambda a, **kw: tensor(np.zeros_like(a), **kw),
        zeros=lambda shape, **kw: tensor(np.zeros(shape), **kw),
        nan_to_num=lambda a, **kw: tensor(np.nan_to_num(a, **kw)))
    names = {'group_standardize', 'cue_group_advantages', 'acoustic_value_token_indices',
             'build_cue_token_advantages', 'completion_span_masks', 'compute_span_split_advantages',
             'build_span_split_token_advantages', 'dynamic_completion_loss'}
    names.update({'fixed_completion_loss', 'reduce_completion_loss', 'validate_credit_assignment'})
    tree = ast.parse(cell_source('class AudioGRPOTrainer('))
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                            type_ignores=[]), '<notebook cue helpers>', 'exec'), env)
    return env


class CharTokenizer:
    def encode(self, text, **kwargs): return list(map(ord, text))
    def decode(self, ids, **kwargs): return ''.join(map(chr, ids))
    def __call__(self, text, **kwargs):
        return {'input_ids': self.encode(text), 'offset_mapping': [(i, i+1) for i in range(len(text))]}


def traces(names, dynamics=(.55, -.6, -.61, -.6)):
    return [dict(r1=float(i == 0), r2=.5 + i/20, r3=1., r2_cues={
        c: dict(raw=dynamics[i] if c == 'pitch_dynamics' else .1, truth='low',
                pred='low' if i == 0 else 'high', valid_gt=True, correct=i == 0)
        for c in names}) for i in range(4)]


def test_group_std_and_cue_isolation(code):
    rows = traces(code['R2_CUE_NAMES'])
    a = code['cue_group_advantages'](rows, 4, 2.)
    raw = np.array([.55, -.6, -.61, -.6])
    expected = 2/5 * (raw - raw.mean()) / (raw.std(ddof=1) + 1e-8)
    np.testing.assert_allclose(a['pitch_dynamics'], expected)
    assert all(np.count_nonzero(a[c]) == 0 for c in a if c != 'pitch_dynamics')
    other = copy.deepcopy(rows)
    for i, row in enumerate(other):
        row['r1'] = 100*i
        row['r2_cues']['voice_quality']['raw'] = i**2
    b = code['cue_group_advantages'](other, 4, 2.)
    np.testing.assert_array_equal(a['pitch_dynamics'], b['pitch_dynamics'])
    missing = copy.deepcopy(rows)
    for row in missing:
        row['r2_cues']['pitch_dynamics'].update(valid_gt=False, truth=None)
    assert not code['cue_group_advantages'](missing, 4)['pitch_dynamics'].any()
    missing[0]['r2_cues']['pitch_dynamics']['valid_gt'] = True
    with pytest.raises(ValueError, match='Mismatched'):
        code['cue_group_advantages'](missing, 4)


def test_grpo_span_tensor_is_standardized_and_scalar_matches_total(code):
    rows = traces(code['R2_CUE_NAMES'])
    result = code['compute_span_split_advantages'](rows, 4, loss_type='grpo')
    total = np.array([r['r1']+r['r2']+.2*r['r3'] for r in rows])
    np.testing.assert_allclose(result['adv_total'], code['group_standardize'](total, 4), atol=2e-7)
    assert result['adv_total'].std(ddof=1) == pytest.approx(1, abs=1e-6)
    ids = tensor([[ord('x')]*4]*4); mask = tensor([[1]*4]*4)
    actual, *_ = code['build_span_split_token_advantages'](ids, mask, result['adv_acoustic'],
        result['adv_emotion'], adv_total=result['adv_total'], use_split_advantage=False)
    np.testing.assert_allclose(actual[:, 0], result['adv_total'])
    source = cell_source('class AudioGRPOTrainer(')
    assert 'adv_total=advantages,' in source
    assert 'device=device, loss_type=self.args.loss_type' in source


def test_binary_affine_rewards_cancel_but_distance_can_rank_wrong_answers(code):
    z = code['group_standardize']
    hits = np.array([1., 0., 0., 1.])
    np.testing.assert_allclose(z(hits-.4, 4), z(hits*3.1, 4), atol=1e-7)
    assert not z(np.zeros(4), 4).any()
    assert z([-.6, -.61, -.6, -.61], 4)[0] > 0
    # Separate audio groups, never one normalization over the whole batch.
    np.testing.assert_allclose(z([0, 1, 100, 101], 2), [-2**-.5, 2**-.5]*2, atol=1e-7)


def test_mapping_context_duplicates_truncation_and_no_reasoning_leak(code):
    env = load_reward_environment()
    tok = CharTokenizer()
    text = env['build_sample_completion'](pitch_dynamics='moderate', speaking_rate='moderate')
    text = text.replace('The speaker', 'Pitch Dynamics: wide. The speaker')
    mapping, status = code['acoustic_value_token_indices'](tok.encode(text), tok)
    assert all(s == 'ok' for s in status.values())
    assert tok.decode([ord(text[i]) for i in mapping['pitch_dynamics']]) == 'moderate'
    assert set(mapping['pitch_dynamics']).isdisjoint(mapping['speaking_rate'])
    assert max(j for indices in mapping.values() for j in indices) < text.index('[Prosodic Reasoning]')
    duplicate = text.replace('- Pitch Dynamics:', '- Pitch Dynamics: wide\n- Pitch Dynamics:')
    mapping, status = code['acoustic_value_token_indices'](tok.encode(duplicate), tok)
    assert status['pitch_dynamics'] == 'duplicate_field' and not mapping['pitch_dynamics']
    partial = text[:text.index('- Voice Quality:')] + '- Voice Quality:'
    mapping, status = code['acoustic_value_token_indices'](tok.encode(partial), tok)
    assert status['voice_quality'] == 'empty_value'
    assert status['pitch_height'] == 'ok'
    mapping, status = code['acoustic_value_token_indices'](tok.encode('Pitch Dynamics: wide'), tok)
    assert not any(mapping.values())


def test_value_masks_get_only_own_reward_and_equal_total_weight(code):
    env = load_reward_environment(); tok = CharTokenizer()
    texts = [env['build_sample_completion'](pitch_dynamics=x) for x in ['wide', 'moderate']]
    width = max(map(len, texts))+2
    ids = tensor([tok.encode(t)+[0]*(width-len(t)) for t in texts])
    mask = tensor([[1]*len(t)+[0]*(width-len(t)) for t in texts])
    cues = {c: np.array([.4, -.4]) if c == 'pitch_dynamics' else np.zeros(2) for c in code['R2_CUE_NAMES']}
    result, _, _, _, counts, statuses = code['build_cue_token_advantages'](
        ids, mask, cues, tensor([.7, .7]), tensor([9., 9.]), tok)
    for i, text in enumerate(texts):
        positions, _ = code['acoustic_value_token_indices'](tok.encode(text), tok)
        # The outer original-GRPO completion mean cancels the explicit length factor.
        assert result[i, positions['pitch_dynamics']].sum()/len(text) == pytest.approx(cues['pitch_dynamics'][i])
        assert all(result[i, positions[c]].sum() == 0 for c in positions if c != 'pitch_dynamics')
        assert result[i, text.index('[Prosodic Reasoning]')+2] == 9
        assert result[i, text.index('[Acoustic Inventory]')+2] == pytest.approx(.7)
        assert not np.any(result[i, len(text):])


def test_grpo_loss_is_mean_of_completion_means(code):
    losses = tensor([[1., 3., 99.], [6., 99., 99.]])
    mask = tensor([[1., 1., 0.], [1., 0., 0.]])
    assert code['dynamic_completion_loss'](losses, mask) == pytest.approx(4.)
    assert code['dynamic_completion_loss'](losses, mask, 2) == pytest.approx(2.)


def test_hybrid_preserves_emotion_and_format_but_not_cue_std(code):
    rows = traces(code['R2_CUE_NAMES'])
    # Nonconstant format tests that hybrid leaves the baseline R1/R3 interaction intact.
    rows[1]['r3'] = .5
    grpo = code['compute_span_split_advantages'](rows, 4, loss_type='grpo')
    hybrid = code['compute_span_split_advantages'](rows, 4, loss_type='hybrid')
    for key in ('adv_r1', 'adv_r3', 'adv_emotion'):
        np.testing.assert_array_equal(hybrid[key], grpo[key])
    cues = code['cue_group_advantages'](rows, 4, 2., loss_type='hybrid')
    raw = np.array([.55, -.6, -.61, -.6])
    np.testing.assert_allclose(cues['pitch_dynamics'], 2/5 * (raw - raw.mean()))
    assert all(not cues[c].any() for c in cues if c != 'pitch_dynamics')
    changed = copy.deepcopy(rows)
    for i, row in enumerate(changed):
        row['r1'] = 100 * i
        row['r3'] = i**2
        row['r2_cues']['pitch_dynamics']['raw'] *= .01
        row['r2_cues']['voice_quality']['raw'] = i
    scaled = code['cue_group_advantages'](changed, 4, 2., loss_type='hybrid')
    np.testing.assert_allclose(scaled['pitch_dynamics'], .01*cues['pitch_dynamics'])


def test_hybrid_tiny_wrong_answer_gap_is_not_amplified(code):
    rows = [copy.deepcopy(row) for row in traces(code['R2_CUE_NAMES']) * 2]
    for i, row in enumerate(rows):
        row['r2_cues']['vocal_energy'].update(raw=-.628273 if i < 6 else -.638273)
    hybrid = code['cue_group_advantages'](rows, 8, loss_type='hybrid')
    np.testing.assert_allclose(hybrid['vocal_energy'], [.0005]*6 + [-.0015]*2)
    grpo = code['cue_group_advantages'](rows, 8, loss_type='grpo')
    assert grpo['vocal_energy'][0] > .1
    for row in rows:
        row['r2_cues']['vocal_energy'].update(valid_gt=False, truth=None)
    assert not code['cue_group_advantages'](rows, 8, loss_type='hybrid')['vocal_energy'].any()
    rows[0]['r2_cues']['pitch_height']['raw'] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        code['cue_group_advantages'](rows, 8, loss_type='hybrid')


@pytest.mark.parametrize('split,r2', [(False, False), (False, True), (True, False)])
def test_hybrid_requires_both_splits(code, split, r2):
    with pytest.raises(ValueError, match='hybrid requires'):
        code['validate_credit_assignment']('hybrid', split, r2)


def test_loss_modes_reject_unsupported_credit_and_preserve_old_modes(code):
    validate = code['validate_credit_assignment']
    validate('hybrid', True, True)
    for mode in ('grpo', 'dr_grpo'):
        validate(mode, False, False)
        validate(mode, True, False)
    validate('grpo', True, True)
    with pytest.raises(ValueError):
        validate('dr_grpo', True, True)
    with pytest.raises(ValueError):
        validate('typo', True, True)


def test_hybrid_gradient_coefficients_dynamic_lengths_and_microbatches(code):
    """Finite differences of actual masked surrogate: answer dilution, cue invariance."""
    env = load_reward_environment()
    tok = CharTokenizer()
    first = env['build_sample_completion'](pitch_dynamics='wide')
    second = env['build_sample_completion'](pitch_dynamics='moderate')
    second = second.replace('[Prosodic Reasoning]', '[Prosodic Reasoning]' + ' Extra prose.'*20)
    texts = [first, second]
    width = max(map(len, texts)) + 4
    ids = tensor([tok.encode(t) + [0]*(width-len(t)) for t in texts])
    mask = tensor([[1]*len(t) + [0]*(width-len(t)) for t in texts])
    cue_adv = {c: np.array([.02, -.02]) if c == 'pitch_dynamics' else np.zeros(2)
               for c in code['R2_CUE_NAMES']}
    adv, *_ = code['build_cue_token_advantages'](
        ids, mask, cue_adv, tensor([.3, .3]), tensor([.7, .7]), tok)
    reduce = code['reduce_completion_loss']
    def objective(logps, maximum=2048):
        return reduce(-np.exp(logps)*adv, mask, 'hybrid', maximum)
    zero = tensor(np.zeros_like(ids, dtype=float))
    for i, text in enumerate(texts):
        positions, _ = code['acoustic_value_token_indices'](tok.encode(text), tok)
        cue = positions['pitch_dynamics']
        answer = text.index('<answer>') + len('<answer>')
        for pos, expected in [(cue[0], -cue_adv['pitch_dynamics'][i]/len(cue)/2),
                              (answer, -.7/len(text)/2), (len(text), 0.)]:
            perturb = zero.copy()
            perturb[i, pos] = 1e-5
            derivative = (objective(perturb)-objective(-perturb))/(2e-5)
            assert derivative == pytest.approx(expected, abs=1e-7)
        assert -adv[i, cue].sum()/len(text)/2 == pytest.approx(-cue_adv['pitch_dynamics'][i]/2)
    assert objective(zero, 512) == pytest.approx(objective(zero, 2048))
    full = objective(zero)
    split = sum(reduce(-adv[i:i+1], mask[i:i+1], 'hybrid', 2048, 2) for i in range(2))
    assert full == pytest.approx(split)
    np.testing.assert_allclose(reduce(-adv, mask, 'grpo', 2048), full)
    assert reduce(-adv, mask, 'dr_grpo', 2048) == pytest.approx((-adv*mask).sum()/(2*2048))


def test_hybrid_rollout_uses_cue_credit_and_never_scalar_reward(code):
    import textwrap
    source = cell_source('class AudioGRPOTrainer(')
    start = source.index('      span_adv_dict = compute_span_split_advantages(')
    end = source.index('      if self.accelerator.is_main_process:', start)
    tok = CharTokenizer()
    text = load_reward_environment()['build_sample_completion']()
    ids = tensor([tok.encode(text)]*4)
    mask = tensor([[1]*len(text)]*4)
    def forbidden(*args):
        raise AssertionError('Hybrid must not compute a scalar composite reward advantage')
    trainer = SimpleNamespace(args=SimpleNamespace(loss_type='hybrid'),
                              use_split_advantage=True, use_split_r2=True,
                              processor=SimpleNamespace(tokenizer=tok),
                              _compute_advantages=forbidden)
    env = dict(code, self=trainer, reward_trace=traces(code['R2_CUE_NAMES']), g=4,
               device=None, completion_ids=ids, completion_mask=mask)
    exec(textwrap.dedent(source[start:end]), env)
    positions, _ = code['acoustic_value_token_indices'](tok.encode(text), tok)
    for cue, tokens in positions.items():
        np.testing.assert_allclose(env['token_advantages'][:, tokens].sum(dim=1)/len(text),
                                   env['cue_advantages'][cue], atol=1e-7)
    answer = text.index('<answer>') + len('<answer>')
    np.testing.assert_array_equal(env['token_advantages'][:, answer], env['span_adv_dict']['adv_emotion'])


@pytest.mark.parametrize('mode', ['grpo', 'dr_grpo', 'hybrid'])
def test_trainer_compute_loss_uses_selected_reduction(code, mode):
    """Execute actual trainer method with CPU tensor shims and fixed log probabilities."""
    from collections import defaultdict
    from contextlib import nullcontext
    from typing import Any, Dict
    tree = ast.parse(cell_source('class AudioGRPOTrainer('))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AudioGRPOTrainer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_loss')
    code.update(Dict=Dict, Any=Any, completion_logps=lambda model, inputs: tensor([[0., 0., 0.], [0., 0., 0.]]),
                reference_policy=lambda *args: nullcontext())
    code['torch'].no_grad = nullcontext
    code['torch'].exp = lambda x: tensor(np.exp(x))
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<notebook compute_loss>', 'exec'), code)
    model = SimpleNamespace(training=True)
    trainer = SimpleNamespace(accelerator=SimpleNamespace(unwrap_model=lambda m: m),
        args=SimpleNamespace(loss_type=mode, beta=0., max_completion_length=512),
        reference_adapter_name=None, current_gradient_accumulation_steps=2,
        _metrics=defaultdict(lambda: defaultdict(list)))
    inputs = dict(completion_mask=tensor([[1., 1., 0.], [1., 0., 0.]]),
                  token_advantages=tensor([[.4, .8, 99.], [-.2, 99., 99.]]),
                  advantages=tensor([99., 99.]))
    loss = code['compute_loss'](trainer, model, inputs)
    expected = -.1 if mode in ('grpo', 'hybrid') else -1./(2*512*2)
    assert loss == pytest.approx(expected)
    if mode == 'hybrid':
        del inputs['token_advantages']
        with pytest.raises(ValueError, match='cue-specific token_advantages'):
            code['compute_loss'](trainer, model, inputs)


def test_real_reward_trace_matches_per_cue_evaluation(code):
    env = load_reward_environment(use_distance_based_payoff=True)
    raw, train, val = calibration_data(env)
    cal = env['calibrate_acoustic_payoffs'](train, raw, val)
    kwargs = install_calibration(env, train, cal)
    trace=[]; text=env['build_sample_completion']()
    env['unified_gated_grpo_reward'](['p']*len(train), [text]*len(train), _reward_trace_sink=trace, **kwargs)
    for i, row in train.iterrows():
        check = env['evaluate_single_prompt_acoustics'](text, row.audio_path, row.speaker, row_data=row.to_dict())
        for cue, details in trace[i]['r2_cues'].items():
            assert details['raw'] == check['cue_raw_scores'][cue]
            assert details['correct'] == check['cue_matches'][cue]
    json.dumps(trace, allow_nan=False)


def test_replay_uses_saved_applied_advantages():
    source = cell_source('class AudioGRPOTrainer(')
    replay = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_replay_completion')
    text = ast.get_source_segment(source, replay)
    assert text.index('if "token_advantages" in group:') < text.index('elif "r1" in group')
    assert 'Cue-split replay requires saved per-token advantages.' in text


def test_context_sensitive_tokenization_uses_checked_prefix_fallback(code):
    class PieceTokenizer(CharTokenizer):
        pieces = ['[Acoustic Inventory]\n- Pitch Dynamics:', ' wi', 'de',
                  '\n[Prosodic Reasoning]\nPitch Dynamics: narrow']
        def decode(self, ids, **kw): return ''.join(self.pieces[i] for i in ids)
        def __call__(self, text, **kw):
            # Re-encoding differs from sampled IDs: these offsets must not be used.
            return {'input_ids': [999], 'offset_mapping': [(0, len(text))]}
    indices, status = code['acoustic_value_token_indices']([0, 1, 2, 3], PieceTokenizer())
    assert indices['pitch_dynamics'] == [1, 2]
    assert status['pitch_dynamics'] == 'ok'
    class BrokenTokenizer(PieceTokenizer):
        def decode(self, ids, **kw):
            return 'bad prefix' if len(ids) == 1 else super().decode(ids)
    indices, status = code['acoustic_value_token_indices']([0, 1, 2, 3], BrokenTokenizer())
    assert not any(indices.values())
    assert set(status.values()) == {'unaligned_tokens'}


def test_cue_reporting_reads_actual_advantages(tmp_path, code):
    from tests.test_grpo_health_report import build
    env = load_reward_environment(use_distance_based_payoff=True)
    raw, train, val = calibration_data(env)
    cal = env['calibrate_acoustic_payoffs'](train, raw, val)
    install_calibration(env, train, cal)
    exec(cell_source('def build_grpo_health_report('), env)
    rows = traces(code['R2_CUE_NAMES'])
    adv = code['cue_group_advantages'](rows, 4)
    group = dict(step_before_update=0, predictions=['neutral']*4, target='neutral',
                 r1=[r['r1'] for r in rows], r2=[r['r2'] for r in rows], r3=[1.]*4,
                 r2_cues=[r['r2_cues'] for r in rows], use_split_r2=True,
                 advantages_cues={k: v.tolist() for k,v in adv.items()},
                 cue_mask_status=[{c:'ok' for c in code['R2_CUE_NAMES']}]*4,
                 cue_value_tokens={c:[1]*4 for c in code['R2_CUE_NAMES']})
    import pandas as pd
    text = env['build_sample_completion']()
    row = train.iloc[0]
    ac = env['evaluate_single_prompt_acoustics'](text, row.audio_path, row.speaker, row_data=row.to_dict())
    record = dict(audio_path='heldout.wav', speaker=row.speaker, truth=row.emotion, pred='neutral',
                  r1_reward=0., r2_acoustic_reward=ac['r2_acoustic_reward'], r3_format_reward=1.,
                  r_unified_reward=ac['r2_acoustic_reward']+.2, valid_slots_count=5,
                  acoustic_exact_match_ratio=ac['acoustic_exact_match_ratio'])
    for cue in code['R2_CUE_NAMES']:
        record[f'truth_{cue}']=ac['cue_truths'][cue]
        record[f'pred_{cue}']=ac['cue_preds'][cue]
    report, metrics = build(env, train, pd.DataFrame([record]), [group])
    by_cue={r[0]: r for r in metrics['cue_diagnostics']}
    assert by_cue['pitch_dynamics'][2] == 0.
    assert by_cue['vocal_energy'][2] == 1.
    assert 'Per-cue R2 training signal' in report
