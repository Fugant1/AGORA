"""Model-free regression checks for notebook training and evaluation paths."""
import ast
from collections import Counter
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads((ROOT / 'ser_grpo_v2.ipynb').read_text())
SOURCE = ''.join(NB['cells'][78]['source'])


class Array(np.ndarray):
    def eq(self, value):
        return (self == value).view(Array)

    def ne(self, value):
        return (self != value).view(Array)

    def long(self):
        return self.astype(np.int64).view(Array)

    def cumsum(self, dim):
        return np.asarray(self).cumsum(axis=dim).view(Array)


def helper(name, **env):
    node = next(x for x in ast.parse(SOURCE).body if isinstance(x, ast.FunctionDef) and x.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook helper>', 'exec'), env)
    return env[name]


def mask(ids, eos, pad):
    torch = SimpleNamespace(bool=bool, zeros_like=lambda a, dtype: np.zeros_like(a, dtype=dtype).view(Array))
    return helper('completion_boundary_mask', torch=torch)(np.array(ids).view(Array), eos, pad)


def test_shared_eos_pad_includes_first_eos_and_truncated_response():
    actual = mask([[9, 2, 2, 2], [2, 2, 2, 2], [3, 4, 5, 6]], [2], 2)
    np.testing.assert_array_equal(actual, [[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]])


def test_multiple_stop_tokens_and_distinct_padding():
    actual = mask([[9, 7, 0, 0], [9, 2, 3, 0], [3, 4, 0, 0]], [2, 7], 0)
    np.testing.assert_array_equal(actual, [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]])


def test_fixed_loss_and_derivative_are_invariant_to_microbatch_partition():
    loss = helper('fixed_completion_loss')
    coefficients = np.array([[1., 2., 3., 4.], [-2., -1., 2., 3.], [4., 2., -3., 1.]])
    active = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
    full = lambda x: loss(x * coefficients, active, 4)
    split = lambda x: sum(loss(x * coefficients[i:i+1], active[i:i+1], 4, 3) for i in range(3))
    assert full(.3) == pytest.approx(split(.3))
    for fn in [full, split]:
        derivative = (fn(.30001) - fn(.29999)) / .00002
        assert derivative == pytest.approx((coefficients * active).sum() / 12)
    # Actual length must not change the denominator for a fixed numerator.
    assert loss(np.ones((1, 4)), np.array([[1, 1, 0, 0]]), 4) == .5


def test_prompt_mask_is_preserved_even_for_eos_inside_prompt():
    node = next(n for n in ast.parse(SOURCE).body if isinstance(n, ast.FunctionDef) and n.name == 'completion_logps')
    prefix = ast.Module(body=node.body[:next(i for i, n in enumerate(node.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'outputs' for t in n.targets))], type_ignores=[])
    inputs = {'generated_ids': np.array([[2, 8, 2, 9, 2, 2]]),
              'completion_ids': np.array([[9, 2, 2]]), 'advantages': np.array([1.]),
              'prompt_input_ids': np.array([[2, 8, 2]]),
              'prompt_attention_mask': np.array([[0, 1, 1]]),
              'completion_mask': np.array([[1, 1, 0]]), 'input_features': np.array([7.])}
    env = {'inputs': inputs, 'torch': SimpleNamespace(cat=lambda xs, dim: np.concatenate(xs, axis=dim))}
    exec(compile(prefix, '<forward mask>', 'exec'), env)
    np.testing.assert_array_equal(env['forward_inputs']['attention_mask'], [[0, 1, 1, 1, 1, 0]])
    assert 'completion_mask' not in env['forward_inputs']
    assert env['forward_inputs']['input_features'] is inputs['input_features']


def test_legacy_replay_refuses_to_guess_audio_identity(tmp_path):
    path = tmp_path / 'old.jsonl'
    path.write_text(json.dumps({'target': 'joy', 'correct': [True, False],
                               'completions': ['a', 'b'], 'advantages': [1, -1]}) + '\n')
    load = helper('load_replay_groups', Path=Path, json=json, Counter=Counter, VALID_EMOTIONS={'joy'})
    with pytest.raises(ValueError, match='no audio_path'):
        load(path)
    audio = tmp_path / 'explicit.wav'
    audio.touch()
    groups = load(path, audio_overrides={0: str(audio)})
    assert groups[0]['audio_path'] == str(audio)
    assert groups[0]['token_ids_reconstructed']


def test_evaluation_uses_training_callable_and_no_old_gate():
    source = next(''.join(c['source']) for c in NB['cells'] if 'def evaluate_grpo_validation(' in ''.join(c['source']))
    assert 'batch_totals = unified_gated_grpo_reward(' in source
    assert 'r_unified_val = batch_totals[idx_in_batch]' in source
    assert 'gated_r1' not in source


def test_config_matches_loss_scaling_and_replay_never_generates():
    assert 'loss_is_scaled_for_ga = True' in SOURCE
    assert 'loss_type="dr_grpo"' in ''.join(NB['cells'][79]['source'])
    for name in ['check_one_replay_update', 'compare_replay_checkpoints']:
        node = next(n for n in ast.parse(SOURCE).body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'generate'
                       for n in ast.walk(node))
        assert any(isinstance(n, ast.Try) and n.finalbody for n in ast.walk(node))
    for i in (67, 68, 71, 78, 79, 80, 101):
        compile(''.join(NB['cells'][i]['source']), f'<cell {i}>', 'exec')


def test_current_reward_math_validation_runs_with_current_weights():
    env = {'display': lambda *args: None}
    for i in (51, 52, 53, 54, 55, 56, 57, 61, 63, 65, 67, 68):
        exec(compile(''.join(NB['cells'][i]['source']), f'<cell {i}>', 'exec'), env)
    report = env['validation_math_report']
    baseline = .5 * env['REWARD_WEIGHTS']['r2'] + env['REWARD_WEIGHTS']['r3']
    np.testing.assert_allclose(report['r_unified_matrix']['All Moderate + Modal'], baseline)
