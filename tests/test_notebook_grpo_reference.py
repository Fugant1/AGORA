"""CPU checks for notebook rollout alignment and reference adapter isolation."""
import ast
from contextlib import contextmanager
import json
from pathlib import Path
import textwrap

import pytest

NB = json.loads((Path(__file__).resolve().parents[1] / 'ser_grpo_v2.ipynb').read_text())
SOURCE = ''.join(NB['cells'][78]['source'])


def reference_context():
    node = next(n for n in ast.parse(SOURCE).body if isinstance(n, ast.FunctionDef) and n.name == 'reference_policy')
    env = {'contextmanager': contextmanager}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<reference_policy>', 'exec'), env)
    return env['reference_policy']


class Parameter:
    def __init__(self, trainable):
        self.requires_grad = trainable

    def requires_grad_(self, value):
        self.requires_grad = value


class AdapterModel:
    def __init__(self):
        self.active_adapter = 'default'
        self.adapters = {'default': Parameter(True), 'sft_reference': Parameter(False)}
        self.base = Parameter(False)

    def parameters(self):
        return [self.base, *self.adapters.values()]

    def set_adapter(self, name):
        self.active_adapter = name
        for key, parameter in self.adapters.items():
            parameter.requires_grad_(key == name)

    def set_requires_grad(self, name, value):
        self.adapters[name].requires_grad_(value)


@pytest.mark.parametrize('fail', [False, True])
def test_reference_restores_policy_and_keeps_sft_frozen(fail):
    model = AdapterModel()
    try:
        with reference_context()(model, 'sft_reference'):
            assert model.active_adapter == 'sft_reference'
            assert not any(p.requires_grad for p in model.parameters())
            if fail:
                raise RuntimeError('reference forward failed')
    except RuntimeError:
        assert fail
    assert model.active_adapter == 'default'
    assert model.adapters['default'].requires_grad
    assert not model.adapters['sft_reference'].requires_grad
    assert not model.base.requires_grad


def rollout_env():
    return {'g': 4, 'inputs': {
        'prompt': ['prompt'] * 8,
        'audio_path': ['a.wav'] * 4 + ['b.wav'] * 4,
        'label': ['joy'] * 4 + ['anger'] * 4,
        'speaker': ['Joey'] * 4 + ['Ross'] * 4,
        'feature': [1.] * 4 + [2.] * 4,
    }, 'processed': {'input_ids': type('Tensor', (), {'shape': (8, 20)})()}, 'raw_audios': []}


def prepare(env):
    start = SOURCE.index('      expanded_inputs = dict(processed)')
    end = SOURCE.index('      unwrapped_model = ', start)
    exec(textwrap.dedent(SOURCE[start:end]), env)


def test_sampler_rows_and_metadata_are_preserved():
    env = rollout_env()
    prepare(env)
    assert env['prompt_input_ids'].shape[0] == 8
    for key in ('audio_path', 'label', 'speaker', 'feature'):
        assert env['replicated_kwargs'][key] is env['inputs'][key]
    assert 'repeat_interleave' not in SOURCE
    assert 'expanded_prompts = list(inputs["prompt"])' in SOURCE


def test_mixed_audio_group_is_rejected():
    env = rollout_env()
    env['inputs']['audio_path'][1] = 'wrong.wav'
    with pytest.raises(ValueError, match='Misaligned GRPO group'):
        prepare(env)


def test_changed_cells_compile_and_frozen_reference_is_wired():
    for i in (72, 78, 79):
        compile(''.join(NB['cells'][i]['source']), f'<cell {i}>', 'exec')
    assert 'reference_adapter_name=SFT_REFERENCE_ADAPTER' in ''.join(NB['cells'][79]['source'])
    assert 'with reference_policy(unwrapped_model, self.reference_adapter_name):' in SOURCE
