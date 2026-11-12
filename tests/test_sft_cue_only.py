"""Exercise actual notebook cue tasks, collator and generation without GPU/Drive."""
import ast
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from tests.test_sft_speaker_calibration import env, training_data, cell_with, exec_nodes


@pytest.fixture
def cue_env(env):
    raw, val = training_data(env)
    env['SPEAKER_ACOUSTIC_QUANTILES'] = env['compute_top_speaker_acoustic_quantiles'](
        env['build_acoustic_reference'](raw, val))
    env.update(SFT_CUE_ONLY=True, SFT_CUE_FLAGS={'pitch_height': True},
               SFT_CUE_PERSONA='Analyze the requested acoustic cue.',
               torch=SimpleNamespace(no_grad=nullcontext))
    exec(cell_with('def selected_sft_cues('), env)
    return env


def test_selected_cues_are_independent_tasks_without_teacher_context(cue_env):
    e = cue_env
    raw, _ = training_data(e)
    raw['reasoning'] = 'stale explanation'
    raw['sft_gt'] = '<answer>joy</answer>'
    e['SFT_CUE_FLAGS'] = dict(pitch_height=True, vocal_energy=True)
    result = e['build_sft_cue_examples'](raw.iloc[[1, 26]])
    assert len(result) == 4
    assert result.cue_truth.tolist() == ['low', 'quiet', 'elevated', 'loud']
    assert 'emotion' not in result and 'reasoning' not in result
    assert result.instruction.nunique() == 2
    assert all('explanation' not in target for target in result.sft_gt)
    assert all('Voice Quality' not in prompt for prompt in result.instruction)
    assert e['sft_task_tag']() == 'cues_pitch_height_vocal_energy'
    e['SFT_CUE_FLAGS'] = {'typo': True}
    with pytest.raises(ValueError):
        e['selected_sft_cues']()
    e['SFT_CUE_FLAGS'] = {'pitch_height': False}
    with pytest.raises(ValueError):
        e['selected_sft_cues']()


def test_missing_targets_are_not_imputed_and_other_cues_survive(cue_env):
    e = cue_env
    raw, _ = training_data(e)
    df = raw.iloc[[1, 26]].copy()
    df.iloc[0, df.columns.get_loc(e['CUE_FEATURE_MAP']['pitch_height'])] = np.nan
    result = e['build_sft_cue_examples'](df, cues=['pitch_height', 'vocal_energy'])
    assert len(result) == 3
    assert result.attrs['missing_targets'] == {'pitch_height': 1, 'vocal_energy': 0}
    df[e['CUE_FEATURE_MAP']['pitch_height']] = 0.
    with pytest.raises(ValueError, match='No valid targets'):
        e['build_sft_cue_examples'](df, cues=['pitch_height'])
    df[e['VOICE_QUALITY_FEATURE_MAP']['jitter']] = np.inf
    with pytest.raises(ValueError):
        e['build_sft_cue_examples'](df, cues=['voice_quality'])


def test_cue_mode_skips_bridge_generation(cue_env):
    e = cue_env
    raw, _ = training_data(e)
    exec(cell_with('def generate_sft_train('), e)
    def forbidden(*a, **kw):
        raise AssertionError('Cue-only targets must not generate reasoning')
    result = e['generate_sft_train'](SimpleNamespace(generate=forbidden), None, raw.iloc[[1]])
    assert result.sft_gt.tolist() == ['<answer>low</answer>']


def test_eligible_pool_is_disjoint_deduplicated_and_not_emotion_balanced(cue_env):
    e = cue_env
    raw, val = training_data(e)
    grpo = raw.iloc[[2]]
    e.update(raw_train_set=raw, val_set=val, grpo_train_set=grpo, SFT_CUE_SOURCE='eligible_pool')
    pool = e['sft_cue_training_source']()
    assert not set(pool.audio_path) & (set(val.audio_path) | set(grpo.audio_path))
    assert not pool.audio_path.duplicated().any()
    assert pool.emotion.value_counts().nunique() > 1


def test_metrics_detect_majority_collapse_and_invalid_answers(cue_env):
    e = cue_env
    rows = [dict(cue_task='pitch_height', cue_truth=t, prediction='moderate')
            for t in ['low', 'moderate', 'elevated']]
    m = e['sft_cue_metrics'](rows)['pitch_height']
    assert m['accuracy'] == pytest.approx(1/3)
    assert m['macro_recall'] == pytest.approx(1/3)
    assert m['per_class']['low']['recall'] == 0.
    for bad in ['low', '<answer>joy</answer>', '<answer>low</answer><answer>low</answer>']:
        assert e['parse_sft_cue_answer'](bad, 'pitch_height') == 'invalid'
    assert e['parse_sft_cue_answer']('<answer> BREATHY </answer>', 'voice_quality') == 'breathy'


def test_actual_collator_routes_each_question_and_masks_prompt(cue_env):
    e = cue_env
    class Tensor(np.ndarray):
        def clone(self): return self.copy().view(Tensor)
    def tensor(x): return np.array(x).view(Tensor)
    seen = []
    class Processor:
        tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda x: {'<|im_start|>': 1, 'assistant': 2}[x])
        def apply_chat_template(self, messages, **kw):
            seen.append(messages)
            return str(messages)
        def __call__(self, **kw):
            return dict(input_ids=tensor([[1, 2, 198, 10, 11, 0]]*2),
                        attention_mask=tensor([[1, 1, 1, 1, 1, 0]]*2))
    e.update(Dataset=object, torch=SimpleNamespace(Tensor=Tensor), load_audio_fast=lambda p: np.zeros(16))
    exec_nodes(cell_with('train_csv_path ='), {'SERMultimodalSFTDataset', 'SERMultimodalDataCollator'}, e)
    raw, _ = training_data(e)
    examples = e['build_sft_cue_examples'](raw.iloc[[1]], cues=['pitch_height', 'vocal_energy'])
    dataset = e['SERMultimodalSFTDataset'](examples)
    batch = e['SERMultimodalDataCollator'](Processor(), 'persona', 'full-inventory prompt')([dataset[0], dataset[1]])
    assert seen[0][1]['content'][1]['text'] == e['sft_cue_instruction']('pitch_height')
    assert seen[1][1]['content'][1]['text'] == e['sft_cue_instruction']('vocal_energy')
    np.testing.assert_array_equal(batch['labels'], [[-100, -100, -100, 10, 11, -100]]*2)


@pytest.mark.parametrize('fail', [False, True])
def test_generation_uses_only_audio_and_question_and_restores_state(cue_env, fail):
    e = cue_env
    class Batch(dict):
        def to(self, device): return self
    class Tokenizer:
        padding_side='right'
        pad_token_id=0
        eos_token_id=2
        def batch_decode(self, ids, **kwargs): return ['<answer>low</answer>']*len(ids)
    class Processor:
        tokenizer=Tokenizer()
        def apply_chat_template(self, messages, **kwargs):
            assert len(messages) == 2 and all(m['role'] != 'assistant' for m in messages)
            assert 'Target Emotion' not in str(messages)
            return 'question'
        def __call__(self, text, **kwargs):
            return Batch(input_ids=np.ones((len(text), 3)))
    class Model:
        training=True
        device='cpu'
        def eval(self): self.training=False
        def train(self, state=True): self.training=state
        def generate(self, **kwargs):
            assert not kwargs['do_sample'] and 'labels' not in kwargs
            if fail: raise RuntimeError('generation failure')
            return np.ones((len(kwargs['input_ids']), 5))
    model, processor = Model(), Processor()
    e['load_audio_fast'] = lambda p: np.zeros(16)
    raw, _ = training_data(e)
    examples = e['build_sft_cue_examples'](raw.iloc[[1]])
    if fail:
        with pytest.raises(RuntimeError): e['evaluate_sft_cues'](model, processor, examples)
    else:
        records, metrics = e['evaluate_sft_cues'](model, processor, examples)
        assert metrics['pitch_height']['accuracy'] == 1.
    assert model.training and processor.tokenizer.padding_side == 'right'


def test_teacher_forced_metric_uses_causal_shift(cue_env):
    e = cue_env
    exec_nodes(cell_with('train_csv_path ='), {'compute_sft_metrics'}, e)
    predictions=np.array([[9, 10, 11, 2, 0]])
    labels=np.array([[-100, -100, 10, 11, 2]])
    assert e['compute_sft_metrics']((predictions, labels)) == {'teacher_forced_token_accuracy': 1.}


def test_same_speaker_shuffle_preserves_targets_and_has_no_fixed_points(cue_env):
    e = cue_env
    rows = [dict(audio_path=f'{speaker}{i}', speaker=speaker, cue_task=cue,
                 cue_truth=['low', 'moderate', 'elevated'][i], instruction='question',
                 sft_gt=f'<answer>{i}</answer>')
            for speaker in ['A', 'B'] for cue in ['pitch_height', 'pitch_dynamics'] for i in range(3)]
    rows.append(dict(rows[0]))
    rows.append(dict(rows[0], audio_path='singleton', speaker='C'))
    examples = pd.DataFrame(rows)
    result = e['permute_sft_audio_within_speaker'](examples, 42)
    assert len(result) == 12 and result.attrs['excluded_tasks'] == 1
    assert (result.audio_path != result.original_audio_path).all()
    pd.testing.assert_frame_equal(result, e['permute_sft_audio_within_speaker'](examples, 42))
    for (cue, speaker), group in result.groupby(['cue_task', 'speaker']):
        assert set(group.audio_path) == set(group.original_audio_path)
        for row in group.itertuples():
            original = examples[(examples.audio_path == row.original_audio_path) & (examples.cue_task == cue)].iloc[0]
            assert row.cue_truth == original.cue_truth and row.sft_gt == original.sft_gt
            assert row.audio_path.startswith(speaker)


def test_shuffle_report_compares_same_eligible_tasks(cue_env, tmp_path):
    e = cue_env
    examples = pd.DataFrame([
        dict(audio_path='A0', speaker='A', cue_task='pitch_height', cue_truth='low'),
        dict(audio_path='A1', speaker='A', cue_task='pitch_height', cue_truth='elevated'),
        dict(audio_path='B0', speaker='B', cue_task='pitch_height', cue_truth='low')])
    calls = []
    def evaluate(model, processor, df, **kwargs):
        calls.append(df.copy())
        records = df.copy()
        records['prediction'] = records.audio_path.map({'A0': 'low', 'A1': 'elevated'})
        return records, e['sft_cue_metrics'](records)
    e['evaluate_sft_cues'] = evaluate
    paired, report = e['evaluate_sft_same_speaker_shuffle'](None, None, examples, output_dir=tmp_path)
    assert len(calls) == 2 and all(len(df) == 2 for df in calls)
    assert calls[0].cue_truth.tolist() == calls[1].cue_truth.tolist()
    assert report['cues']['pitch_height']['drop']['accuracy'] == 1.
    assert report['cues']['pitch_height']['donor_label_changed_fraction'] == 1.
    assert report['excluded_tasks'] == 1
    assert (tmp_path / 'paired_predictions.jsonl').exists()
    assert (tmp_path / 'comparison.json').exists()
    with pytest.raises(ValueError, match='at least two'):
        e['permute_sft_audio_within_speaker'](examples.iloc[[2]])
