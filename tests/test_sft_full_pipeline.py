"""Actual full-SFT notebook prompts, teacher cache, masking and validation scoring."""
from contextlib import nullcontext
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from tests.test_sft_partitions import env, train, NOTEBOOK
from tests.test_sft_v2 import tensor


def test_teacher_gets_transcript_gender_and_targets_student_gets_only_gender(env, train):
    row = train.iloc[0].to_dict()
    teacher = env['messages_for'](row, teacher=True)[1]['content'][1]['text']
    student = env['messages_for'](row)[1]['content'][1]['text']
    assert 'F (female)' in teacher and 'F (female)' in student
    assert row['text'] in teacher and row['text'] not in student
    assert f"Target emotion: {row['emotion']}" in teacher and 'Target emotion:' not in student
    assert '3 numbered steps' in teacher
    assert f"- timbre: {row['timbre_cue']}" in teacher
    row.update(effective_gender='__GLOBAL__', text='UNTRUSTED', emotion='joy', pitch_level_cue='high')
    prompt = env['student_instruction'](row)
    assert 'pooled reference' in prompt and 'UNTRUSTED' not in prompt
    # All allowed classes occur in instructions, but no row's ground truth affects student input.
    row.update(text='CHANGED', emotion='anger', pitch_level_cue='low')
    assert env['student_instruction'](row) == prompt


def test_teacher_checkpoint_reuses_targets_and_resumes_pending_rows(env, train, tmp_path):
    frame = train.head(3)
    path = tmp_path/'sft.csv'
    calls = []
    def generate(model, processor, rows, teacher=False):
        assert teacher
        calls.extend(row['audio_path'] for row in rows)
        if len(calls) > 2:
            raise RuntimeError('interrupted')
        return ['1. Delivery. 2. Evidence. 3. Emotion.'] * len(rows)
    env['generate_completions'] = generate
    with pytest.raises(RuntimeError, match='interrupted'):
        env['generate_sft_train'](None, None, frame, path, batch_size=2)
    cached = pd.read_csv(path)
    assert cached.reasoning.notna().sum() == 2
    assert set(frame.columns) <= set(cached.columns)
    resumed = []
    def finish(model, processor, rows, teacher=False):
        resumed.extend(row['audio_path'] for row in rows)
        return ['1. Remaining. 2. Evidence. 3. Emotion.'] * len(rows)
    # Rerun the execution cell with the stale in-memory frame left by the exception.
    env.update(generate_completions=finish, model=None, processor=None, sft_train=frame,
               SFT_CSV=path, GENERATION_BATCH_SIZE=2, clean_train_set=train, val_set=pd.DataFrame(),
               load_or_create_partitions=lambda *args: (pd.read_csv(path), pd.DataFrame(), {}),
               validate_partitions=lambda *args: None)
    cell = next(c for c in json.loads(NOTEBOOK.read_text())['cells']
                if c['id'] == 'ser-sft-15')
    exec(''.join(cell['source']), env)
    result = env['sft_train']
    assert resumed == [frame.iloc[2].audio_path]
    for _, row in result.iterrows():
        emotion, cues = env['parse_completion'](row.sft_gt)
        assert emotion == row.emotion
        assert all(cues[cue] == row[f'{cue}_cue'] for cue in env['CUES'])
    resumed.clear()
    env['generate_sft_train'](None, None, result, path)
    assert not resumed
    result.loc[0, 'bridge_prompt'] = 'old prompt'
    with pytest.raises(ValueError, match='mismatch'):
        env['generate_sft_train'](None, None, result, path)


def test_empty_or_structurally_invalid_teacher_response_fails(env, train):
    for reasoning in ['', '<answer>joy</answer>', '[Acoustic Inventory] fake']:
        with pytest.raises(ValueError):
            env['format_sft_target'](train.iloc[0], reasoning)


def test_collator_masks_gender_prompt_and_preserves_complete_target(env, train):
    row = train.iloc[0].to_dict()
    row['sft_gt'] = env['format_sft_target'](row, '1. A. 2. B. 3. C.')
    seen = []
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            seen.append(messages)
            return 'full' if len(messages) == 3 else 'prompt'
        def __call__(self, text, **kwargs):
            ids = [[1, 2, 3, 4, 0, 0]] if text == ['full'] else [[1, 2, 3]]
            mask = [[1, 1, 1, 1, 1, 0]] if text == ['full'] else [[1, 1, 1]]
            return dict(input_ids=tensor(ids), attention_mask=tensor(mask))
    env['load_audio'] = lambda path: np.zeros(160)
    batch = env['SERCollator'](Processor())([row])
    assert 'F (female)' in seen[0][1]['content'][1]['text']
    assert row['text'] not in str(seen[0])
    assert seen[1][-1]['content'] == row['sft_gt']
    np.testing.assert_array_equal(batch['labels'], [[-100, -100, -100, 4, 0, -100]])


def test_validation_scores_emotion_and_saved_cues_with_missing_targets(env, train, tmp_path):
    frame = train.head(2).copy()
    frame.loc[frame.index[1], 'pitch_level_cue'] = None
    outputs = [env['format_sft_target'](frame.iloc[0], '1. A. 2. B. 3. C.'), 'invalid response']
    def generate(model, processor, rows, teacher=False):
        assert not teacher
        assert len(rows) == 2
        return outputs
    env.update(generate_completions=generate, GENERATION_BATCH_SIZE=2)
    report = env['evaluate_validation'](None, None, frame, 'after', tmp_path)
    assert report['emotion']['n'] == 2 and report['emotion']['accuracy'] == .5
    assert report['cues']['pitch_level']['n'] == 1
    assert report['cues']['pitch_level']['missing_targets'] == 1
    assert report['cues']['pitch_level']['accuracy'] == 1
    assert report['cues']['timbre']['accuracy'] == .5
    assert (tmp_path/'after_validation_predictions.csv').exists()
    summary = pd.read_csv(tmp_path/'after_emotion_by_gender.csv')
    assert summary.set_index('gender').loc['F', 'accuracy'] == .5
    assert report['emotion_by_gender']['groups']['F']['n'] == 2
    saved = json.loads((tmp_path/'after_validation_metrics.json').read_text())
    assert saved['emotion_by_gender'] == report['emotion_by_gender']
    for text in [outputs[0] + '<answer>joy</answer>', '<answer>garbage</answer>']:
        assert env['parse_completion'](text)[0] == 'invalid'


@pytest.mark.parametrize('teacher, outcome, budgets', [
    (True, 'complete', [10]),
    (True, 'error', [10]),
    (True, 'retry', [10, 20]),
    (True, 'truncated', [10, 20]),
    (True, 'eos_at_limit', [10]),
    (False, 'truncated', [10]),
])
def test_multimodal_generation_restores_state_and_handles_truncation(env, train, tmp_path, teacher, outcome, budgets):
    calls = []
    class Batch(dict):
        def to(self, device): return self
    class Processor:
        tokenizer = SimpleNamespace(padding_side='right', pad_token_id=0,
                                    batch_decode=lambda *a, **kw: ['explanation', 'explanation'])
        def apply_chat_template(self, messages, **kwargs):
            assert 'F (female)' in messages[1]['content'][1]['text']
            return 'prompt'
        def __call__(self, **kwargs):
            assert 'audio' in kwargs
            return Batch(input_ids=np.ones((2, 3)))
    class Model:
        training = True
        device = 'cuda'
        config = SimpleNamespace(use_cache=False)
        generation_config = SimpleNamespace(eos_token_id=[2, 3] if outcome == 'eos_at_limit' else 2)
        def eval(self): self.training = False
        def train(self, flag): self.training = flag
        def generate(self, **kwargs):
            assert 'labels' not in kwargs and not kwargs['do_sample']
            limit = kwargs['max_new_tokens']
            calls.append(limit)
            if outcome == 'error': raise RuntimeError('failure')
            if outcome == 'truncated' or (outcome == 'retry' and len(calls) == 1):
                sequence = [9] * limit
            elif outcome == 'eos_at_limit':
                sequence = [9] * (limit - 1) + [3]
            else:
                sequence = [9, 2]
            # The other sequence already ended and is padded to the batch's length.
            return np.array([[1, 1, 1, 9, 2] + [0] * (len(sequence) - 2), [1, 1, 1] + sequence])
    env.update(torch=SimpleNamespace(inference_mode=nullcontext, autocast=lambda *a, **k: nullcontext()),
               compute_dtype='bf16', MAX_NEW_TOKENS=10, load_audio=lambda p: np.zeros(16))
    model, processor = Model(), Processor()
    rows = train.head(2).to_dict('records')
    if outcome == 'error':
        with pytest.raises(RuntimeError, match='failure'):
            env['generate_completions'](model, processor, rows, teacher=teacher)
    elif teacher and outcome == 'truncated':
        path = tmp_path / 'sft.csv'
        train.head(2).to_csv(path, index=False)
        saved = path.read_bytes()
        with pytest.raises(ValueError, match='still incomplete after 20 tokens') as error:
            env['generate_sft_train'](model, processor, train.head(2), path)
        assert rows[1]['audio_path'] in str(error.value)
        assert rows[0]['audio_path'] not in str(error.value)
        assert path.read_bytes() == saved
    else:
        assert env['generate_completions'](model, processor, rows, teacher=teacher) == ['explanation', 'explanation']
    assert calls == budgets
    assert model.training and not model.config.use_cache
    assert processor.tokenizer.padding_side == 'right'


def test_notebook_removes_legacy_custom_validation_and_refitting():
    notebook = json.loads(NOTEBOOK.read_text())
    text = '\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'train_test_split' not in text and 'compute_top_speaker_acoustic_quantiles' not in text
    assert "VAL_CSV = CLEAN_DIR / 'clean_val_set_with_cues.csv'" in text
    assert 'train_dataset=SERDataset(sft_train)' in text
    assert "evaluate_validation(model, processor, val_set, 'after'" in text


def test_gender_metrics_reveal_difficulty_and_do_not_impute_unknown_gender(env):
    # Equal seven-emotion support: male predictions are correct, female predictions invalid.
    rows = [dict(gender=gender, effective_gender=gender, emotion=emotion,
                 predicted_emotion=emotion if gender == 'M' else 'invalid')
            for gender in ('F', 'M') for emotion in env['EMOTIONS']]
    rows.append(dict(gender=None, effective_gender='F', emotion='joy', predicted_emotion='joy'))
    report = env['gender_emotion_metrics'](pd.DataFrame(rows))
    assert report['group_column'] == 'gender'
    assert report['groups']['F']['n'] == report['groups']['M']['n'] == 7
    assert report['groups']['F']['accuracy'] == report['groups']['F']['macro_f1'] == 0
    assert report['groups']['M']['accuracy'] == report['groups']['M']['macro_f1'] == 1
    assert report['groups']['unknown']['n'] == 1
    assert report['female_minus_male'] == dict(accuracy=-1., macro_f1=-1.)
    assert report['groups']['F']['per_class']['joy']['support'] == 1


def test_gender_metrics_use_fixed_emotion_classes_and_null_for_absent_group(env):
    predictions = pd.DataFrame([dict(gender='F', emotion='joy', predicted_emotion='joy')])
    report = env['gender_emotion_metrics'](predictions)
    assert report['groups']['F']['accuracy'] == 1
    assert report['groups']['F']['macro_f1'] == pytest.approx(1 / 7)
    assert report['groups']['M']['n'] == 0
    assert report['groups']['M']['accuracy'] is None
    assert report['groups']['M']['macro_f1'] is None
    assert report['female_minus_male'] == dict(accuracy=None, macro_f1=None)
    assert 'null' in json.dumps(report, allow_nan=False)
