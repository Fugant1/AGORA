"""Execute v2 notebook helpers without downloading the model or requiring CUDA."""
import ast
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

NOTEBOOK = Path(__file__).resolve().parents[1] / 'cue_learnability.ipynb'


@pytest.fixture
def env():
    import re
    import tarfile
    scope = dict(tarfile=tarfile, Path=Path, pd=pd, np=np, re=re, accuracy_score=accuracy_score,
                 precision_recall_fscore_support=precision_recall_fscore_support,
                 Dataset=object, torch=SimpleNamespace(inference_mode=nullcontext))
    constants = {'CUES', 'CUE_LABELS', 'CUE_DESCRIPTIONS', 'PERSONA'}
    for cell in json.loads(NOTEBOOK.read_text())['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(line for line in cell['source'] if not line.lstrip().startswith(('%', '!')))
        tree = ast.parse(source)
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 or (isinstance(node, ast.Assign) and any(
                     (isinstance(t, ast.Name) and t.id in constants)
                     or (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                         and t.value.id in constants) for t in node.targets))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), 'exec'), scope)
    return scope


def test_manifests_rebase_without_resplitting_and_preserve_labels(env, tmp_path):
    rows = []
    for i in range(3):
        audio = tmp_path / 'emo_tag_clean' / 'val' / f'dia0_utt{i}.wav'
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.touch()
        rows.append(dict(audio_path=f'/old/ssd/meld/emo_tag_clean/val/{audio.name}',
                         effective_gender=['F', 'M', '__GLOBAL__'][i],
                         **{f'{c}_cue': env['CUE_LABELS'][c][i] for c in env['CUES']}))
    rows[0]['pitch_level_cue'] = None
    csv = tmp_path / 'val.csv'
    pd.DataFrame(rows).to_csv(csv, index=False)
    frame = env['load_manifest'](csv, 'val', tmp_path)
    examples, coverage = env['build_examples'](frame, env['CUES'])
    assert len(frame) == 3 and len(examples) == 14
    assert coverage.set_index('cue').loc['pitch_level', 'missing'] == 1
    assert examples.loc[examples.cue == 'timbre', 'truth'].tolist() == list(env['CUE_LABELS']['timbre'])
    assert 'emotion' not in examples and 'speaker' not in examples
    assert all(s == f'<answer>{t}</answer>' for s, t in zip(examples.sft_gt, examples.truth))
    with pytest.raises(ValueError, match='overlap'):
        env['check_disjoint'](frame, frame)
    with pytest.raises(ValueError, match='Unexpected train path'):
        env['load_manifest'](csv, 'train', tmp_path)
    pd.concat([frame, frame]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match='duplicate'):
        env['load_manifest'](csv, 'val', tmp_path)


def test_missing_training_labels_and_unknown_labels_fail(env, tmp_path):
    path = tmp_path / 'emo_tag_clean/train/a.wav'
    path.parent.mkdir(parents=True)
    path.touch()
    row = dict(audio_path=str(path), effective_gender='F', **{f'{c}_cue': env['CUE_LABELS'][c][0] for c in env['CUES']})
    csv = tmp_path / 'train.csv'
    for value, error in [(None, 'missing training'), ('elevated', 'unknown')]:
        row['pitch_level_cue'] = value
        pd.DataFrame([row]).to_csv(csv, index=False)
        with pytest.raises(ValueError, match=error):
            env['load_manifest'](csv, 'train', tmp_path)


def test_metrics_count_invalid_and_detect_collapse(env):
    rows = pd.DataFrame(dict(cue=['pitch_level'] * 3, truth=['low', 'moderate', 'high'],
                             prediction=['moderate'] * 3))
    result = env['cue_metrics'](rows)['pitch_level']
    assert result['accuracy'] == pytest.approx(1 / 3)
    assert result['macro_recall'] == pytest.approx(1 / 3)
    assert result['per_class']['high']['recall'] == 0
    rows['prediction'] = ['invalid', 'moderate', 'invalid']
    assert env['cue_metrics'](rows)['pitch_level']['invalid_rate'] == pytest.approx(2 / 3)
    for bad in ['low', '<answer>low</answer><answer>high</answer>', '<answer>elevated</answer>']:
        assert env['parse_answer'](bad, 'pitch_level') == 'invalid'
    assert env['parse_answer']('<answer>high (tense/bright)</answer>', 'timbre') == 'high (tense/bright)'


class Tensor(np.ndarray):
    def clone(self):
        return self.copy().view(Tensor)

    def bool(self):
        return self.astype(bool)

    def nonzero(self, as_tuple=False):
        return np.asarray(self).nonzero()


def tensor(value):
    return np.asarray(value).view(Tensor)


def test_loss_masks_prompt_and_padding_but_preserves_eos(env):
    full = dict(input_ids=tensor([[1, 2, 3, 4, 0, 0], [1, 2, 4, 0, 0, 0]]),
                attention_mask=tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]]))
    prompt = dict(input_ids=tensor([[1, 2, 3], [1, 2, 0]]),
                  attention_mask=tensor([[1, 1, 1], [1, 1, 0]]))
    labels = env['mask_answer_labels'](full, prompt)
    np.testing.assert_array_equal(labels, [[-100, -100, -100, 4, 0, -100],
                                          [-100, -100, 4, 0, -100, -100]])
    prompt['input_ids'][0, 0] = 99
    with pytest.raises(ValueError, match='exact prefix'):
        env['mask_answer_labels'](full, prompt)


@pytest.mark.parametrize('fail', [False, True])
def test_generation_has_no_targets_and_restores_state(env, fail):
    bars = []
    class Progress:
        def __init__(self, **kwargs):
            self.settings = kwargs
            self.updates = []
            self.closed = False
            bars.append(self)
        def __enter__(self): return self
        def __exit__(self, *args): self.closed = True
        def update(self, amount): self.updates.append(amount)
    env['tqdm'] = Progress

    class Batch(dict):
        def to(self, device): return self

    class Processor:
        tokenizer = SimpleNamespace(padding_side='right', pad_token_id=0,
                                    batch_decode=lambda ids, **kw: ['<answer>low</answer>'] * len(ids))

        def apply_chat_template(self, messages, **kwargs):
            assert len(messages) == 2
            assert messages[1]['content'][1]['text'] == env['cue_instruction']('pitch_level', 'F')
            return 'prompt'

        def __call__(self, **kwargs):
            return Batch(input_ids=np.ones((len(kwargs['text']), 3)))

    class Model:
        training = True
        config = SimpleNamespace(use_cache=False)
        device = 'cpu'
        def eval(self): self.training = False
        def train(self, value): self.training = value
        def generate(self, **kwargs):
            assert 'labels' not in kwargs and not kwargs['do_sample']
            if fail: raise RuntimeError('generation failed')
            return np.ones((len(kwargs['input_ids']), 5))

    model, processor = Model(), Processor()
    env['load_audio'] = lambda p: np.zeros(16)
    frame = pd.DataFrame([dict(audio_path='a.wav', cue='pitch_level', truth='low',
                               effective_gender='F', instruction=env['cue_instruction']('pitch_level', 'F'),
                               sft_gt='must never enter prompt')] * 3)
    if fail:
        with pytest.raises(RuntimeError):
            env['evaluate_generated'](model, processor, frame)
    else:
        _, metrics = env['evaluate_generated'](model, processor, frame)
        assert metrics['pitch_level']['accuracy'] == 1.0
    assert bars[0].settings['total'] == 3
    assert '{remaining}' in bars[0].settings['bar_format']
    assert bars[0].updates == ([] if fail else [2, 1])
    assert bars[0].closed
    assert model.training and not model.config.use_cache
    assert processor.tokenizer.padding_side == 'right'


def write_drive_export(root):
    clean = root / 'emo_tag_clean'
    for split in ('train', 'val'):
        audio = clean / split / 'sample.wav'
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(split.encode())
        # Original preprocessing CSVs can point at a previous runtime.
        pd.DataFrame([{'audio_path': f'/old/runtime/meld_16k_mono/emo_tag_clean/{split}/sample.wav'}]).to_csv(
            clean / f'clean_{split}_set_with_cues.csv', index=False)
    (clean / 'gender_acoustic_quartiles.json').write_text('{"reference": "saved"}')


def test_missing_archive_is_created_on_drive_and_audio_is_staged_locally(env, tmp_path):
    import tarfile
    drive = tmp_path / 'Drive/meld_16k_mono'
    local = tmp_path / 'runtime/meld_16k_mono'
    write_drive_export(drive)
    env['stage_clean_data'](drive, local)
    archive = drive / 'meld_16k_mono_clean.tar'
    assert archive.is_file()
    with tarfile.open(archive) as bundle:
        assert 'meld_16k_mono/emo_tag_clean/train/sample.wav' in bundle.getnames()
        assert 'meld_16k_mono/emo_tag_clean/gender_acoustic_quartiles.json' in bundle.getnames()
    assert (local / 'emo_tag_clean/train/sample.wav').read_bytes() == b'train'
    assert (local / 'emo_tag_clean/val/sample.wav').read_bytes() == b'val'
    assert (local.parent / archive.name).read_bytes() == archive.read_bytes()
    assert not list(tmp_path.rglob('*.partial'))
    frame = pd.read_csv(local / 'emo_tag_clean/clean_train_set_with_cues.csv')
    resolved = env['resolve_audio_path'](frame.audio_path.iloc[0], 'train', local)
    assert resolved == str(local / 'emo_tag_clean/train/sample.wav')
    original_archive = archive.read_bytes()
    # New runtime must restore using the archive even when unpacked Drive WAVs are gone.
    (drive / 'emo_tag_clean/train/sample.wav').unlink()
    (drive / 'emo_tag_clean/val/sample.wav').unlink()
    fresh = tmp_path / 'fresh_runtime/meld_16k_mono'
    env['stage_clean_data'](drive, fresh)
    assert archive.read_bytes() == original_archive
    assert (fresh / 'emo_tag_clean/train/sample.wav').read_bytes() == b'train'


def test_existing_preprocessing_tar_excludes_raw_audio_and_unsafe_members(env, tmp_path):
    import io
    import tarfile
    drive = tmp_path / 'Drive/meld_16k_mono'
    local = tmp_path / 'runtime/meld_16k_mono'
    write_drive_export(drive)
    archive = drive / 'meld_16k_mono_clean.tar'
    with tarfile.open(archive, 'w') as bundle:
        bundle.add(drive / 'emo_tag_clean', arcname='meld_16k_mono/emo_tag_clean')
        for name in ['meld_16k_mono/train/raw.wav', 'meld_16k_mono/emo_tag_clean/val/../../escape.wav',
                     '/tmp/escape.wav']:
            member = tarfile.TarInfo(name)
            member.size = 5
            bundle.addfile(member, io.BytesIO(b'audio'))
        link = tarfile.TarInfo('meld_16k_mono/emo_tag_clean/val/link.wav')
        link.type = tarfile.SYMTYPE
        link.linkname = '/tmp/escape.wav'
        bundle.addfile(link)
    env['stage_clean_data'](drive, local)
    assert len(list(local.rglob('*.wav'))) == 2
    assert not (local / 'train').exists()
    assert not (local / 'emo_tag_clean/val/link.wav').exists()


def test_missing_drive_audio_fails_before_archive_is_published(env, tmp_path):
    drive = tmp_path / 'Drive/meld_16k_mono'
    write_drive_export(drive)
    (drive / 'emo_tag_clean/train/sample.wav').unlink()
    with pytest.raises(FileNotFoundError):
        env['stage_clean_data'](drive, tmp_path / 'runtime/meld_16k_mono')
    assert not (drive / 'meld_16k_mono_clean.tar').exists()
    with pytest.raises(ValueError, match='outside Drive'):
        env['stage_clean_data'](drive, drive / 'local')


@pytest.mark.parametrize('gender', ['F', 'M', '__GLOBAL__'])
def test_reference_group_reaches_training_and_generation_prompts(env, gender):
    frame = pd.DataFrame([dict(audio_path='audio.wav', effective_gender=gender,
                               gender='M' if gender == 'F' else 'F',
                               pitch_level_cue='low', emotion='joy', speaker='private')])
    examples, _ = env['build_examples'](frame, ['pitch_level'])
    item = env['CueDataset'](examples)[0]
    assert item['effective_gender'] == gender
    assert item['sft_gt'] == '<answer>low</answer>'
    instruction = item['instruction']
    assert f': {gender} (' in instruction
    assert 'joy' not in instruction and 'private' not in instruction
    if gender == '__GLOBAL__':
        assert 'pooled reference' in instruction
    generation_messages = env['messages_for'](item)
    seen = []

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            seen.append(messages)
            return 'full' if len(messages) == 3 else 'prompt'

        def __call__(self, text, **kwargs):
            ids = [[1, 2, 3, 4]] if text == ['full'] else [[1, 2]]
            return dict(input_ids=tensor(ids), attention_mask=tensor(np.ones_like(ids)))

    env['load_audio'] = lambda path: np.zeros(16)
    env['CueCollator'](Processor())([item])
    assert seen[0] == generation_messages
    assert seen[1][:-1] == generation_messages
    assert generation_messages[1]['content'][1]['text'] == instruction


@pytest.mark.parametrize('gender', [None, 'unknown', 'female'])
def test_manifest_rejects_missing_or_unknown_reference_group(env, tmp_path, gender):
    row = dict(audio_path='val/a.wav', effective_gender=gender,
               **{f'{cue}_cue': env['CUE_LABELS'][cue][0] for cue in env['CUES']})
    csv = tmp_path / 'val.csv'
    pd.DataFrame([row]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match='effective_gender'):
        env['load_manifest'](csv, 'val', tmp_path)
    pd.DataFrame([row]).drop(columns='effective_gender').to_csv(csv, index=False)
    with pytest.raises(ValueError, match='missing columns'):
        env['load_manifest'](csv, 'val', tmp_path)


def test_stale_local_audio_cannot_hide_an_incomplete_archive(env, tmp_path):
    import tarfile
    drive = tmp_path / 'Drive/meld_16k_mono'
    local = tmp_path / 'runtime/meld_16k_mono'
    write_drive_export(drive)
    write_drive_export(local)
    # This archive omits train/sample.wav, which is already present locally.
    with tarfile.open(drive / 'meld_16k_mono_clean.tar', 'w') as bundle:
        bundle.add(drive / 'emo_tag_clean/val', arcname='meld_16k_mono/emo_tag_clean/val')
    with pytest.raises(ValueError, match='Archive is missing'):
        env['stage_clean_data'](drive, local)


def test_archive_creation_failure_does_not_publish_partial_tar(env, tmp_path, monkeypatch):
    import tarfile
    drive = tmp_path / 'Drive/meld_16k_mono'
    write_drive_export(drive)
    def interrupted(*args, **kwargs):
        raise OSError('interrupted archive write')
    monkeypatch.setattr(tarfile.TarFile, 'add', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        env['stage_clean_data'](drive, tmp_path / 'runtime/meld_16k_mono')
    assert not (drive / 'meld_16k_mono_clean.tar').exists()
    assert not list(drive.rglob('*.partial'))
