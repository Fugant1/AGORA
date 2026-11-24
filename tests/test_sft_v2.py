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
                 Dataset=object, Trainer=object,
                 torch=SimpleNamespace(inference_mode=nullcontext, float32=np.float32,
                                       tensor=lambda value, dtype=None: tensor(value)))
    constants = {'CUES', 'CUE_COLUMNS', 'GLOBAL_CUES', 'CUE_LABELS', 'CUE_DESCRIPTIONS', 'PERSONA'}
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
                         speaking_rate_cue='low',
                         **{env['CUE_COLUMNS'][c]: env['CUE_LABELS'][c][i] for c in env['CUES']}))
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
    row = dict(audio_path=str(path), effective_gender='F',
               **{env['CUE_COLUMNS'][c]: env['CUE_LABELS'][c][0] for c in env['CUES']})
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
    for bad in ['low', '<answer>low</answer><answer>high</answer>', '<answer>elevated</answer>',
                '<answer>low', '<answer>low</answer>Human: continue']:
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
                                    get_vocab=lambda: {'<|im_end|>': 9, '<|endoftext|>': 10},
                                    batch_decode=lambda ids, skip_special_tokens: [
                                        '<answer>low</answer>' + ('' if skip_special_tokens else '<|im_end|>')
                                    ] * len(ids))

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
            assert kwargs['use_cache'] is True
            assert kwargs['eos_token_id'] == [9, 10]
            assert kwargs['stop_strings'] == ['</answer>']
            assert kwargs['tokenizer'] is processor.tokenizer
            if fail: raise RuntimeError('generation failed')
            return np.array([[1, 1, 1, 4, 9]] * len(kwargs['input_ids']))

    model, processor = Model(), Processor()
    env['load_audio'] = lambda p: np.zeros(16)
    frame = pd.DataFrame([dict(audio_path='a.wav', cue='pitch_level', truth='low',
                               effective_gender='F', instruction=env['cue_instruction']('pitch_level', 'F'),
                               sft_gt='must never enter prompt')] * 3)
    if fail:
        with pytest.raises(RuntimeError):
            env['evaluate_generated'](model, processor, frame, batch_size=2)
    else:
        records, metrics = env['evaluate_generated'](model, processor, frame, batch_size=2)
        assert metrics['pitch_level']['accuracy'] == 1.0
        assert records.raw_completion.str.endswith('<|im_end|>').all()
        assert not records.hit_token_limit.any()
    assert bars[0].settings['total'] == 3
    assert '{remaining}' in bars[0].settings['bar_format']
    assert bars[0].updates == ([] if fail else [2, 1])
    assert bars[0].closed
    assert model.training and not model.config.use_cache
    assert processor.tokenizer.padding_side == 'right'


def test_end_of_turn_is_required_even_if_default_eos_exists(env):
    with pytest.raises(ValueError, match='im_end'):
        env['generation_eos_ids'](SimpleNamespace(get_vocab=lambda: {'<|endoftext|>': 10}))


def test_inverse_frequency_weights_balance_each_cue_from_training_counts(env):
    examples = pd.DataFrame([
        dict(cue=cue, truth=label)
        for cue, counts in [('pitch_level', (2, 4, 2)), ('timbre', (1, 2, 3))]
        for label, count in zip(env['CUE_LABELS'][cue], counts)
        for _ in range(count)
    ])
    weights = env['inverse_frequency_weights'](examples)
    pitch = weights[weights.cue == 'pitch_level'].set_index('label')
    assert pitch.loc['moderate', 'weight'] == pytest.approx(2 / 3)
    assert pitch.loc['low', 'weight'] == pytest.approx(4 / 3)
    assert pitch.loc['high', 'weight'] == pytest.approx(4 / 3)
    for _, group in weights.groupby('cue'):
        assert (group['count'] * group.weight).values == pytest.approx([group['count'].sum() / 3] * 3)
        assert np.average(group.weight, weights=group['count']) == pytest.approx(1.)
    assert len(weights[weights.cue == 'timbre']) == 3
    for bad in [examples.iloc[:0], examples[examples.truth != 'high'],
                pd.DataFrame([dict(cue='pitch_level', truth='unknown')])]:
        with pytest.raises(ValueError):
            env['inverse_frequency_weights'](bad)


def test_weighted_loss_shifts_masks_normalizes_lengths_and_preserves_singleton_weights(env):
    torch = pytest.importorskip('torch')
    env['torch'] = torch
    fn = env['weighted_answer_loss']
    torch.manual_seed(42)
    logits = torch.randn(2, 5, 4, requires_grad=True)
    # Token 0 is both EOS and padding; only the attention-derived -100 mask excludes padding.
    labels = torch.tensor([[-100, -100, 2, 0, -100], [-100, 3, 2, 1, 0]])
    weights = torch.tensor([4 / 3, 2 / 3])
    nll = -logits.log_softmax(-1)
    expected = ((nll[0, 1, 2] + nll[0, 2, 0]) / 2 * weights[0]
                + (nll[1, 0, 3] + nll[1, 1, 2] + nll[1, 2, 1] + nll[1, 3, 0]) / 4 * weights[1]) / 2
    loss = fn(logits, labels, weights)
    torch.testing.assert_close(loss, expected)
    singleton = fn(logits[:1], labels[:1], weights[:1])
    torch.testing.assert_close(singleton, fn(logits[:1], labels[:1], torch.ones(1)) * weights[0])
    split = sum(fn(logits[i:i+1], labels[i:i+1], weights[i:i+1]) for i in range(2)) / 2
    torch.testing.assert_close(loss, split)
    grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    torch.testing.assert_close(grad, torch.autograd.grad(split, logits)[0])
    assert grad[0, 0].abs().sum() == 0  # Prompt.
    assert grad[0, 3:].abs().sum() == 0  # Padding and final unused next-token position.
    assert grad[0, 2].abs().sum() > 0  # EOS is supervised.
    for invalid in [torch.tensor([0., 1.]), torch.tensor([float('nan'), 1.]), torch.ones(3)]:
        with pytest.raises(ValueError, match='weight'):
            fn(logits, labels, invalid)
    with pytest.raises(ValueError, match='supervised'):
        fn(logits, torch.full_like(labels, -100), weights)


def test_weighted_trainer_consumes_weights_without_exposing_them_to_model(env):
    torch = pytest.importorskip('torch')
    env['torch'] = torch
    logits = torch.zeros(1, 3, 4, requires_grad=True)
    def model(**kwargs):
        assert set(kwargs) == {'input_ids', 'use_cache'}
        assert kwargs['use_cache'] is False
        return SimpleNamespace(logits=logits)
    inputs = dict(input_ids=torch.tensor([[1, 2, 0]]), labels=torch.tensor([[-100, 2, 0]]),
                  loss_weights=torch.tensor([4 / 3]))
    trainer = env['CueWeightedTrainer']()
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True, num_items_in_batch=100)
    assert trainer.loss_is_scaled_for_ga is False
    torch.testing.assert_close(loss, torch.tensor(np.log(4) * 4 / 3, dtype=torch.float32))
    assert outputs.logits is logits and 'labels' in inputs and 'loss_weights' in inputs
    loss.backward()
    assert logits.grad.abs().sum() > 0


def test_real_trainer_accumulation_matches_full_batch(env, tmp_path):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    env.update(torch=torch, Trainer=transformers.Trainer)
    cells = json.loads(NOTEBOOK.read_text())['cells']
    tree = ast.parse(next(''.join(c['source']) for c in cells
                         if 'class CueWeightedTrainer(' in ''.join(c['source'])))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CueWeightedTrainer')
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(NOTEBOOK), 'exec'), env)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16)

        def forward(self, input_ids, use_cache=False):
            assert use_cache is False
            return transformers.modeling_outputs.CausalLMOutput(logits=self.table[input_ids])

    dataset = [dict(input_ids=[1, 2, 0], labels=[-100, 2, 0], loss_weights=4 / 3),
               dict(input_ids=[1, 3, 0], labels=[-100, 3, 0], loss_weights=2 / 3)]
    tables = []
    for batch_size, accumulation in [(2, 1), (1, 2)]:
        model = Model()
        args = transformers.TrainingArguments(
            output_dir=str(tmp_path / str(batch_size)), use_cpu=True, max_steps=1,
            per_device_train_batch_size=batch_size, gradient_accumulation_steps=accumulation,
            remove_unused_columns=False, label_names=['labels'], prediction_loss_only=True,
            save_strategy='no', report_to='none', disable_tqdm=True, dataloader_pin_memory=False,
            max_grad_norm=0., lr_scheduler_type='constant', seed=42)
        trainer = env['CueWeightedTrainer'](
            model=model, args=args, train_dataset=dataset, eval_dataset=dataset,
            optimizers=(torch.optim.SGD(model.parameters(), lr=.1), None))
        trainer.train()
        assert np.isfinite(trainer.evaluate()['eval_loss'])
        tables.append(model.table.detach().clone())
    torch.testing.assert_close(tables[0], tables[1])


def test_real_batched_generation_stops_at_answer_or_qwen_eos(env):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    tokenizers = pytest.importorskip('tokenizers')
    from functools import partial
    from tqdm.auto import tqdm

    special = ['<pad>', '<|im_end|>', '<|endoftext|>']
    vocab = {text: i for i, text in enumerate(special + sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet()))}
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab, []))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token='<pad>', eos_token='<|im_end|>',
        additional_special_tokens=['<|endoftext|>'])
    # No checkpoints or tokenizer files are downloaded: force tokens from a tiny random local model.
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=len(tokenizer), n_positions=128, n_embd=8, n_layer=1, n_head=1,
        eos_token_id=vocab['<|endoftext|>'], pad_token_id=vocab['<pad>']))
    emitted = ['<answer>low</answer>Human: continue',
               '<answer>high (tense/bright)</answer>Human: continue',
               '<answer>low<|im_end|>Human: continue',
               '<answer>high<|endoftext|>Human: continue',
               'x' * 60]
    streams = [tokenizer.encode(text, add_special_tokens=False) for text in emitted]
    prompt_length = len(tokenizer.encode('prompt', add_special_tokens=False))

    class ForceTokens(transformers.LogitsProcessor):
        def __call__(self, input_ids, scores):
            step = input_ids.shape[1] - prompt_length
            scores.fill_(-float('inf'))
            for i, stream in enumerate(streams):
                scores[i, stream[step] if step < len(stream) else vocab['<|im_end|>']] = 0
            return scores

    model.generate = partial(model.generate, logits_processor=[ForceTokens()])

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return 'prompt'

        def __call__(self, text, **kwargs):
            return tokenizer(text, padding=True, return_tensors='pt', return_token_type_ids=False)

    processor = Processor()
    processor.tokenizer = tokenizer
    env.update(torch=torch, tqdm=tqdm, load_audio=lambda path: np.zeros(16))
    frame = pd.DataFrame([dict(audio_path='a.wav', effective_gender='F', instruction='cue?',
                               cue='timbre' if i == 1 else 'pitch_level',
                               truth='high (tense/bright)' if i == 1 else 'low') for i in range(5)])
    records, _ = env['evaluate_generated'](model, processor, frame)
    assert records.prediction.tolist() == ['low', 'high (tense/bright)', 'invalid', 'invalid', 'invalid']
    assert records.hit_token_limit.tolist() == [False, False, False, False, True]
    assert not records.completion.str.contains('Human').any()
    assert '<|im_end|>' in records.raw_completion.iloc[2]
    assert '<|endoftext|>' in records.raw_completion.iloc[3]
    assert tokenizer.padding_side == 'right' and model.training


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
    item['loss_weight'] = 1.25
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
    batch = env['CueCollator'](Processor())([item])
    assert batch['loss_weights'][0] == 1.25
    assert seen[0] == generation_messages
    assert seen[1][:-1] == generation_messages
    assert generation_messages[1]['content'][1]['text'] == instruction


@pytest.mark.parametrize('gender', [None, 'unknown', 'female'])
def test_manifest_rejects_missing_or_unknown_reference_group(env, tmp_path, gender):
    row = dict(audio_path='val/a.wav', effective_gender=gender,
               **{env['CUE_COLUMNS'][cue]: env['CUE_LABELS'][cue][0] for cue in env['CUES']})
    csv = tmp_path / 'val.csv'
    pd.DataFrame([row]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match='effective_gender'):
        env['load_manifest'](csv, 'val', tmp_path)
    pd.DataFrame([row]).drop(columns='effective_gender').to_csv(csv, index=False)
    with pytest.raises(ValueError, match='missing columns'):
        env['load_manifest'](csv, 'val', tmp_path)


def test_speaking_rate_uses_whisperx_syllable_rate_cue_and_global_prompt(env, tmp_path):
    assert env['CUE_COLUMNS']['speaking_rate'] == 'whisperx_syllable_rate_cue'
    assert 'speaking_rate' in env['GLOBAL_CUES']
    audio = tmp_path / 'emo_tag_clean/val/a.wav'
    audio.parent.mkdir(parents=True)
    audio.touch()
    # Deliberately set legacy speaking_rate_cue='low' while whisperx_syllable_rate_cue='high'
    row = dict(
        audio_path=str(audio),
        effective_gender='F',
        speaking_rate_cue='low',
        **{env['CUE_COLUMNS'][c]: 'high' if c == 'speaking_rate' else env['CUE_LABELS'][c][0] for c in env['CUES']},
    )
    csv = tmp_path / 'val.csv'
    pd.DataFrame([row]).to_csv(csv, index=False)
    frame = env['load_manifest'](csv, 'val', tmp_path)
    examples, _ = env['build_examples'](frame, ['speaking_rate'])
    assert len(examples) == 1
    assert examples.loc[0, 'cue'] == 'speaking_rate'
    assert examples.loc[0, 'truth'] == 'high'
    assert examples.loc[0, 'sft_gt'] == '<answer>high</answer>'
    instr_f = env['cue_instruction']('speaking_rate', 'F')
    instr_m = env['cue_instruction']('speaking_rate', 'M')
    instr_g = env['cue_instruction']('speaking_rate', '__GLOBAL__')
    assert instr_f == instr_m == instr_g
    assert 'global (all speakers, gender-agnostic)' in instr_f
    assert 'syllables per second, including pauses' in instr_f
    assert 'F (female)' not in instr_f
    assert 'F (female)' in env['cue_instruction']('pitch_level', 'F')
    # Missing whisperx_syllable_rate_cue must fail even when legacy speaking_rate_cue is present
    pd.DataFrame([row]).drop(columns=['whisperx_syllable_rate_cue']).to_csv(csv, index=False)
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
