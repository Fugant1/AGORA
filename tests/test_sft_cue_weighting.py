"""Exercise notebook cue weights, inventory span alignment, and actual CPU gradients."""
import ast
import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.test_sft_partitions import env, train, NOTEBOOK


def test_weights_use_each_cues_own_sft_counts(env, train):
    frame = train.head(8).copy()
    for cue in env['CUES']:
        low, mid, high = env['CUE_LABELS'][cue]
        frame[env['CUE_COLUMNS'][cue]] = [low] * 2 + [mid] * 4 + [high] * 2
    frame['timbre_cue'] = [env['CUE_LABELS']['timbre'][i] for i in [0, 1, 1, 2, 2, 2, 2, 2]]
    # Legacy openSMILE speaking_rate_cue is deliberately skewed and must be ignored
    frame['speaking_rate_cue'] = ['moderate'] * 8
    weights = env['inverse_frequency_cue_weights'](frame)
    lookup = weights.set_index(['cue', 'label']).weight
    assert lookup['pitch_level', 'low'] == pytest.approx(4 / 3)
    assert lookup['pitch_level', 'moderate'] == pytest.approx(2 / 3)
    assert lookup['timbre', 'low (muffled/soft)'] == pytest.approx(8 / 3)
    for _, group in weights.groupby('cue'):
        assert (group['count'] * group.weight).tolist() == pytest.approx([8 / 3] * 3)
    for invalid in [frame.iloc[:0], frame.assign(pitch_level_cue='moderate'),
                    frame.assign(timbre_cue=None), frame.assign(vocal_energy_cue='loud')]:
        with pytest.raises(ValueError):
            env['inverse_frequency_cue_weights'](invalid)


def test_only_inventory_values_are_weighted_including_subwords_and_leading_spaces(env, train):
    class Tokenizer:
        is_fast = True
        def __call__(self, text, **kwargs):
            # A tokenizer can include the preceding space in the same token as the label.
            offsets = [m.span() for m in re.finditer(r' ?[A-Za-z]+|[\s\S]', text)]
            return dict(input_ids=list(range(len(offsets))), offset_mapping=offsets)

    row = train.iloc[0].to_dict()
    row['sft_gt'] = env['format_sft_target'](
        row, '1. Moderate pitch_level: low. 2. Timbre high (tense/bright). 3. Neutral.')
    lookup = {(cue, label): .5 + i for i, cue in enumerate(env['CUES']) for label in env['CUE_LABELS'][cue]}
    tokenizer = Tokenizer()
    ids, weights = env['inventory_token_weights'](tokenizer, row, lookup)
    encoded = tokenizer(row['sft_gt'])
    assert ids == encoded['input_ids']
    inventory_end = row['sft_gt'].index('[Prosodic Reasoning]')
    spans = {m.group(1): m.span(2) for m in re.finditer(
        r'^- (\w+): (.+)$', row['sft_gt'][:inventory_end], flags=re.MULTILINE)}
    for (start, end), weight in zip(encoded['offset_mapping'], weights):
        expected = 1.
        for cue, (a, b) in spans.items():
            if start < b and end > a:
                expected = lookup[cue, row[env['CUE_COLUMNS'][cue]]]
        assert weight == expected
        if start >= inventory_end:
            assert weight == 1.  # Repeated cue words in reasoning and final emotion stay unweighted.
    with pytest.raises(ValueError, match='differs'):
        env['inventory_token_weights'](tokenizer, dict(row, sft_gt=row['sft_gt'].replace('- pitch_level:', '- fake:')), lookup)
    tokenizer.is_fast = False
    with pytest.raises(ValueError, match='fast tokenizer'):
        env['inventory_token_weights'](tokenizer, row, lookup)


@pytest.mark.parametrize('padding_side', ['left', 'right'])
def test_real_tokenizer_collator_aligns_expanded_audio_and_keeps_eos(env, train, padding_side):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    tokenizers = pytest.importorskip('tokenizers')
    special = ['<|im_start|>', '<|im_end|>', '<|audio|>']
    vocab = {text: i for i, text in enumerate(special + sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet()))}
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab, []))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token='<|im_end|>', eos_token='<|im_end|>',
        additional_special_tokens=['<|im_start|>', '<|audio|>'], padding_side=padding_side)
    rows = train.head(2).to_dict('records')
    for i, row in enumerate(rows):
        row['sft_gt'] = env['format_sft_target'](row, 'moderate high low. ' * (i + 1))
    weights = env['inverse_frequency_cue_weights'](train).set_index(['cue', 'label']).weight.to_dict()

    class Processor:
        def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
            result = '<|im_start|>user\n<|audio|>audio and gender<|im_end|>\n<|im_start|>assistant\n'
            if not add_generation_prompt:
                result += messages[-1]['content'] + '<|im_end|>\n'
            return result

        def __call__(self, text, **kwargs):
            expanded = [value.replace('<|audio|>', '<|audio|>' * (i + 3)) for i, value in enumerate(text)]
            return tokenizer(expanded, padding=True, return_tensors='pt', return_token_type_ids=False)

    processor = Processor()
    processor.tokenizer = tokenizer
    env.update(torch=torch, load_audio=lambda path: np.zeros(160))
    batch = env['SERCollator'](processor, weights)(rows)
    for i, row in enumerate(rows):
        labels, loss_weights = batch['labels'][i], batch['token_weights'][i]
        active = labels != -100
        assert loss_weights[~active].sum() == 0
        decoded = tokenizer.decode(labels[active], skip_special_tokens=False)
        assert decoded == row['sft_gt'] + '<|im_end|>\n'
        assert loss_weights[labels == tokenizer.eos_token_id].tolist() == [1.]
        # Every token after the inventory (reasoning, emotion and EOS) is weight one.
        suffix_length = len(tokenizer.encode('[Prosodic Reasoning]\n' + row['sft_gt'].split('[Prosodic Reasoning]\n')[1]
                                             + '<|im_end|>\n', add_special_tokens=False))
        assert loss_weights[active][-suffix_length:].eq(1).all()

    original = env['inventory_token_weights']
    def misaligned(*args):
        ids, weights = original(*args)
        return [99999] + ids[1:], weights
    env['inventory_token_weights'] = misaligned
    with pytest.raises(ValueError, match='do not align'):
        env['SERCollator'](processor, weights)(rows)


def test_weighted_loss_changes_only_cue_gradients_and_survives_batch_one(env):
    torch = pytest.importorskip('torch')
    env['torch'] = torch
    torch.manual_seed(12)
    logits = torch.randn(2, 5, 4, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 0, -100], [-100, 3, 2, 1, 0]])
    weights = torch.tensor([[0., 0., 4 / 3, 1., 0.], [0., 2 / 3, 1., 1., 1.]])
    fn = env['weighted_sft_loss']
    nll = -logits.log_softmax(-1)
    expected = ((nll[0, 1, 2] * 4 / 3 + nll[0, 2, 0]) / 2
                + (nll[1, 0, 3] * 2 / 3 + nll[1, 1, 2] + nll[1, 2, 1] + nll[1, 3, 0]) / 4) / 2
    loss = fn(logits, labels, weights)
    torch.testing.assert_close(loss, expected)
    split = sum(fn(logits[i:i+1], labels[i:i+1], weights[i:i+1]) for i in range(2)) / 2
    torch.testing.assert_close(loss, split)
    grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    torch.testing.assert_close(grad, torch.autograd.grad(split, logits, retain_graph=True)[0])
    unweighted = fn(logits, labels, (labels != -100).float())
    ordinary = torch.autograd.grad(unweighted, logits)[0]
    torch.testing.assert_close(grad[0, 1], ordinary[0, 1] * 4 / 3)
    torch.testing.assert_close(grad[1, 0], ordinary[1, 0] * 2 / 3)
    torch.testing.assert_close(grad[0, 2], ordinary[0, 2])  # EOS.
    torch.testing.assert_close(grad[1, 1:], ordinary[1, 1:])  # Non-cue assistant tokens.
    assert grad[0, 0].abs().sum() == 0 and grad[0, 3:].abs().sum() == 0
    for value in [0., -1., float('nan')]:
        bad = weights.clone()
        bad[0, 2] = value
        with pytest.raises(ValueError, match='positive'):
            fn(logits, labels, bad)
    with pytest.raises(ValueError, match='one loss weight'):
        fn(logits, labels, weights[:, :-1])
    with pytest.raises(ValueError, match='supervised'):
        fn(logits, torch.full_like(labels, -100), weights)


def test_real_trainer_accumulation_and_forward_contract(env, tmp_path):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    env.update(torch=torch, Trainer=transformers.Trainer)
    source = next(''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells']
                  if 'class CueWeightedSFTTrainer(' in ''.join(c['source']))
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'CueWeightedSFTTrainer')
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(NOTEBOOK), 'exec'), env)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16)

        def forward(self, input_ids, use_cache=False):
            # labels and token_weights must be consumed by the trainer, not passed to the model.
            assert use_cache is False
            return transformers.modeling_outputs.CausalLMOutput(logits=self.table[input_ids])

    dataset = [dict(input_ids=[1, 2, 0, 0], labels=[-100, 2, 0, -100], token_weights=[0., 4 / 3, 1., 0.]),
               dict(input_ids=[1, 3, 2, 0], labels=[-100, 3, 2, 0], token_weights=[0., 2 / 3, 1., 1.])]
    tables = []
    for batch_size, accumulation in [(2, 1), (1, 2)]:
        model = Model()
        args = transformers.TrainingArguments(
            output_dir=str(tmp_path / str(batch_size)), use_cpu=True, max_steps=1,
            per_device_train_batch_size=batch_size, gradient_accumulation_steps=accumulation,
            remove_unused_columns=False, label_names=['labels'], prediction_loss_only=True,
            save_strategy='no', report_to='none', disable_tqdm=True, dataloader_pin_memory=False,
            max_grad_norm=0., lr_scheduler_type='constant', seed=42)
        trainer = env['CueWeightedSFTTrainer'](model=model, args=args, train_dataset=dataset, eval_dataset=dataset,
                                             optimizers=(torch.optim.SGD(model.parameters(), lr=.1), None))
        assert trainer.loss_is_scaled_for_ga is False
        trainer.train()
        assert np.isfinite(trainer.evaluate()['eval_loss'])
        tables.append(model.table.detach().clone())
    torch.testing.assert_close(tables[0], tables[1])


def test_run_exports_train_only_counts_and_weighting_metadata(env, train, tmp_path):
    val_csv = tmp_path / 'val.csv'
    val_csv.write_text('untouched validation')
    env.update(sft_train=train, OUTPUT_DIR=tmp_path / 'run', EPOCHS=1, split_manifest={'saved': True},
               VAL_CSV=val_csv, quartiles={'saved': True})
    code = next(''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells']
                if "cue_loss_weights = inverse_frequency_cue_weights(sft_train)" in ''.join(c['source']))
    exec(code.split('if EVALUATE_BEFORE:')[0], env)
    saved = pd.read_csv(env['OUTPUT_DIR'] / 'cue_loss_weights.csv')
    pd.testing.assert_frame_equal(saved, env['inverse_frequency_cue_weights'](train))
    manifest = json.loads((env['OUTPUT_DIR'] / 'experiment.json').read_text())
    assert manifest['loss']['weights_source'] == 'sft_train_only'
    assert manifest['loss']['weighted_spans'] == 'inventory_values_only'
    assert manifest['loss']['other_assistant_token_weight'] == 1.
    assert len(manifest['loss']['weights']) == len(env['CUES']) * 3
