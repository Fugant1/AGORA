"""Exercise full-SFT local staging and permanent split paths across runtimes."""
import pandas as pd

from tests.test_sft_partitions import env, train
# Both standalone Colab notebooks use this staging helper; exercise the same contract.
from tests.test_sft_v2 import (
    test_missing_archive_is_created_on_drive_and_audio_is_staged_locally,
    test_existing_preprocessing_tar_excludes_raw_audio_and_unsafe_members,
    test_missing_drive_audio_fails_before_archive_is_published,
    test_stale_local_audio_cannot_hide_an_incomplete_archive,
    test_archive_creation_failure_does_not_publish_partial_tar,
)


def test_saved_drive_partitions_rebase_locally_without_changing_targets(env, train, tmp_path):
    drive = tmp_path / 'Drive/meld_16k_mono'
    local = tmp_path / 'runtime/meld_16k_mono'
    clean = drive / 'emo_tag_clean'
    frames = {}
    for split, frame in [('train', train.groupby(['emotion', 'gender']).head(5).copy()),
                         ('val', train.head(1).copy())]:
        paths = []
        for i in range(len(frame)):
            path = clean / split / f'dia{i}_utt0.wav'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'audio')
            paths.append(str(path))
        frame['audio_path'] = paths
        frame.to_csv(clean / f'clean_{split}_set_with_cues.csv', index=False)
        frames[split] = frame.reset_index(drop=True)
    (clean / 'gender_acoustic_quartiles.json').write_text('{}')
    splits = tmp_path / 'Drive/splits'
    env.update(DATA_DIR=drive, DRIVE_DATA_DIR=drive,
               TRAIN_CSV=clean / 'clean_train_set_with_cues.csv',
               QUARTILES_JSON=clean / 'gender_acoustic_quartiles.json',
               SPLIT_DIR=splits, SFT_CSV=splits / 'sft_train.csv',
               GRPO_CSV=splits / 'grpo_train.csv', SPLIT_META=splits / 'split_manifest.json')
    sft, _, _ = env['load_or_create_partitions'](frames['train'], frames['val'])
    row = sft.iloc[0]
    sft['reasoning'] = ''
    sft['sft_gt'] = ''
    sft['bridge_prompt'] = ''
    sft.loc[0, 'reasoning'] = '1. Delivery. 2. Evidence. 3. Emotion.'
    sft.loc[0, 'sft_gt'] = env['format_sft_target'](row, sft.loc[0, 'reasoning'])
    sft.loc[0, 'bridge_prompt'] = env['build_bridge_generation_prompt'](row, concise=False)
    env['atomic_csv'](sft, env['SFT_CSV'])

    env['stage_clean_data'](drive, local)
    env.update(DATA_DIR=local,
               TRAIN_CSV=local / 'emo_tag_clean/clean_train_set_with_cues.csv',
               QUARTILES_JSON=local / 'emo_tag_clean/gender_acoustic_quartiles.json')
    train_local = env['load_manifest'](env['TRAIN_CSV'], 'train', local)
    val_local = env['load_manifest'](local / 'emo_tag_clean/clean_val_set_with_cues.csv', 'val', local)
    def forbidden(*args): raise AssertionError('Existing splits must be reused')
    env['make_partitions'] = forbidden
    loaded, grpo, _ = env['load_or_create_partitions'](train_local, val_local)
    assert loaded.audio_path.str.startswith(str(local)).all()
    assert grpo.audio_path.str.startswith(str(local)).all()
    for column in ['reasoning', 'sft_gt', 'bridge_prompt']:
        assert loaded.loc[0, column] == sft.loc[0, column]
    pd.testing.assert_series_equal(loaded.raw_audio_path, sft.raw_audio_path)
    env['validate_partitions'](loaded, grpo, train_local, val_local)

    # Saving local working rows must write durable paths without mutating the working frame.
    env['atomic_csv'](loaded, env['SFT_CSV'])
    saved = pd.read_csv(env['SFT_CSV'])
    assert saved.audio_path.tolist() == sft.audio_path.tolist()
    assert loaded.audio_path.str.startswith(str(local)).all()
    assert saved.loc[0, 'bridge_prompt'] == sft.loc[0, 'bridge_prompt']
