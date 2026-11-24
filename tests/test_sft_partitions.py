"""Saved gender labels, constrained partitions and Drive cache contract (CPU)."""
import ast
import hashlib
import json
from pathlib import Path
import re
import tarfile

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix, eye, hstack, vstack
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

NOTEBOOK = Path(__file__).resolve().parents[1] / 'ser_sft.ipynb'


@pytest.fixture
def env():
    scope = dict(Path=Path, hashlib=hashlib, json=json, re=re, tarfile=tarfile, np=np, pd=pd,
                 DATA_DIR=Path('/clean'), DRIVE_DATA_DIR=Path('/clean'),
                 milp=milp, Bounds=Bounds, LinearConstraint=LinearConstraint,
                 csr_matrix=csr_matrix, eye=eye, hstack=hstack, vstack=vstack,
                 accuracy_score=accuracy_score, precision_recall_fscore_support=precision_recall_fscore_support,
                 Dataset=object, Trainer=object, display=lambda *args: None)
    constants = {'CUES', 'CUE_COLUMNS', 'GLOBAL_CUES', 'CUE_LABELS', 'EMOTIONS', 'PERSONA', 'PROMPT_VERSION', 'MODEL_ID', 'SEED'}
    for cell in json.loads(NOTEBOOK.read_text())['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(line for line in cell['source'] if not line.lstrip().startswith(('%', '!')))
        tree = ast.parse(source)
        nodes = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                nodes.append(node)
            elif isinstance(node, ast.Assign):
                target = node.targets[0]
                name = target.id if isinstance(target, ast.Name) else getattr(getattr(target, 'value', None), 'id', None)
                if name in constants:
                    nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), 'exec'), scope)
    return scope


@pytest.fixture
def train(env):
    rows = []
    rng = np.random.default_rng(19)
    for e, emotion in enumerate(env['EMOTIONS']):
        for gender in ('F', 'M'):
            for i in range(15 + e * 2 + (5 if gender == 'M' else 0)):
                row = dict(audio_path=f'/clean/emo_tag_clean/train/{e}_{gender}_{i}.wav',
                           emotion=emotion, gender=gender, effective_gender=gender,
                           text='A private training transcript.', speaker=f'speaker-{gender}',
                           duration=2., dialogue=len(rows), utterance=0,
                           raw_audio_path=f'/raw/{len(rows)}.wav', measured_feature=1.2345,
                           laughter_intervals_removed=i % 2)
                for cue in env['CUES']:
                    row[env['CUE_COLUMNS'][cue]] = rng.choice(env['CUE_LABELS'][cue], p=[.2, .6, .2])
                # Legacy openSMILE speaking_rate_cue intentionally differs from whisperx_syllable_rate_cue
                row['speaking_rate_cue'] = 'low' if row['whisperx_syllable_rate_cue'] != 'low' else 'high'
                rows.append(row)
    return pd.DataFrame(rows)


def test_balancing_precedes_exact_disjoint_split_and_preserves_all_metadata(env, train):
    sft, grpo, details = env['make_partitions'](train, 42)
    assert len(sft) == 126 and len(grpo) == 84
    assert details['quota_per_emotion_gender'] == 15
    assert set(sft.columns) == set(train.columns) == set(grpo.columns)
    val = pd.DataFrame({'audio_path': ['/clean/emo_tag_clean/val/a.wav']})
    env['validate_partitions'](sft, grpo, train, val)
    pool = pd.concat([sft, grpo])
    assert pool.groupby(['emotion', 'effective_gender']).size().eq(15).all()
    again, _, _ = env['make_partitions'](train.sample(frac=1, random_state=2), 42)
    assert sft.audio_path.tolist() == again.audio_path.tolist()


def test_cue_imbalance_matches_reference_when_feasible(env, train):
    # Correlated cue profiles, with a nonuniform 20/60/20 reference in every stratum.
    pool = train.groupby(['emotion', 'effective_gender'], group_keys=False).head(15).copy()
    for _, group in pool.groupby(['emotion', 'effective_gender']):
        for cue in env['CUES']:
            labels = env['CUE_LABELS'][cue]
            pool.loc[group.index, env['CUE_COLUMNS'][cue]] = [labels[0]] * 3 + [labels[1]] * 9 + [labels[2]] * 3
    selected, fit = env['select_cue_stratified'](pool, 5, pool, 42)
    assert fit['absolute_cue_count_deviation'] == pytest.approx(0)
    for gender in ('F', 'M'):
        for cue in env['CUES']:
            counts = selected.loc[selected.effective_gender == gender, env['CUE_COLUMNS'][cue]].value_counts()
            assert counts[env['CUE_LABELS'][cue][1]] == 21  # 60%, not uniform thirds
    audit = env['distribution_audit'](dict(pool=pool, selected=selected))
    assert set(audit.loc[audit.variable == 'speaking_rate', 'column']) == {'whisperx_syllable_rate_cue'}


def test_small_stratum_fails_instead_of_duplicating_or_relaxing(env, train):
    keep = ~((train.emotion == 'fear') & (train.effective_gender == 'M'))
    rare = train.loc[~keep].head(4)
    with pytest.raises(ValueError, match='at least five'):
        env['make_partitions'](pd.concat([train[keep], rare]), 42)


def test_partition_validation_detects_leakage_and_stale_labels(env, train):
    sft, grpo, _ = env['make_partitions'](train, 42)
    val = pd.DataFrame({'audio_path': []})
    with pytest.raises(ValueError, match='overlap'):
        env['validate_partitions'](sft, grpo, train, sft.iloc[:1])
    corrupted = sft.copy()
    corrupted.loc[0, 'measured_feature'] = -1
    with pytest.raises(ValueError, match='metadata/labels differ'):
        env['validate_partitions'](corrupted, grpo, train, val)
    with pytest.raises(ValueError, match='lost preprocessing'):
        env['validate_partitions'](sft.drop(columns='text'), grpo, train, val)


def test_drive_cache_reloads_preserves_reasoning_and_rejects_stale_provenance(env, train, tmp_path):
    data = tmp_path / 'data'
    train = train.copy()
    paths = []
    for old in train.audio_path:
        path = data / 'emo_tag_clean/train' / Path(old).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        paths.append(str(path))
    train['audio_path'] = paths
    train_csv = tmp_path / 'clean.csv'
    train.to_csv(train_csv, index=False)
    quartiles = tmp_path / 'quartiles.json'
    quartiles.write_text('{}')
    split = tmp_path / 'splits'
    env.update(DATA_DIR=data, DRIVE_DATA_DIR=data, TRAIN_CSV=train_csv, QUARTILES_JSON=quartiles,
               SPLIT_DIR=split, SFT_CSV=split/'sft_train.csv', GRPO_CSV=split/'grpo_train.csv',
               SPLIT_META=split/'split_manifest.json')
    val = pd.DataFrame({'audio_path': []})
    current_prompt = env['PROMPT_VERSION']
    env['PROMPT_VERSION'] = 'gender_transcript_teacher_v1'
    sft, grpo, _ = env['load_or_create_partitions'](train, val)
    env['PROMPT_VERSION'] = current_prompt
    sft['reasoning'] = 'cached explanation'
    sft.to_csv(env['SFT_CSV'], index=False)
    def forbidden(*args): raise AssertionError('Must load saved splits')
    env['make_partitions'] = forbidden
    loaded, _, _ = env['load_or_create_partitions'](train, val)
    assert loaded.reasoning.eq('cached explanation').all()
    quartiles.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='provenance changed'):
        env['load_or_create_partitions'](train, val)
    env['GRPO_CSV'].unlink()
    with pytest.raises(ValueError, match='Incomplete saved split'):
        env['load_or_create_partitions'](train, val)
