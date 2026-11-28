"""MILP cue-stratified 60/40 SFT/GRPO partitioning (supports semi-balanced ~4,300 and uniform V4 modes)."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, eye, hstack, vstack

from src.config import CUE_COLUMNS, CUE_LABELS, CUES, EMOTIONS, MODEL_ID, PROMPT_VERSION
from src.utils.io import atomic_csv, load_manifest

StratumKey = Tuple[str, str]
QuotaSpec = Union[int, Dict[StratumKey, int]]


def _resolve_quota_vector(quota: QuotaSpec) -> Dict[StratumKey, int]:
    strata = [(emotion, gender) for emotion in EMOTIONS for gender in ("F", "M")]
    if isinstance(quota, dict):
        missing = [s for s in strata if s not in quota]
        if missing:
            raise ValueError(f"Missing quota entries for strata: {missing}")
        return {s: int(quota[s]) for s in strata}
    q = int(quota)
    return {s: q for s in strata}


def select_cue_stratified(
    pool: pd.DataFrame,
    quota: QuotaSpec,
    reference: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select exact per-(emotion, gender) quotas while preserving reference cue marginals within each gender."""
    strata = [(emotion, gender) for emotion in EMOTIONS for gender in ("F", "M")]
    quota_map = _resolve_quota_vector(quota)
    quota_vec = np.array([quota_map[s] for s in strata], dtype=int)
    if (quota_vec < 1).any():
        raise ValueError(f"Every emotion/gender stratum requires quota >= 1, got {quota_map}")

    columns = ["emotion", "effective_gender"] + [CUE_COLUMNS[cue] for cue in CUES]
    pool = pool.sort_values("audio_path").reset_index(drop=True)
    grouped = pool.groupby(columns, sort=True, dropna=False).indices
    profiles = list(grouped)
    capacity = np.array([len(grouped[key]) for key in profiles])
    hard = np.array([[int(key[:2] == stratum) for key in profiles] for stratum in strata])
    if (hard @ capacity < quota_vec).any():
        shortfall = {
            strata[i]: (int((hard @ capacity)[i]), int(quota_vec[i]))
            for i in range(len(strata))
            if (hard @ capacity)[i] < quota_vec[i]
        }
        raise ValueError(f"Insufficient rows for requested emotion/gender quotas: {shortfall}")

    marginal_rows, targets = [], []
    for gender in ("F", "M"):
        ref = reference.loc[reference.effective_gender == gender]
        gender_total = sum(quota_map[(e, gender)] for e in EMOTIONS)
        for j, cue in enumerate(CUES):
            for label in CUE_LABELS[cue]:
                marginal_rows.append([int(key[1] == gender and key[2 + j] == label) for key in profiles])
                targets.append(gender_total * float(ref[CUE_COLUMNS[cue]].eq(label).mean()))

    marginal = csr_matrix(marginal_rows)
    n, k = len(profiles), len(targets)
    matrix = vstack(
        [
            hstack([csr_matrix(hard), csr_matrix((len(strata), k))]),
            hstack([marginal, -eye(k)]),
            hstack([marginal, eye(k)]),
        ],
        format="csr",
    )
    lower = np.r_[quota_vec, np.full(k, -np.inf), targets]
    upper = np.r_[quota_vec, targets, np.full(k, np.inf)]
    rng = np.random.default_rng(seed)
    result = milp(
        c=np.r_[rng.random(n) * 1e-7, np.ones(k)],
        integrality=np.r_[np.ones(n), np.zeros(k)],
        bounds=Bounds(np.zeros(n + k), np.r_[capacity, np.full(k, np.inf)]),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": 120, "mip_rel_gap": 0.001},
    )
    if result.x is None:
        raise RuntimeError(f"Cue-stratified selection failed: {result.message}")
    counts = np.rint(result.x[:n]).astype(int)
    if not np.allclose(result.x[:n], counts, atol=1e-5) or not (hard @ counts == quota_vec).all():
        raise RuntimeError("Solver did not return an integer selection satisfying quotas")
    if (counts < 0).any() or (counts > capacity).any():
        raise RuntimeError("Invalid selected profile counts")

    indices = []
    for key, count in zip(profiles, counts):
        indices.extend(rng.choice(grouped[key], size=count, replace=False).tolist())
    selected = pool.iloc[indices].sample(frac=1, random_state=seed).reset_index(drop=True)
    return selected, dict(
        solver_status=int(result.status),
        message=result.message,
        absolute_cue_count_deviation=float(np.abs(marginal @ counts - targets).sum()),
    )


def compute_semi_balanced_quotas(
    train: pd.DataFrame,
    neutral_target_fraction: float = 0.25,
) -> Dict[StratumKey, int]:
    """Compute exact 60/40-divisible quotas: ALL non-neutral clips + neutral sized to `neutral_target_fraction`.

    Each stratum quota is rounded down to a multiple of 5 so 60% SFT and 40% GRPO are exact integers.
    Non-neutral strata keep 100% of available clips (mod 5); neutral is split 50/50 across F and M
    (with overflow to the other gender if one is smaller) so neutral/(non_neutral + neutral) ~= 0.25.
    """
    if not (0.0 < float(neutral_target_fraction) < 0.8):
        raise ValueError(f"neutral_target_fraction must be in (0.0, 0.8), got {neutral_target_fraction!r}")

    sizes = train.groupby(["emotion", "effective_gender"]).size().reindex(
        pd.MultiIndex.from_product([EMOTIONS, ("F", "M")]), fill_value=0
    )
    if int(sizes.min()) < 5:
        raise ValueError("Need at least five clean utterances in every emotion/gender group")

    quotas: Dict[StratumKey, int] = {}
    non_neutral_total = 0
    for emotion in EMOTIONS:
        if emotion == "neutral":
            continue
        for gender in ("F", "M"):
            q = (int(sizes[(emotion, gender)]) // 5) * 5
            quotas[(emotion, gender)] = q
            non_neutral_total += q

    # Target neutral total so neutral / (non_neutral + neutral) == neutral_target_fraction
    target_neutral = int(round(non_neutral_total * (neutral_target_fraction / (1.0 - neutral_target_fraction))))
    avail_f = (int(sizes[("neutral", "F")]) // 5) * 5
    avail_m = (int(sizes[("neutral", "M")]) // 5) * 5
    target_neutral = min(target_neutral, avail_f + avail_m)
    target_neutral = max(10, (target_neutral // 5) * 5)

    half = ((target_neutral // 2) // 5) * 5
    q_f = min(avail_f, half)
    q_m = min(avail_m, target_neutral - q_f)
    # Rebalance any remainder to whichever gender still has headroom
    rem = target_neutral - (q_f + q_m)
    if rem >= 5 and q_f + rem <= avail_f:
        q_f += (rem // 5) * 5
    elif rem >= 5 and q_m + rem <= avail_m:
        q_m += (rem // 5) * 5
    quotas[("neutral", "F")] = max(5, q_f)
    quotas[("neutral", "M")] = max(5, q_m)
    return quotas


def make_partitions(
    train: pd.DataFrame,
    seed: int = 42,
    partition_mode: str = "semi_balanced",
    neutral_target_fraction: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build disjoint 60/40 SFT/GRPO splits in either 'semi_balanced' (all non-neutral + 25% neutral) or 'uniform' mode."""
    if partition_mode not in ("semi_balanced", "uniform"):
        raise ValueError(f"Unsupported partition_mode={partition_mode!r}; expected 'semi_balanced' or 'uniform'.")

    if partition_mode == "uniform":
        sizes = train.groupby(["emotion", "effective_gender"]).size().reindex(
            pd.MultiIndex.from_product([EMOTIONS, ("F", "M")]), fill_value=0
        )
        quota = int(sizes.min()) // 5 * 5
        if quota == 0:
            raise ValueError("Need at least five clean utterances in every emotion/gender group")
        pool_quotas: QuotaSpec = quota
        sft_quotas: QuotaSpec = quota * 3 // 5
        quota_meta: Dict[str, Any] = dict(partition_mode="uniform", quota_per_emotion_gender=quota)
    else:
        pool_quotas = compute_semi_balanced_quotas(train, neutral_target_fraction=neutral_target_fraction)
        sft_quotas = {k: (v * 3) // 5 for k, v in pool_quotas.items()}
        total_pool = sum(pool_quotas.values())
        neu_total = pool_quotas[("neutral", "F")] + pool_quotas[("neutral", "M")]
        quota_meta = dict(
            partition_mode="semi_balanced",
            neutral_target_fraction=float(neutral_target_fraction),
            actual_neutral_fraction=float(neu_total / max(1, total_pool)),
            quotas_by_stratum={f"{e}_{g}": int(v) for (e, g), v in pool_quotas.items()},
        )

    balanced, pool_fit = select_cue_stratified(train, pool_quotas, train, seed)
    sft, split_fit = select_cue_stratified(balanced, sft_quotas, balanced, seed + 1)
    grpo = balanced.loc[~balanced.audio_path.isin(sft.audio_path)].reset_index(drop=True)
    return sft, grpo, dict(**quota_meta, pool_fit=pool_fit, split_fit=split_fit)


def distribution_audit(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        for gender in ("F", "M"):
            group = frame.loc[frame.effective_gender == gender]
            variables = [("emotion", "emotion", EMOTIONS)] + [
                (cue, CUE_COLUMNS[cue], CUE_LABELS[cue]) for cue in CUES
            ]
            for variable, column, labels in variables:
                for label in labels:
                    count = int(group[column].eq(label).sum())
                    rows.append(
                        dict(
                            dataset=name,
                            gender=gender,
                            variable=variable,
                            column=column,
                            label=label,
                            count=count,
                            fraction=count / len(group),
                        )
                    )
    return pd.DataFrame(rows)


def validate_partitions(
    sft: pd.DataFrame,
    grpo: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    require_uniform_quotas: bool = False,
) -> None:
    source = train.set_index("audio_path")
    for name, frame in [("SFT", sft), ("GRPO", grpo)]:
        if frame.empty or frame.audio_path.duplicated().any():
            raise ValueError(f"{name}: empty or duplicate audio")
        if not set(frame.audio_path) <= set(source.index):
            raise ValueError(f"{name}: audio outside saved clean training source")
        if set(frame.audio_path) & set(val.audio_path):
            raise ValueError(f"{name}: validation overlap")
        missing = set(train.columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: lost preprocessing columns {missing}")
        try:
            pd.testing.assert_frame_equal(
                frame[train.columns].reset_index(drop=True),
                source.loc[frame.audio_path].reset_index()[train.columns],
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-10,
            )
        except AssertionError as exc:
            raise ValueError(f"{name}: cached metadata/labels differ from preprocessing") from exc
        counts = frame.groupby(["emotion", "effective_gender"]).size().reindex(
            pd.MultiIndex.from_product([EMOTIONS, ("F", "M")]), fill_value=0
        )
        if counts.min() == 0:
            raise ValueError(f"{name}: missing one or more emotion/gender strata: {counts.to_dict()}")
        if require_uniform_quotas and counts.nunique() != 1:
            raise ValueError(f"{name}: emotion/gender quotas are not balanced: {counts.to_dict()}")
    if set(sft.audio_path) & set(grpo.audio_path):
        raise ValueError("SFT/GRPO audio overlap")
    if len(sft) * 2 != len(grpo) * 3:
        raise ValueError("Expected an exact 60/40 partition")


def load_or_create_partitions(
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    split_dir: Path,
    train_csv: Path,
    quartiles_json: Path,
    data_dir: Path,
    drive_data_dir: Optional[Path] = None,
    seed: int = 42,
    prompt_version: str = PROMPT_VERSION,
    model_id: str = MODEL_ID,
    partition_mode: str = "semi_balanced",
    neutral_target_fraction: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    split_dir = Path(split_dir)
    sft_csv = split_dir / "sft_train.csv"
    grpo_csv = split_dir / "grpo_train.csv"
    split_meta = split_dir / "split_manifest.json"
    identity = dict(
        version=2 if partition_mode == "semi_balanced" else 1,
        seed=seed,
        sft_fraction=0.6,
        partition_mode=partition_mode,
        neutral_target_fraction=float(neutral_target_fraction) if partition_mode == "semi_balanced" else None,
        prompt_version=prompt_version,
        teacher_model=model_id,
        cues=list(CUES),
        cue_columns=dict(CUE_COLUMNS),
        train_sha256=hashlib.sha256(Path(train_csv).read_bytes()).hexdigest(),
        quartiles_sha256=hashlib.sha256(Path(quartiles_json).read_bytes()).hexdigest(),
    )
    require_uniform = partition_mode == "uniform"
    paths = [sft_csv, grpo_csv, split_meta]
    if any(p.exists() for p in paths):
        if not all(p.exists() for p in paths):
            raise ValueError("Incomplete saved split: recover its files or choose a new split_dir")
        meta = json.loads(split_meta.read_text(encoding="utf-8"))
        cached_identity = dict(meta["identity"])
        if cached_identity.get("prompt_version") == "gender_transcript_teacher_v1":
            cached_identity["prompt_version"] = prompt_version
        # Allow reading legacy V4 uniform split manifests that lacked partition_mode
        if partition_mode == "uniform" and "partition_mode" not in cached_identity:
            cached_identity["partition_mode"] = "uniform"
            cached_identity["neutral_target_fraction"] = None
        if cached_identity != identity:
            raise ValueError("Saved split provenance changed: choose a new split_dir")
        sft = load_manifest(sft_csv, "train", data_dir, require_transcripts=True)
        grpo = load_manifest(grpo_csv, "train", data_dir)
    else:
        sft, grpo, details = make_partitions(
            train, seed, partition_mode=partition_mode, neutral_target_fraction=neutral_target_fraction
        )
        validate_partitions(sft, grpo, train, val, require_uniform_quotas=require_uniform)
        split_dir.mkdir(parents=True, exist_ok=True)
        atomic_csv(sft, sft_csv, data_dir=data_dir, drive_data_dir=drive_data_dir)
        atomic_csv(grpo, grpo_csv, data_dir=data_dir, drive_data_dir=drive_data_dir)
        meta = dict(
            identity=identity,
            **details,
            source_rows=len(train),
            balanced_rows=len(sft) + len(grpo),
            sft_rows=len(sft),
            grpo_rows=len(grpo),
        )
        split_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    validate_partitions(sft, grpo, train, val, require_uniform_quotas=require_uniform)
    return sft, grpo, meta
