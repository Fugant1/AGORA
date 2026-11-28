"""Lexical speaker gender classification (F/M) and training split filtering (exact match to data_processing.ipynb)."""

from typing import Dict, List, Optional

import pandas as pd

KNOWN_SPEAKERS_GENDER: Dict[str, str] = {
    # Protagonists (~83% of MELD)
    "Rachel": "F", "Monica": "F", "Phoebe": "F",
    "Joey": "M", "Chandler": "M", "Ross": "M",
    # Recurring secondary (Female)
    "Janice": "F", "Carol": "F", "Susan": "F", "Judy": "F", "Emily": "F",
    "Ursula": "F", "Mona": "F", "Charlie": "F", "Amy": "F", "Jill": "F",
    "Mindy": "F", "Estelle": "F", "Kathy": "F", "Elizabeth": "F", "Nora": "F",
    "Julie": "F", "Alice": "F", "Joanna": "F", "Lydia": "F", "Caitlin": "F",
    "Bonnie": "F", "Erin": "F", "Chloe": "F", "Dina": "F", "Kim": "F",
    "Leslie": "F", "ginger": "F", "Ginger": "F", "Aurora": "F", "Kristen": "F",
    # Recurring secondary (Male)
    "Gunther": "M", "Richard": "M", "Mike": "M", "Jack": "M", "Ben": "M",
    "Pete": "M", "David": "M", "Paul": "M", "Frank": "M", "Tag": "M",
    "Joshua": "M", "Gary": "M", "Paolo": "M", "Eddie": "M", "Mark": "M",
    "Barry": "M", "Treeger": "M", "Gavin": "M", "Danny": "M", "Will": "M",
    "Leonard": "M", "parker": "M", "Parker": "M", "Steve": "M", "stu": "M",
    "Stu": "M", "Doug": "M", "Terry": "M", "Allesandro": "M", "Wayne": "M",
}

FEMALE_KEYWORDS: List[str] = [
    "mrs.", "ms.", "miss", "woman", "girl", "waitress", "hostess",
    "nurse", "mother", "mom", "grandmother", "sister", "lady",
    "stewardess", "actress", "bride", "her", "she",
]

MALE_KEYWORDS: List[str] = [
    "mr.", "man", "guy", "boy", "waiter", "policeman", "fireman",
    "father", "dad", "grandfather", "brother", "gentleman", "actor",
    "groom", "bellhop", "doorman",
]


def get_lexical_gender(speaker_name: str) -> Optional[str]:
    """Resolve speaker gender ('F'/'M') via protagonist/secondary lookup or explicit lexical role."""
    spk_clean = str(speaker_name).strip()
    if spk_clean in KNOWN_SPEAKERS_GENDER:
        return KNOWN_SPEAKERS_GENDER[spk_clean]

    spk_lower = spk_clean.lower()
    for kw in FEMALE_KEYWORDS:
        if kw in spk_lower.split() or spk_lower.startswith(kw):
            return "F"
    for kw in MALE_KEYWORDS:
        if kw in spk_lower.split() or spk_lower.startswith(kw):
            return "M"
    return None


def classify_and_filter_gender(
    df: pd.DataFrame,
    split_name: str,
    drop_unknown: bool = False,
) -> pd.DataFrame:
    """Add 'gender' column ('F'/'M'/None) via get_lexical_gender and optionally drop unresolved rows."""
    out = df.copy()
    out["gender"] = out["speaker"].map(get_lexical_gender)
    n_total = len(out)
    n_f = int((out["gender"] == "F").sum())
    n_m = int((out["gender"] == "M").sum())
    n_unk = int(out["gender"].isna().sum())

    if drop_unknown:
        out = out.loc[out["gender"].isin(["F", "M"])].reset_index(drop=True)
        print(
            f"[{split_name}] Gender classification: F={n_f:,}, M={n_m:,} | "
            f"Removed unresolved: {n_unk:,}/{n_total:,} -> Retained: {len(out):,} "
            f"({100.0 * len(out) / max(1, n_total):.2f}%)"
        )
    else:
        print(
            f"[{split_name}] Gender classification: F={n_f:,}, M={n_m:,}, "
            f"Unresolved={n_unk:,}/{n_total:,} (kept for eval)"
        )
    return out
