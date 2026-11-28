"""Acoustic feature extraction and gender-quartile classification."""

from src.acoustics.features import (
    extract_egemaps_cues,
    filter_training_set_by_duration_and_nonzero_cues,
)
from src.acoustics.quartiles import (
    apply_gender_quartile_classification,
    classify_value_by_quartile,
    compute_gender_acoustic_quartiles,
)

__all__ = [
    "extract_egemaps_cues",
    "filter_training_set_by_duration_and_nonzero_cues",
    "compute_gender_acoustic_quartiles",
    "classify_value_by_quartile",
    "apply_gender_quartile_classification",
]
