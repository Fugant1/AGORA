"""PyTorch SFT Dataset and HuggingFace GRPO Dataset builders (exact match to V4 notebooks)."""

from typing import Any, Dict, List

import pandas as pd

from src.config import CUE_COLUMNS, CUES
from src.prompts.templates import student_instruction


class SERDataset:
    """Lightweight record-list dataset for CueWeightedSFTTrainer (matches ser_sft.ipynb)."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.records: List[Dict[str, Any]] = df.to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


def prepare_grpo_dataset(df: pd.DataFrame) -> Any:
    """Convert grpo_train.csv into a TRL GRPO Dataset preserving per-row gender prompts and 4 cue targets."""
    from datasets import Dataset

    records = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        rec = {
            "prompt": student_instruction(row_dict),
            "label": str(row_dict["emotion"]).lower().strip(),
            "audio_path": str(row_dict["audio_path"]),
            "speaker": str(row_dict.get("speaker", "")),
            "effective_gender": str(row_dict["effective_gender"]),
            "gender": str(row_dict.get("gender", "")),
        }
        for cue in CUES:
            col = CUE_COLUMNS[cue]
            rec[col] = str(row_dict[col]).lower().strip()
        records.append(rec)
    return Dataset.from_list(records)
