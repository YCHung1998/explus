"""Append-only recorder for live detection states (edge data collection).

Default OFF. When enabled, each finalized segment is appended in the dataset
CSV schema (split="record") so recorded states round-trip into future
evaluation datasets. No frame pixels are stored.
"""

from datetime import datetime
from typing import Optional

from src.io.dataset_csv import DatasetRow, append_dataset


class StateRecorder:
    def __init__(self, path: str, enabled: bool = False):
        self.enabled = enabled
        self.path = path.replace("{date}", datetime.now().strftime("%Y%m%d"))

    def record_segment(
        self,
        video_id: str,
        start: float,
        end: float,
        label: int,
        score: float = 1.0,
        boundary: Optional[list] = None,
    ) -> None:
        if not self.enabled:
            return
        # NOTE: `score` is accepted for API symmetry with detection events but
        # is not persisted -- the dataset CSV schema (DatasetRow) has no score
        # column. Wire score into the schema in a later phase if needed.
        row = DatasetRow(
            video_path=video_id,
            start=start,
            end=end,
            label=label,
            split="record",
            enabled=True,
            boundary=boundary,
        )
        append_dataset([row], self.path)

    def close(self) -> None:
        # Append-only: nothing buffered. Present for a symmetric lifecycle.
        return
