"""Dataset CSV: a derived, regenerable index over the ActivityNet GT json.

The VIDAT2-derived ActivityNet json is the source of truth; this CSV is a
one-way flattened view (one row per GT segment) used for selection / split /
enable, and as the shared schema for edge state recording.
"""

import csv
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import path_utils


CSV_FIELDS = [
    "video_path", "start", "end", "label", "split", "enabled", "boundary",
]

ACTIVITYNET_SUFFIX = "_annotations_ActivityNet.json"


@dataclass
class DatasetRow:
    video_path: str
    start: float
    end: float
    label: int
    split: str = "test"
    enabled: bool = True
    boundary: Optional[List[float]] = None  # [x1, y1, x2, y2]


def _boundary_to_str(boundary: Optional[List[float]]) -> str:
    if not boundary:
        return ""
    return ",".join(str(x) for x in boundary)


def _boundary_from_str(text: Optional[str]) -> Optional[List[float]]:
    text = (text or "").strip()
    if not text:
        return None
    return [float(x) for x in text.split(",")]


def _row_to_dict(row: DatasetRow) -> dict:
    return {
        "video_path": row.video_path,
        "start": row.start,
        "end": row.end,
        "label": row.label,
        "split": row.split,
        "enabled": int(row.enabled),
        "boundary": _boundary_to_str(row.boundary),
    }


def _row_from_dict(d: dict) -> DatasetRow:
    return DatasetRow(
        video_path=d["video_path"],
        start=float(d["start"]),
        end=float(d["end"]),
        label=int(d["label"]),
        split=(d.get("split") or "test"),
        enabled=bool(int(d.get("enabled", "1") or 1)),
        boundary=_boundary_from_str(d.get("boundary")),
    )


def write_dataset(rows: List[DatasetRow], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_dict(row))


def read_dataset(csv_path: str) -> List[DatasetRow]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [_row_from_dict(d) for d in csv.DictReader(f)]


def append_dataset(rows: List[DatasetRow], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_dict(row))


def video_id_from_path(video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    if stem.endswith("_mp4v"):
        stem = stem[: -len("_mp4v")]
    return stem


def unique_videos(
    rows: List[DatasetRow], enabled_only: bool = True
) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for row in rows:
        if enabled_only and not row.enabled:
            continue
        if row.video_path not in seen:
            seen.add(row.video_path)
            ordered.append(row.video_path)
    return ordered


def rows_to_gt_data(
    rows: List[DatasetRow], enabled_only: bool = True
) -> dict:
    """Group segment rows back into eval-compatible GT data.json structure.

    annotations (segment + label) is all eval_custom consumes; video_local_path
    and boundary are additionally emitted for eval_dashboard / visual_results.
    """
    data: dict = {}
    for row in rows:
        if enabled_only and not row.enabled:
            continue
        video_id = video_id_from_path(row.video_path)
        if video_id not in data:
            data[video_id] = {
                "annotations": [],
                "video_local_path": row.video_path,
                "boundary": (list(row.boundary)
                             if row.boundary is not None else None),
            }
        data[video_id]["annotations"].append(
            {"segment": [row.start, row.end], "label": row.label,
             "score": 1.0}
        )
    return data


def activitynet_dir_to_rows(src_dir: str) -> List[DatasetRow]:
    """Flatten every ActivityNet GT json under src_dir into segment rows.

    One row per annotation segment. video_path is derived from the json
    filename using the existing naming convention.
    """
    rows: List[DatasetRow] = []
    json_files = path_utils.get_files_recursive(
        src_dir, supported_extensions=ACTIVITYNET_SUFFIX
    )
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        video_path = json_file[: -len(ACTIVITYNET_SUFFIX)] + ".mp4"
        for key, info in data.items():
            if key == "version":
                continue
            boundary = info.get("boundary")
            for anno in info.get("annotations", []):
                segment = anno["segment"]
                rows.append(
                    DatasetRow(
                        video_path=video_path,
                        start=float(segment[0]),
                        end=float(segment[1]),
                        label=int(anno["label"]),
                        boundary=list(boundary) if boundary is not None else None,
                    )
                )
    return rows
