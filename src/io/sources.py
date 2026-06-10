"""Input-source abstraction: dataset / video / camera all expose clips().

A Clip carries the argument to feed cv2.VideoCapture() in Phase B (a path for
files, an int device id for cameras). Phase A only enumerates clips; it never
opens a capture, so these classes are pure and unit-testable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

from src.config.run_config import SourceConfig
from src.io.dataset_csv import read_dataset, unique_videos


@dataclass
class Clip:
    video_path: Optional[str]          # None for a live camera
    capture_arg: Union[str, int]       # passed to cv2.VideoCapture(...)


class SourceProvider(ABC):
    @abstractmethod
    def clips(self) -> List[Clip]:
        ...


class VideoSource(SourceProvider):
    def __init__(self, video_path: str):
        self.video_path = video_path

    def clips(self) -> List[Clip]:
        return [Clip(video_path=self.video_path,
                     capture_arg=self.video_path)]


class CameraSource(SourceProvider):
    def __init__(self, device: int = 0):
        self.device = device

    def clips(self) -> List[Clip]:
        return [Clip(video_path=None, capture_arg=self.device)]


class DatasetSource(SourceProvider):
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def clips(self) -> List[Clip]:
        rows = read_dataset(self.csv_path)
        return [Clip(video_path=v, capture_arg=v)
                for v in unique_videos(rows)]


def build_source(source: SourceConfig) -> SourceProvider:
    if source.type == "video":
        if not source.video:
            raise ValueError("source.video is required when type == 'video'")
        return VideoSource(source.video)
    if source.type == "camera":
        return CameraSource(device=source.device or 0)
    if source.type == "dataset":
        if not source.dataset:
            raise ValueError(
                "source.dataset is required when type == 'dataset'")
        return DatasetSource(source.dataset)
    raise ValueError(f"Unknown source.type: {source.type!r}")
