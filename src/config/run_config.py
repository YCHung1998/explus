"""Experiment-layer configuration that wraps the algorithm-layer PipelineConfig.

run.yaml is the single entry point for all runners. The `pipeline` block maps
1:1 to PipelineConfig.from_dict(); this class adds source / output / record.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml

from src.pipeline_config import PipelineConfig


@dataclass
class SourceConfig:
    type: str  # "dataset" | "video" | "camera"
    dataset: Optional[str] = None  # type == "dataset"
    video: Optional[str] = None    # type == "video"
    device: Optional[int] = None   # type == "camera"


@dataclass
class OutputConfig:
    dir: str


@dataclass
class RecordConfig:
    enabled: bool = False
    path: str = "records/{date}_states.csv"


@dataclass
class RunConfig:
    source: SourceConfig
    output: OutputConfig
    record: RecordConfig = field(default_factory=RecordConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        known_keys = {"source", "output", "record", "pipeline"}
        unknown = set(data) - known_keys
        if unknown:
            raise ValueError(f"Unrecognized run config keys: {sorted(unknown)}")
        source = SourceConfig(**data["source"])
        output = OutputConfig(**data["output"])
        record = RecordConfig(**data.get("record", {}))
        pipeline = PipelineConfig.from_dict(data.get("pipeline", {}))
        return cls(source=source, output=output, record=record,
                   pipeline=pipeline)

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def eval_paths(self) -> tuple[str, str]:
        """Derive (gt, pd) json paths from output.dir."""
        gt = os.path.join(self.output.dir, "ground_truth", "data.json")
        pd = os.path.join(self.output.dir, "predictions", "merge_data.json")
        return gt, pd
