import os

import pytest

from src.config.run_config import RunConfig
from src.pipeline_config import PipelineConfig


def test_from_dict_builds_layers():
    cfg = RunConfig.from_dict({
        "source": {"type": "dataset", "dataset": "datasets/bread.csv"},
        "output": {"dir": "output/exp1"},
        "record": {"enabled": True, "path": "records/x.csv"},
        "pipeline": {"signal": {"fps": 15}, "trigger": {"mode": "ema_phash"}},
    })

    assert cfg.source.type == "dataset"
    assert cfg.source.dataset == "datasets/bread.csv"
    assert cfg.output.dir == "output/exp1"
    assert cfg.record.enabled is True
    assert isinstance(cfg.pipeline, PipelineConfig)
    assert cfg.pipeline.signal.fps == 15
    assert cfg.pipeline.trigger.mode == "ema_phash"


def test_defaults_when_sections_absent():
    cfg = RunConfig.from_dict({
        "source": {"type": "camera", "device": 0},
        "output": {"dir": "output/edge"},
    })
    assert cfg.source.device == 0
    assert cfg.record.enabled is False           # default off
    assert isinstance(cfg.pipeline, PipelineConfig)
    assert cfg.pipeline.signal.fps == 10.0        # PipelineConfig default


def test_from_yaml(tmp_path):
    yaml_text = (
        "source:\n"
        "  type: video\n"
        "  video: a/b.mp4\n"
        "output:\n"
        "  dir: output/exp2\n"
        "pipeline:\n"
        "  signal:\n"
        "    fps: 12\n"
    )
    p = tmp_path / "run.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    cfg = RunConfig.from_yaml(str(p))
    assert cfg.source.type == "video"
    assert cfg.source.video == "a/b.mp4"
    assert cfg.pipeline.signal.fps == 12


def test_example_yaml_loads():
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    path = os.path.join(here, "configs", "run.example.yaml")
    cfg = RunConfig.from_yaml(path)
    assert cfg.source.type in {"dataset", "video", "camera"}
    assert cfg.output.dir
    assert cfg.record.enabled is False  # default reference keeps recording off
    assert isinstance(cfg.pipeline, PipelineConfig)


def test_unknown_top_level_key_raises():
    with pytest.raises(ValueError, match="Unrecognized run config keys"):
        RunConfig.from_dict({
            "source": {"type": "camera", "device": 0},
            "output": {"dir": "output/x"},
            "logging": {"level": "debug"},   # not a known section
        })


def test_eval_paths_derives_from_output_dir():
    from src.config.run_config import (
        RunConfig, SourceConfig, OutputConfig,
    )
    cfg = RunConfig(
        source=SourceConfig(type="dataset", dataset="d.csv"),
        output=OutputConfig(dir="output/exp1"),
    )
    gt, pd = cfg.eval_paths()
    assert gt == "output/exp1/ground_truth/data.json"
    assert pd == "output/exp1/predictions/merge_data.json"


def test_trigger_model_path_overridable_via_config():
    from src.pipeline_config import PipelineConfig
    cfg = PipelineConfig.from_dict(
        {"trigger": {"model_path": "/vol/08822801/AutoTrigger/model/custom.onnx"}})
    assert cfg.trigger.model_path == "/vol/08822801/AutoTrigger/model/custom.onnx"
    assert cfg.trigger.resolved_model_path == "/vol/08822801/AutoTrigger/model/custom.onnx"


def test_trigger_model_path_falls_back_to_feature_position():
    from src.pipeline_config import PipelineConfig
    neck = PipelineConfig.from_dict({"trigger": {"feature_position": "Neck"}})
    assert neck.trigger.model_path is None
    assert neck.trigger.resolved_model_path == "./models/best_vis_with_8400_3.onnx"
    back = PipelineConfig.from_dict({"trigger": {"feature_position": "Backbone"}})
    assert back.trigger.resolved_model_path == "./models/best_vis_with_8400_b3.onnx"
