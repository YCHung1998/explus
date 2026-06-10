import json

import pytest

from src.config.run_config import RunConfig, SourceConfig, OutputConfig


def test_batch_infer_rejects_non_dataset_source(tmp_path):
    import batch_infer
    cfg = RunConfig(
        source=SourceConfig(type="camera", device=0),
        output=OutputConfig(dir=str(tmp_path / "out")),
    )
    with pytest.raises(ValueError):
        batch_infer.run_batch(cfg, resume=False)


def test_batch_infer_records_skipped_clips(tmp_path, monkeypatch):
    import batch_infer
    from src.io.dataset_csv import DatasetRow, write_dataset
    from src.runtime.clip_runner import ClipOpenError

    csv_path = str(tmp_path / "ds.csv")
    write_dataset([
        DatasetRow(video_path="/data/good.mp4", start=1.0, end=2.0, label=5),
        DatasetRow(video_path="/data/bad.mp4", start=1.0, end=2.0, label=5),
    ], csv_path)

    out_dir = str(tmp_path / "out")
    cfg = RunConfig(
        source=SourceConfig(type="dataset", dataset=csv_path),
        output=OutputConfig(dir=out_dir),
    )

    class FakePipeline:
        def get_results(self):
            return [{"segment": [1.0, 2.0], "label": 5, "score": 0.8}]

    def fake_run_clip(clip, run_cfg, recorder=None, **kw):
        if clip.video_path.endswith("bad.mp4"):
            raise ClipOpenError("boom")
        return FakePipeline()

    monkeypatch.setattr(batch_infer, "run_clip", fake_run_clip)

    batch_infer.run_batch(cfg, resume=False)

    with open(f"{out_dir}/skipped.json") as f:
        skipped = json.load(f)
    assert any("bad" in s["video_id"] for s in skipped)
    with open(f"{out_dir}/predictions/merge_data.json") as f:
        merged = json.load(f)
    assert "good" in merged["results"]
    # GT derived from CSV
    with open(f"{out_dir}/ground_truth/data.json") as f:
        gt = json.load(f)
    assert "good" in gt and "bad" in gt
