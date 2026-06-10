import pytest

from src.config.run_config import RunConfig, SourceConfig, OutputConfig
from src.io.sources import Clip
from src.runtime.clip_runner import run_clip, ClipOpenError


def _cfg():
    return RunConfig(
        source=SourceConfig(type="video", video="x.mp4"),
        output=OutputConfig(dir="output/test"),
    )


def test_run_clip_raises_on_unopenable_source():
    clip = Clip(video_path="/no/such/file.mp4",
                capture_arg="/no/such/file.mp4")
    with pytest.raises(ClipOpenError):
        run_clip(clip, _cfg())


def test_run_clip_records_segments_when_recorder_enabled(monkeypatch, tmp_path):
    import src.runtime.clip_runner as cr

    # Stub out the heavy capture loop; return a fake pipeline with results.
    class FakePipeline:
        def finalize(self, frame_index=None):
            pass

        def get_results(self):
            return [{"segment": [1.0, 2.0], "label": 5, "score": 0.9}]

    def fake_process(cap, pipeline, **kwargs):
        return 0  # last frame index

    class _Cap:
        def isOpened(self):
            return True

    monkeypatch.setattr(cr, "_process_capture", fake_process)
    monkeypatch.setattr(cr, "_open_capture", lambda arg: _Cap())
    monkeypatch.setattr(cr, "DetectionPipeline",
                        lambda config: FakePipeline())

    from src.io.state_recorder import StateRecorder
    rec_path = str(tmp_path / "rec.csv")
    recorder = StateRecorder(rec_path, enabled=True)

    clip = Clip(video_path="/data/clipA.mp4", capture_arg="/data/clipA.mp4")
    run_clip(clip, _cfg(), recorder)

    from src.io.dataset_csv import read_dataset
    rows = read_dataset(recorder.path)
    assert len(rows) == 1
    assert rows[0].label == 5
    assert rows[0].split == "record"
