from src.io.dataset_csv import read_dataset
from src.io.state_recorder import StateRecorder


def test_disabled_recorder_writes_nothing(tmp_path):
    path = tmp_path / "rec.csv"
    rec = StateRecorder(str(path), enabled=False)
    rec.record_segment(video_id="cam0", start=0.0, end=1.0, label=5)
    rec.close()
    assert not path.exists()


def test_enabled_recorder_appends_rows(tmp_path):
    path = tmp_path / "rec.csv"
    rec = StateRecorder(str(path), enabled=True)
    rec.record_segment(video_id="cam0", start=0.0, end=1.0, label=5)
    rec.record_segment(video_id="cam0", start=2.0, end=3.0, label=5,
                       score=0.9)
    rec.close()

    rows = read_dataset(str(path))
    assert len(rows) == 2
    assert rows[0].video_path == "cam0"
    assert rows[0].start == 0.0 and rows[0].end == 1.0
    assert rows[0].label == 5
    assert rows[0].split == "record"


def test_date_placeholder_is_resolved(tmp_path):
    template = str(tmp_path / "{date}_states.csv")
    rec = StateRecorder(template, enabled=True)
    rec.record_segment(video_id="cam0", start=0.0, end=1.0, label=5)
    rec.close()
    assert "{date}" not in rec.path
    assert rec.path.endswith("_states.csv")
