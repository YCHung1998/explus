import json

from src.io.dataset_csv import (
    DatasetRow,
    activitynet_dir_to_rows,
    append_dataset,
    read_dataset,
    rows_to_gt_data,
    unique_videos,
    video_id_from_path,
    write_dataset,
)


def test_write_then_read_roundtrip(tmp_path):
    rows = [
        DatasetRow(video_path="a/100130.mp4", start=1.37, end=4.03, label=5,
                   split="test", enabled=True, boundary=[0.0, 0.0, 1.0, 1.0]),
        DatasetRow(video_path="a/100130.mp4", start=6.70, end=9.25, label=5,
                   split="test", enabled=False, boundary=None),
    ]
    csv_path = tmp_path / "d.csv"
    write_dataset(rows, str(csv_path))
    back = read_dataset(str(csv_path))

    assert back == rows
    assert back[0].boundary == [0.0, 0.0, 1.0, 1.0]
    assert back[1].boundary is None
    assert back[1].enabled is False


def test_append_writes_header_once(tmp_path):
    csv_path = tmp_path / "rec.csv"
    r1 = DatasetRow(video_path="cam0", start=0.0, end=1.0, label=5)
    r2 = DatasetRow(video_path="cam0", start=2.0, end=3.0, label=5)

    append_dataset([r1], str(csv_path))
    append_dataset([r2], str(csv_path))

    with open(csv_path, encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln]
    assert lines[0].startswith("video_path,")  # exactly one header
    assert sum(1 for ln in lines if ln.startswith("video_path,")) == 1
    assert len(read_dataset(str(csv_path))) == 2


def test_video_id_strips_extension_and_mp4v():
    assert video_id_from_path("x/y/Clip_100130.mp4") == "Clip_100130"
    assert video_id_from_path("x/y/Clip_100130_mp4v.mp4") == "Clip_100130"


def _write_activitynet(dir_path, video_id, segments, boundary=None):
    payload = {
        "version": "VIDAT-to-ActivityNet_v1.0",
        video_id: {
            "annotations": [
                {"segment": [s, e], "label": lbl, "score": 1.0}
                for (s, e, lbl) in segments
            ],
            "boundary": boundary if boundary is not None else [0, 0, 1, 1],
        },
    }
    fn = dir_path / f"{video_id}_annotations_ActivityNet.json"
    fn.write_text(json.dumps(payload), encoding="utf-8")
    return fn


def test_activitynet_dir_to_rows(tmp_path):
    _write_activitynet(tmp_path, "Clip_100130",
                       [(1.37, 4.03, 5), (6.70, 9.25, 5)],
                       boundary=[0, 0, 1, 1])

    rows = activitynet_dir_to_rows(str(tmp_path))

    assert len(rows) == 2
    assert rows[0].video_path.endswith("Clip_100130.mp4")
    assert rows[0].start == 1.37 and rows[0].end == 4.03
    assert rows[0].label == 5
    assert rows[0].boundary == [0, 0, 1, 1]
    # "version" key must be skipped, not treated as a video
    assert all(r.video_path.endswith("Clip_100130.mp4") for r in rows)


def _rows():
    return [
        DatasetRow(video_path="a/Clip_1.mp4", start=1.0, end=2.0, label=5,
                   enabled=True),
        DatasetRow(video_path="a/Clip_1.mp4", start=3.0, end=4.0, label=5,
                   enabled=True),
        DatasetRow(video_path="a/Clip_2.mp4", start=1.0, end=2.0, label=5,
                   enabled=False),
    ]


def test_unique_videos_dedups_and_respects_enabled():
    rows = _rows()
    assert unique_videos(rows) == ["a/Clip_1.mp4"]
    assert unique_videos(rows, enabled_only=False) == [
        "a/Clip_1.mp4", "a/Clip_2.mp4",
    ]


def test_rows_to_gt_data_is_eval_compatible():
    data = rows_to_gt_data(_rows())
    assert set(data.keys()) == {"Clip_1"}  # disabled Clip_2 excluded
    annos = data["Clip_1"]["annotations"]
    assert annos == [
        {"segment": [1.0, 2.0], "label": 5, "score": 1.0},
        {"segment": [3.0, 4.0], "label": 5, "score": 1.0},
    ]


def test_rows_to_gt_data_includes_video_meta():
    rows = [
        DatasetRow(video_path="/data/clipA.mp4", start=1.0, end=2.0, label=5,
                   split="test", enabled=True, boundary=[0.0, 0.0, 1.0, 1.0]),
        DatasetRow(video_path="/data/clipA.mp4", start=3.0, end=4.0, label=5,
                   split="test", enabled=True, boundary=[0.0, 0.0, 1.0, 1.0]),
    ]
    data = rows_to_gt_data(rows)
    info = data["clipA"]
    # eval-compatible annotations 不變
    assert len(info["annotations"]) == 2
    assert info["annotations"][0]["segment"] == [1.0, 2.0]
    assert info["annotations"][0]["label"] == 5
    # 新增的 dashboard/visual 欄位
    assert info["video_local_path"] == "/data/clipA.mp4"
    assert info["boundary"] == [0.0, 0.0, 1.0, 1.0]
