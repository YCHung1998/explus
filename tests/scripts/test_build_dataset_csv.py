import json

from scripts.build_dataset_csv import build_dataset_csv
from src.io.dataset_csv import read_dataset


def test_build_dataset_csv_end_to_end(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    payload = {
        "version": "VIDAT-to-ActivityNet_v1.0",
        "Clip_1": {
            "annotations": [{"segment": [1.0, 2.0], "label": 5, "score": 1.0}],
            "boundary": [0, 0, 1, 1],
        },
    }
    (src / "Clip_1_annotations_ActivityNet.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    out = tmp_path / "out" / "dataset.csv"

    count = build_dataset_csv(str(src), str(out))

    assert count == 1
    rows = read_dataset(str(out))
    assert len(rows) == 1
    assert rows[0].video_path.endswith("Clip_1.mp4")
    assert rows[0].label == 5
