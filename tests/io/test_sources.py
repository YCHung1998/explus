import json

from src.config.run_config import SourceConfig
from src.io.sources import (
    Clip, VideoSource, CameraSource, DatasetSource, build_source,
)


def test_video_source_single_clip():
    src = VideoSource("a/b.mp4")
    clips = src.clips()
    assert clips == [Clip(video_path="a/b.mp4", capture_arg="a/b.mp4")]


def test_camera_source_single_clip():
    src = CameraSource(device=0)
    clips = src.clips()
    assert clips == [Clip(video_path=None, capture_arg=0)]


def test_dataset_source_lists_unique_enabled_videos(tmp_path):
    csv_path = tmp_path / "d.csv"
    csv_path.write_text(
        "video_path,start,end,label,split,enabled,boundary\n"
        "a/Clip_1.mp4,1.0,2.0,5,test,1,\n"
        "a/Clip_1.mp4,3.0,4.0,5,test,1,\n"
        "a/Clip_2.mp4,1.0,2.0,5,test,0,\n",
        encoding="utf-8",
    )
    src = DatasetSource(str(csv_path))
    clips = src.clips()
    assert [c.video_path for c in clips] == ["a/Clip_1.mp4"]
    assert clips[0].capture_arg == "a/Clip_1.mp4"


def test_build_source_dispatches_on_type(tmp_path):
    assert isinstance(
        build_source(SourceConfig(type="video", video="a.mp4")), VideoSource)
    assert isinstance(
        build_source(SourceConfig(type="camera", device=0)), CameraSource)

    csv_path = tmp_path / "d.csv"
    csv_path.write_text(
        "video_path,start,end,label,split,enabled,boundary\n"
        "a/Clip_1.mp4,1.0,2.0,5,test,1,\n", encoding="utf-8")
    assert isinstance(
        build_source(SourceConfig(type="dataset", dataset=str(csv_path))),
        DatasetSource)
