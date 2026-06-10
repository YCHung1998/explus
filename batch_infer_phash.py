"""Legacy phash batch line. Config-driven via run.yaml.

    python batch_infer_phash.py --config configs/run.yaml

Wrapped in main() + __main__ guard so importing this module no longer runs
inference or calls exit() (fixes #2). GT is derived from the dataset CSV,
consistent with batch_infer.py.
"""

import argparse
import json
import logging
import os

import cv2
from tqdm import tqdm

from src.config.run_config import RunConfig
from src.io.dataset_csv import read_dataset, rows_to_gt_data, video_id_from_path
from src.io.sources import build_source

MERGE_FILENAME = "merge_data.json"


def single_process_video(_fn, frame_history, constant):
    camera = cv2.VideoCapture(_fn)
    frame_rate = 10
    original_fps = camera.get(cv2.CAP_PROP_FPS)
    if frame_rate >= original_fps:
        frame_skip_interval = 1
        frame_step_time = 1 if original_fps < 1 else 1 / original_fps
    else:
        frame_skip_interval = max(1, round(original_fps / frame_rate))
        frame_step_time = 1.0 / frame_rate

    current_fake_timestamp = 0.0
    boundary = constant.BOUNDARY
    final_stop = False
    while True:
        for _ in range(frame_skip_interval):
            ret, frame = camera.read()
            if not ret:
                final_stop = True
                break
        if final_stop:
            break
        y1 = int(frame.shape[0] * boundary[0])
        y2 = int(frame.shape[0] * boundary[1])
        x1 = int(frame.shape[1] * boundary[2])
        x2 = int(frame.shape[1] * boundary[3])
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[y1:y2, x1:x2]
        frame_history.observe(
            rgb_frame,
            stable_threshold=constant.THRESHOLD,
            trigger_threshold=constant.TRIGGER_THRESHOLD,
            distance_lower_bound_threshold=constant.DISTANCE_LOWER_BOUND,
            distance_upper_bound_threshold=constant.DISTANCE_UPPER_BOUND,
            trigger_wait_time=constant.TRIGGER_WAIT_TIME,
            ema_distance_alpha=constant.EMA_ALPHA,
            checkout_region=[0, 0, -1, -1],
            export_external_j_feature=False,
            timestamp=current_fake_timestamp,
            is_monitor_unstable_interval=True,
        )
        current_fake_timestamp += frame_step_time
    frame_history.finalize()
    return frame_history


def run_batch(run_cfg: RunConfig, resume: bool = True) -> None:
    if run_cfg.source.type != "dataset":
        raise ValueError(
            "batch_infer_phash requires source.type == 'dataset', got "
            f"{run_cfg.source.type!r}")
    # Import here so module import has no side effects (fixes #2).
    from FrameHistory_phash import FrameHistory, Constant

    output_dir = run_cfg.output.dir
    predictions_dir = os.path.join(output_dir, "predictions")
    gt_dir = os.path.join(output_dir, "ground_truth")
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    rows = read_dataset(run_cfg.source.dataset)
    with open(os.path.join(gt_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(rows_to_gt_data(rows), f, ensure_ascii=False, indent=4)

    source = build_source(run_cfg.source)
    merge_data_results = {"results": {}}
    for clip in tqdm(source.clips()):
        _fn = clip.video_path
        video_id = video_id_from_path(_fn)
        save_path = os.path.join(predictions_dir, video_id + ".json")
        if resume and os.path.exists(save_path):
            with open(save_path, "r") as f:
                data = json.load(f)
            merge_data_results["results"].update(data["results"])
            continue
        frame_history = FrameHistory(
            alpha=Constant.EMA_ALPHA, history_size=300,
            threshold=Constant.THRESHOLD,
            trigger_threshold=Constant.TRIGGER_THRESHOLD,
            method=Constant.METHOD, log_level=logging.INFO)
        frame_history = single_process_video(_fn, frame_history, Constant)
        results = []
        for iv in frame_history.unstable_frame_intervals:
            results.append({
                "segment": iv.get("segment", []),
                "label": iv.get("label", -1),
                "score": iv.get("score", 0),
            })
        data = {"results": {video_id: results}}
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)
        merge_data_results["results"].update(data["results"])

    with open(os.path.join(predictions_dir, MERGE_FILENAME), "w") as fp:
        json.dump(merge_data_results, fp, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Legacy phash batch (config-driven)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args()
    run_cfg = RunConfig.from_yaml(args.config)
    run_batch(run_cfg, resume=args.resume)


if __name__ == "__main__":
    main()
