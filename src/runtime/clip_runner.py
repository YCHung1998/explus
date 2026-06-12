"""Single-clip runner: open one source, run DetectionPipeline, optionally record.

Shared by main.py (single source) and batch_infer.py (loop over dataset clips).
The capture loop lives here so both entrypoints stay thin.
"""

import copy
import json
import os
import time
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np

from src.config.run_config import RunConfig
from src.detection_pipeline import DetectionPipeline
from src.io.dataset_csv import video_id_from_path
from src.io.sources import Clip
from src.io.state_recorder import StateRecorder
from src.preprocessor import ImagePreprocessor
from src.snapshot_manager import SnapshotController
from src.visual import auto_image_collage_with_padding


DEFAULT_BOUNDARY = [0.01, 0.0, 0.99, 1.0]


class ClipOpenError(RuntimeError):
    """cv2 could not open the clip's capture source."""


def load_boundary(video_path: Optional[str],
                  default_boundary: Optional[List[float]] = None) -> List[float]:
    """Load boundary from the video's ActivityNet json, else default."""
    if default_boundary is None:
        default_boundary = DEFAULT_BOUNDARY
    if video_path is None:
        return default_boundary
    annotation_fn = video_path
    if annotation_fn.endswith("_mp4v.mp4"):
        annotation_fn = annotation_fn.replace(
            "_mp4v.mp4", "_annotations_ActivityNet.json")
    elif annotation_fn.endswith(".mp4"):
        annotation_fn = annotation_fn.replace(
            ".mp4", "_annotations_ActivityNet.json")
    if not os.path.isfile(annotation_fn):
        return default_boundary
    try:
        with open(annotation_fn, "r") as fp:
            data = json.load(fp)
        boundary = default_boundary
        for k, v in data.items():
            if k == "version":
                continue
            boundary = v.get("boundary", default_boundary)
            break
        return boundary
    except Exception as e:
        print(f"Error loading boundary: {e}")
        return default_boundary


def visualize_payload(controller: SnapshotController,
                      prep: ImagePreprocessor,
                      gray_frame: np.ndarray) -> List[np.ndarray]:
    """Generate visualization images for the current frame payload."""
    if not controller.payload:
        return []
    payload = controller.payload
    signal_vis_list = [payload.curr_meta.ori_img, payload.prev_meta.ori_img]
    if (payload.curr_meta.infer_result is not None
            and payload.prev_meta.infer_result is not None):
        diff_maps = []
        for curr_feat, prev_feat in zip(
                payload.curr_meta.infer_result["feature_maps"],
                payload.prev_meta.infer_result["feature_maps"]):
            px_curr_norm = np.linalg.norm(curr_feat, axis=0, keepdims=True)
            px_prev_norm = np.linalg.norm(prev_feat, axis=0, keepdims=True)
            sim_map = np.sum(
                (curr_feat / (px_curr_norm + 1e-6))
                * (prev_feat / (px_prev_norm + 1e-6)),
                axis=0, keepdims=False)
            diff_map = np.clip(1.0 - sim_map, 0, 2)
            diff_map[0, 0] = 0
            diff_map[0, 1] = 2
            norm_val = np.clip(diff_map * 255.0, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(norm_val, cv2.COLORMAP_JET)
            diff_maps.append(heatmap)
        resized_diffs = [
            cv2.resize(prep.minmax_normalize(feat), (640, 640),
                       interpolation=cv2.INTER_LINEAR)[140:500, :, :]
            for feat in diff_maps]
        signal_vis_list.extend(resized_diffs)
    return [
        cv2.resize(img, (gray_frame.shape[1], gray_frame.shape[0]),
                   interpolation=cv2.INTER_NEAREST)
        for img in signal_vis_list]


def _open_capture(capture_arg):
    """Wrapped for test seam."""
    return cv2.VideoCapture(capture_arg)


def _camera_video_id() -> str:
    return f"camera_{datetime.now().strftime('%Y%m%d')}"


def _record_results(pipeline, recorder: Optional[StateRecorder],
                    clip: Clip, boundary: Optional[list]) -> None:
    if recorder is None or not recorder.enabled:
        return
    if clip.video_path is None:
        video_id = _camera_video_id()
    else:
        video_id = video_id_from_path(clip.video_path)
    for seg in pipeline.get_results():
        segment = seg.get("segment", [])
        if len(segment) != 2:
            continue
        recorder.record_segment(
            video_id=video_id, start=float(segment[0]), end=float(segment[1]),
            label=int(seg.get("label", -1)), score=float(seg.get("score", 1.0)),
            boundary=boundary)


def _process_capture(cap, pipeline, *, frame_rate, source_is_file,
                     debug_visual, save_video, video_path) -> int:
    """Run the capture loop. Returns the last frame index processed."""
    if not source_is_file:
        sleep_time = 1.0 / frame_rate
        frame_skip_interval = 1
    else:
        sleep_time = 0
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip_interval = max(1, round(original_fps / frame_rate))
        print(f"Original FPS: {original_fps}, Skip Interval: {frame_skip_interval}")

    frame_index = -1
    paused = False
    video_writer = None
    output_name = "visual_output_v3.mp4"
    if save_video and video_path:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_name = f"{base}_vis_v3.mp4"

    while cap.isOpened():
        if not paused:
            time.sleep(sleep_time)
            ret = True
            frame = None
            for _ in range(frame_skip_interval):
                ret, frame = cap.read()
                if not ret:
                    break
            if not ret:
                break
            frame_index += 1
            results = pipeline.observe(frame, timestamp=time.time())
            vis_data = results["visualization_data"]
            gray = vis_data["gray"]
            score_map = vis_data["score_map"]
            valid_mask = vis_data["valid_mask"]
            is_motion = results["is_motion"]
            is_stable = results["is_stable"]
            signal_is_trigger = results["is_trigger"]

            if debug_visual or save_video:
                signal_vis_list = visualize_payload(
                    pipeline.controller, pipeline.prep, gray)
                image_list = [
                    cv2.resize(frame, (gray.shape[1], gray.shape[0])),
                    cv2.resize(pipeline.prep.clip_minmax_normalize(score_map, 0, 100),
                               (gray.shape[1], gray.shape[0]),
                               interpolation=cv2.INTER_NEAREST),
                    cv2.resize(pipeline.prep.minmax_normalize(valid_mask),
                               (gray.shape[1], gray.shape[0]),
                               interpolation=cv2.INTER_NEAREST),
                ]
                image_list = [np.repeat(img[:, :, None], 3, axis=2)
                              if img.ndim == 2 else img for img in image_list]
                image_list += signal_vis_list
                debug_view = auto_image_collage_with_padding(image_list)
                cv2.putText(debug_view, f"Frame: {frame_index:3d}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                status_text = (f"Motion: {'YES' if is_motion else 'NO'} | "
                               f"{'STABLE' if is_stable else 'UNSTABLE'} "
                               f"\n{'[TRIGGER]' if signal_is_trigger else ''}")
                cv2.putText(debug_view, status_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if save_video:
                    if video_writer is None:
                        slot_h, slot_w = gray.shape[:2]
                        video_writer = cv2.VideoWriter(
                            output_name, cv2.VideoWriter_fourcc(*'mp4v'),
                            frame_rate, (slot_w * 3, slot_h * 3))
                    canvas = np.zeros((gray.shape[0] * 3, gray.shape[1] * 3, 3),
                                      dtype=np.uint8)
                    h, w = debug_view.shape[:2]
                    canvas[:h, :w] = debug_view
                    video_writer.write(canvas)
                if debug_visual:
                    cv2.imshow("Detection Pipeline V3", debug_view)
                    key = cv2.waitKey(30) & 0xFF
                    if key == 27:
                        break
                    elif key == ord(" "):
                        paused = not paused
        else:
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            elif key == ord(" "):
                paused = not paused

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    return frame_index


def run_clip(clip: Clip, run_cfg: RunConfig,
             recorder: Optional[StateRecorder] = None, *,
             debug_visual: bool = False,
             save_video: bool = False) -> DetectionPipeline:
    """Process a single clip end-to-end. Raises ClipOpenError if unopenable."""
    config = copy.deepcopy(run_cfg.pipeline)
    boundary = load_boundary(clip.video_path, default_boundary=DEFAULT_BOUNDARY)
    print(f"Active Boundary: {boundary}")
    config.adaptive.boundary = boundary
    config.trigger.boundary = boundary

    pipeline = DetectionPipeline(config)

    cap = _open_capture(clip.capture_arg)
    source_is_file = clip.video_path is not None
    if cap is None or not cap.isOpened():
        raise ClipOpenError(f"Could not open source: {clip.capture_arg!r}")

    frame_index = _process_capture(
        cap, pipeline, frame_rate=config.signal.fps,
        source_is_file=source_is_file, debug_visual=debug_visual,
        save_video=save_video, video_path=clip.video_path)

    print("Processing complete.")
    pipeline.finalize(frame_index=frame_index)
    _record_results(pipeline, recorder, clip, boundary)
    return pipeline
