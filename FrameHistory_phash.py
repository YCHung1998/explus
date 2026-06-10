# pylint: disable=missing-docstring
import math
import time
import logging

import altair as alt
import cv2
import numpy as np
import pandas as pd
import scipy.fftpack
import streamlit as st

from constants import LABEL_NAME2ID, LabelMap


class Constant:
    SOURCE = "Selected Video"  # 'Camera', 'Selected Image', 'Selected Video'
    METHOD = "phash"  # 'yolo12n_40x40', 'ssd_mobilenet_v2'
    THRESHOLD = 1
    TRIGGER_THRESHOLD = 1.1

    DISTANCE_LOWER_BOUND = 5.0
    DISTANCE_UPPER_BOUND = 50.0
    TRIGGER_WAIT_TIME = 0.1  # seconds
    EMA_ALPHA = 0.33
    # [y_top, y_bottom, x_left, x_right], None means full frame
    # BOUNDARY = [0.4, 0.9, 0.1, 0.9]  # checkout region
    # BOUNDARY = [0.0, 1.0, 0.0, 1.0]  # checkout region
    BOUNDARY = [0.0, 0.85, 0.2, 0.75]  # checkout region bdry
    # BOUNDARY = [0.0, 1.0, 0.2, 0.85]  # checkout region bdry_v2

    TRIGGER_REGION_SCALE = 1  # 1.0 means same as checkout


def setup_logger(log_level=logging.INFO, log_file="application.log"):
    logger = logging.getLogger("MyAppLogger")
    logger.setLevel(log_level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


@st.cache_resource
def get_frame_history(ema_alpha, history_size, threshold, trigger_threshold, method):
    # Load ONNX model for yolo12n method
    if method == "yolo12n_40x40":
        import onnxruntime as ort

        interpreter = ort.InferenceSession("models/best_vis_with_feature_map.onnx")
    elif method.startswith("ssd_conv"):
        import tensorflow as tf

        model_path = "models/ssd_mobilenet_v2_oid_v4_300x300_full_integer_quant.tflite"
        interpreter = tf.lite.Interpreter(
            model_path, experimental_preserve_all_tensors=True
        )
        interpreter.allocate_tensors()
    elif method == "mobilenet_7_7_1280_relu6":
        import tensorflow as tf

        model_path = "models/mobilenet_v2_224_dm10_full_integer_quant.tflite"
        interpreter = tf.lite.Interpreter(
            model_path, experimental_preserve_all_tensors=True
        )
        interpreter.allocate_tensors()
    else:
        interpreter = None
    return FrameHistory(
        ema_alpha, history_size, threshold, trigger_threshold, method, interpreter
    )


def sigmoid(x):
    return 1 / (1 + 1 / (np.exp(-1 * x + 1e-7)))


class FrameHistory:

    def __init__(
        self,
        alpha: float,
        history_size: int,
        threshold: float,
        trigger_threshold: float,
        method: str,
        interpreter = None,
        distance_lower_bound_threshold: float = 0.4,
        distance_upper_bound_threshold: float = 0.8,
        # logger: logging.Logger = logging.getLogger("FrameHistory"),
        log_level=logging.ERROR,
    ):
        # self.logger = setup_logger(
        #     log_level=log_level, log_file="frame_history_output.log"
        # )
        # self.logger.info("====Initializing FrameHistory...====")

        self._alpha = alpha
        self._history_size = history_size
        self._threshold = threshold
        self._trigger_threshold = trigger_threshold
        self._method = method
        if interpreter:
            self._interpreter = interpreter
        # Distances plateau region (for jitter suppression)
        self._distance_lower_bound_threshold = distance_lower_bound_threshold
        self._distance_upper_bound_threshold = distance_upper_bound_threshold

        # Fixed Parameters
        self._target_fps = 10
        self._ema_dct_mean_alpha = 0.1
        self._ema_phash_alpha = 0.33
        # In seconds
        self._trigger_wait_time = 0.1  # 100ms
        self._trigger_timeout_duration = 0.8  # 800ms

        # frame start index
        self.frame_index = -1  # start from 0

        # State
        self._distances = [0.0 for _ in range(history_size)]
        self._ema_distances = [0.0 for _ in range(history_size)]
        self._stable = True
        self._unstable_events = [0 for _ in range(history_size)]
        self._trigger = False
        self._trigger_events = [0 for _ in range(history_size)]
        self._previous_stable = True
        self._previous_phash = None
        self._previous_stable_phash = None
        self._previous_stable_ema_phash = None
        self._stable_ema_phash = None
        self._unstable_to_stable_timestamp = None
        self._stable_timestamp = None
        self._previous_dct = None
        self._previous_timestamp = None
        self.stable_distance = 0
        self.ema_dct_mean = None
        self._previous_ema_dct_mean = None
        self.dct_medians = [0.0 for _ in range(history_size)]

        self._is_timing = False
        self._unstable_intervals = []

    @property
    def stable(self):
        return self._stable

    @property
    def trigger(self):
        return self._trigger

    @property
    def distances(self):
        return self._distances

    @property
    def ema_distances(self):
        return self._ema_distances

    @property
    def unstable_events(self):
        return self._unstable_events

    @property
    def trigger_events(self):
        return self._trigger_events

    @property
    def unstable_frame_intervals(self):
        return self._unstable_intervals

    def observe(
        self,
        frame,
        stable_threshold: float = None,
        trigger_threshold: float = None,
        distance_lower_bound_threshold: float = None,
        distance_upper_bound_threshold: float = None,
        trigger_wait_time: float = None,
        ema_distance_alpha: float = None,
        checkout_region: np.ndarray[int] = [0, 0, -1, -1],  # xyxy int
        export_external_j_feature: bool = False,  # not used
        is_monitor_unstable_interval: bool = False,
        timestamp: float = None,  # (second)
    ) -> tuple:
        self.frame_index += 1  # index start from 0

        # Update parameters if provided
        if stable_threshold is not None:
            self._threshold = stable_threshold
        if trigger_threshold is not None:
            self._trigger_threshold = trigger_threshold
        if distance_lower_bound_threshold is not None:
            self._distance_lower_bound_threshold = distance_lower_bound_threshold
        if distance_upper_bound_threshold is not None:
            self._distance_upper_bound_threshold = distance_upper_bound_threshold
        if trigger_wait_time is not None:
            self._trigger_wait_time = trigger_wait_time
        if ema_distance_alpha is not None:
            self._alpha = ema_distance_alpha

        time_partition = 1 / self._target_fps
        current_time_part = math.floor(timestamp / time_partition)
        if self._previous_timestamp is None:
            self._previous_timestamp = timestamp - time_partition
            previous_time_part = current_time_part - 1
        else:
            previous_time_part = math.floor(self._previous_timestamp / time_partition)

        # Not change any state if the current time partition
        # is the same as the previous time partition
        if current_time_part == previous_time_part:
            return self._previous_dct, self._previous_phash
        missed_time_part = current_time_part - previous_time_part

        if self._method == "phash":
            dct = self._compute_dct(frame)
            if self._previous_ema_dct_mean is not None:
                self.ema_dct_mean = np.around(
                    self._ema_dct_mean_alpha * np.mean(dct)
                    + (1 - self._ema_dct_mean_alpha) * self._previous_ema_dct_mean
                ).astype(int)
            else:
                self.ema_dct_mean = np.mean(dct)
            phash = self._compute_j_hash(dct).astype(np.float32)
            if self._previous_phash is None:
                self._previous_phash = phash
            if self._stable_ema_phash is None:
                self._stable_ema_phash = phash
            distance = self._compute_log_l2_distance(phash, self._previous_phash)
            # Add an interpolation frame if the current time partition is larger than the previous one
            if missed_time_part > 1:

                # XXX DEBUG
                print("Enter missed_time_part here")

                distance_partition = (distance - self._distances[-1]) / missed_time_part
                for _ in range(missed_time_part - 1):
                    missed_distance = self._distances[-1] + distance_partition
                    ema_distance = (
                        self._alpha * missed_distance
                        + (1 - self._alpha) * self._ema_distances[-1]
                    )
                    self._distances.append(missed_distance)
                    self._ema_distances.append(ema_distance)
                    self._unstable_events.append(int(not self._previous_stable))
                    self._trigger_events.append(False)
            ema_distance = (
                self._alpha * distance + (1 - self._alpha) * self._ema_distances[-1]
            )
            self._stable = ema_distance < self._threshold

            # Update stable EMA phash when stable
            if self._stable:
                self._stable_ema_phash = self._compute_matrices_ema(
                    phash, self._stable_ema_phash, self._ema_phash_alpha
                )

            # Stable to Unstable
            if self._previous_stable and not self._stable:
                if self._unstable_to_stable_timestamp is None:
                    self._previous_stable_ema_phash = self._stable_ema_phash
                self._stable_ema_phash = None
                self._unstable_to_stable_timestamp = None
                self._stable_timestamp = None

            # Unstable to Stable
            if not self._previous_stable and self._stable:
                self._unstable_to_stable_timestamp = time.time()

            # Update unstable interval (Low-Level record)
            if is_monitor_unstable_interval:
                self.monitor_unstable_intervals()

            # Start check trigger
            self._trigger = False
            if (
                self._unstable_to_stable_timestamp is not None
                and self._stable_ema_phash is not None
                and self._previous_stable_ema_phash is not None
            ):

                # Check if distance drops below lower bound
                if (
                    self._stable_timestamp is None
                    and ema_distance < self._distance_lower_bound_threshold
                ):
                    self._stable_timestamp = time.time()

                if self._stable_timestamp is not None:
                    stable_time = time.time() - self._stable_timestamp

                    if stable_time > self._trigger_timeout_duration:
                        # Timeout
                        self._previous_stable_ema_phash = None
                        self._unstable_to_stable_timestamp = None
                        self._stable_timestamp = None
                        print("DEBUG: Trigger timeout")
                    elif stable_time > self._trigger_wait_time:
                        # Check trigger condition
                        self.stable_distance = self._compute_log_l2_distance(
                            self._stable_ema_phash.flatten(),
                            self._previous_stable_ema_phash.flatten(),
                        )
                        self._trigger = self.stable_distance >= self._trigger_threshold
                        print(f"DEBUG: StableDistance: {self.stable_distance}")

                        if self._trigger:
                            self._previous_stable_ema_phash = None
                            self._unstable_to_stable_timestamp = None
                            self._stable_timestamp = None
                            print("DEBUG: !!!!Triggered!!!!")
                            # update trigger sign (Mid Level record update) if is trigger
                            print("✅-✅ Immediate Trigger")
                            self._unstable_intervals[-1]["label"] = \
                                LABEL_NAME2ID[LabelMap.TRIGGER]

        elif self._method == "yolo12n_40x40":
            # YOLO12n P4 feature map (40x40x128) from neck
            # Layer name: /model/model.11/cv2/act/Mul_output_0
            # Get the feature map output (second output): [1, 128, 40, 40]
            feature_map = self.invoke_onnx(frame)
            # Average across channels to get spatial representation: [40, 40]
            spatial_feature = np.mean(feature_map, axis=1).squeeze(0)
            # Use DCT on averaged spatial features for better frequency domain representation
            dct = self._compute_mean_dct(spatial_feature, hash_size=20)
            if self._previous_ema_dct_mean is not None:
                self.ema_dct_mean = np.around(
                    self._ema_dct_mean_alpha * np.mean(dct)
                    + (1 - self._ema_dct_mean_alpha) * self._previous_ema_dct_mean
                ).astype(int)
            else:
                self.ema_dct_mean = np.mean(dct)
            # Compute hash for comparison
            phash = self._compute_j_hash(dct).astype(np.float32)
            if self._previous_phash is None:
                self._previous_phash = phash
            if self._stable_ema_phash is None:
                self._stable_ema_phash = phash
            distance = self._compute_l2_distance(phash, self._previous_phash)
            # Add an interpolation frame if the current time partition is larger than the previous one
            if missed_time_part > 1:
                distance_partition = (distance - self._distances[-1]) / missed_time_part
                for _ in range(missed_time_part - 1):
                    missed_distance = self._distances[-1] + distance_partition
                    ema_distance = (
                        self._alpha * missed_distance
                        + (1 - self._alpha) * self._ema_distances[-1]
                    )
                    self._distances.append(missed_distance)
                    self._ema_distances.append(ema_distance)
                    self._unstable_events.append(int(not self._previous_stable))
                    self._trigger_events.append(False)
            ema_distance = (
                self._alpha * distance + (1 - self._alpha) * self._ema_distances[-1]
            )
            self._stable = ema_distance < self._threshold

            # Update stable EMA phash when stable
            if self._stable:
                self._stable_ema_phash = self._compute_matrices_ema(
                    phash, self._stable_ema_phash, self._ema_phash_alpha
                )

            # Stable to Unstable
            if self._previous_stable and not self._stable:
                if self._unstable_to_stable_timestamp is None:
                    self._previous_stable_ema_phash = self._stable_ema_phash
                self._stable_ema_phash = None
                self._unstable_to_stable_timestamp = None
                self._stable_timestamp = None

            # Unstable to Stable
            if not self._previous_stable and self._stable:
                # self._unstable_to_stable_timestamp = time.time()
                self._unstable_to_stable_timestamp = timestamp

            # Update unstable interval (Low-Level record)
            if is_monitor_unstable_interval:
                self.monitor_unstable_intervals()

            # Start check trigger
            self._trigger = False
            if (
                self._unstable_to_stable_timestamp is not None
                and self._stable_ema_phash is not None
                and self._previous_stable_ema_phash is not None
            ):

                # Check if distance drops below lower bound
                if (
                    self._stable_timestamp is None
                    and ema_distance < self._distance_lower_bound_threshold
                ):
                    # self._stable_timestamp = time.time()
                    self._stable_timestamp = timestamp

                if self._stable_timestamp is not None:
                    # stable_time = time.time() - self._stable_timestamp
                    stable_time_duration = timestamp - self._stable_timestamp

                    print("stable_time_duration > self._trigger_timeout_duration")
                    print(stable_time_duration, self._trigger_timeout_duration)

                    if stable_time_duration > self._trigger_timeout_duration:
                        # Timeout
                        self._previous_stable_ema_phash = None
                        self._unstable_to_stable_timestamp = None
                        self._stable_timestamp = None
                        self.logger.debug("DEBUG: Trigger timeout")
                    elif stable_time_duration > self._trigger_wait_time:
                        # Check trigger condition
                        self.stable_distance = self._compute_l2_distance(
                            self._stable_ema_phash.flatten(),
                            self._previous_stable_ema_phash.flatten(),
                        )
                        self._trigger = self.stable_distance >= self._trigger_threshold
                        self.logger.debug(
                            f"DEBUG: StableDistance: {self.stable_distance}"
                        )
                        if self._trigger:
                            self._previous_stable_ema_phash = None
                            self._unstable_to_stable_timestamp = None
                            self._stable_timestamp = None
                            self.logger.debug("DEBUG: !!!!Triggered!!!!")
                            self._unstable_intervals[-1]["label"] = \
                                LABEL_NAME2ID[LabelMap.TRIGGER]
                            print("✅-✅ Immediate Trigger")

        else:
            raise ValueError(f"Method {self._method} not supported.")


        # update trigger sign (Mid Level record update) if is trigger
        if self._trigger:
            # update unstable intrevals
            self._unstable_intervals[-1]["label"] = \
                LABEL_NAME2ID[LabelMap.TRIGGER]
            print("✅-✅ Immediate Trigger")

        self._previous_phash = phash
        self._previous_dct = dct
        self._previous_timestamp = None
        self._previous_stable = self._stable
        self._previous_ema_dct_mean = self.ema_dct_mean
        self._distances.append(distance)
        self._ema_distances.append(ema_distance)
        self._unstable_events.append(int(not self._stable))
        self._trigger_events.append(int(self._trigger))
        self.dct_medians.append(self.ema_dct_mean)
        self._distances = self._truncate_history(self._distances)
        self._ema_distances = self._truncate_history(self._ema_distances)
        self._unstable_events = self._truncate_history(self._unstable_events)
        self._trigger_events = self._truncate_history(self._trigger_events)
        self.dct_medians = self._truncate_history(self.dct_medians)
        return dct, phash

    def invoke_tensorflow(self, frame):
        input_details = self._interpreter.get_input_details()
        frame = cv2.resize(frame, tuple(input_details[0]["shape"][1:3]))
        frame = frame[np.newaxis, :, :, :].astype(input_details[0]["dtype"])
        self._interpreter.set_tensor(input_details[0]["index"], frame)
        self._interpreter.invoke()

    def invoke_onnx(self, frame):
        """Run ONNX model inference and return the feature map."""
        # Resize frame to 640x640 for YOLO input
        frame_resized = cv2.resize(frame, (640, 640))
        # Convert RGB to BGR and normalize to [0, 1] as float32
        frame_input = frame_resized.astype(np.float32) / 255.0
        # Add batch dimension: [1, H, W, C]
        frame_input = frame_input[np.newaxis, :, :, :]
        # Transpose to NCHW format: [1, C, H, W]
        frame_input = np.transpose(frame_input, (0, 3, 1, 2))
        # Run inference
        outputs = self._interpreter.run(None, {"images": frame_input})
        # outputs[0] are detection output [1, 300, 6]
        # outputs[1] is feature map [1, 128, 40, 40]
        return outputs[1]

    def clear(self):
        self.frame_index = -1  # start from 0
        self._distances = [0.0 for _ in range(self._history_size)]
        self._ema_distances = [0.0 for _ in range(self._history_size)]
        self._stable = True
        self._trigger = False
        self._previous_stable = True
        self._previous_phash = None
        self._previous_stable_ema_phash = None
        self._stable_ema_phash = None
        self._unstable_to_stable_timestamp = None
        self._stable_timestamp = None
        self._previous_dct = None
        self._previous_timestamp = None
        self.ema_dct_mean = 0
        self._previous_ema_dct_mean = None

    def _truncate_history(self, series) -> list:
        if len(series) > self._history_size:
            series = series[-1 * self._history_size :]
        return series

    @staticmethod
    def _compute_matrices_ema(
        current_matrix: np.ndarray, previous_matrix: np.ndarray, alpha: float
    ) -> np.ndarray:
        """Compute exponential moving average of two matrices."""
        if current_matrix.shape != previous_matrix.shape:
            raise ValueError("Matrices must have the same dimensions")
        return alpha * current_matrix + (1 - alpha) * previous_matrix

    @staticmethod
    def _compute_dct(image, hash_size=32, img_size=64):
        image = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
            (img_size, img_size),
        )
        dct = scipy.fftpack.dct(scipy.fftpack.dct(image, axis=0), axis=1)
        return dct[:hash_size, :hash_size]

    @staticmethod
    def _compute_mean_dct(image, hash_size=7):
        dct = scipy.fftpack.dct(scipy.fftpack.dct(image, axis=0), axis=1)
        return dct[:hash_size, :hash_size]

    @staticmethod
    def _compute_phash(dct):
        return dct > np.mean(dct)

    def _compute_j_hash(self, dct):
        return dct - self.ema_dct_mean

    @staticmethod
    def _compute_median_phash(dct):
        return dct > np.median(dct, axis=[0, 1])[np.newaxis, np.newaxis, :]

    @staticmethod
    def _compute_hamming_distance(phash1, phash2):
        return np.count_nonzero(
            phash1.flatten().astype(bool) != phash2.flatten().astype(bool)
        )

    def _compute_log_l2_distance(self, phash1, phash2):
        """
        Compute log L2 distance with optional clamping to lower bound.
        Matches TypeScript implementation.
        """
        distance_offset = 1
        distance_scale_factor = 0.1
        distance = np.log10(
            (np.linalg.norm(phash1 - phash2) / phash1.size) * distance_scale_factor
            + distance_offset
        )

        # Clamp distance if within bounds
        if (
            self._distance_lower_bound_threshold
            < distance
            < self._distance_upper_bound_threshold
        ):
            distance = self._distance_lower_bound_threshold
        return distance

    @staticmethod
    def _compute_l2_distance(phash1, phash2):
        return np.linalg.norm(phash1 - phash2) / phash1.size

    @staticmethod
    def _compute_norm_l2_distance(phash1, phash2):
        return np.linalg.norm(
            phash1 / np.linalg.norm(phash1) - phash2 / np.linalg.norm(phash2)
        )

    @staticmethod
    def _compute_avg_l2_distance(phash1, phash2):
        return np.mean(np.linalg.norm(phash1 - phash2, axis=-1))

    def monitor_unstable_intervals(self):
        new_stable = self._stable
        # 1. Transition from Stable -> Unstable (Start timing)
        if self._previous_stable and not new_stable:
            # State: True -> False
            self._is_timing = True
            self._start_frame_unstable = self.frame_index
            print(
            # self.logger.info(
                f"[Frame {self.frame_index:4d}] ** State Change: UNSTABLE "
                f"(Start timing from this frame) **"
            )

        # 2. Transition from Unstable -> Stable (Stop timing and record)
        elif not self._previous_stable and new_stable:
            # State: False -> True
            if self._is_timing:
                # The frame index where stability returned (exclusive end)
                end_frame = self.frame_index

                # Calculate metrics based on frames
                duration_frame = end_frame - self._start_frame_unstable

                interval_data = {
                    # frame records
                    # "interval_id": len(self._unstable_intervals) + 1,
                    # "start_frame": self._start_frame_unstable,
                    # # Note: This is the frame where stability returned
                    # "end_frame": end_frame,
                    # "duration_frames": duration_frame,
                    "frame_rate": self._target_fps,
                    # time records
                    "start_time": (self._start_frame_unstable / self._target_fps),
                    "end_time": end_frame / self._target_fps,
                    "duration": duration_frame / self._target_fps,
                    # ActivityNet labels and scores
                    "segment": [
                        self._start_frame_unstable / self._target_fps,
                        end_frame / self._target_fps,
                    ],
                    "label": LABEL_NAME2ID[
                        LabelMap.POSITIVE_UNSTABLE
                    ],  # 2 repre Positive Unstable
                    "score": 1.0,  # default prediction confidence 1.0
                }
                print(
                # self.logger.info(
                    f"[Frame {end_frame:4d}] ** State Change: STABLE "
                    f"(Interval recorded: {duration_frame:.3f} frames) **"
                )
                self._unstable_intervals.append(interval_data)
                self._is_timing = False  # Stop timing

    def finalize(self):
        """處理結尾截斷問題"""
        if self._is_timing:
            # 強制關閉區間
            # 注意：這裡的時間計算也要小心，最好紀錄最後一次的 timestamp
            if self._previous_timestamp:
                end_time = self._previous_timestamp
            else:
                end_time = 0
            # ... (寫入 unstable_intervals)
            self._is_timing = False
            print("TBD : final unstable to be write in interval")
        # self.clear()


def single_process_video(fn, frame_history, debug_visual=False):
    cap = cv2.VideoCapture(fn)
    source=''

    frame_rate = 10  # target frame rate

    frame_step_time = 1.0 / frame_rate
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # Off-Stream: 時間從 0 開始
    current_fake_timestamp = 0.0

    # if frame_rate >= original_fps:
    #     print(f"注意：目標 FPS ({frame_rate}) >= 原始 FPS ({original_fps})，將逐幀處理。")
    #     frame_skip_interval = 1
    #     frame_step_time = 1.0 / original_fps
    # else:
    #     # 計算跳幀的間隔：例如 30 FPS -> 10 FPS，則間隔為 30/10 = 3 幀
    #     frame_skip_interval = max(1, round(original_fps / frame_rate))
    #     print(
    #         f"原始 FPS: {original_fps:.2f}, "
    #         f"目標 FPS: {frame_rate}，將每 {frame_skip_interval} 幀處理一次。"
    #     )
    # final_stop = False

    last_time = time.time()

    if source == 0:
        sleep_time = frame_step_time  # (sec)
        frame_skip_interval = 1
    else:
        # [Off-Stream 時間計算變數]
        sleep_time = 0
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip_interval = max(1, round(original_fps / frame_rate))
        frame_step_time = 1.0 / frame_rate
        print("original_fps", original_fps)

    frame_index = -1
    while cap.isOpened():
        start_time = time.time()
        time.sleep(sleep_time)
        # Skip frames
        for _ in range(frame_skip_interval):
            ret, frame = cap.read()
            if not ret:
                break
        if not ret:
            print("End of video stream.")
            break
        frame_index += 1
        curr_time = time.time()
        dt = curr_time - last_time

        y1 = int(frame.shape[0] * Constant.BOUNDARY[0])  # pylint: disable=invalid-name
        y2 = int(frame.shape[0] * Constant.BOUNDARY[1])  # pylint: disable=invalid-name
        x1 = int(frame.shape[1] * Constant.BOUNDARY[2])  # pylint: disable=invalid-name
        x2 = int(frame.shape[1] * Constant.BOUNDARY[3])  # pylint: disable=invalid-name

        checkout_region = [0, 0, -1, -1]  # xyxy

        # with crop image (boundary)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[y1:y2, x1:x2]
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        export_external_j_feature = False
        frame_history.observe(
            rgb_frame,
            stable_threshold=Constant.THRESHOLD,
            trigger_threshold=Constant.TRIGGER_THRESHOLD,
            distance_lower_bound_threshold=Constant.DISTANCE_LOWER_BOUND,
            distance_upper_bound_threshold=Constant.DISTANCE_UPPER_BOUND,
            trigger_wait_time=Constant.TRIGGER_WAIT_TIME,
            ema_distance_alpha=Constant.EMA_ALPHA,
            checkout_region=checkout_region,
            export_external_j_feature=export_external_j_feature,
            timestamp=current_fake_timestamp,
            is_monitor_unstable_interval=True,  # record unstable interval
        )
        # ==================== Visualize ====================
        if debug_visual:
            from src import visual
            image_list = [
                rgb_frame,
            ]
            is_motion = None
            is_stable = frame_history.stable
            signal_is_trigger = frame_history.trigger
            distances = frame_history.distances
            ema_distances = frame_history.ema_distances
            # print("image_list:", [m.shape for m in image_list])
            debug_view = visual.auto_image_collage_with_padding(image_list)
            frame_index_string = f"Frame id: {frame_index:3d} | "
            description = " "
            description += frame_index_string
            description += (
                f"distance: {distances[-1]:.2f}"
                + f"| ema_distance: {ema_distances[-1]:.2f}"
            )

            # Text info
            # status = "STABLE" if self._is_stable else "UNSTABLE"
            is_motion_string = "Physical Motion Detected!" if is_motion else ""
            stable_or_not = "Stable" if is_stable else "Unstable"
            trigger_or_not = " | Trigger!!" if signal_is_trigger else " | "
            stable_or_not += trigger_or_not
            cv2.putText(
                debug_view,
                description,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                debug_view,
                f"is_motion: {is_motion_string}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                debug_view,
                f"{stable_or_not}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(  # boundary
                debug_view,
                "boundary mask",
                (10, 180 * 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (128, 128, 128),
                2,
            )

            cv2.imshow(f"Phash", debug_view)
            if cv2.waitKey(10) & 0xFF == 27:
                pass
        # ==================== Visualize ====================

        current_fake_timestamp += frame_step_time
    print(_fn)
    frame_history.finalize()
    print("Finished processing video.")
    print(*frame_history.unstable_frame_intervals, sep="\n")
    return frame_history


if __name__ == "__main__":
    SOURCE = Constant.SOURCE
    METHOD = Constant.METHOD
    THRESHOLD = Constant.THRESHOLD
    TRIGGER_THRESHOLD = Constant.TRIGGER_THRESHOLD
    DISTANCE_LOWER_BOUND = Constant.DISTANCE_LOWER_BOUND
    DISTANCE_UPPER_BOUND = Constant.DISTANCE_UPPER_BOUND
    TRIGGER_WAIT_TIME = Constant.TRIGGER_WAIT_TIME  # seconds
    EMA_ALPHA = Constant.EMA_ALPHA
    # [y_top, y_bottom, x_left, x_right], None means full frame
    BOUNDARY = Constant.BOUNDARY  # checkout region
    TRIGGER_REGION_SCALE = Constant.TRIGGER_REGION_SCALE  # 1.0 means same as checkout
    frame_history = FrameHistory(
        alpha=EMA_ALPHA,
        history_size=300,
        threshold=THRESHOLD,
        trigger_threshold=TRIGGER_THRESHOLD,
        method=METHOD,
        log_level=logging.INFO,
    )
    FNS = [
        # "/Users/eason.hung/Documents/Projects/test-something/external_camera/formal_output_light/Viscovery_Bread_DemoRoom_20251107_102031_mp4v.mp4",
        "/Users/eason.hung/Documents/Projects/explus/external_camera/formal_output_進_單入_new/Viscovery_Bread_DemoRoom_20251107_105228.mp4",
    ]

    for _fn in FNS:
        # main(fn=None, debug_visual=True)
        single_process_video(_fn, frame_history, debug_visual=True)
