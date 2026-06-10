import queue
import threading
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cv2
import math
import numpy as np
import onnxruntime as ort

import utils
from src.schemas import VerificationPayload


class AsyncTriggerVerifier:
    """Asynchronous verification engine using ONNX models and signal logic.

    Attributes:
        on_result_callback: Function called after verification completion.
    """

    def __init__(
        self,
        model_path: str,
        mode: Optional[str] = "yolo",
        feature_abstraction_level: Optional[str] = "P4",
        context_mirror_ratio: float = 0.0,
        similarity_threshold: float = 0.6,
        min_trigger_area: int = 2,
        vertical_focus_range: list = [140, 500],
        ema_threshold: float = None,
    ) -> None:
        self._support_mode = ["yolo", "ema_phash"]
        if mode not in self._support_mode:
            raise ValueError(f"Not support {mode}, Only {self._support_mode}")

        if mode == "yolo":
            self._interpreter: ort.InferenceSession = ort.InferenceSession(
                model_path
            )
            self._select_process_verification = self._process_verification
            self.feature_abstraction_level = feature_abstraction_level
            self.context_mirror_ratio = context_mirror_ratio
            self.similarity_threshold = similarity_threshold
            self.min_trigger_area = min_trigger_area
            self.vertical_focus_range = vertical_focus_range

        elif mode == "ema_phash":
            self._interpreter = None
            self._select_process_verification = self.phash_process_verification
            self._ema_threshold = ema_threshold

        # Optimization Cache
        self._last_processed_meta_index: int = -1
        self._last_processed_result: Optional[Dict[str, Any]] = None  ## TODO

        # Threading resources
        self.task_queue: queue.Queue = queue.Queue()
        self.on_result_callback: Optional[Callable[[bool, str, int], None]] = (
            None
        )

        # self._worker_thread: threading.Thread = threading.Thread(
        #     target=self._worker_loop, daemon=True
        # )
        # self._worker_thread.start()

    def set_callback(self, callback: Callable[[bool, str, int], None]) -> None:
        """Assigns a callback function for asynchronous results."""
        self.on_result_callback = callback

    def close(self) -> None:
        """Stops the worker thread and releases resources."""
        if hasattr(self, "_worker_thread") and self._worker_thread.is_alive():
            self.task_queue.put(None)  # Poison pill
            self._worker_thread.join(timeout=2.0)
        
        # Explicitly release ONNX session if possible
        if hasattr(self, "_interpreter"):
            del self._interpreter
            self._interpreter = None

    def submit(self, payload: Optional[VerificationPayload]) -> None:
        """Puts a verification task into the processing queue."""
        self.task_queue.put(payload)

    def verify_sync(self, payload: VerificationPayload) -> Tuple[bool, str]:
        """Performs a blocking verification on the current thread."""
        if self._check_has_data(payload):
            return self._select_process_verification(payload)
        return False, "No Data", payload

    def _check_has_data(self, payload):
        if payload.prev_meta.index == -1 or payload.curr_meta.index == -1:
            # does not have data
            return False
        return True

    def _worker_loop(self) -> None:
        """Continuous loop for background task processing."""
        while True:
            payload: Optional[VerificationPayload] = self.task_queue.get()
            if payload is None:
                self.task_queue.task_done()
                break
            try:
                triggered, reason, payload = self._process_verification(payload)
                if self.on_result_callback:
                    self.on_result_callback(
                        triggered, reason, payload.interval_id
                    )
            except Exception as e:
                print(f"[Verifier Error] {e}")
            finally:
                self.task_queue.task_done()

    def _process_verification(
        self, payload: VerificationPayload
    ) -> Tuple[bool, str]:
        """Orchestrates the verification steps and cache management."""
        # 1. Handle previous image inference (with cache)
        if self._last_processed_meta_index != payload.prev_meta.index:
            res_pre = self._inference(payload.prev_meta.ori_img)
        else:
            res_pre = self._last_processed_result

        # 2. Handle current image inference
        res_post = self._inference(payload.curr_meta.ori_img)

        # 3. Decision Logic
        if self.feature_abstraction_level == "P4":
            triggered, reason = self._compare_logic_m2(
                res_pre, res_post, payload.boundary
            )
        elif self.feature_abstraction_level == "fusion":
            triggered, reason = self._compare_logic_m3(
                res_pre, res_post, payload.boundary
            )
        else:
            raise ValueError(
                f"Not support verify Trigger version: {self.feature_abstraction_level}"
            )

        # 4. Update Cache for next chain
        self._last_processed_result = res_post
        self._last_processed_meta_index = payload.curr_meta.index

        # 5. output debug payload inference result
        payload.prev_meta.infer_result = res_pre
        payload.curr_meta.infer_result = res_post

        return triggered, reason, payload

    def _inference(self, img: np.ndarray) -> Dict[str, Any]:  ## TODO
        """Prepares image and executes model inference."""
        transformer = utils.LetterBox_new(
                    new_shape=(640, 640),
                    border_mode=cv2.BORDER_REFLECT_101,
                    mirror_ratio=self.context_mirror_ratio,
                    padding_value=114,
                )
        model_img = self._prepare_model_io(
            frame=img,
            transformer=transformer,
        )
        _, detection_result, feature_maps = self.invoke_onnx(model_img)
        return {
            "result": detection_result,
            "ori_img": img,
            "img": model_img,
            "feature_maps": feature_maps,
        }

    def invoke_onnx(self, frame: np.ndarray) -> Tuple[Any, Any, Any]:  ## TODO
        """Low-level ONNX execution and tensor slicing."""
        outputs = self._interpreter.run(None, {"images": frame})

        # Process main detection output
        output_main = outputs[0].squeeze(0)
        conf_mask = output_main[:, 4] > 0.5

        conf_mask0 = output_main[:, 4] > 0
        print("conf:", output_main[conf_mask0, 4])

        filtered_bboxes = output_main[conf_mask, :4].astype(int)

        # Extract multi-scale feature maps
        # TODO(Eason): 我們在取得 onnx export 要在乾淨精煉一點.
        if len(outputs) == 4:
            indices = [1, 2, 3]  # open source yolo12n
        elif len(outputs) == 5:
            indices = [2, 3, 4]  # custom tuning
        else:
            raise ValueError(
                f"Indice is Not define: output len is {len(outputs)} != 4 or 5"
            )
        feature_maps = (
            outputs[indices[0]].squeeze(0),
            outputs[indices[1]].squeeze(0),
            outputs[indices[2]].squeeze(0),
        )

        return (outputs[1], filtered_bboxes, feature_maps)

    @staticmethod
    def _prepare_model_io(
        frame: np.ndarray,
        transformer: Optional[Any] = None,
        mirror_ratio: float = 0.0,
    ) -> np.ndarray:
        """Standardizes image format for model input."""
        if transformer is None:
            # letterbox_mirror version
            if mirror_ratio > 0.0:
                transformer = utils.LetterBox_new(
                    new_shape=(640, 640),
                    border_mode=cv2.BORDER_REFLECT_101,
                    mirror_ratio=mirror_ratio,
                    padding_value=114,
                )
            else:
                transformer = utils.LetterBox(new_shape=(640, 640))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transformer(rgb_frame).astype(np.float32) / 255.0
        img = np.transpose(img[np.newaxis, :, :, :], (0, 3, 1, 2))
        return img

    def _compare_logic_m3(
        self,
        res_prev: Dict[str, Any],
        res_post: Dict[str, Any],
        boundary: Optional[list] = None,  ## TODO finish this process
    ) -> Tuple[bool, str]:
        # 多種的 scale 整合在一起去比較. (更大根的向量)
        resize_shape = res_prev["feature_maps"][1].shape[1:3]  # (H, W)
        print("[Feature] resize_shape", resize_shape)
        prev_concat_feats = []
        for prev in res_prev["feature_maps"]:
            prev_hwc = np.transpose(prev, (1, 2, 0))  # (C, H, W) -> (H, W, C)
            prev_hwc = cv2.resize(
                prev_hwc,
                (resize_shape[1], resize_shape[0]),
                interpolation=cv2.INTER_LINEAR,  # NEAREST
            )
            prev_chw = np.transpose(
                prev_hwc, (2, 0, 1)
            )  # (H, W, C) -> (C, H, W)
            prev_concat_feats.append(prev_chw)
        prevs = np.concatenate(prev_concat_feats, axis=0)  # (C_total, H, W)

        post_concat_feats = []
        for post in res_post["feature_maps"]:
            post_hwc = np.transpose(post, (1, 2, 0))  # (C, H, W) -> (H, W, C)
            post_hwc = cv2.resize(
                post_hwc,
                (resize_shape[1], resize_shape[0]),
                interpolation=cv2.INTER_LINEAR,  # NEAREST
            )
            post_chw = np.transpose(
                post_hwc, (2, 0, 1)
            )  # (H, W, C) -> (C, H, W)
            post_concat_feats.append(post_chw)
        posts = np.concatenate(post_concat_feats, axis=0)  # (C_total, H, W)

        Px_prev_norm = np.linalg.norm(prevs, axis=0, keepdims=True)
        Px_post_norm = np.linalg.norm(posts, axis=0, keepdims=True)

        _sim_map = np.sum(
            (prevs / (Px_prev_norm + 1e-6)) * (posts / (Px_post_norm + 1e-6)),
            axis=0,
            keepdims=False,  # False: (H, W) / True: (1, H, W)
        )

        # [sim (-1 ~ 1)] -> [1-sim (0 ~ 2)]
        diff_map = np.clip(1.0 - _sim_map, 0, 2)  # 越不相似越大

        # Hotfix for value normalize
        diff_map[0, 0] = 0
        diff_map[0, 1] = 2

        # NOTE(Eason): 這裡相當於是 clip 掉 差異很大的都轉換為 255.
        # 直接轉 0~1 且超出1 都飽和色
        norm_val = np.clip(diff_map * 255.0, 0, 255).astype(np.uint8)  # (H, W)
        # channel (H, W) 1 -> 3 (H, W, 3)
        norm_val = np.repeat(norm_val[:, :, np.newaxis], 3, axis=2)
        norm_val = cv2.resize(
            norm_val,  # (H, W, 1)
            (640, 640),  # width, height
            interpolation=cv2.INTER_LINEAR,
        )[self.vertical_focus_range[0]:self.vertical_focus_range[1], :, :]
        x_left, x_right = int(640 * boundary[0]), int(640 * boundary[2])
        y_top, y_bottom = int(640 * boundary[1]), int(640 * boundary[3])
        # Adjusted for vertical crop
        y_top = max(0, y_top - self.vertical_focus_range[0])
        y_bottom = max(0, y_bottom - self.vertical_focus_range[0])
        
        norm_val_roi = norm_val[y_top:y_bottom, x_left:x_right, 0]
        self.similarity_threshold = 0.4
        threshold = self.similarity_threshold * 255
        if np.any(norm_val_roi > threshold):
            return True, "Significant change detected in ROI"
        return False, "No significant change"

    def _compare_logic_m2(
        self,
        res_prev: Dict[str, Any],
        res_post: Dict[str, Any],
        boundary: Optional[list] = None,  ## TODO finish this process
        # xyxy [0011]
    ) -> Tuple[bool, str]:
        """
        Featurep map base and point-wise similarity comparison. (do not mean or max aggregate)
        Channel-wise Cosine Similarity.
        """

        h, w, c = res_prev["ori_img"].shape

        # Select 'P4'
        feat_map_prev = res_prev["feature_maps"][1]
        feat_map_post = res_post["feature_maps"][1]

        is_normalize = True
        if is_normalize:
            Px_prev_norm = np.linalg.norm(feat_map_prev, axis=0, keepdims=True)
            Px_post_norm = np.linalg.norm(feat_map_post, axis=0, keepdims=True)
        else:
            Px_prev_norm = 1
            Px_post_norm = 1

        # similarity
        _sim_map = np.sum(
            (feat_map_prev / (Px_prev_norm + 1e-6))
            * (feat_map_post / (Px_post_norm + 1e-6)),
            axis=0,
            keepdims=False,  # False: (H, W) / True: (1, H, W)
        )
        print(_sim_map.min(), _sim_map.max(), _sim_map.shape)
        # [sim (-1 ~ 1)] -> [1-sim (0 ~ 2)]
        diff_map = np.clip(1.0 - _sim_map, 0, 2)  # 越不相似越大

        # Hotfix for value normalize
        diff_map[0, 0] = 0
        diff_map[0, 1] = 2

        # NOTE(Eason): 這裡相當於是 clip 掉 差異很大的都轉換為 255.
        # 直接轉 0~1 且超出1 都飽和色
        norm_val = np.clip(diff_map * 255.0, 0, 255).astype(np.uint8)  # (H, W)
        # channel (H, W) 1 -> 3 (H, W, 3)
        norm_val = np.repeat(norm_val[:, :, np.newaxis], 3, axis=2)
        norm_val = cv2.resize(
            norm_val,  # (H, W, 1)
            (640, 640),  # width, height
            interpolation=cv2.INTER_LINEAR,
        )[self.vertical_focus_range[0]:self.vertical_focus_range[1], :, :]
        x_left, x_right = int(640 * boundary[0]), int(640 * boundary[2])
        y_top, y_bottom = int(640 * boundary[1]), int(640 * boundary[3])
        # Adjusted for vertical crop
        y_top = max(0, y_top - self.vertical_focus_range[0])
        y_bottom = max(0, y_bottom - self.vertical_focus_range[0])

        norm_val_roi = norm_val[y_top:y_bottom, x_left:x_right, 0]

        threshold = self.similarity_threshold * 255

        if np.count_nonzero(norm_val_roi > threshold) >= self.min_trigger_area:
            return True, "Significant change detected in ROI"
        return False, "No significant change"

    def phash_process_verification(self, payload: VerificationPayload):
        """Orchestrates the verification steps and cache management."""
        if payload.boundary:
            h, w = payload.prev_meta.ori_img.shape[:2]
            y1 = math.ceil(h * payload.boundary[1])
            y2 = math.floor(h * payload.boundary[3])
            x1 = math.ceil(w * payload.boundary[0])
            x2 = math.floor(w * payload.boundary[2])

        # 1. Handle previous image inference (with cache)
        if self._last_processed_meta_index != payload.prev_meta.index:
            crop_img = payload.prev_meta.ori_img[y1:y2, x1:x2, :]
            # frame to phash
            res_pre = compute_dct(
                # payload.prev_meta.ori_img,
                crop_img,
                hash_size=32,
                img_size=64,
            )
        else:
            res_pre = self._last_processed_result

        # 2. Handle current image inference
        crop_img = payload.curr_meta.ori_img[y1:y2, x1:x2, :]
        res_post = compute_dct(
            crop_img,
            hash_size=32,
            img_size=64,
        )

        # 3. Decision Logic
        triggered, reason = phash_compare_logic(
            res_pre, res_post, trigger_threshold=self._ema_threshold
        )

        # 4. Update Cache for next chain
        self._last_processed_result = res_post
        self._last_processed_meta_index = payload.curr_meta.index

        return triggered, reason, payload


def phash_compare_logic(res_pre, res_post, trigger_threshold=None):
    if (res_pre is None) or (res_post is None):
        print("res_pre or res_post", res_pre, res_post)
        return None
    stable_distance = compute_l2_distance(
        res_pre.flatten(),
        res_post.flatten(),
    )
    _trigger = stable_distance >= trigger_threshold
    if _trigger:
        print(stable_distance, trigger_threshold)
        return True, "Trigger"
    return False, "Not Trigger"


def compute_l2_distance(phash1, phash2):
    return np.linalg.norm(phash1 - phash2) / phash1.size


def compute_dct(image, hash_size=32, img_size=64):
    import scipy

    image = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
        (img_size, img_size),
    )
    dct = scipy.fftpack.dct(scipy.fftpack.dct(image, axis=0), axis=1)
    return dct[:hash_size, :hash_size]
