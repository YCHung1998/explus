import math
import cv2
import numpy as np

from typing import Optional


class AdaptiveDecisionUnit:
    def __init__(
        self,
        rows,
        cols,
        motion_sensitivity=30,
        boundary=None,
    ):
        self.sens_map = np.full(
            (rows, cols),
            motion_sensitivity,
            dtype=np.float32,
        )
        self.motion_sensitivity = motion_sensitivity  # color difference lower bound
        self.boundary = boundary
        self.kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def decide(
        self,
        score_map,  # block-base score value
        hamming_map,  # [v] pixel-base score value (LBSP)

        prep,  # preprocess
        gray,  # input gray frame
        background,
        blocks_uint8=None,  # current frame blocks
        dt=None,

        # Runtime parameter overrides (optional)
        motion_sensitivity: Optional[float] = None,
        boundary: Optional[list] = None,
        pixel_diff_threshold: Optional[float] = None,
        global_motion_ratio: Optional[float] = None,
    ):

        # 1. LBSP Denoising
        # Remove flickering edges
        open_is_true = True
        if open_is_true:
            # remove flikering edge and find the real object edge
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            hamming_map_open = cv2.morphologyEx(
                hamming_map.astype(np.uint8).copy(),
                cv2.MORPH_OPEN,
                kernel,
            )
            hamming_map = hamming_map_open
        else:
            hamming_map = hamming_map.astype(np.uint8)

        # Texture change
        # active_mask_blocks = prep._to_blocks(hamming_map)  # (nr, nx, gh, gw)
        # region_mask = np.any(
        #     active_mask_blocks,
        #     axis=(2, 3),
        # ).astype(
        #     np.uint8
        # )  # (nr, nc)

        # Intensity to strong
        # 定義太亮或是過曝, 如果物體就這麼亮, 就是靠他的邊界了,
        # 如果整個物體都在反光, 那他會被吃掉也很正常.
        # block_max = np.max(blocks_uint8, axis=(2, 3)) < 255  # 255 就是全部
        # highlight_mask = region_mask & block_max  # (nr, nc)
        # highlight_mask = cv2.resize(
        #     highlight_mask,
        #     (hamming_map.shape[1], hamming_map.shape[0]),
        #     interpolation=cv2.INTER_NEAREST,
        # )

        # 2. Parameter Setup
        effective_motion_sensitivity = (
            motion_sensitivity if motion_sensitivity is not None else self.motion_sensitivity
        )
        effective_boundary = boundary if boundary is not None else self.boundary
        effective_pixel_diff_threshold = (
            pixel_diff_threshold if pixel_diff_threshold is not None else 15.0
        )
        effective_global_motion_ratio = (
            global_motion_ratio if global_motion_ratio is not None else 0.85
        )

        # # Block Base score map:
        # # --- 3. Statistical Metrics ---
        max_motion_score = np.max(score_map)
        active_grids = score_map > effective_pixel_diff_threshold
        active_count = np.sum(active_grids)
        total_grids = np.prod(score_map.shape)  # 16 x 9
        active_ratio = active_count / total_grids

        # boundary filter
        valid_mask = np.ones_like(score_map)  # Default: all valid
        score_map_valid = score_map.copy()
        if effective_boundary:
            h, w = score_map.shape
            y1 = math.ceil(h * effective_boundary[1])  # 取窄一點
            y2 = math.floor(h * effective_boundary[3])
            x1 = math.ceil(w * effective_boundary[0])
            x2 = math.floor(w * effective_boundary[2])

            valid_mask = np.zeros_like(score_map)
            valid_mask[y1:y2, x1:x2] = 1
            score_map_valid = score_map * valid_mask
            max_motion_score = np.max(score_map_valid)

        if (
            max_motion_score > effective_motion_sensitivity
            and active_ratio < effective_global_motion_ratio
        ):
            print("[[Block-Based Unstable]]")
            return True, score_map_valid, valid_mask
        if active_ratio > effective_global_motion_ratio:
            print("[Light reject]")

        # return False, score_map_valid, valid_mask

        # TODO BOUNDARY usage in low-level or mid level feature
        # (Texture) boundary filter
        valid_mask = np.ones_like(hamming_map)  # Default: all valid
        hamming_map_valid = hamming_map.copy()
        if effective_boundary:
            self.h, self.w = hamming_map.shape
            y1 = int(self.h * effective_boundary[1])
            y2 = int(self.h * effective_boundary[3])
            x1 = int(self.w * effective_boundary[0])
            x2 = int(self.w * effective_boundary[2])
            valid_mask = np.zeros_like(hamming_map)
            valid_mask[y1:y2, x1:x2] = 1

            # boundary * hamming result * filter out too light region
            hamming_map_valid = valid_mask * hamming_map
            # * highlight_mask

        # decision, hamming_map, boundary mask
        if np.any(hamming_map > 0):
            print("[[LBSP Unstable]]")
            return True, hamming_map_valid, valid_mask

        return (
            False,  # is motion
            hamming_map_valid,  # active mask
            valid_mask,  # valid (boundary) mask
        )
