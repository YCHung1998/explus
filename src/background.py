import numpy as np


class BackgroundManager:
    def __init__(self, bg_update_rate=0.3):
        self.bg_update_rate = bg_update_rate
        self.bg = None
        self.prev_centered_blocks = None

    def update_and_get_score(
        self,
        gray_frame,
        blocks,
        lower_update_mask=None,
        lower_update_alpha=None
    ):
        """
        Output: score_mape, center_blocks, background, previous_center_blocks
        """
        # 1. Init
        if self.bg is None:
            self.bg = gray_frame.copy().astype(np.float32)
            # 初次計算 Centered Blocks
            means = np.mean(blocks, axis=(2, 3), keepdims=True)
            self.prev_centered_blocks = blocks - means
            return (
                np.zeros((blocks.shape[0], blocks.shape[1])),
                self.prev_centered_blocks,
                self.bg,
                self.prev_centered_blocks,
            )

        # 1. EMA Update
        if lower_update_mask is None:
            lower_update_mask = np.zeros_like(gray_frame)
        
        # Determine the update rate for each pixel
        if lower_update_alpha is not None:
            per_pixel_alpha = np.where(
                lower_update_mask < 0.5,
                self.bg_update_rate,
                lower_update_alpha
            ).astype(np.float32)
        else:
            per_pixel_alpha = self.bg_update_rate

        # 2. 滾動式修正公式: B_t = (1 - α) * B_{t-1} + α * I_t
        self.bg = (
            (1.0 - per_pixel_alpha) * self.bg +
            per_pixel_alpha * gray_frame.astype(np.float32)
        )

        # 2. DC Removal & Energy Score
        means = np.mean(blocks, axis=(2, 3), keepdims=True)
        curr_centered = blocks - means
        score_map = np.mean(
            np.abs(
                curr_centered
                - self.prev_centered_blocks
            ),
            axis=(2, 3)
        )

        # 3. 更新背景緩存
        self.prev_centered_blocks = (
            (1 - self.bg_update_rate) * self.prev_centered_blocks
            + self.bg_update_rate * curr_centered
        )
        return (
            score_map,
            curr_centered,
            self.bg,
            self.prev_centered_blocks,
        )

    def get_background(self):
        return (
            self.bg,
            self.prev_centered_blocks
        )

    def reset(self):
        self.bg = None
        self.prev_centered_blocks = None
