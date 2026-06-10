import cv2
import numpy as np


class ImagePreprocessor:
    def __init__(self, process_wh=(320, 180), block_size=(20, 20)):
        self.w, self.h = process_wh
        self.bw, self.bh = block_size
        self.rows = self.h // self.bh
        self.cols = self.w // self.bw

    def process(self, frame):
        # 1. Resize & Gray
        small_gray = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        if len(small_gray.shape) == 3:
            small_gray = cv2.cvtColor(small_gray, cv2.COLOR_BGR2GRAY)

        # 2. Split into blocks: (Rows, Cols, BH, BW)
        blocks = self._to_blocks(small_gray)

        # 3. zero mean blocks: (Rows, Cols, BH, BW)
        means = np.mean(blocks, axis=(2, 3), keepdims=True)
        centered_blocks = blocks - means
        return small_gray, blocks, centered_blocks

    def _to_Ycrcb(self, color_image):
        return cv2.cvtColor(color_image, cv2.COLOR_BGR2YCR_CB)

    def _to_blocks(self, frame):
        # Input: (H, W), Output: (Rows, Cols, BH, BW)
        blocks = frame[:self.rows*self.bh, :self.cols*self.bw]
        blocks = (
            blocks.reshape(self.rows, self.bh, self.cols, self.bw)
            .transpose(0, 2, 1, 3)
        )
        return blocks

    def _back_to_frame(self, blocks):
        return (
            blocks.transpose(0, 2, 1, 3)
            .reshape(self.rows*self.bh, self.cols*self.bw)
        )

    @staticmethod
    def minmax_normalize(img):
        _min, _max = np.min(img), np.max(img)
        if _max == _min:
            return np.zeros(img.shape, dtype=np.uint8)
        return ((img - _min) / (_max - _min) * 255).astype(np.uint8)


    @staticmethod
    def clip_minmax_normalize(img, min_value=0, max_value=255):
        img = np.clip(img, min_value, max_value)
        img[0,0] = min_value
        img[0,1] = max_value
        _min, _max = np.min(img), np.max(img)
        if _max == _min:
            return np.zeros(img.shape, dtype=np.uint8)
        return ((img - _min) / (_max - _min) * 255).astype(np.uint8)

