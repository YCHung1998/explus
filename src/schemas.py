from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

import enum


class LabelMap(str, enum.Enum):
    STABLE = 'Stable'
    POSITIVE_UNSTABLE = 'Positive Unstable'
    NEGATIVE_UNSTABLE_EXTERNAL_DISTURBANCES = \
        'Negative Unstable External Disturbances'
    NEGATIVE_UNSTABLE_OVEREXPOSURE = 'Negative Unstable Overexposure'
    TRIGGER = 'Trigger'

    def __str__(self) -> str:
        return self.value


# signal_processor.py
class SignalState(str, enum.Enum):
    STABLE = 0
    UNSTABLE = 1

    def __str__(self) -> str:
        return self.value

# ==========================================
# 2. Data Classes (資料結構)
# ==========================================
@dataclass
class ImageMeta:
    """Trigger Engine 接收的標準圖片封裝"""
    index: int = -1
    ori_img: np.ndarray = None  # BGR Image
    # 未來可擴充: timestamp, features 等
    ema_frame: np.ndarray = None
    ema_phash: np.ndarray = None
    # model inference results
    infer_result: dict = None



@dataclass
class VerificationPayload:
    """發送給 Verifier 的打包請求"""
    prev_meta: ImageMeta
    curr_meta: ImageMeta
    interval_id: int          # 用來 callback 時對應回原本的區間
    extra_info: Optional[Dict[str, Any]] = None
    # relative boundary [xyxy] for original frame
    boundary: np.ndarray = None

@dataclass
class IntervalData:
    """單一區間的資料結構 (讓 SignalIntegrator 產出標準格式)"""
    start_frame: int
    end_frame: int
    duration: float
    label: int
    score: float
    segment: list  # [start_sec, end_sec]


LABEL_MAP = {
    1: LabelMap.STABLE,
    2: LabelMap.POSITIVE_UNSTABLE,
    3: LabelMap.NEGATIVE_UNSTABLE_EXTERNAL_DISTURBANCES,
    4: LabelMap.NEGATIVE_UNSTABLE_OVEREXPOSURE,
    5: LabelMap.TRIGGER
}
LABEL_NAME2ID = dict(zip(LABEL_MAP.values(), LABEL_MAP.keys()))


if __name__ == '__main__':

    print(LabelMap.STABLE)
    print("LABEL_NAME2ID[LabelMap.STABLE]", LABEL_NAME2ID[LabelMap.STABLE])
