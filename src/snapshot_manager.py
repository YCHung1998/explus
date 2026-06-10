from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar

import numpy as np

from constants import LABEL_NAME2ID, LabelMap, SignalState

from .schemas import ImageMeta
from .signal_processor import SignalIntegrator
from .trigger_verifier import AsyncTriggerVerifier, VerificationPayload


class SnapshotController:
    def __init__(
        self,
        integrator: SignalIntegrator,
        verifier: AsyncTriggerVerifier,
        settle_frames=3,
    ):
        self.integrator = integrator
        self.verifier = verifier
        self.settle_frames = settle_frames  # 等待轉 stable 穩定幀數

        # State
        self.reference_snapshot: Optional[ImageMeta] = None  # Pre-Event
        self.pending_interval_idx = None
        self.waiting_for_settle = False
        self.payload = None

        # system is start
        self.is_initialized = False

        # Setup Callback
        self.verifier.set_callback(self._on_verification_result)

    def close(self):
        """Releases resources held by the controller and verifier."""
        if hasattr(self, "verifier") and self.verifier:
            self.verifier.close()

    def on_frame(
        self,
        frame_index,
        frame,
        is_motion,
        ema_frame=None,
        ema_phash=None,
        boundary=None,
    ):
        """Main Loop 每幀呼叫此函式"""

        is_triggered_now = False  # 預設回傳 False
        reason = ""

        if boundary is None:  #
            boundary = [0,0,1,1]

        # --- A. 初始啟動：抓取第一幀作為起點基準 ---
        if not self.is_initialized:
            self.reference_snapshot = ImageMeta(
                index=frame_index,
                ori_img=frame.copy(),
                ema_frame=ema_frame if ema_frame is not None else None,
                ema_phash=ema_phash if ema_phash is not None else None,
            )
            self.is_initialized = True
            print(
                "[SnapshotController] Initialized Reference at Frame " f"{frame_index}"
            )
            # 初始化時也要跑一次 Integrator 以同步計數器，但不處理結果
            self.integrator.update(frame_index, is_motion)
            return False, reason

        # 1. 更新 Signal (Low Level)
        new_interval_idx = self.integrator.update(frame_index, is_motion)

        # --- B. 偵測到 Unstable 結束，開啟拍照等待 ---
        #    當 Integrator 回傳 index，代表剛發生 Edge: Unstable -> Stable
        if new_interval_idx is not None:
            self.pending_interval_idx = new_interval_idx
            self.waiting_for_settle = True
            # 注意：此時不拍照，也不更新 Reference，保持上一次的狀態

        # --- C. 拍照點到達 (檢查沉澱) ---
        if self.waiting_for_settle:

            # 防禦機制：如果在等待沉澱的過程中，突然又動了 (Stable -> Unstable)
            # 那這次等待就失效了 (被打斷)
            if not self.integrator.is_currently_stable():
                print(f"[SnapshotController] Settle interrupted at frame {frame_index}")
                self.waiting_for_settle = False
                self.pending_interval_idx = None
                return False, reason

            # 檢查是否達到指定的穩定幀數 (例如第 3 幀)
            # 這就是你說的 "趨前取得": 我們等到它穩定滿 N 幀的那一刻才抓
            if self.integrator.get_stable_duration() >= self.settle_frames:
                # 沉澱完成！拍攝 Post-Event Snapshot
                current_snapshot = ImageMeta(
                    index=frame_index,
                    ori_img=frame.copy(),
                    ema_frame=ema_frame if ema_frame is not None else None,
                    ema_phash=ema_phash if ema_phash is not None else None,
                )

                # 發送給 Verifier
                if self.reference_snapshot and self.pending_interval_idx is not None:
                    self.payload = VerificationPayload(
                        prev_meta=self.reference_snapshot,
                        curr_meta=current_snapshot,
                        interval_id=self.pending_interval_idx,
                        boundary=boundary
                    )
                    # ==== async ====
                    # self.verifier.submit(self.payload)
                    # ==== async ====

                    # ==== sync ====
                    # 這行會卡住 (Block)，直到 AI 算完
                    triggered, reason, self.payload = self.verifier.verify_sync(
                        self.payload
                    )
                    # [立即更新] 直接修改 Integrator interval 的資料
                    if triggered:
                        self.integrator.update_intervals(
                            interval_index=self.pending_interval_idx,
                            update_id=LABEL_NAME2ID[LabelMap.TRIGGER],
                        )
                        print(f"✅ Immediate Trigger: {reason}")
                        is_triggered_now = True  # 標記這一幀觸發了！
                    else:
                        print(f"ℹ️ Verified Negative: {reason}")
                    # ==== sync ====

                # Reset
                self.waiting_for_settle = False
                self.pending_interval_idx = None
                # Update Reference (Post 變成了新的 Pre)
                self.reference_snapshot = current_snapshot

        return is_triggered_now, reason

    def _on_verification_result(self, triggered, reason, interval_idx):
        """接收 Verifier 的結果並更新 Integrator 的資料"""
        intervals = self.integrator.intervals
        if interval_idx < len(intervals):
            if triggered:
                intervals[interval_idx]["label"] = LABEL_NAME2ID[LabelMap.TRIGGER]
                print(f"✅ Interval {interval_idx} CONFIRMED: {reason}")
            else:
                print(f"ℹ️ Interval {interval_idx} IGNORED: {reason}")
