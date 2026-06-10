from typing import List, Optional, Union, Any
from constants import SignalState, LabelMap, LABEL_NAME2ID


class SignalIntegrator:
    """Consolidates motion signals into stable/unstable state intervals.

    Attributes:
        fps: Frames per second of the video source.
        onset_frames: Threshold to enter Unstable state.
        cooldown_frames: Tolerance threshold to return to Stable state.
        state: Current logical state of the signal.
        prev_state: State of the signal in the previous frame.
        unstable_counter: Consecutive frames with motion.
        stable_counter: Consecutive frames without motion.
        start_frame_index: Recording start point of the current event.
        intervals: List of identified motion segments.
    """

    def __init__(
        self,
        fps: float,
        onset_frames: int = 3,
        cooldown_frames: int = 5
    ) -> None:
        self.fps: float = fps
        self.onset_frames: int = onset_frames
        self.cooldown_frames: int = cooldown_frames

        # Internal State
        self.state: SignalState = SignalState.STABLE
        self.prev_state: SignalState = SignalState.STABLE
        self.unstable_counter: int = 0
        self.stable_counter: int = 0

        self.start_frame_index: Optional[int] = None
        self._intervals: List[dict[str, Any]] = []

    def update(self, frame_index: int, is_motion: bool) -> Optional[int]:
        """Updates internal state and returns new interval index if finished.

        目前邏輯僅在 Unstable -> Stable 轉換時才建立區間。如果影片結束時仍處於 
        UNSTABLE 狀態，需呼叫 finalize() 來確保最後一段運動被記錄。

        Args:
            frame_index: Current frame count from video stream.
            is_motion: Boolean flag indicating motion detection
                       in current frame.

        Returns:
            The index of the newly created interval in self._intervals,
            or None if no interval was completed in this frame.
        """
        # 1. Hysteresis Counter Logic
        if is_motion:
            self.unstable_counter += 1
            self.stable_counter = 0
        else:
            self.stable_counter += 1
            self.unstable_counter = 0

        # 2. State Transition Logic
        if (self.state == SignalState.STABLE and
                self.unstable_counter >= self.onset_frames):
            self.state = SignalState.UNSTABLE
        elif (self.state == SignalState.UNSTABLE and
                self.stable_counter >= self.cooldown_frames):
            self.state = SignalState.STABLE

        # 3. Edge Detection & Recording
        new_interval_idx: Optional[int] = None

        # Edge: Stable -> Unstable (Event Start)
        if (self.prev_state == SignalState.STABLE and
                self.state == SignalState.UNSTABLE):
            self.start_frame_index = frame_index

        # Edge: Unstable -> Stable (Event End)
        elif (self.prev_state == SignalState.UNSTABLE and
                self.state == SignalState.STABLE):
            if self.start_frame_index is not None:
                new_interval_idx = self._create_interval(
                    self.start_frame_index,
                    frame_index
                )
                self.start_frame_index = None

        self.prev_state = self.state
        return new_interval_idx

    def finalize(self, last_frame_index: int) -> Optional[int]:
        """Closes any active UNSTABLE interval at the end of the stream.

        Args:
            last_frame_index: The index of the final frame in the stream.

        Returns:
            The index of the newly closed interval, or None if no interval was open.
        """
        if self.state == SignalState.UNSTABLE and self.start_frame_index is not None:
            # Force transition to STABLE logically (or just close it)
            new_interval_idx = self._create_interval(
                self.start_frame_index,
                last_frame_index
            )
            self.state = SignalState.STABLE
            self.start_frame_index = None
            return new_interval_idx
        return None

    def _create_interval(self, start: int, end: int) -> int:
        """Internal helper to package and store interval data."""
        data: dict[str, Union[int, float, list[float]]] = {
            "start_frame": start,
            "end_frame": end,
            "duration": (end - start) / self.fps,
            "label": LABEL_NAME2ID[LabelMap.POSITIVE_UNSTABLE],
            "score": 1.0,
            "segment": [start / self.fps, end / self.fps]
        }
        self._intervals.append(data)
        return len(self._intervals) - 1

    def update_intervals(self, interval_index, update_id):
        self._intervals[interval_index]["label"] = update_id

    @property
    def intervals(self):
        """Returns the current unstable intervals."""
        return self._intervals

    def get_stable_duration(self) -> int:
        """Returns the current consecutive stable frame count."""
        return self.stable_counter

    def is_currently_stable(self) -> bool:
        """Checks if the system is in a stable state."""
        return self.state == SignalState.STABLE
