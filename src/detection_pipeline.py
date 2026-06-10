"""Detection Pipeline with modular architecture and runtime parameter support."""

import json
import time
from typing import Optional, Dict, Any, List
import numpy as np

from src.preprocessor import ImagePreprocessor
from src.background import BackgroundManager
from src.texture_engine import TextureEngine
from src.motion_decider import AdaptiveDecisionUnit
from src.signal_processor import SignalIntegrator
from src.snapshot_manager import SnapshotController
from src.trigger_verifier import AsyncTriggerVerifier
from src.pipeline_config import PipelineConfig


class DetectionPipeline:
    """Facade for all detection modules, providing simple observe() API.

    This class maintains modular architecture while providing a clean interface
    similar to FrameHistory.observe(). Supports runtime parameter updates.
    """

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration.

        Args:
            config: PipelineConfig instance containing all module parameters
        """
        self.config = config

        # Initialize all modules (maintaining modularity)
        self.prep = ImagePreprocessor(
            process_wh=config.preprocessing.process_wh,
            block_size=config.preprocessing.block_size,
        )

        self.texture = TextureEngine(
            texture_sensitivity_ratio=config.texture.texture_sensitivity_ratio,
            texture_diff_threshold=config.texture.texture_diff_threshold,
            dark_region_noise_floor=config.texture.dark_region_noise_floor,
        )

        self.bg_mgr = BackgroundManager(bg_update_rate=config.background.bg_update_rate)

        self.adu = AdaptiveDecisionUnit(
            rows=self.prep.rows,
            cols=self.prep.cols,
            motion_sensitivity=config.adaptive.motion_sensitivity,
            boundary=config.adaptive.boundary,
        )

        self.signal_int = SignalIntegrator(
            fps=config.signal.fps,
            onset_frames=config.signal.onset_frames,
            cooldown_frames=config.signal.cooldown_frames,
        )

        self.trigger_verifier = AsyncTriggerVerifier(
            model_path=config.trigger.resolved_model_path,
            mode=config.trigger.mode,
            feature_abstraction_level=config.trigger.feature_abstraction_level,
            context_mirror_ratio=config.trigger.context_mirror_ratio,
            similarity_threshold=config.trigger.similarity_threshold,
            min_trigger_area=config.trigger.min_trigger_area,
            vertical_focus_range=config.trigger.vertical_focus_range,
            ema_threshold=config.trigger.ema_threshold,
        )

        self.controller = SnapshotController(
            integrator=self.signal_int,
            verifier=self.trigger_verifier,
            settle_frames=int(config.signal.fps * config.settle_frames_ratio),
        )

        # History management (similar to FrameHistory)
        self.history_size = config.history_size
        self._distances = []
        self._ema_distances = []
        self._unstable_events = []
        self._trigger_events = []
        self._score_maps = []

        # State tracking
        self.frame_index = -1
        self.is_initial = True
        self.last_time = None

        # Previous state for background update
        self.prev_centered_blocks = None
        self.ema_background = None

    def observe(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        # Runtime parameter overrides (optional, similar to FrameHistory)
        runtime_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a single frame and return detection results.

        Args:
            frame: Input frame (BGR format)
            timestamp: Optional timestamp for the frame
            runtime_params: Optional dict to override parameters at runtime.
                Format: {
                    'adaptive': {'motion_sensitivity': 40.0, 'boundary': [...]},
                    'texture': {'texture_sensitivity_ratio': 0.2, 'texture_diff_threshold': 10},
                    'background': {'bg_update_rate': 0.4},
                    'trigger': {'similarity_threshold': 0.5, 'min_trigger_area': 5},
                    'signal': {'fps': 15.0, 'onset_frames': 5, 'cooldown_frames': 10},
                }

        Returns:
            Dictionary containing:
            - is_motion: bool
            - is_stable: bool
            - is_trigger: bool
            - metrics: dict with various metrics
            - visualization_data: dict with data for visualization
        """
        self.frame_index += 1

        # Apply runtime parameter overrides if provided
        if runtime_params:
            self.config.update(runtime_params)
            # Update modules that need runtime updates
            self._apply_runtime_updates(runtime_params)

        # Calculate time delta
        if timestamp is None:
            timestamp = time.time()
        if self.last_time is None:
            self.last_time = timestamp
        dt = timestamp - self.last_time
        self.last_time = timestamp

        # -------- Processing Pipeline --------
        # 1. Preprocessing
        gray, blocks, centered_blocks = self.prep.process(frame)

        # 2. Texture analysis
        curr_lbsp, curr_diff = self.texture.compute_lbsp(gray)

        # 3. Background management
        if self.is_initial:
            _, _, self.ema_background, self.prev_centered_blocks = (
                self.bg_mgr.update_and_get_score(
                    gray,
                    blocks,
                    lower_update_mask=None,
                    lower_update_alpha=self.config.background.bg_update_rate / 2.0,
                )
            )
            self.is_initial = False
        else:
            self.ema_background, self.prev_centered_blocks = (
                self.bg_mgr.get_background()
            )

        # Background LBSP
        bg_lbsp, _ = self.texture.compute_lbsp(self.ema_background.astype(np.uint8))

        # 4. Block-based color difference
        diff_blocks = np.abs(centered_blocks - self.prev_centered_blocks)
        score_map = np.mean(diff_blocks, axis=(2, 3))

        # 5. Hamming distance map
        h_map, h_dist = self.texture.get_hamming_map(curr_lbsp, bg_lbsp)

        # 6. Adaptive Decision (with runtime parameter support)
        adaptive_params = {}
        if runtime_params and "adaptive" in runtime_params:
            adaptive_params = runtime_params["adaptive"]

        decision_results = self.adu.decide(
            prep=self.prep,
            gray=gray,
            background=self.ema_background,
            score_map=score_map,
            hamming_map=h_map,
            blocks_uint8=np.maximum(blocks, self.prev_centered_blocks).astype(np.uint8),
            dt=dt,
            **adaptive_params,  # Unpack runtime parameters
        )
        is_motion, active_mask, valid_mask = decision_results

        # 7. Signal processing
        signal_is_trigger, reason = self.controller.on_frame(
            frame_index=self.frame_index,
            frame=frame,
            is_motion=is_motion,
            ema_phash=frame,  # For ema_phash mode

            # TODO add boundary inside
            boundary=self.config.adaptive.boundary,
        )
        is_stable = self.controller.integrator.is_currently_stable()

        # 8. Update background
        score_map, curr_centered, self.ema_background, self.prev_centered_blocks = (
            self.bg_mgr.update_and_get_score(
                gray,
                blocks,
                lower_update_mask=None,
                lower_update_alpha=np.clip(self.config.background.bg_update_rate * 2, 0, 1),
            )
        )

        # 9. Update history
        # Calculate distance metric (similar to FrameHistory)
        distance = np.mean(h_dist) if len(h_dist) > 0 else 0.0
        ema_alpha = 0.33  # Could be configurable
        if len(self._ema_distances) > 0:
            ema_distance = (
                ema_alpha * distance + (1 - ema_alpha) * self._ema_distances[-1]
            )
        else:
            ema_distance = distance

        self._distances.append(distance)
        self._ema_distances.append(ema_distance)
        self._unstable_events.append(int(not is_stable))
        self._trigger_events.append(int(signal_is_trigger))
        # Only copy if needed for visualization (avoid unnecessary copy)
        self._score_maps.append(
            score_map.copy() if len(self._score_maps) < self.history_size else score_map
        )

        # Truncate history
        self._truncate_history()

        # Prepare return data
        return {
            "is_motion": is_motion,
            "is_stable": is_stable,
            "is_trigger": signal_is_trigger,
            "trigger_reason": reason,
            "metrics": {
                "distance": distance,
                "ema_distance": ema_distance,
                "max_score": np.max(score_map),
                "active_ratio": (
                    np.sum(active_mask > 0) / active_mask.size
                    if active_mask.size > 0
                    else 0
                ),
            },
            "visualization_data": {
                "gray": gray,
                "score_map": score_map,
                "active_mask": active_mask,
                "valid_mask": valid_mask,
                "hamming_map": h_map,
                "background": self.ema_background,
            },
        }

    def _apply_runtime_updates(self, runtime_params: Dict[str, Any]) -> None:
        """Apply runtime parameter updates to modules."""
        # Update texture engine
        if "texture" in runtime_params:
            texture_params = runtime_params["texture"]
            if "texture_sensitivity_ratio" in texture_params:
                self.texture.texture_sensitivity_ratio = texture_params["texture_sensitivity_ratio"]
            if "texture_diff_threshold" in texture_params:
                self.texture.texture_diff_threshold = texture_params["texture_diff_threshold"]
            if "dark_region_noise_floor" in texture_params:
                self.texture.dark_region_noise_floor = texture_params["dark_region_noise_floor"]

        # Update background manager
        if "background" in runtime_params:
            bg_params = runtime_params["background"]
            if "bg_update_rate" in bg_params:
                self.bg_mgr.bg_update_rate = bg_params["bg_update_rate"]

        # Update signal integrator (requires recalculation)
        if "signal" in runtime_params:
            signal_params = runtime_params["signal"]
            if "fps" in signal_params:
                self.signal_int.fps = signal_params["fps"]
            if (
                "stable_hold_time" in signal_params
                or "unstable_hold_time" in signal_params
            ):
                # Recalculate frames
                self.signal_int.onset_frames = self.config.signal.onset_frames
                self.signal_int.cooldown_frames = self.config.signal.cooldown_frames

    def _truncate_history(self) -> None:
        """Truncate history to maintain size limit."""
        if len(self._distances) > self.history_size:
            self._distances = self._distances[-self.history_size :]
        if len(self._ema_distances) > self.history_size:
            self._ema_distances = self._ema_distances[-self.history_size :]
        if len(self._unstable_events) > self.history_size:
            self._unstable_events = self._unstable_events[-self.history_size :]
        if len(self._trigger_events) > self.history_size:
            self._trigger_events = self._trigger_events[-self.history_size :]
        if len(self._score_maps) > self.history_size:
            self._score_maps = self._score_maps[-self.history_size :]

    # Properties for accessing history (similar to FrameHistory)
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
    def score_maps(self):
        return self._score_maps

    @property
    def stable(self):
        return self.controller.integrator.is_currently_stable()

    @property
    def trigger(self):
        return self.controller.payload is not None  # Simplified

    def clear(self) -> None:
        """Reset pipeline state."""
        self.frame_index = -1
        self.is_initial = True
        self.last_time = None
        self._distances = []
        self._ema_distances = []
        self._unstable_events = []
        self._trigger_events = []
        self._score_maps = []
        self.bg_mgr.reset()
        self.signal_int = SignalIntegrator(
            fps=self.config.signal.fps,
            onset_frames=self.config.signal.onset_frames,
            cooldown_frames=self.config.signal.cooldown_frames,
        )

    def finalize(self, frame_index: Optional[int] = None) -> None:
        """Finalize the detection process and close any open intervals.

        Args:
            frame_index: Index of the final frame. If None, uses the last processed index.
        """
        if frame_index is None:
            frame_index = self.frame_index
        self.signal_int.finalize(frame_index)

    def get_results(self) -> List[Dict[str, Any]]:
        """Return the detected intervals in a standardized format.

        Returns:
            List of dictionaries with 'segment', 'label', and 'score'.
        """
        return [
            {
                "segment": interval.get("segment", []),
                "label": interval.get("label", -1),
                "score": interval.get("score", 0),
            }
            for interval in self.signal_int.intervals
        ]

    def save_results(self, output_path: str, filename: Optional[str] = None) -> None:
        """Helper to save results to a JSON file.

        Args:
            output_path: Path to the JSON file or directory.
            filename: If output_path is a directory, the key name for the results.
        """
        results = self.get_results()
        if filename:
            data = {"results": {filename: results}}
        else:
            data = {"results": results}

        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Results saved to {output_path}")
