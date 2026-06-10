"""Pipeline configuration management with support for runtime parameter updates."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class AdaptiveDecisionConfig:
    """Configuration for AdaptiveDecisionUnit"""
    motion_sensitivity: float = 30.0
    pixel_diff_threshold: float = 15.0  # Block-based score threshold
    global_motion_ratio: float = 0.85  # Active grid ratio threshold
    boundary: Optional[list] = None  # [x1, y1, x2, y2] in relative coordinates


@dataclass
class TextureConfig:
    """Configuration for TextureEngine (LBSP-based)"""

    texture_sensitivity_ratio: float = 0.15  # SuBSENSE R_rel
    texture_diff_threshold: int = 5          # SuBSENSE T (Hamming distance)
    dark_region_noise_floor: int = 50        # Noise floor for low intensity


@dataclass
class BackgroundConfig:
    """Configuration for BackgroundManager"""

    bg_update_rate: float = 0.3


@dataclass
class SignalConfig:
    """Configuration for SignalIntegrator"""

    fps: float = 10.0
    stable_hold_time: float = 0.1  # seconds
    unstable_hold_time: float = 0.5  # seconds

    @property
    def onset_frames(self) -> int:
        # TODO (Eason): Stable state to Unstable state hold on time
        # stable2unstable_hold_time
        """Calculate onset_frames from stable_hold_time and fps"""
        return int(self.stable_hold_time * self.fps)

    @property
    def cooldown_frames(self) -> int:
        # TODO (Eason): Unstable state to Stable state hold on time
        # unstable2stable_hold_time
        """Calculate cooldown_frames from unstable_hold_time and fps"""
        return int(self.unstable_hold_time * self.fps)


@dataclass
class TriggerConfig:
    """Configuration for AsyncTriggerVerifier (Mid-Level)"""

    mode: str = "yolo"  # "yolo" or "ema_phash"
    feature_position: str = "Neck" # "Neck" or "Backbone"
    feature_abstraction_level: str = "P4"  # Formerly 'version'. e.g., 'P4' or 'fusion'
    context_mirror_ratio: float = 0.0  # Formerly 'mirror_ratio'

    # Explicit ONNX model path (run.yaml: pipeline.trigger.model_path).
    # If None, resolved_model_path falls back to a default by feature_position.
    # Point this at e.g. /vol/08822801/AutoTrigger/model/<name>.onnx
    model_path: Optional[str] = None

    # YOLO Feature Logic
    similarity_threshold: float = 0.6  # Default for P4 is 0.6; fusion usually 0.4
    min_trigger_area: int = 2          # Min change spots on feature map
    vertical_focus_range: list = field(default_factory=lambda: [140, 500])

    boundary: Optional[list] = None  # [x1, y1, x2, y2]
    ema_threshold: float = 100.0     # for ema_phash mode

    # Model Path Resolution: explicit model_path wins; else by feature_position.
    @property
    def resolved_model_path(self) -> str:
        if self.model_path:
            return self.model_path
        if self.feature_position == "Backbone":
            return "./models/best_vis_with_8400_b3.onnx"
        return "./models/best_vis_with_8400_3.onnx"


@dataclass
class PreprocessingConfig:
    """Configuration for ImagePreprocessor"""

    process_wh: tuple = (320, 180)
    block_size: tuple = (20, 20)


@dataclass
class PipelineConfig:
    """Main pipeline configuration container"""

    adaptive: AdaptiveDecisionConfig = field(default_factory=AdaptiveDecisionConfig)
    texture: TextureConfig = field(default_factory=TextureConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    # Snapshot controller
    settle_frames_ratio: float = 0.3  # ratio of frame_rate

    # History management
    history_size: int = 300

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with nested dictionary.

        Example:
            config.update({
                'adaptive': {'base_sens': 40.0, 'boundary': [0.1, 0.1, 0.9, 0.9]},
                'texture': {'rel_threshold': 0.2},
                'signal': {'fps': 15.0}
            })
        """
        for key, value in updates.items():
            if hasattr(self, key):
                config_obj = getattr(self, key)
                if isinstance(value, dict):
                    # Update nested config
                    for nested_key, nested_value in value.items():
                        if hasattr(config_obj, nested_key):
                            setattr(config_obj, nested_key, nested_value)
                        else:
                            raise ValueError(f"Unknown parameter: {key}.{nested_key}")
                else:
                    # Update top-level attribute
                    setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config section: {key}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            "adaptive": {
                "motion_sensitivity": self.adaptive.motion_sensitivity,
                "pixel_diff_threshold": self.adaptive.pixel_diff_threshold,
                "global_motion_ratio": self.adaptive.global_motion_ratio,
                "boundary": self.adaptive.boundary,
            },
            "texture": {
                "texture_sensitivity_ratio": self.texture.texture_sensitivity_ratio,
                "texture_diff_threshold": self.texture.texture_diff_threshold,
                "dark_region_noise_floor": self.texture.dark_region_noise_floor,
            },
            "background": {
                "bg_update_rate": self.background.bg_update_rate,
            },
            "signal": {
                "fps": self.signal.fps,
                "stable_hold_time": self.signal.stable_hold_time,
                "unstable_hold_time": self.signal.unstable_hold_time,
            },
            "trigger": {
                "boundary": self.trigger.boundary,
                "mode": self.trigger.mode,
                # yolo
                "model_path": self.trigger.model_path,
                "resolved_model_path": self.trigger.resolved_model_path,
                "feature_position": self.trigger.feature_position,
                "feature_abstraction_level": self.trigger.feature_abstraction_level,
                "context_mirror_ratio": self.trigger.context_mirror_ratio,
                "similarity_threshold": self.trigger.similarity_threshold,
                "min_trigger_area": self.trigger.min_trigger_area,
                "vertical_focus_range": self.trigger.vertical_focus_range,
                # ema_phash
                "ema_threshold": self.trigger.ema_threshold,
            },
            "preprocessing": {
                "process_wh": self.preprocessing.process_wh,
                "block_size": self.preprocessing.block_size,
            },
            "settle_frames_ratio": self.settle_frames_ratio,
            "history_size": self.history_size,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary"""
        config = cls()
        config.update(config_dict)
        return config
