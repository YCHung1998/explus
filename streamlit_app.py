"""Streamlit application for interactive motion detection."""

import argparse
import os
import time

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.detection_pipeline import DetectionPipeline
from src.pipeline_config import PipelineConfig
from src.performance_monitor import PerformanceMonitor
import path_utils


@st.cache_resource
def get_detection_pipeline(config: PipelineConfig) -> DetectionPipeline:
    """Create and cache detection pipeline instance."""
    return DetectionPipeline(config)


@st.cache_resource
def load_config_defaults():
    """Load trigger/model defaults from an optional run.yaml.

    Run with:  streamlit run streamlit_app.py -- --config configs/neck_p4.yaml
    The yaml's `pipeline.trigger` (model_path, feature_position, ...) pre-fills
    the sidebar so the demo reuses a working model path instead of relying on
    the built-in default (which may not exist on this machine).

    Returns a dict {"path", "pipeline", "error"} or None when no --config given.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()
    if not args.config:
        return None
    try:
        from src.config.run_config import RunConfig

        return {"path": args.config,
                "pipeline": RunConfig.from_yaml(args.config).pipeline,
                "error": None}
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        return {"path": args.config, "pipeline": None, "error": str(exc)}


def compose_altair_heatmap(image, chart_size: int = 400):
    """Compose an Altair heatmap chart with consistent physical size."""
    h, w = image.shape[:2]
    x, y = np.meshgrid(range(w), range(h))

    source = pd.DataFrame(
        {
            "x": x.ravel(),
            "y": y.ravel(),
            "z": image.ravel(),
        }
    )

    x_enc = alt.X(
        "x:O",
        scale=alt.Scale(domain=list(range(w)), paddingInner=0, paddingOuter=0),
        axis=None,
    )
    y_enc = alt.Y(
        "y:O",
        scale=alt.Scale(domain=list(range(h)), paddingInner=0, paddingOuter=0),
        axis=None,
    )

    chart = (
        alt.Chart(source)
        .mark_rect()
        .encode(x=x_enc, y=y_enc, color=alt.Color("z:Q"))
        .properties(
            width=chart_size,
            height=chart_size,
        )
    )

    return chart


def main():
    st.set_page_config(layout="wide", page_title="Motion Detection System")
    st.header("🎯 Motion Detection System - Interactive Demo")

    # Optional run.yaml supplying model/trigger defaults (-- --config <yaml>).
    cfg_file = load_config_defaults()
    trig_defaults = cfg_file["pipeline"].trigger if cfg_file and cfg_file["pipeline"] else None
    if cfg_file and cfg_file["error"]:
        st.warning(f"⚠️ Failed to load --config `{cfg_file['path']}`: {cfg_file['error']}")
    elif cfg_file:
        st.caption(f"📄 Model/trigger defaults loaded from `{cfg_file['path']}`")

    # ========== Settings Panel ==========
    with st.sidebar:
        st.header("⚙️ Settings")

        with st.form("Settings"):
            # Input source
            source = st.selectbox(
                "Input Source",
                options=["Camera", "Selected Video", "Selected Image"],
                index=0,
                help="Choose the input source",
            )

            camera_id = st.number_input(
                "Camera Device ID",
                min_value=0,
                max_value=10,
                value=0,
                help="Camera device index (usually 0 for default camera)",
            )

            # Video file selection
            selected_video_relative = None
            video_root_dir = None
            if source == "Selected Video":
                video_root_dir = st.text_input(
                    "Video Directory Path",
                    value="/Users/eason.hung/Documents/Projects/explus/external_camera",
                    help="Directory containing video files",
                )
                if video_root_dir and os.path.exists(video_root_dir):
                    video_files_relative = path_utils.get_files_recursive(
                        video_root_dir,
                        supported_extensions=(".mp4", ".avi", ".mov"),
                        is_relative=True,
                    )
                    if video_files_relative:
                        selected_video_relative = st.selectbox(
                            f"Select Video ({len(video_files_relative)} found)",
                            options=video_files_relative,
                        )
                else:
                    st.warning("Please enter a valid directory path.")

            # Image file selection
            selected_image = None
            if source == "Selected Image":
                try:
                    image_files = sorted(
                        [
                            f
                            for f in os.listdir("images")
                            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                        ]
                    )
                    if image_files:
                        selected_image = st.selectbox(
                            "Select Image",
                            options=image_files,
                        )
                except Exception:
                    st.warning("Images directory not found.")

            frame_rate = st.number_input(
                "Target FPS",
                min_value=1,
                max_value=30,
                value=10,
                help="Target frames per second for processing",
            )

            st.divider()
            st.subheader("📊 Adaptive Decision Parameters")

            motion_sensitivity = st.number_input(
                "Motion Sensitivity",
                min_value=0.0,
                max_value=200.0,
                value=30.0,
                step=1.0,
                help="Motion sensitivity threshold for detection",
            )

            pixel_diff_threshold = st.number_input(
                "Pixel Difference Threshold",
                min_value=0.0,
                max_value=100.0,
                value=15.0,
                step=0.5,
                help="Block-based pixel difference threshold",
            )

            global_motion_ratio = st.number_input(
                "Global Motion Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.05,
                format="%.2f",
                help="Global motion ratio threshold",
            )

            # Boundary selection
            n_grids = 20
            boundary_options = [i / n_grids for i in range(n_grids)] + [1.0]
            boundary_left, boundary_right = st.select_slider(
                "X-axis Boundary",
                options=boundary_options,
                value=(0.0, 1.0),
                help="Horizontal boundary region (relative coordinates)",
            )
            boundary_top, boundary_bottom = st.select_slider(
                "Y-axis Boundary",
                options=boundary_options,
                value=(0.0, 1.0),
                help="Vertical boundary region (relative coordinates)",
            )

            adaptive_boundary = [
                boundary_left,
                boundary_top,
                boundary_right,
                boundary_bottom,
            ]

            st.divider()
            st.subheader("🎨 Texture Parameters")

            texture_sensitivity_ratio = st.number_input(
                "Texture Sensitivity Ratio",
                min_value=0.01,
                max_value=1.0,
                value=0.15,
                step=0.01,
                format="%.2f",
                help="LBSP relative intensity threshold (Sensitivity)",
            )

            texture_diff_threshold = st.number_input(
                "Texture Difference Threshold",
                min_value=1,
                max_value=16,
                value=5,
                step=1,
                help="Hamming distance threshold for texture change",
            )

            st.divider()
            st.subheader("🖼️ Background Parameters")

            bg_update_rate = st.number_input(
                "Background Update Rate",
                min_value=0.01,
                max_value=1.0,
                value=0.3,
                step=0.01,
                format="%.2f",
                help="Rate at which background adapts to current frame",
            )

            dark_region_noise_floor = st.number_input(
                "Dark Region Noise Floor",
                min_value=1,
                max_value=255,
                value=50,
                step=1,
                help="Minimum threshold to suppress noise in dark areas",
            )

            st.divider()
            st.subheader("📡 Signal Parameters")

            signal_stable_hold = st.number_input(
                "Stable Hold Time (sec)",
                min_value=0.01,
                max_value=2.0,
                value=0.1,
                step=0.01,
                format="%.2f",
                help="Time to maintain stable state",
            )

            signal_unstable_hold = st.number_input(
                "Unstable Hold Time (sec)",
                min_value=0.01,
                max_value=2.0,
                value=0.5,
                step=0.01,
                format="%.2f",
                help="Time to maintain unstable state",
            )

            st.divider()
            st.subheader("🔔 Trigger Parameters")

            mode_options = ["yolo", "ema_phash"]
            mode_default = trig_defaults.mode if trig_defaults else "yolo"
            trigger_mode = st.selectbox(
                "Trigger Mode",
                options=mode_options,
                index=mode_options.index(mode_default) if mode_default in mode_options else 0,
                help="Verification method for triggers",
            )

            if trigger_mode == "yolo":
                fp_options = ["Neck", "Backbone"]
                fp_default = trig_defaults.feature_position if trig_defaults else "Neck"
                feature_position = st.selectbox(
                    "Feature Position (model)",
                    options=fp_options,
                    index=fp_options.index(fp_default) if fp_default in fp_options else 0,
                    help="Selects the default ONNX model: "
                    "Neck -> models/best_vis_with_8400_3.onnx, "
                    "Backbone -> models/best_vis_with_8400_b3.onnx",
                )

                model_path = st.text_input(
                    "Model Path (override)",
                    value=(trig_defaults.model_path or "") if trig_defaults else "",
                    help="Explicit ONNX path. Leave empty to use the default for "
                    "the selected Feature Position. Pre-filled from --config when given.",
                )

                fl_options = ["P4", "fusion"]
                fl_default = trig_defaults.feature_abstraction_level if trig_defaults else "P4"
                feature_level = st.selectbox(
                    "Feature Abstraction Level",
                    options=fl_options,
                    index=fl_options.index(fl_default) if fl_default in fl_options else 0,
                    help="Model feature layer to extract",
                )

                context_mirror_ratio = st.slider(
                    "Context Mirror Ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1,
                    help="Expansion ratio for image boundaries",
                )

                similarity_threshold = st.slider(
                    "Feature Change Threshold",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.6 if feature_level == "P4" else 0.4,
                    step=0.05,
                    help="Similarity threshold for feature map comparison",
                )

                min_trigger_area = st.number_input(
                    "Minimum Trigger Spots",
                    min_value=1,
                    max_value=100,
                    value=2,
                    step=1,
                    help="Minimum required changed pixels on feature map",
                )

                v_focus_min, v_focus_max = st.slider(
                    "Vertical Focus ROI",
                    min_value=0,
                    max_value=640,
                    value=(140, 500),
                    step=10,
                    help="Vertical pixel range for feature comparison",
                )
                vertical_focus_range = [v_focus_min, v_focus_max]
            else:
                trigger_ema_threshold = st.number_input(
                    "EMA Threshold",
                    min_value=0.0,
                    max_value=200.0,
                    value=100.0,
                    step=1.0,
                    help="EMA phash distance threshold (ema_phash mode)",
                )

            st.divider()
            st.subheader("📈 Display Options")

            history_size = st.number_input(
                "History Size",
                min_value=50,
                max_value=1000,
                value=300,
                step=50,
                help="Number of frames to keep in history",
            )

            visualization_mode = st.selectbox(
                "Visualization Mode",
                options=[
                    "Original Frame",
                    "Score Map Overlay",
                    "Active Mask Overlay",
                ],
                index=0,
            )

            # Performance options
            st.divider()
            st.subheader("⚡ Performance Options")

            update_charts_every = st.number_input(
                "Update Charts Every N frames",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="Reduce chart update frequency for better performance",
            )

            update_heatmaps_every = st.number_input(
                "Update Score Heatmap Every N frames",
                min_value=1,
                max_value=20,
                value=10,
                step=1,
                help="Reduce score heatmap update frequency",
            )

            # Image compression for display
            display_image_scale = st.slider(
                "Display Image Scale",
                min_value=0.3,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Scale down images for display to improve performance (0.6 = 60% size)",
            )

            enable_perf_monitor = st.checkbox(
                "Enable Performance Monitor",
                value=False,
                help="Track timing for each operation (for debugging)",
            )

            submitted = st.form_submit_button("💾 Save & Start", type="primary")

    # ========== Initialize Pipeline ==========
    if not submitted:
        st.info('👆 Adjust settings in the sidebar and click "Save & Start" to begin.')
        return

    # Create configuration
    config = PipelineConfig()
    config.adaptive.motion_sensitivity = motion_sensitivity
    config.adaptive.pixel_diff_threshold = pixel_diff_threshold
    config.adaptive.global_motion_ratio = global_motion_ratio
    config.adaptive.boundary = adaptive_boundary
    config.texture.texture_sensitivity_ratio = texture_sensitivity_ratio
    config.texture.texture_diff_threshold = texture_diff_threshold
    config.texture.dark_region_noise_floor = dark_region_noise_floor
    config.background.bg_update_rate = bg_update_rate
    config.signal.fps = frame_rate
    config.signal.stable_hold_time = signal_stable_hold
    config.signal.unstable_hold_time = signal_unstable_hold
    config.trigger.mode = trigger_mode
    if trigger_mode == "yolo":
        config.trigger.feature_position = feature_position
        config.trigger.model_path = model_path.strip() or None
        config.trigger.feature_abstraction_level = feature_level
        config.trigger.context_mirror_ratio = context_mirror_ratio
        config.trigger.similarity_threshold = similarity_threshold
        config.trigger.min_trigger_area = min_trigger_area
        config.trigger.vertical_focus_range = vertical_focus_range
    else:
        config.trigger.ema_threshold = trigger_ema_threshold
    
    config.trigger.boundary = adaptive_boundary
    config.history_size = history_size

    # Fail fast with a clear message if the (config-driven) model is missing,
    # instead of a deep onnxruntime error inside the pipeline.
    if trigger_mode == "yolo":
        resolved_model = config.trigger.resolved_model_path
        if not os.path.exists(resolved_model):
            st.error(
                f"❌ ONNX model not found: `{resolved_model}`. "
                "Pass `-- --config <run.yaml>` or set **Model Path (override)** "
                "in the sidebar."
            )
            return

    # Initialize pipeline
    pipeline = get_detection_pipeline(config)

    # ========== Setup Input Source ==========
    camera = None
    full_video_path = None
    image = None
    frame_skip_interval = 1

    if source == "Camera":
        camera = cv2.VideoCapture(camera_id)
        if not camera.isOpened():
            st.error(f"❌ Cannot open camera {camera_id}")
            st.stop()
    elif source == "Selected Video":
        if selected_video_relative and video_root_dir:
            full_video_path = os.path.join(video_root_dir, selected_video_relative)
            if not os.path.exists(full_video_path):
                st.error(f"❌ Video file not found: {full_video_path}")
                st.stop()
            camera = cv2.VideoCapture(full_video_path)
            original_fps = camera.get(cv2.CAP_PROP_FPS)
            if frame_rate < original_fps:
                frame_skip_interval = max(1, round(original_fps / frame_rate))
                st.info(
                    f"📹 Original FPS: {original_fps:.2f}, Target: {frame_rate}, Skip: {frame_skip_interval}"
                )
        else:
            st.error("❌ No video selected")
            st.stop()
    elif source == "Selected Image":
        if selected_image:
            image_path = os.path.join("images", selected_image)
            image = cv2.imread(image_path)
            if image is None:
                st.error(f"❌ Cannot load image: {image_path}")
                st.stop()
        else:
            st.error("❌ No image selected")
            st.stop()

    # ========== Main Display Area ==========
    st.header("📺 Live Detection")

    # Status display
    status_placeholder = st.empty()
    status_placeholder.write("### ❓ Initializing...")

    current_metrics = st.empty()

    # Layout: Main view + Charts
    top_cols = st.columns([2, 1])
    with top_cols[0]:
        display_window = st.image([])
    with top_cols[1]:
        event_chart_placeholder = st.empty()
        distance_chart_placeholder = st.empty()
        score_chart_placeholder = st.empty()

    # Bottom: Comparison View (Prev vs Curr frames when stable transition)
    bottom_cols = st.columns([2, 2, 2])
    with bottom_cols[0]:
        st.caption("📸 Previous Frame (Before Stable)")
        prev_frame_placeholder = st.empty()
    with bottom_cols[1]:
        st.caption("📸 Current Frame (After Stable)")
        curr_frame_placeholder = st.empty()
    with bottom_cols[2]:
        score_heatmap_placeholder = st.empty()
        metrics_placeholder = st.empty()

    # ========== Main Processing Loop ==========
    start_time = time.time()
    sleep_time = 1.0 / (frame_rate + 1e-7)

    # Performance monitoring
    perf_monitor = PerformanceMonitor(enabled=enable_perf_monitor)

    # Frame counters for selective updates
    frame_count = 0

    try:
        while True:
            frame_count += 1
            should_update_charts = frame_count % update_charts_every == 0
            should_update_heatmaps = frame_count % update_heatmaps_every == 0
            # Get frame
            if source == "Camera":
                time.sleep(sleep_time)
                ret, frame = camera.read()
                if not ret:
                    break
            elif source == "Selected Video":
                for _ in range(frame_skip_interval):
                    ret, frame = camera.read()
                    if not ret:
                        if source == "Selected Video":
                            st.success("✅ Video processing completed!")
                            st.stop()
                        break
                if not ret:
                    break
            elif source == "Selected Image":
                frame = image.copy()
                time.sleep(0.1)  # Slow down for image mode

            # Process frame
            perf_monitor.start("total_frame")
            timestamp = time.time() - start_time

            perf_monitor.start("pipeline_observe")
            result = pipeline.observe(frame, timestamp=timestamp)
            perf_monitor.end()

            # Extract results
            is_stable = result["is_stable"]
            is_trigger = result["is_trigger"]
            trigger_reason = result["trigger_reason"]
            metrics = result["metrics"]
            viz_data = result["visualization_data"]

            # Update status
            if is_stable:
                status_text = "### ✅ Stable"
            else:
                status_text = "### 🚨 Unstable"
            if is_trigger:
                status_text += " 🔔 **TRIGGER!**"
                st.toast("🔔 Trigger Detected!", icon="🔔")
                st.toast(f"{trigger_reason}", icon="🔔")

            status_placeholder.write(status_text)

            # Update metrics
            current_metrics.text(
                f"📊 EMA Distance: {metrics['ema_distance']:.6f} | "
                f"Max Score: {metrics['max_score']:.2f} | "
                f"Active Ratio: {metrics['active_ratio']:.2%}"
            )

            # Prepare visualization frame (with compression)
            display_frame = frame.copy()

            # Compress display frame for better performance
            if display_image_scale < 1.0:
                h, w = display_frame.shape[:2]
                new_w = int(w * display_image_scale)
                new_h = int(h * display_image_scale)
                display_frame = cv2.resize(
                    display_frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                )

            if visualization_mode == "Score Map Overlay":
                score_map = viz_data["score_map"]
                score_map_resized = cv2.resize(
                    score_map,
                    (display_frame.shape[1], display_frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                score_map_norm = cv2.normalize(
                    score_map_resized, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                heat_colormap = cv2.applyColorMap(score_map_norm, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(display_frame, 0.6, heat_colormap, 0.4, 0)
                display_frame = overlay
            elif visualization_mode == "Active Mask Overlay":
                active_mask = viz_data["active_mask"]
                active_mask_resized = cv2.resize(
                    active_mask.astype(np.uint8) * 255,
                    (display_frame.shape[1], display_frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                active_mask_colored = cv2.applyColorMap(
                    active_mask_resized, cv2.COLORMAP_HOT
                )
                overlay = cv2.addWeighted(
                    display_frame, 0.7, active_mask_colored, 0.3, 0
                )
                display_frame = overlay

            # Draw boundary if specified
            if adaptive_boundary and adaptive_boundary != [0.0, 0.0, 1.0, 1.0]:
                h, w = display_frame.shape[:2]
                x1 = int(w * adaptive_boundary[0])
                y1 = int(h * adaptive_boundary[1])
                x2 = int(w * adaptive_boundary[2])
                y2 = int(h * adaptive_boundary[3])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display main frame
            perf_monitor.start("display_frame")
            display_window.image(
                display_frame, channels="BGR", use_container_width=True
            )
            perf_monitor.end()

            # Update charts (selective)
            if should_update_charts:
                perf_monitor.start("update_charts")
                event_chart_placeholder.bar_chart(
                    {
                        "trigger": [float(e) for e in pipeline.trigger_events],
                        "unstable": [float(-e) for e in pipeline.unstable_events],
                    },
                    use_container_width=True,
                )

                distance_chart_placeholder.bar_chart(
                    {
                        "distance": pipeline.distances,
                        "ema_distance": [-1*v for v in  pipeline.ema_distances],
                    },
                    use_container_width=True,
                )

                # Score map chart
                if len(pipeline.score_maps) > 0:
                    max_scores = [np.max(sm) for sm in pipeline.score_maps]
                    score_chart_placeholder.bar_chart(
                        {"max_score": max_scores},
                        use_container_width=True,
                    )

                # Score map heatmap (optional, less frequent)
                if should_update_heatmaps:
                    perf_monitor.start("update_score_heatmap")
                    score_map = viz_data["score_map"]
                    if score_map.size > 0:
                        score_heatmap_placeholder.altair_chart(
                            compose_altair_heatmap(score_map),
                            use_container_width=False,
                        )
                    perf_monitor.end()
                perf_monitor.end()

            # Display comparison frames (prev vs curr) when payload exists
            if pipeline.controller.payload:
                perf_monitor.start("update_comparison")
                payload = pipeline.controller.payload

                # Get previous and current frames
                prev_img = payload.prev_meta.ori_img
                curr_img = payload.curr_meta.ori_img

                # Compress images for display
                if display_image_scale < 1.0:
                    h_prev, w_prev = prev_img.shape[:2]
                    h_curr, w_curr = curr_img.shape[:2]
                    new_w_prev = int(w_prev * display_image_scale)
                    new_h_prev = int(h_prev * display_image_scale)
                    new_w_curr = int(w_curr * display_image_scale)
                    new_h_curr = int(h_curr * display_image_scale)

                    prev_img = cv2.resize(
                        prev_img, (new_w_prev, new_h_prev), interpolation=cv2.INTER_AREA
                    )
                    curr_img = cv2.resize(
                        curr_img, (new_w_curr, new_h_curr), interpolation=cv2.INTER_AREA
                    )

                # Display comparison
                prev_frame_placeholder.image(
                    prev_img, channels="BGR", use_container_width=True
                )
                curr_frame_placeholder.image(
                    curr_img, channels="BGR", use_container_width=True
                )
                perf_monitor.end()
            else:
                # Clear placeholders when no payload
                prev_frame_placeholder.empty()
                curr_frame_placeholder.empty()

            perf_monitor.end()  # End total_frame

            # Display performance stats if enabled
            if enable_perf_monitor and frame_count % 30 == 0:
                stats = perf_monitor.get_stats()
                if stats:
                    perf_text = " | ".join(
                        [
                            f"{k}: {v['mean']*1000:.1f}ms"
                            for k, v in sorted(
                                stats.items(), key=lambda x: x[1]["mean"], reverse=True
                            )[:5]
                        ]
                    )
                    metrics_placeholder.text(f"⚡ Performance: {perf_text}")

    except KeyboardInterrupt:
        st.info("⏹️ Processing stopped by user")
    finally:
        if camera:
            camera.release()


if __name__ == "__main__":
    main()
