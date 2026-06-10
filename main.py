"""Single-source runtime entry. Config-driven via run.yaml.

    python main.py --config configs/run.yaml [-f VIDEO] [-sign_record]
                   [--no-visual] [--save-video]

-f overrides the config source to a single video (fixes the old dead
VIDEO_FILES_TO_TEST list). -sign_record force-enables edge state recording.
Algorithm / model settings (mode, feature, fps) come from run.yaml's pipeline:.
"""

import argparse

from src.config.run_config import RunConfig
from src.io.sources import VideoSource, build_source
from src.io.state_recorder import StateRecorder
from src.runtime.clip_runner import run_clip


def main():
    parser = argparse.ArgumentParser(description="AutoTrigger single-source runtime")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to run.yaml")
    parser.add_argument("-f", "--file", type=str, default=None,
                        help="Override source with a single video path")
    parser.add_argument("-sign_record", "--sign_record", action="store_true",
                        help="Force-enable edge state recording (append-only)")
    parser.add_argument("--no-visual", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    run_cfg = RunConfig.from_yaml(args.config)

    if args.file:
        source = VideoSource(args.file)
    else:
        source = build_source(run_cfg.source)

    recorder = StateRecorder(
        run_cfg.record.path,
        enabled=run_cfg.record.enabled or args.sign_record)

    for clip in source.clips():
        run_clip(clip, run_cfg, recorder,
                 debug_visual=not args.no_visual,
                 save_video=args.save_video)


if __name__ == "__main__":
    main()
