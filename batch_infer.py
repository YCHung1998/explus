"""Batch inference over a dataset CSV. Config-driven via run.yaml.

    python batch_infer.py --config configs/run.yaml [--no-resume]

GT (ground_truth/data.json) is derived from the dataset CSV (replacing the
hardcoded DEFAULT_DIRECTORY_FNS list -> fixes #3). Clips that cannot be opened
are recorded in skipped.json instead of silently dropped (fixes #1).
"""

import argparse
import json
import os
from typing import List

from tqdm import tqdm

from src.config.run_config import RunConfig
from src.io.dataset_csv import read_dataset, rows_to_gt_data, video_id_from_path
from src.io.sources import build_source
from src.runtime.clip_runner import ClipOpenError, run_clip

MERGE_FILENAME = "merge_data.json"


def run_batch(run_cfg: RunConfig, resume: bool = True) -> None:
    if run_cfg.source.type != "dataset":
        raise ValueError(
            "batch_infer requires source.type == 'dataset', got "
            f"{run_cfg.source.type!r}")

    output_dir = run_cfg.output.dir
    predictions_dir = os.path.join(output_dir, "predictions")
    gt_dir = os.path.join(output_dir, "ground_truth")
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    merge_filepath = os.path.join(predictions_dir, MERGE_FILENAME)

    # 1. GT from CSV
    rows = read_dataset(run_cfg.source.dataset)
    gt_data = rows_to_gt_data(rows)
    with open(os.path.join(gt_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=4)
    print(f"GT written for {len(gt_data)} videos -> {gt_dir}/data.json")

    # 2. Inference per clip
    source = build_source(run_cfg.source)
    clips = source.clips()
    print(f"Total Videos: {len(clips)}")

    merge_data_results = {"results": {}}
    skipped: List[dict] = []

    for clip in tqdm(clips, desc="Processing"):
        video_id = video_id_from_path(clip.video_path)
        save_path = os.path.join(predictions_dir, video_id + ".json")

        if resume and os.path.exists(save_path):
            with open(save_path, "r") as f:
                data = json.load(f)
            merge_data_results["results"].update(data["results"])
            continue

        try:
            pipeline = run_clip(clip, run_cfg, recorder=None,
                                debug_visual=False, save_video=False)
        except ClipOpenError as e:
            print(f"SKIP {video_id}: {e}")
            skipped.append({"video_id": video_id, "reason": str(e)})
            continue

        results = pipeline.get_results()
        data = {"results": {video_id: results}}
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)
        merge_data_results["results"].update(data["results"])

    # 3. Merged + skip list
    with open(merge_filepath, "w") as fp:
        json.dump(merge_data_results, fp, indent=4)
    with open(os.path.join(output_dir, "skipped.json"), "w") as fp:
        json.dump(skipped, fp, indent=4)
    print(f"Merged results -> {merge_filepath}")
    if skipped:
        print(f"Skipped {len(skipped)} clips -> {output_dir}/skipped.json")


def main():
    parser = argparse.ArgumentParser(description="Batch inference (config-driven)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args()
    run_cfg = RunConfig.from_yaml(args.config)
    run_batch(run_cfg, resume=args.resume)


if __name__ == "__main__":
    main()
