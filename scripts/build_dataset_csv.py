"""Fixed reference script: ActivityNet GT json dir -> dataset CSV index.

One-way and regenerable. The json stays the source of truth.

IMPORTANT: pass --src as an ABSOLUTE path. The CSV's video_path column copies
--src's form verbatim; downstream (cv2.VideoCapture, boundary-json lookup, eval
dashboard video serving) all consume it directly, so a relative path fails once
the data is not under the current working directory (e.g. on a server mount).

Usage:
    python scripts/build_dataset_csv.py \\
        --src /vol/08822801/AutoTrigger/dataset/external_camera/ \\
        --out datasets/bread_demo.csv
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io.dataset_csv import activitynet_dir_to_rows, write_dataset


def build_dataset_csv(src_dir: str, out_csv: str) -> int:
    rows = activitynet_dir_to_rows(src_dir)
    write_dataset(rows, out_csv)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten ActivityNet GT json into a dataset CSV index."
    )
    parser.add_argument("--src", required=True,
                        help="ABSOLUTE path to the dir containing "
                             "*_annotations_ActivityNet.json (relative paths "
                             "break once data moves off the current CWD)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    count = build_dataset_csv(args.src, args.out)
    print(f"Wrote {count} segment rows to {args.out}")


if __name__ == "__main__":
    main()
