import json
import argparse
from pathlib import Path
from tqdm import tqdm
import path_utils
import os
import numpy as np


def convert_vidat_to_thumos(json_file_path, output_txt_path=None):
    """
    Convert Vidat JSON format to THUMOS txt format.

    Args:
        json_file_path: Path to an input Vidat JSON file
        output_txt_path: Path to output THUMOS txt file (optional)
                        If not provided, uses the same name as input with .txt
                        extension

    THUMOS Format: [video name] [starting time] [ending time] [class label] [confidence score]
    """

    # Load JSON file
    with open(json_file_path, "r") as f:
        vidat_data = json.load(f)

    # Extract video information
    video_filename = Path(json_file_path).stem

    # Extract action annotations
    action_list = vidat_data["annotation"]["actionAnnotationList"]

    # Determine the output file path
    if output_txt_path is None:
        output_txt_path = json_file_path.replace(".json", ".txt")

    # Write THUMOS format
    with open(output_txt_path, "w") as f:
        for action in action_list:
            start_time = action["start"]
            end_time = action["end"]
            class_label = action["action"]  # Class label from action ID
            confidence_score = 1  # Fixed confidence score

            # Format: video_name, start_time, end_time, class_label, confidence_score
            line = f"{video_filename}, {start_time}, {end_time}, {class_label}, {confidence_score}"
            f.write(line + "\n")

    print(f"Successfully converted {json_file_path} to {output_txt_path}")
    print(f"Total actions exported: {len(action_list)}")


def convert_vidat_to_ActivityNet(json_file_path, output_json_path=None):
    """
    Convert Vidat JSON format to ActivityNet-style JSON result format.

    Args:
        json_file_path: Path to an input Vidat JSON file
        output_json_path: Path to output ActivityNet JSON file (optional)
                        If not provided, uses the input filename appended
                        with 'ActivityNet.json'
    """
    # Determine the output file path
    if output_json_path is None:
        output_json_path = Path(json_file_path).parent / (
            Path(json_file_path).stem + "_ActivityNet.json"
        )
    output_json_path = str(output_json_path)
    try:
        with open(json_file_path, "r") as f:
            vidat_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return

    # Extract video information
    try:
        # NOTE(Eason): Vidat data change the name. we get the video name from
        # json_file_path directly
        video_filename = Path(json_file_path).stem
        if "_annotations" in video_filename:
            video_filename = video_filename.replace("_annotations", "")
    except KeyError:
        print(
            "Warning: Missing 'annotation' or 'video' structure in JSON. \
            Using 'unknown_video' as filename."
        )
        video_filename = "unknown_video"

    # video meta
    video_meta = vidat_data["annotation"].get("video", {})
    width, height = video_meta["width"], video_meta["height"]
    # Extract video boundary if is None [xyxy] -> [0011]
    # [x_left, y_top, x_right, y_bottom]
    # TODO BOUNDARY
    object_bbox_in_frame = vidat_data["annotation"].get("objectAnnotationListMap", {})
    if len(object_bbox_in_frame)==0:
        boundary_info = [0,0,1,1]

    # NOTE(Eason): we will set the boundary in the video first frame.
    for k_frame_index, v_obj_list in object_bbox_in_frame.items():
        # TODO refactor
        BOUNDARY_FRAME_INDEX = "0"
        # Our setiing is in the first frame to label boundary
        if (
            k_frame_index == BOUNDARY_FRAME_INDEX
            or k_frame_index == int(BOUNDARY_FRAME_INDEX)
        ):
            v_obj = v_obj_list[0]
            x_left = np.clip(v_obj["x"], 0, width) / width
            y_top = np.clip(v_obj["y"], 0, height) / height
            x_right = np.clip(v_obj["x"] + v_obj["width"], 0, width) / width
            y_bottom = np.clip(v_obj["y"] + v_obj["height"], 0, height) / height
            boundary_info = [
                x_left,
                y_top,
                x_right,
                y_bottom,
            ]
            break

    # Extract action annotations (temporal interval)
    action_list = vidat_data["annotation"].get("actionAnnotationList", [])
    configs_actLabels = vidat_data["config"].get("actionLabelData", {})
    if not action_list:
        print(
            "Warning: No actions found in 'actionAnnotationList'. Output \
            file will contain an empty result list."
        )
    video_meta = vidat_data["annotation"].get("video", [])
    if not video_meta:
        print(f"Path: {json_file_path}")
        print(
            "Warning: No actions found in 'actionAnnotationList'. Output \
            file will contain an empty result list."
        )
        video_meta["duration"] = 0.0
    video_meta["src"] = video_filename

    # ActivityNet format requires results keyed by video ID (filename)
    results = []
    export_count = 0
    for action in action_list:
        segment = [action.get("start", 0), action.get("end", 0)]
        class_label = action.get("action", "unknown_action")
        confidence_score = 1.0
        results.append(
            {"segment": segment, "label": class_label, "score": confidence_score}
        )
        export_count += 1

    # increasing temport interval
    results = sorted(results, key=lambda x: x["segment"][0])
    # The final ActivityNet result JSON structure
    data = {
        "version": "VIDAT-to-ActivityNet_v1.0",
        video_filename: {
            "annotations": results,
            "video": video_meta,
            "config": configs_actLabels,
            "boundary": boundary_info
        },
    }
    try:
        with open(output_json_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error writing output file: {e}")
        return


def convert_all_json_under_dir(directory, mode):
    if not os.path.exists(directory):
        print(f"[File or path does not exists] {directory}")
        return
    if mode not in ["thumos", "activitynet"]:
        print(f"[Convert mode {mode} does not exists]")
        return
    elif mode == "thumos":
        process = convert_vidat_to_thumos
    elif mode == "activitynet":
        process = convert_vidat_to_ActivityNet

    all_anno_video_paths = path_utils.get_files_recursive(
        directory,
        # NOTE(Eason): vidat 精標之後取的後綴名稱
        supported_extensions=("_annotations.json"),
    )

    for anno_path in tqdm(all_anno_video_paths):
        process(anno_path)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Vidat JSON annotations to different video action \
            recognition formats (THUMOS or ActivityNet)."
    )

    # Positional Arguments
    parser.add_argument(
        "-m",
        "--mode",
        choices=["thumos", "activitynet"],
        help="The target format for conversion: 'thumos' (TXT output) or \
            'activitynet' (JSON output).",
    )
    parser.add_argument(
        "-i",
        "--input_json",
        type=str,
        help="Path to the required input Vidat JSON annotation file.",
    )

    # Optional Argument
    parser.add_argument(
        "-o",
        "--output_file",
        nargs="?",  # Makes it optional. If not provided, it will be None.
        default=None,
        help="Optional path to the desired output file. If omitted, a path \
            will be inferred based on the input filename and mode.",
    )

    parser.add_argument(
        "-d",
        "--directory",
        nargs="?",  # Makes it optional. If not provided, it will be None.
        default=None,
    )

    args = parser.parse_args()

    if args.directory:
        convert_all_json_under_dir(args.directory, args.mode)
    else:
        if args.mode == "thumos":
            convert_vidat_to_thumos(args.input_json, args.output_file)
        elif args.mode == "activitynet":
            convert_vidat_to_ActivityNet(args.input_json, args.output_file)
        else:
            print("Example:")
            print(
                """
    python convert_vidat.py \\
        -m activitynet \\
        -i /Users/eason.hung/Documents/Projects/test-something/external_camera/formal_output_出/Viscovery_Bread_DemoRoom_20251107_100211_annotations.json
    """
            )  # noqa
            print(
                """
    python convert_vidat.py \\
        -m activitynet \\
        -d /Users/eason.hung/Documents/Projects/test-something/external_camera/formal_output_出/
    """
            )  # noqa

# python convert_vidat.py
# Example:

# python convert_vidat.py \
#     -m activitynet \
#     -i /Users/eason.hung/Documents/Projects/test-something/external_camera/formal_output_出/Viscovery_Bread_DemoRoom_20251107_100211_annotations.json  # noqa

# convert vidat annotation refine file
# (*_annotation.json) -> (*_AcitivityNet.json)

# python convert_vidat.py \
#     -m activitynet \
#     -d /Users/eason.hung/Documents/Projects/test-something/external_camera/formal_output_出/  # noqa
