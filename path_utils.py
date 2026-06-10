import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_files_recursive(
    root_directory: str, supported_extensions=None, is_relative=False
):
    if not supported_extensions:
        supported_extensions = ".mp4"
    found_videos = []
    if not os.path.isdir(root_directory):
        return []

    for dirpath, dirnames, filenames in os.walk(root_directory):
        for f in filenames:
            # if f.lower().endswith(supported_extensions):
            if f.endswith(supported_extensions):
                full_path = os.path.join(dirpath, f)
                if is_relative:
                    relative_path = os.path.relpath(full_path, root_directory)
                    found_videos.append(relative_path)  # relative path
                else:
                    found_videos.append(full_path)  # full path

    return sorted(found_videos)


def convert_mp4v_to_h264(input_path, output_path=None, preset="ultrafast", crf=23):
    """
    Convert mp4v encoded MP4 to H.264 (avc1)

    Parameters:
        input_path: Path to input mp4v MP4 file
        output_path: Output file path (default: input_path with 'h264_' prefix)
        preset: FFmpeg encoding speed ('ultrafast', 'superfast', 'veryfast', 'fast', 'medium')
        crf: Video quality (0=highest, 51=lowest, default 23 for balanced)
    """

    # Check if FFmpeg is installed
    if shutil.which("ffmpeg") is None:
        print("❌ FFmpeg is not installed!")
        print("\nPlease install FFmpeg:")
        print("  - Windows: Download from https://ffmpeg.org/download.html")
        print("  - macOS: brew install ffmpeg")
        print("  - Linux (Ubuntu/Debian): sudo apt-get install ffmpeg")
        return None

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_h264{ext}"

    # FFmpeg conversion command
    cmd = [
        "ffmpeg",
        "-i",
        input_path,  # Input file
        "-c:v",
        "libx264",  # Use H.264 encoder
        "-preset",
        preset,  # Encoding speed
        "-crf",
        str(crf),  # Quality
        "-c:a",
        "aac",  # Audio codec
        "-y",  # Overwrite output file
        output_path,
    ]

    try:
        print(f"Converting: {input_path} -> {output_path}")
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Conversion complete: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"✗ Conversion failed: {e.stderr.decode()}")
        return None


def convert_dir_mp4v_to_h264(
    input_dir="/vol/08822801/AutoTrigger/dataset", max_workers=None
):
    all_mp4v_video_paths = get_files_recursive(
        root_directory=input_dir, supported_extensions=("_mp4v.mp4")
    )
    # from 'Viscovery_Bread_DemoRoom_20251107_110731_mp4v.mp4'
    # to -> 'Viscovery_Bread_DemoRoom_20251107_110731.mp4'

    print(len(all_mp4v_video_paths))
    print(all_mp4v_video_paths)

    # mp4v_video_path = os.path.join(video_dir, base_name + '_mp4v.mp4')
    # video_path = os.path.join(video_dir, base_name + '.mp4')

    if max_workers is None:
        # 根據您的描述，這裡會是 40 個左右
        max_workers = os.cpu_count()
        print(
            f"Detected CPU count: {max_workers}. Using {max_workers} parallel workers."
        )

    total_files = len(all_mp4v_video_paths)
    if total_files == 0:
        print("No '_mp4v.mp4' files found. Exiting.")
        return

    print(f"Found {total_files} files for conversion.")
    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for mp4v_video_path in all_mp4v_video_paths:
            video_path = mp4v_video_path.replace("_mp4v.mp4", ".mp4")
            if os.path.exists(video_path):
                print(f"Skipping: Output file already exists: {video_path}")
                continue

            future = executor.submit(
                convert_mp4v_to_h264,
                input_path=mp4v_video_path,
                output_path=video_path,
                preset="fast",
                crf=20,
            )
            tasks.append(future)

        # 收集結果並印出進度
        for i, future in enumerate(as_completed(tasks)):
            result = future.result()
            print(f"[{i+1}/{total_files}] {result}")

    # for mp4v_video_path in all_mp4v_video_paths:
    #     video_path = mp4v_video_path.replace('_mp4v.mp4', '.mp4')
    #     if os.path.exists(video_path):
    #         continue
    #     convert_mp4v_to_h264(
    #         mp4v_video_path,
    #         output_path=video_path,
    #         preset='fast',
    #         crf=20,
    #     )
