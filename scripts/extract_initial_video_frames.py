"""
extract_initial_video_frames.py
--------------------------------
Extract the first N frames from every video file in a given input folder and
save them as JPEG images in a structured output folder.

Usage example (Windows):

    python scripts/extract_initial_video_frames.py ^
      --input-dir "C:\\Users\\talif\\Desktop\\videos" ^
      --output-dir "tests/recognition_pipeline_manager/assets/extracted_video_frames" ^
      --max-frames 30

Output layout:
    <output_dir>/
        <video_name>/
            frame_000001.jpg
            frame_000002.jpg
            ...
            frame_000030.jpg

Notes:
- Only the first --max-frames frames are extracted (default: 30).
- If the per-video output folder already exists, the video is skipped unless
  --overwrite is passed, in which case the folder is deleted and recreated.
- This script does NOT run RPM or any image_processing module.
  It is only responsible for converting videos into extracted frame images.
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the first N frames from each video in a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input video files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory where extracted frame folders will be written.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        metavar="N",
        help="Maximum number of frames to extract from each video.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If set, existing per-video output folders are deleted and recreated.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def extract_frames(video_path: Path, output_dir: Path, max_frames: int) -> int:
    """Open video_path, extract up to max_frames frames, write to output_dir.

    Returns the number of frames actually written.
    Prints a warning if the video contains fewer frames than max_frames.
    Raises RuntimeError if the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = output_dir / f"frame_{count + 1:06d}.jpg"
        cv2.imwrite(str(frame_filename), frame)
        count += 1

    cap.release()

    if count < max_frames:
        print(
            f"  [WARNING] {video_path.name} has only {count} frame(s) "
            f"(fewer than --max-frames {max_frames})."
        )

    return count


def process_video(
    video_path: Path,
    output_root: Path,
    max_frames: int,
    overwrite: bool,
) -> dict:
    """Process a single video file.

    Returns a result dict with keys:
        name, frames_extracted, output_dir, skipped, warning
    """
    video_output_dir = output_root / video_path.stem
    result = {
        "name": video_path.name,
        "frames_extracted": 0,
        "output_dir": str(video_output_dir),
        "skipped": False,
        "warning": None,
    }

    if video_output_dir.exists():
        if not overwrite:
            print(
                f"  [SKIP] Output folder already exists for '{video_path.name}'. "
                f"Pass --overwrite to replace it.\n"
                f"         {video_output_dir}"
            )
            result["skipped"] = True
            return result
        # Overwrite: remove existing folder
        shutil.rmtree(video_output_dir)

    try:
        frames_written = extract_frames(video_path, video_output_dir, max_frames)
    except RuntimeError as exc:
        print(f"  [ERROR] {exc}")
        result["warning"] = str(exc)
        return result

    result["frames_extracted"] = frames_written
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"[ERROR] Input directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.input_dir.is_dir():
        print(f"[ERROR] Input path is not a directory: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if missing
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect video files
    video_files = sorted(
        p for p in args.input_dir.iterdir() if _is_video_file(p)
    )

    if not video_files:
        print(f"[INFO] No video files found in: {args.input_dir}")
        sys.exit(0)

    print(f"Found {len(video_files)} video(s) in: {args.input_dir}")
    print(f"Output root: {args.output_dir}")
    print(f"Max frames per video: {args.max_frames}")
    print(f"Overwrite: {args.overwrite}")
    print("-" * 60)

    results = []
    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")
        result = process_video(video_path, args.output_dir, args.max_frames, args.overwrite)
        results.append(result)

        if not result["skipped"] and result["warning"] is None:
            print(f"  Frames extracted : {result['frames_extracted']}")
            print(f"  Output dir       : {result['output_dir']}")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_extracted = 0
    for r in results:
        status = "SKIPPED" if r["skipped"] else ("ERROR" if r["warning"] and r["frames_extracted"] == 0 else "OK")
        print(f"  {r['name']:<40} {status:<8} frames={r['frames_extracted']}")
        total_extracted += r["frames_extracted"]

    print("-" * 60)
    print(f"  Total frames extracted: {total_extracted}")
    print(f"  Videos processed      : {sum(1 for r in results if not r['skipped'])}")
    print(f"  Videos skipped        : {sum(1 for r in results if r['skipped'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
