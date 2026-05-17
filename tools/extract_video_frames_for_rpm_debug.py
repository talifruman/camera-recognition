"""
extract_video_frames_for_rpm_debug.py
--------------------------------------
Phase 1 of RPM visual-inspection tooling.

Scans a directory of video files, extracts a fixed number of frames from each
video using a chosen strategy, and writes them into a deterministic folder
structure that a separate RPM visual-inspection runner can consume later.

This script does NOT run RecognitionPipelineManager.
This script does NOT import any image_processing modules.
It is only responsible for preparing frame images and per-video metadata.

Output layout:
    <output_dir>/
        <video_name>/
            frames/
                frame_000000.jpg
                frame_000030.jpg
            metadata/
                frames_manifest.json
        summary_all_videos.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from video files for RPM visual-debug input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path(r"C:\Users\talif\Desktop\videos"),
        help="Directory containing input video files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "debug_inputs" / "rpm_frames",
        help="Root directory for extracted frames and manifests.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=5,
        metavar="N",
        help="Number of frames to extract from each video.",
    )
    parser.add_argument(
        "--strategy",
        choices=["evenly_spaced", "first_n", "every_n"],
        default="evenly_spaced",
        help="Frame selection strategy.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=30,
        metavar="N",
        help="Step size used only when --strategy is every_n.",
    )
    parser.add_argument(
        "--image-format",
        choices=["jpg", "png"],
        default="jpg",
        help="Image format for saved frames.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=None,
        metavar="PX",
        help=(
            "If set, downscale each frame so the longest side is at most PX "
            "while preserving aspect ratio. For debug-speed optimisation only."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frame files. Without this flag, existing files are skipped.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper: find video files
# ---------------------------------------------------------------------------

def find_video_files(video_dir: Path) -> list:
    """Return a sorted list of video file paths in video_dir."""
    if not video_dir.is_dir():
        print(f"[ERROR] Video directory does not exist: {video_dir}")
        return []

    files = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    print(f"[INFO] Found {len(files)} video file(s) in: {video_dir}")
    return files


# ---------------------------------------------------------------------------
# Helper: frame index selection
# ---------------------------------------------------------------------------

def select_frame_indexes(
    total_frames: int,
    strategy: str,
    frames_per_video: int,
    every_n: int,
) -> list:
    """
    Return a list of frame indexes to extract.

    Guarantees:
    - All indexes are in [0, total_frames - 2] to avoid unreliable last-frame seeks.
    - No duplicates.
    - At most frames_per_video entries.
    """
    if total_frames <= 0:
        return []

    # Use total_frames - 2 as the safe upper bound (avoid last frame).
    safe_max = max(0, total_frames - 2)

    if strategy == "first_n":
        count = min(frames_per_video, safe_max + 1)
        return list(range(count))

    if strategy == "every_n":
        indexes = []
        i = 0
        while len(indexes) < frames_per_video:
            idx = i * every_n
            if idx > safe_max:
                break
            indexes.append(idx)
            i += 1
        return indexes

    # evenly_spaced (default)
    n = min(frames_per_video, safe_max + 1)
    if n <= 0:
        return []
    if n == 1:
        return [0]
    seen = set()
    indexes = []
    for i in range(n):
        raw = int(i * safe_max / (n - 1))
        idx = min(raw, safe_max)
        if idx not in seen:
            seen.add(idx)
            indexes.append(idx)
    return indexes


# ---------------------------------------------------------------------------
# Helper: optional resize
# ---------------------------------------------------------------------------

def resize_if_needed(frame, max_dim):
    """
    Return frame unchanged when max_dim is None.
    Otherwise downscale so the longest side is at most max_dim.
    """
    if max_dim is None:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame
    scale = max_dim / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Helper: write JSON
# ---------------------------------------------------------------------------

def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# Core: extract frames for a single video
# ---------------------------------------------------------------------------

def extract_frames_for_video(
    video_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """
    Extract frames from video_path and write them to output_dir/<video_name>/.

    Returns a per-video result dict for the global summary.
    """
    video_name = video_path.stem
    print(f"\n{'=' * 60}")
    print(f"[VIDEO] {video_name}  ({video_path.name})")

    # ------------------------------------------------------------------
    # Open video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return {
            "video_name": video_name,
            "video_path": str(video_path),
            "success": False,
            "error": "Cannot open video",
            "frames_extracted": 0,
            "frames_skipped": 0,
            "frames_failed": 0,
        }

    # ------------------------------------------------------------------
    # Read metadata
    # ------------------------------------------------------------------
    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_raw if fps_raw and fps_raw > 0 else 0.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Resolution : {original_width}x{original_height}")
    print(f"  FPS        : {fps:.3f}")
    print(f"  Frames     : {total_video_frames}")

    # ------------------------------------------------------------------
    # Select frame indexes
    # ------------------------------------------------------------------
    indexes = select_frame_indexes(
        total_video_frames,
        args.strategy,
        args.frames_per_video,
        args.every_n,
    )
    print(f"  Strategy   : {args.strategy}")
    print(f"  Selected   : {indexes}")

    # ------------------------------------------------------------------
    # Prepare output directories
    # ------------------------------------------------------------------
    video_out_dir = output_dir / video_name
    frames_dir = video_out_dir / "frames"
    metadata_dir = video_out_dir / "metadata"
    frames_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Extract each frame
    # ------------------------------------------------------------------
    extracted_frames = []
    frames_extracted = 0
    frames_skipped = 0
    frames_failed = 0
    extracted_width = original_width
    extracted_height = original_height

    for idx in indexes:
        frame_filename = f"frame_{idx:06d}.{args.image_format}"
        frame_path = frames_dir / frame_filename

        # Skip existing files when --overwrite is not set
        if not args.overwrite and frame_path.exists():
            print(f"  [SKIP] {frame_filename} (already exists)")
            # Still include in manifest so it is visible to downstream tools
            extracted_frames.append({
                "frame_id": f"{video_name}_frame_{idx:06d}",
                "frame_index": idx,
                "timestamp_ms": round(idx / fps * 1000) if fps > 0 else None,
                "image_path": str(frame_path.relative_to(output_dir)).replace("\\", "/"),
                "width": None,
                "height": None,
                "skipped": True,
            })
            frames_skipped += 1
            continue

        # Seek to requested frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  [WARN] Cannot read frame {idx} — skipping.")
            extracted_frames.append({
                "frame_id": f"{video_name}_frame_{idx:06d}",
                "frame_index": idx,
                "timestamp_ms": round(idx / fps * 1000) if fps > 0 else None,
                "image_path": None,
                "width": None,
                "height": None,
                "read_failed": True,
            })
            frames_failed += 1
            continue

        # Optional resize
        frame = resize_if_needed(frame, args.max_dim)

        # Write frame
        success = cv2.imwrite(str(frame_path), frame)
        if not success:
            print(f"  [WARN] cv2.imwrite failed for frame {idx} — skipping.")
            frames_failed += 1
            continue

        frame_h, frame_w = frame.shape[:2]
        extracted_width = frame_w
        extracted_height = frame_h
        timestamp_ms = round(idx / fps * 1000) if fps > 0 else None

        print(f"  [SAVE] {frame_filename}  ({frame_w}x{frame_h}  t={timestamp_ms}ms)")

        extracted_frames.append({
            "frame_id": f"{video_name}_frame_{idx:06d}",
            "frame_index": idx,
            "timestamp_ms": timestamp_ms,
            "image_path": str(frame_path.relative_to(output_dir)).replace("\\", "/"),
            "width": frame_w,
            "height": frame_h,
        })
        frames_extracted += 1

    cap.release()

    # ------------------------------------------------------------------
    # Write frames_manifest.json
    # ------------------------------------------------------------------
    manifest = {
        "video_name": video_name,
        "video_path": str(video_path),
        "fps": fps,
        "original_width": original_width,
        "original_height": original_height,
        "extracted_width": extracted_width,
        "extracted_height": extracted_height,
        "total_video_frames": total_video_frames,
        "extraction_strategy": args.strategy,
        "frames_per_video": args.frames_per_video,
        "extracted_frames": extracted_frames,
    }
    write_json(metadata_dir / "frames_manifest.json", manifest)
    print(f"  [META] Manifest written -> {metadata_dir / 'frames_manifest.json'}")

    return {
        "video_name": video_name,
        "video_path": str(video_path),
        "success": True,
        "frames_extracted": frames_extracted,
        "frames_skipped": frames_skipped,
        "frames_failed": frames_failed,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    video_dir: Path = args.video_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    print(f"[INFO] Video directory : {video_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Frames per video: {args.frames_per_video}")
    print(f"[INFO] Strategy        : {args.strategy}")
    if args.strategy == "every_n":
        print(f"[INFO] Every-N step    : {args.every_n}")
    print(f"[INFO] Image format    : {args.image_format}")
    if args.max_dim is not None:
        print(f"[INFO] Max dimension   : {args.max_dim}px")
    print(f"[INFO] Overwrite       : {args.overwrite}")

    # Discover videos
    video_files = find_video_files(video_dir)
    if not video_files:
        print("[INFO] No video files found. Nothing to do.")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each video
    per_video_results = []
    failed_videos = []

    for video_path in video_files:
        result = extract_frames_for_video(video_path, output_dir, args)
        per_video_results.append(result)
        if not result.get("success", False):
            failed_videos.append(result["video_name"])

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_extracted = sum(r.get("frames_extracted", 0) for r in per_video_results)
    total_processed = sum(1 for r in per_video_results if r.get("success", False))

    print(f"\n{'=' * 60}")
    print(f"[DONE] Videos found    : {len(video_files)}")
    print(f"[DONE] Videos processed: {total_processed}")
    print(f"[DONE] Frames extracted: {total_extracted}")
    if failed_videos:
        print(f"[DONE] Failed videos   : {', '.join(failed_videos)}")

    summary = {
        "video_dir": str(video_dir),
        "output_dir": str(output_dir),
        "total_videos_found": len(video_files),
        "total_videos_processed": total_processed,
        "total_frames_extracted": total_extracted,
        "failed_videos": failed_videos,
        "per_video": per_video_results,
    }
    summary_path = output_dir / "summary_all_videos.json"
    write_json(summary_path, summary)
    print(f"[DONE] Summary written -> {summary_path}")


if __name__ == "__main__":
    main()
