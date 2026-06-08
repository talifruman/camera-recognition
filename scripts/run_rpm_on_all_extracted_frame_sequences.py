"""
Run RPM on all extracted frame sequences under a root folder.

Usage example (Windows):

python scripts/run_rpm_on_all_extracted_frame_sequences.py ^
  --frames-root-dir "tests/recognition_pipeline_manager/assets/extracted_video_frames" ^
  --output-root-dir "debug_outputs" ^
  --rpm-config "config/image_processing_service/local_debug.yaml" ^
  --real-object-detection ^
  --real-face-detection ^
  --real-face-recognition
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from run_rpm_on_frames import IMAGE_EXTENSIONS, run_sequence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run run_rpm_on_frames.py over all extracted sequence folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--frames-root-dir",
        type=Path,
        default=Path("tests/recognition_pipeline_manager/assets/extracted_video_frames"),
        help="Root directory containing one subdirectory per extracted video sequence.",
    )
    parser.add_argument(
        "--output-root-dir",
        type=Path,
        default=Path("debug_outputs"),
        help="Root directory for per-sequence outputs.",
    )
    parser.add_argument(
        "--rpm-config",
        type=Path,
        default=Path("config/image_processing_service/local_debug.yaml"),
        help="Path to root RPM YAML config.",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="camera_01",
        help="Camera ID to pass to run_rpm_on_frames.",
    )

    parser.add_argument("--real-motion-detection", action="store_true", help="Pass-through flag.")
    parser.add_argument("--real-object-detection", action="store_true", help="Pass-through flag.")
    parser.add_argument("--real-face-detection", action="store_true", help="Pass-through flag.")
    parser.add_argument("--real-face-recognition", action="store_true", help="Pass-through flag.")
    parser.add_argument("--overwrite", action="store_true", help="Delete each sequence output dir before running.")
    return parser.parse_args()


def _has_image_frames(sequence_dir: Path) -> tuple[bool, int]:
    count = sum(
        1
        for p in sequence_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return count > 0, count


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def main() -> None:
    args = _parse_args()

    frames_root = args.frames_root_dir.resolve()
    output_root = args.output_root_dir.resolve()
    rpm_config = args.rpm_config.resolve()

    if not frames_root.is_dir():
        print(f"[ERROR] frames root directory not found: {frames_root}")
        sys.exit(1)
    if not rpm_config.exists():
        print(f"[ERROR] rpm-config file not found: {rpm_config}")
        sys.exit(1)

    sequence_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])
    if not sequence_dirs:
        print(f"[INFO] No sequence subdirectories found in: {frames_root}")
        sys.exit(0)

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[BATCH] frames_root_dir={frames_root}")
    print(f"[BATCH] output_root_dir={output_root}")
    print(f"[BATCH] rpm_config={rpm_config}")
    print(f"[BATCH] total_candidate_sequences={len(sequence_dirs)}")

    successful_sequences = 0
    failed_sequences = 0
    per_sequence: list[dict[str, Any]] = []

    for index, sequence_dir in enumerate(sequence_dirs, start=1):
        sequence_name = sequence_dir.name
        print(f"\n[SEQUENCE {index}/{len(sequence_dirs)}] {sequence_name}")

        has_frames, frame_count = _has_image_frames(sequence_dir)
        sequence_output_dir = output_root / sequence_name

        if not has_frames:
            print("  [SKIP] no .jpg/.jpeg/.png frames found")
            failed_sequences += 1
            per_sequence.append(
                {
                    "sequence_name": sequence_name,
                    "frames_dir": str(sequence_dir),
                    "output_path": str(sequence_output_dir),
                    "frame_count": 0,
                    "failures": 1,
                    "persons_detected": 0,
                    "recognized_faces": 0,
                    "success": False,
                    "error": "No image frames found",
                }
            )
            continue

        try:
            summary = run_sequence(
                frames_dir=sequence_dir,
                output_dir=sequence_output_dir,
                rpm_config=rpm_config,
                camera_id=args.camera_id,
                real_motion_detection=bool(args.real_motion_detection),
                real_object_detection=bool(args.real_object_detection),
                real_face_detection=bool(args.real_face_detection),
                real_face_recognition=bool(args.real_face_recognition),
                overwrite=bool(args.overwrite),
            )
            successful_sequences += 1

            per_sequence.append(
                {
                    "sequence_name": sequence_name,
                    "frames_dir": str(sequence_dir),
                    "output_path": str(sequence_output_dir),
                    "frame_count": int(summary.get("total_frames", frame_count)),
                    "failures": int(summary.get("failures_count", 0)),
                    "persons_detected": int(summary.get("total_persons_detected", 0)),
                    "recognized_faces": int(summary.get("total_recognized_faces", 0)),
                    "success": True,
                    "error": None,
                }
            )
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            failed_sequences += 1
            per_sequence.append(
                {
                    "sequence_name": sequence_name,
                    "frames_dir": str(sequence_dir),
                    "output_path": str(sequence_output_dir),
                    "frame_count": frame_count,
                    "failures": frame_count,
                    "persons_detected": 0,
                    "recognized_faces": 0,
                    "success": False,
                    "error": str(exc),
                }
            )
            continue

    batch_summary = {
        "frames_root_dir": str(frames_root),
        "output_root_dir": str(output_root),
        "rpm_config": str(rpm_config),
        "total_sequences": len(sequence_dirs),
        "successful_sequences": successful_sequences,
        "failed_sequences": failed_sequences,
        "sequences": per_sequence,
    }

    summary_path = output_root / "batch_run_summary.json"
    _write_json(summary_path, batch_summary)

    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"  total sequences     : {len(sequence_dirs)}")
    print(f"  successful sequences: {successful_sequences}")
    print(f"  failed sequences    : {failed_sequences}")
    print(f"  summary file        : {summary_path}")


if __name__ == "__main__":
    main()
