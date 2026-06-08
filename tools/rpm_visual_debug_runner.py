#!/usr/bin/env python3
"""RPM Visual Debug Runner
==========================
Run the implemented RecognitionPipelineManager on all video files in a
directory and produce annotated video, debug crops, per-video JSON reports,
and a global markdown summary.

Usage:
    python tools/rpm_visual_debug_runner.py ^
        --video-dir "C:\\Users\\talif\\Desktop\\videos" ^
        --output-dir "C:\\...\\debug_outputs\\rpm_visual" ^
        --max-frames 100 --save-crops

    # Offline overlay-flow validation only (no real models needed):
    python tools/rpm_visual_debug_runner.py ... --use-fakes

Output tree:
    <output_dir>/
      <video_name>/
        annotated_video.mp4
        frames/               annotated JPEG snapshots
        crops/
          persons/            person-bbox crops (--save-crops)
          faces/              face-bbox crops   (--save-crops)
        reports/
          summary.json
      summary_all_videos.md   global markdown table
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.path bootstrap — must happen before any image_processing imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_TESTS_DIR = _PROJECT_ROOT / "tests"

for _p in [str(_SRC_DIR), str(_TESTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Third-party imports (validate early)
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError:
    print("[FATAL] opencv-python is not installed.  Run: pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("[FATAL] numpy is not installed.  Run: pip install numpy")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Core pipeline imports
# ---------------------------------------------------------------------------
from image_processing.recognition_pipeline_manager import (
    RecognitionPipelineManager,
    RecognitionPipelineOutput,
)
from image_processing.frame_transformation_layer.contracts import FramePacket
from image_processing.shared.contracts import BoundingBox

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE_LABEL = 0.5
_FONT_SCALE_HUD = 0.4
_THICKNESS = 1

# BGR colours
_COLOR_PERSON = (0, 220, 0)       # green  — person bbox
_COLOR_FACE = (255, 120, 0)       # blue   — face bbox
_COLOR_HUD = (200, 200, 200)      # light gray — HUD text


# ---------------------------------------------------------------------------
# Per-video statistics
# ---------------------------------------------------------------------------
@dataclass
class VideoStats:
    video_name: str
    total_frames_processed: int = 0
    cold_start_frames: int = 0
    frames_with_persons: int = 0
    frames_with_recognized_faces: int = 0
    error_count: int = 0
    error_frame_ids: list[str] = field(default_factory=list)
    processing_times_ms: list[float] = field(default_factory=list)

    @property
    def avg_processing_ms(self) -> float:
        if not self.processing_times_ms:
            return 0.0
        return sum(self.processing_times_ms) / len(self.processing_times_ms)

    @property
    def max_processing_ms(self) -> float:
        if not self.processing_times_ms:
            return 0.0
        return max(self.processing_times_ms)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def find_video_files(video_dir: Path) -> list[Path]:
    """Return sorted list of video files with supported extensions."""
    files: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(video_dir.glob(f"*{ext}"))
        files.extend(video_dir.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def resize_bgr(bgr_frame: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale frame so its longest side is at most max_dim; preserves aspect ratio."""
    h, w = bgr_frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return bgr_frame
    scale = max_dim / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(bgr_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def bgr_to_frame_packet(
    bgr_frame: np.ndarray,
    frame_id: str,
    camera_id: str,
    timestamp_ms: int,
) -> FramePacket:
    """Convert a BGR cv2 frame to a FramePacket (RGB/HWC/uint8)."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return FramePacket(
        frame_id=frame_id,
        camera_id=camera_id,
        timestamp_ms=timestamp_ms,
        width=w,
        height=h,
        pixel_format="RGB",
        layout="HWC",
        num_color_channels=3,
        bits_per_channel=8,
        image_bytes=rgb.tobytes(),
    )


def _clamp(bbox: BoundingBox, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) clamped to frame bounds."""
    x1 = max(0, bbox["x"])
    y1 = max(0, bbox["y"])
    x2 = min(frame_w, bbox["x"] + bbox["width"])
    y2 = min(frame_h, bbox["y"] + bbox["height"])
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Visual overlay drawing
# ---------------------------------------------------------------------------
def draw_overlays(
    bgr_frame: np.ndarray,
    output: RecognitionPipelineOutput,
) -> np.ndarray:
    """Draw person/face bboxes and HUD text onto bgr_frame (returns same array)."""
    h, w = bgr_frame.shape[:2]

    # HUD — top-left corner
    hud_lines = [
        f"frame:  {output.get('frame_id', '')}",
        f"cam:    {output.get('camera_id', '')}",
        f"ts:     {output.get('timestamp_ms', 0)} ms",
        f"persons:{len(output.get('persons', []))}",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(
            bgr_frame, line,
            (5, 14 + i * 14),
            _FONT, _FONT_SCALE_HUD, _COLOR_HUD, _THICKNESS, cv2.LINE_AA,
        )

    for person in output.get("persons", []):
        pb = person["person_bbox"]
        x1, y1, x2, y2 = _clamp(pb, w, h)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), _COLOR_PERSON, 2)
            cv2.putText(
                bgr_frame, "person",
                (x1, max(0, y1 - 4)),
                _FONT, _FONT_SCALE_LABEL, _COLOR_PERSON, _THICKNESS, cv2.LINE_AA,
            )

        for face in person.get("recognized_faces", []):
            fb = face["face_bbox"]
            fx1, fy1, fx2, fy2 = _clamp(fb, w, h)
            if fx2 > fx1 and fy2 > fy1:
                cv2.rectangle(bgr_frame, (fx1, fy1), (fx2, fy2), _COLOR_FACE, 2)
                pid = face.get("person_id", "UNKNOWN")
                pname = face.get("person_name", "UNKNOWN")
                label = pname if (pname and pname != "UNKNOWN") else pid
                cv2.putText(
                    bgr_frame, label,
                    (fx1, max(0, fy1 - 4)),
                    _FONT, _FONT_SCALE_LABEL, _COLOR_FACE, _THICKNESS, cv2.LINE_AA,
                )
    return bgr_frame


# ---------------------------------------------------------------------------
# Crop saving
# ---------------------------------------------------------------------------
def save_crops_from_output(
    bgr_frame: np.ndarray,
    output: RecognitionPipelineOutput,
    crops_dir: Path,
    frame_id: str,
) -> None:
    """Slice person and face bbox regions to disk.

    Note: motion ROIs and aligned-face crops are NOT available from
    RecognitionPipelineOutput.  No changes were made to RPM to expose them.
    """
    h, w = bgr_frame.shape[:2]
    persons_dir = crops_dir / "persons"
    faces_dir = crops_dir / "faces"
    persons_dir.mkdir(parents=True, exist_ok=True)
    faces_dir.mkdir(parents=True, exist_ok=True)

    for pi, person in enumerate(output.get("persons", [])):
        pb = person["person_bbox"]
        x1, y1, x2, y2 = _clamp(pb, w, h)
        if x2 > x1 and y2 > y1:
            crop = bgr_frame[y1:y2, x1:x2]
            cv2.imwrite(str(persons_dir / f"{frame_id}_p{pi:02d}.jpg"), crop)

        for fi, face in enumerate(person.get("recognized_faces", [])):
            fb = face["face_bbox"]
            fx1, fy1, fx2, fy2 = _clamp(fb, w, h)
            if fx2 > fx1 and fy2 > fy1:
                face_crop = bgr_frame[fy1:fy2, fx1:fx2]
                pid = face.get("person_id", "UNKNOWN")
                cv2.imwrite(
                    str(faces_dir / f"{frame_id}_p{pi:02d}_f{fi:02d}_{pid}.jpg"),
                    face_crop,
                )


# ---------------------------------------------------------------------------
# Factory: build_rpm_for_visual_debug
# ---------------------------------------------------------------------------
def build_rpm_for_visual_debug(
    *,
    gallery_dir: Path,
    person_json: Path | None,
    use_fakes: bool,
) -> RecognitionPipelineManager:
    """Construct RecognitionPipelineManager with real or fake dependencies.

    Real mode  — all actual pipeline modules are initialized.  Any module
                 that fails to initialise prints exactly which dependency is
                 missing and how to configure it, then raises.

    Fakes mode (--use-fakes) — imports the existing test fakes from
                 tests/recognition_pipeline_manager/fakes.py.  Useful for
                 validating overlay drawing and output structure without
                 loading any ML models.
    """
    if use_fakes:
        return _build_rpm_fakes()
    return _build_rpm_real(gallery_dir=gallery_dir, person_json=person_json)


# ---------------------------------------------------------------------------
# Real dependency wiring
# ---------------------------------------------------------------------------
def _build_rpm_real(
    *,
    gallery_dir: Path,
    person_json: Path | None,
) -> RecognitionPipelineManager:
    """Wire all real pipeline modules into RPM."""

    # ---- 1. FrameTransformationLayer ----
    print("[init] FrameTransformationLayer ...")
    try:
        from image_processing.frame_transformation_layer import FrameTransformationLayer
        ftl = FrameTransformationLayer()
    except Exception as exc:
        print(f"[FATAL] FrameTransformationLayer failed: {exc}")
        raise

    # ---- 2. MotionDetection ----
    print("[init] MotionDetectionManager ...")
    try:
        from image_processing.motion_detection import MotionDetectionManager
        motion = MotionDetectionManager()
    except Exception as exc:
        print(f"[FATAL] MotionDetectionManager failed: {exc}")
        raise

    # ---- 3. ObjectDetection ----
    print("[init] ObjectDetectionModule (YOLO) ...")
    try:
        from image_processing.object_detection import ObjectDetectionModule
        object_det = ObjectDetectionModule()
        _od_backend = object_det.get_backend_info()
        print(
            f"[init]   OD backend={_od_backend.get('backend', '?')}  "
            f"device={_od_backend.get('inference_device', '?')}  "
            f"cuda_available={_od_backend.get('cuda_available', '?')}  "
            f"model={_od_backend.get('model_name', '?')}"
            + (f"  why_unknown={_od_backend['why_unknown']}" if _od_backend.get("why_unknown") else "")
        )
    except Exception as exc:
        print(f"[FATAL] ObjectDetectionModule failed: {exc}")
        print("  → Ensure models/object_detection/yolo11m.pt exists.")
        print("    Configure via PersonDetectionConfig(model_path=...) if needed.")
        raise

    # ---- 4. FaceDetection ----
    print("[init] FaceDetectionModule (SCRFD) ...")
    try:
        from image_processing.face_detection import FaceDetectionModule, SCRFDFaceDetector
        scrfd = SCRFDFaceDetector(
            model_dir=_PROJECT_ROOT / "models" / "face_detection"
        )
        face_det = FaceDetectionModule(detector_engine=scrfd)
    except Exception as exc:
        print(f"[FATAL] FaceDetectionModule failed: {exc}")
        print("  → Model: models/face_detection/det_500m.onnx")
        print("    Auto-downloaded from InsightFace buffalo_sc.zip if missing.")
        raise

    # ---- 5. FaceRecognition (requires gallery) ----
    print(f"[init] FaceGalleryLoader from: {gallery_dir}")
    try:
        from image_processing.face_gallery_loader import FaceGalleryLoaderModule
        from image_processing.face_recognition import (
            ArcFaceEmbeddingEngine,
            EnrolledIdentity,
            FaceRecognitionConfig,
            FaceRecognitionModule,
        )

        gallery_loader = FaceGalleryLoaderModule()
        gallery_loader.load_gallery(str(gallery_dir))
        loaded = gallery_loader.get_all_embeddings()
        enrolled: list[EnrolledIdentity] = [
            EnrolledIdentity(person_id=e["person_id"], embedding=e["embedding"])
            for e in loaded
        ]
        print(f"[init]   {len(enrolled)} embeddings loaded ({len(gallery_loader.get_person_ids())} persons)")

        arc_engine = ArcFaceEmbeddingEngine(
            model_dir=_PROJECT_ROOT / "models" / "face_recognition"
        )
        face_rec = FaceRecognitionModule(
            config=FaceRecognitionConfig(),
            embedding_engine=arc_engine,
            gallery_entries=enrolled,
        )
    except Exception as exc:
        print(f"[FATAL] FaceRecognitionModule failed: {exc}")
        print(f"  → Gallery dir: {gallery_dir}")
        print("  → ArcFace model: models/face_recognition/ (auto-downloaded)")
        raise

    # ---- 6. PersonDirectory ----
    print("[init] PersonDirectory ...")
    try:
        from image_processing.person_directory import PersonDirectory, PersonDirectoryConfig

        if person_json is not None:
            pd_config = PersonDirectoryConfig(
                json_file_path=str(person_json),
                is_optional=False,
            )
            print(f"[init]   JSON: {person_json}")
        else:
            # is_optional=True silently skips missing/empty file.
            # Overlay will display person_id when name is not found.
            pd_config = PersonDirectoryConfig(
                json_file_path="",
                is_optional=True,
            )
            print("[init]   No --person-json provided; names will fall back to person_id")

        person_dir = PersonDirectory(config=pd_config)
        person_dir.load()
    except Exception as exc:
        print(f"[FATAL] PersonDirectory failed: {exc}")
        print(f"  → person_json path: {person_json}")
        raise

    # ---- Wire RPM ----
    print("[init] Wiring RecognitionPipelineManager ...")
    rpm = RecognitionPipelineManager(
        ftl=ftl,
        motion=motion,
        object_det=object_det,
        face_det=face_det,
        face_rec=face_rec,
        person_dir=person_dir,
    )
    print("[init] RPM ready.\n")
    return rpm


# ---------------------------------------------------------------------------
# Fake dependency wiring (--use-fakes)
# ---------------------------------------------------------------------------
def _build_rpm_fakes() -> RecognitionPipelineManager:
    """Build RPM with fake dependencies to validate overlay/output flow.

    Uses the existing test fakes from tests/recognition_pipeline_manager/fakes.py.
    Fakes are pre-configured to return realistic-looking data so every stage
    of the pipeline fires and the full drawing/report logic executes.
    """
    print("[init] Building RPM in FAKE mode (--use-fakes)\n")

    from recognition_pipeline_manager.fakes import (
        FakeFTL,
        FakeFaceDetection,
        FakeFaceRecognition,
        FakeMotionDetection,
        FakeObjectDetection,
        FakePersonDirectory,
    )
    from image_processing.face_detection import DetectedFace, FaceDetectionOutput
    from image_processing.face_recognition import FaceRecognitionOutput
    from image_processing.motion_detection import MotionResult
    from image_processing.object_detection import PersonDetectionResult
    from image_processing.person_directory import PersonDirectoryOutput
    from image_processing.shared.contracts import BoundingBox, FaceLandmarks, Point

    _FAKE_PID = "fake-person-001"
    _FAKE_NAME = "Fake Person"

    # Motion: report a 200x200 region starting at (50, 50) — small but visible
    motion_fake = FakeMotionDetection(
        result=MotionResult(
            detected=True,
            bboxes=[BoundingBox(x=50, y=50, width=200, height=200)],
        )
    )

    # Object detection: one person in the ROI (ROI-local coords 0,0)
    object_fake = FakeObjectDetection(
        result=PersonDetectionResult(
            frame_id="",
            person_detected=True,
            persons=[BoundingBox(x=0, y=0, width=200, height=200)],
        )
    )

    # Face detection: one face in the ROI (ROI-local coords)
    # DetectedFace.face_bbox is in ROI-local space; RPM projects to full frame.
    face_det_fake = FakeFaceDetection(
        result=FaceDetectionOutput(
            frame_id="",
            camera_id="",
            timestamp_ms=0,
            detections=[
                DetectedFace(
                    face_bbox=BoundingBox(x=20, y=20, width=80, height=80),
                    landmarks=FaceLandmarks(
                        left_eye=Point(x=35, y=40),
                        right_eye=Point(x=65, y=40),
                        nose=Point(x=50, y=55),
                        mouth_left=Point(x=35, y=70),
                        mouth_right=Point(x=65, y=70),
                    ),
                )
            ],
        )
    )

    face_rec_fake = FakeFaceRecognition(
        result=FaceRecognitionOutput(
            frame_id="",
            camera_id="",
            timestamp_ms=0,
            person_found=True,
            person_id=_FAKE_PID,
        )
    )

    person_dir_fake = FakePersonDirectory(
        lookup_map={
            _FAKE_PID: PersonDirectoryOutput(
                person_id=_FAKE_PID,
                person_name=_FAKE_NAME,
                found=True,
            )
        }
    )

    return RecognitionPipelineManager(
        ftl=FakeFTL(),
        motion=motion_fake,
        object_det=object_fake,
        face_det=face_det_fake,
        face_rec=face_rec_fake,
        person_dir=person_dir_fake,
    )


# ---------------------------------------------------------------------------
# Per-video processor
# ---------------------------------------------------------------------------
def _od_backend_info(rpm: RecognitionPipelineManager) -> dict[str, str]:
    """Extract OD backend diagnostics from the RPM's orchestrator static debug info."""
    try:
        orchestrator = getattr(rpm, "_orchestrator", None)
        if orchestrator is None:
            return {}
        static = getattr(orchestrator, "_stage_static_debug", {})
        return {
            "backend": str(static.get("object_detection_device_provider", "unknown")),
            "inference_device": str(static.get("object_detection_inference_device", "unknown")),
            "cuda_available": str(static.get("object_detection_cuda_available", "unknown")),
            "why_unknown": str(static.get("object_detection_why_unknown", "")),
            "model_name": str(static.get("object_detection_model_name", "unknown")),
            "model_path": str(static.get("object_detection_model_path", "unknown")),
        }
    except Exception:  # pragma: no cover
        return {}


def process_video(
    video_path: Path,
    rpm: RecognitionPipelineManager,
    output_dir: Path,
    *,
    max_frames: int | None,
    show: bool,
    save_crops: bool,
    skip_frames: int,
    max_dim: int | None = None,
) -> VideoStats:
    """Process one video through RPM and write all output artifacts."""
    video_name = video_path.stem
    video_out_dir = output_dir / video_name
    frames_dir = video_out_dir / "frames"
    crops_dir = video_out_dir / "crops"
    reports_dir = video_out_dir / "reports"

    for d in [video_out_dir, frames_dir, crops_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    stats = VideoStats(video_name=video_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed to open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    annotated_path = video_out_dir / "annotated_video.mp4"

    effective_w = frame_w
    effective_h = frame_h
    if max_dim is not None:
        longest = max(frame_w, frame_h)
        if longest > max_dim:
            scale = max_dim / longest
            effective_w = max(1, int(frame_w * scale))
            effective_h = max(1, int(frame_h * scale))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (effective_w, effective_h))

    print(
        f"  {video_path.name}  "
        f"({frame_w}x{frame_h} @ {fps:.1f} fps, ~{total_video_frames} frames)"
        + (f"  [resized → {effective_w}x{effective_h}]" if max_dim else "")
    )

    frame_index = 0    # raw frame counter — drives timestamp_ms
    processed_count = 0

    try:
        while True:
            if max_frames is not None and processed_count >= max_frames:
                break

            # Skip-frames logic: use grab() (no pixel decode) for skipped frames
            # to avoid decoding every 4K frame even when not needed.
            if frame_index % skip_frames != 0:
                ret = cap.grab()
                if not ret:
                    break
                frame_index += 1
                continue

            ret, bgr_frame = cap.read()
            if not ret:
                break

            camera_id = video_name
            frame_id = f"{video_name}_frame_{frame_index:06d}"
            timestamp_ms = int(frame_index * 1000.0 / fps)

            t_start = time.perf_counter()
            try:
                if max_dim is not None:
                    bgr_frame = resize_bgr(bgr_frame, max_dim)
                packet = bgr_to_frame_packet(
                    bgr_frame, frame_id, camera_id, timestamp_ms
                )
                output: RecognitionPipelineOutput = rpm.process_frame(packet)
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0

                stats.total_frames_processed += 1
                stats.processing_times_ms.append(elapsed_ms)
                processed_count += 1

                # Cold-start heuristic: first raw frame always lacks a PREVIOUS
                # frame in FTL, so motion detection cannot fire.
                if frame_index == 0:
                    stats.cold_start_frames += 1

                persons = output.get("persons", [])
                if persons:
                    stats.frames_with_persons += 1
                    if any(p.get("recognized_faces") for p in persons):
                        stats.frames_with_recognized_faces += 1

                # Draw overlays
                annotated = draw_overlays(bgr_frame.copy(), output)
                writer.write(annotated)

                # Save person/face crops on request
                if save_crops and persons:
                    save_crops_from_output(bgr_frame, output, crops_dir, frame_id)

                # Save a JPEG snapshot every 30 processed frames
                if processed_count % 30 == 0:
                    cv2.imwrite(str(frames_dir / f"{frame_id}.jpg"), annotated)

                if show:
                    cv2.imshow(f"RPM Debug: {video_name}", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("    [show] q pressed — skipping to next video")
                        break

                # Print every frame when few frames requested, else every 10
                _progress_interval = 1 if (max_frames or 9999) <= 10 else 10
                if processed_count % _progress_interval == 0:
                    print(
                        f"    frame {frame_index:6d} | "
                        f"processed {processed_count:4d} | "
                        f"{elapsed_ms:6.0f} ms | "
                        f"persons={len(persons)}"
                    )

            except Exception as frame_err:
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                stats.error_count += 1
                stats.error_frame_ids.append(frame_id)
                print(f"    [ERROR] {frame_id}: {frame_err}")

            frame_index += 1

    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyWindow(f"RPM Debug: {video_name}")

    # ---- per-video JSON report ----
    report: dict[str, Any] = {
        "video_name": video_name,
        "video_path": str(video_path),
        "annotated_video_path": str(annotated_path),
        "total_frames_processed": stats.total_frames_processed,
        "cold_start_frames": stats.cold_start_frames,
        "frames_with_motion": (
            "N/A — not directly exposed by RecognitionPipelineOutput; "
            "use frames_with_persons as proxy"
        ),
        "frames_with_persons": stats.frames_with_persons,
        "frames_with_recognized_faces": stats.frames_with_recognized_faces,
        "error_count": stats.error_count,
        "error_frame_ids": stats.error_frame_ids,
        "avg_processing_ms": round(stats.avg_processing_ms, 2),
        "max_processing_ms": round(stats.max_processing_ms, 2),
        "object_detection_backend": _od_backend_info(rpm),
    }
    with open(reports_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(
        f"  -> processed={stats.total_frames_processed} | "
        f"persons_frames={stats.frames_with_persons} | "
        f"face_frames={stats.frames_with_recognized_faces} | "
        f"errors={stats.error_count}"
    )
    print(f"  -> annotated video: {annotated_path}")
    return stats


# ---------------------------------------------------------------------------
# Global markdown summary
# ---------------------------------------------------------------------------
def write_global_summary(
    all_stats: list[VideoStats],
    output_dir: Path,
    video_dir: Path,
    use_fakes: bool,
) -> Path:
    """Write summary_all_videos.md to output_dir."""
    summary_path = output_dir / "summary_all_videos.md"

    mode_str = "FAKES (--use-fakes)" if use_fakes else "REAL modules"
    lines = [
        "# RPM Visual Debug Run — Summary",
        "",
        f"- **Video directory**: `{video_dir}`",
        f"- **Output directory**: `{output_dir}`",
        f"- **Mode**: {mode_str}",
        f"- **Videos processed**: {len(all_stats)}",
        "",
        "## Notes",
        "",
        "- `frames_with_motion` is not directly available from `RecognitionPipelineOutput`.",
        "  `frames_with_persons` is the nearest proxy (persons only detected when motion fires).",
        "- `cold_start_frames` = 1 per video (first frame lacks a PREVIOUS frame in FTL).",
        "- Motion ROIs and aligned-face crops are not exposed by `RecognitionPipelineOutput`.",
        "  No changes were made to the RPM public API.",
        "",
        "## Per-Video Results",
        "",
        "| Video | Frames processed | Persons (frames) | Faces recognised (frames) | Errors | Annotated video |",
        "|-------|-----------------|-----------------|--------------------------|--------|----------------|",
    ]

    for s in all_stats:
        rel_path = f"`debug_outputs/rpm_visual/{s.video_name}/annotated_video.mp4`"
        lines.append(
            f"| {s.video_name} "
            f"| {s.total_frames_processed} "
            f"| {s.frames_with_persons} "
            f"| {s.frames_with_recognized_faces} "
            f"| {s.error_count} "
            f"| {rel_path} |"
        )

    lines += ["", "---", "*Generated by `tools/rpm_visual_debug_runner.py`*", ""]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RPM Visual Debug Runner — feed video files through "
            "RecognitionPipelineManager and save annotated outputs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path(r"C:\Users\talif\Desktop\videos"),
        metavar="PATH",
        help="Directory containing video files (.mp4/.avi/.mov/.mkv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "debug_outputs" / "rpm_visual",
        metavar="PATH",
        help="Root directory where debug outputs will be written",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Maximum frames to process per video (default: unlimited)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated frames in a cv2 window (press q to skip to next video)",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save person and face bbox crops per frame to crops/",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        metavar="N",
        help="Process every Nth frame (default: 1 = every frame)",
    )
    parser.add_argument(
        "--use-fakes",
        action="store_true",
        help=(
            "Use fake pipeline dependencies (no ML models needed). "
            "Validates overlay/output flow only."
        ),
    )
    parser.add_argument(
        "--gallery-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "generated_face_gallery",
        metavar="PATH",
        help="Path to face gallery used by FaceRecognitionModule (default: generated_face_gallery)",
    )
    parser.add_argument(
        "--person-json",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to PersonDirectory JSON file mapping person_id → name. "
            "Optional — omit to run without person names (names fall back to person_id)."
        ),
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=None,
        metavar="PX",
        help=(
            "Downscale frames so the longest side is at most PX before inference "
            "(e.g. 640 makes 4K frames ~30x faster on CPU). Default: no resize."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    print("=" * 60)
    print("RPM Visual Debug Runner")
    print("=" * 60)
    print(f"  video-dir   : {args.video_dir}")
    print(f"  output-dir  : {args.output_dir}")
    print(f"  max-frames  : {args.max_frames or 'unlimited'}")
    print(f"  skip-frames : {args.skip_frames}")
    print(f"  save-crops  : {args.save_crops}")
    print(f"  show        : {args.show}")
    print(f"  use-fakes   : {args.use_fakes}")
    print(f"  gallery-dir : {args.gallery_dir}")
    print(f"  person-json : {args.person_json}")
    print(f"  max-dim     : {args.max_dim or 'original'}")
    print()

    # ---- Validate inputs ----
    if not args.video_dir.is_dir():
        print(f"[FATAL] --video-dir does not exist or is not a directory: {args.video_dir}")
        sys.exit(1)

    if not args.use_fakes and not args.gallery_dir.is_dir():
        print(f"[FATAL] --gallery-dir does not exist: {args.gallery_dir}")
        print("  → Provide a valid gallery directory or use --use-fakes for offline validation.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Find videos ----
    video_files = find_video_files(args.video_dir)
    if not video_files:
        print(f"[WARNING] No video files found in: {args.video_dir}")
        print(f"  Supported extensions: {sorted(VIDEO_EXTENSIONS)}")
        sys.exit(0)

    print(f"Videos found: {len(video_files)}")
    for v in video_files:
        print(f"  • {v.name}")
    print()

    # ---- Build RPM ----
    try:
        rpm = build_rpm_for_visual_debug(
            gallery_dir=args.gallery_dir,
            person_json=args.person_json,
            use_fakes=args.use_fakes,
        )
    except Exception:
        print("\n[FATAL] Could not build RecognitionPipelineManager. See errors above.")
        sys.exit(1)

    # ---- Process each video ----
    all_stats: list[VideoStats] = []
    failed_videos: list[str] = []

    for video_path in video_files:
        print(f"\n[Video] {video_path.name}")
        try:
            stats = process_video(
                video_path=video_path,
                rpm=rpm,
                output_dir=args.output_dir,
                max_frames=args.max_frames,
                show=args.show,
                save_crops=args.save_crops,
                skip_frames=args.skip_frames,
                max_dim=args.max_dim,
            )
            all_stats.append(stats)
        except Exception as vid_err:
            print(f"  [ERROR] {video_path.name}: {vid_err}")
            traceback.print_exc()
            failed_videos.append(video_path.name)
            all_stats.append(VideoStats(video_name=video_path.stem, error_count=1))

    # ---- Global summary ----
    summary_path = write_global_summary(
        all_stats=all_stats,
        output_dir=args.output_dir,
        video_dir=args.video_dir,
        use_fakes=args.use_fakes,
    )

    # ---- Final report ----
    print()
    print("=" * 60)
    print("Run Complete")
    print("=" * 60)
    print(f"  Videos found     : {len(video_files)}")
    print(f"  Videos processed : {len(all_stats) - len(failed_videos)}")
    print(f"  Videos failed    : {len(failed_videos)}")
    if failed_videos:
        for fv in failed_videos:
            print(f"    x {fv}")
    print(f"  Output folder    : {args.output_dir}")
    print()

    for s in all_stats:
        if s.total_frames_processed > 0:
            vid_out = args.output_dir / s.video_name / "annotated_video.mp4"
            print(f"  [{s.video_name}]")
            print(f"    Annotated video   : {vid_out}")
            print(f"    Frames processed  : {s.total_frames_processed}")
            print(f"    Persons (frames)  : {s.frames_with_persons}")
            print(f"    Faces (frames)    : {s.frames_with_recognized_faces}")
            print(f"    Avg time/frame    : {s.avg_processing_ms:.1f} ms")
            print(f"    Max time/frame    : {s.max_processing_ms:.1f} ms")
            print(f"    Errors            : {s.error_count}")
            print()

    print(f"  Summary report   : {summary_path}")


if __name__ == "__main__":
    main()
