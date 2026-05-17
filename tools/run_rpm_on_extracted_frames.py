"""
run_rpm_on_extracted_frames.py
-------------------------------
Phase 2 of RPM visual-inspection tooling.

Reads frame inputs prepared by extract_video_frames_for_rpm_debug.py (phase 1),
feeds each frame sequentially through a fake-only RecognitionPipelineManager, and
writes a per-frame stage-by-stage debug folder showing exactly what image each
pipeline stage received and what each fake returned.

This script does NOT run real inference models.
This script does NOT require a face gallery.
This script does NOT write annotated MP4 video.

RPM API is untouched.  Stage-by-stage interception is done via LoggingFTL
(wraps the real FrameTransformationLayer — delegates ingest/get_frame and logs
temporal diagnostics) and thin per-stage logging wrappers around the standard fakes.

Input layout (produced by extract_video_frames_for_rpm_debug.py):
    <input_dir>/
        <video_name>/
            frames/          frame_XXXXXX.jpg
            metadata/        frames_manifest.json   (optional; fallback: raw scan)

Output layout:
    <output_dir>/
        <video_name>/
            frame_000000/
                00_original.jpg
                01_ftl_0_current_...jpg
                02_motion_output.json
                03_obj_det_input_0.jpg
                03_obj_det_output.json
                04_face_det_input_0.jpg
                04_face_det_output.json
                06_final_annotated.jpg
                metadata.json
        summary_all_frames.json
"""

from __future__ import annotations

import argparse
import json
import sys
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
# Third-party early validation
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

try:
    import yaml
except ImportError:
    print("[FATAL] pyyaml is not installed.  Run: pip install pyyaml")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
from image_processing.recognition_pipeline_manager import (
    RecognitionPipelineManager,
    RecognitionPipelineOutput,
)
from image_processing.frame_transformation_layer.contracts import (
    FramePacket,
    FrameTemporalSelector,
    PreviousFrameNotAvailableError,
    ProcessedFrame,
)
from image_processing.frame_transformation_layer import FrameTransformationLayer
from image_processing.shared.contracts import (
    BoundingBox,
    OutputImageType,
    PipelineStageInputContract,
)
from image_processing.motion_detection.module import MotionResult
from image_processing.object_detection.module import PersonDetectionResult
from image_processing.face_detection.module import DetectedFace, FaceDetectionOutput
from image_processing.shared.contracts import FaceLandmarks, Point
from image_processing.person_directory.module import PersonDirectoryOutput

# Fakes — loaded from tests/ via sys.path
from recognition_pipeline_manager.fakes import (  # type: ignore[import]
    FakeFaceDetection,
    FakeFaceRecognition,
    FakeMotionDetection,
    FakeObjectDetection,
    FakePersonDirectory,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_THICKNESS = 1
_COLOR_MOTION = (0, 255, 255)       # yellow      — motion ROI box (BGR)
_COLOR_PERSON = (0, 220, 0)         # green       — person bbox (BGR)
_COLOR_FACE = (255, 255, 0)         # cyan        — face bbox (BGR)
_COLOR_LANDMARK = (0, 200, 255)     # orange-blue — landmark dot (BGR)
_COLOR_RECOGNITION = (255, 0, 200)  # magenta     — recognition label (BGR)
_COLOR_HUD = (200, 200, 200)        # light-gray  — HUD text


# ---------------------------------------------------------------------------
# Input data model
# ---------------------------------------------------------------------------

@dataclass
class FrameEntry:
    frame_id: str
    frame_index: int
    timestamp_ms: int | None
    image_path: Path


@dataclass
class FrameSet:
    video_name: str
    frames: list[FrameEntry]
    frame_width: int
    frame_height: int


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fake RPM on pre-extracted frames for stage-by-stage inspection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_PROJECT_ROOT / "debug_inputs" / "rpm_frames",
        help="Root directory produced by extract_video_frames_for_rpm_debug.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "debug_outputs" / "rpm_frame_inspection",
        help="Root directory for per-frame debug folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-frame debug folders.",
    )
    parser.add_argument(
        "--person-json",
        type=Path,
        default=_PROJECT_ROOT / "data" / "person_directory.json",
        help="Path to person_directory.json for identity name resolution.",
    )
    parser.add_argument(
        "--rpm-config",
        type=Path,
        default=None,
        help="Path to root YAML config (e.g. config/image_processing_service/local_debug.yaml). "
             "Required when --real-object-detection is set.",
    )
    parser.add_argument(
        "--real-object-detection",
        action="store_true",
        help="Use the real ObjectDetection module instead of FakeObjectDetection. "
             "Requires --rpm-config.",
    )
    parser.add_argument(
        "--real-face-detection",
        action="store_true",
        help="Use the real FaceDetectionModule instead of FakeFaceDetection. "
             "Requires --rpm-config. Normally used together with --real-object-detection.",
    )
    parser.add_argument(
        "--real-face-recognition",
        action="store_true",
        help="Use the real FaceRecognitionModule instead of FakeFaceRecognition. "
             "Requires --rpm-config. Normally used with --real-object-detection and "
             "--real-face-detection.",
    )
    parser.add_argument(
        "--real-motion-detection",
        action="store_true",
        help="Use the real MotionDetectionManager instead of FakeMotionDetection. "
             "Requires --rpm-config.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# LoggingFTL — wraps the real FrameTransformationLayer; logs every call
# ---------------------------------------------------------------------------

class LoggingFTL:
    """Delegates ingest_frame and get_frame to the real FrameTransformationLayer.

    Provides temporal diagnostic logging on every call and captures cropped
    debug images from the returned ProcessedFrame for the debug output folder.

    Cold-start behavior: PreviousFrameNotAvailableError is caught only to set
    cold_start_triggered, then unconditionally re-raised so that
    PipelineOrchestrator handles it correctly (spec §11).
    """

    def __init__(self) -> None:
        self._ftl = FrameTransformationLayer()
        # Shadow per-camera state for ingest logging only — not used for temporal logic
        self._current_frame_id: dict[str, str] = {}
        self._log: list[dict[str, Any]] = []
        self._cold_start_triggered: bool = False

    # ------------------------------------------------------------------
    # FTL interface — all temporal state owned by _ftl
    # ------------------------------------------------------------------

    def ingest_frame(self, packet: FramePacket) -> None:
        self._cold_start_triggered = False
        camera_id = packet.camera_id
        incoming_frame_id = packet.frame_id
        prev_frame_id = self._current_frame_id.get(camera_id, "none")

        self._ftl.ingest_frame(packet)

        self._current_frame_id[camera_id] = incoming_frame_id
        print(
            f"[FTL INGEST] camera={camera_id!r}  incoming={incoming_frame_id!r}  "
            f"prev_after={prev_frame_id!r}  current_after={incoming_frame_id!r}"
        )

    def get_frame(
        self,
        camera_id: str,
        temporal_selector: FrameTemporalSelector,
        region_bbox: BoundingBox,
        output_type: OutputImageType,
        geometry_spec: Any,
    ) -> ProcessedFrame:
        call_index = len(self._log)
        try:
            processed = self._ftl.get_frame(
                camera_id, temporal_selector, region_bbox, output_type, geometry_spec
            )
        except PreviousFrameNotAvailableError:
            self._cold_start_triggered = True
            print(
                f"[FTL GET FRAME] camera={camera_id!r}  "
                f"temporal={temporal_selector.value!r}  -> PreviousFrameNotAvailableError (cold start)"
            )
            raise  # re-raise: PipelineOrchestrator must handle this

        print(
            f"[FTL GET FRAME] camera={camera_id!r}  temporal={temporal_selector.value!r}  "
            f"returned_frame_id={processed.frame_id!r}  region_bbox={dict(region_bbox)}"
        )

        # Capture BGR copy from ProcessedFrame.image.data for debug file output
        data = processed.image["data"]
        color_fmt = processed.image.get("color_format", "RGB")
        if color_fmt == "GRAY":
            gray2d = data[:, :, 0] if data.ndim == 3 else data
            bgr_save: np.ndarray = cv2.cvtColor(gray2d, cv2.COLOR_GRAY2BGR)
        else:
            bgr_save = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        self._log.append({
            "call_index": call_index,
            "temporal_selector": temporal_selector.value,
            "region_bbox": dict(region_bbox),
            "source_bbox_full_frame": {
                "x": processed.source_bbox_full_frame["x"],
                "y": processed.source_bbox_full_frame["y"],
                "width": processed.source_bbox_full_frame["width"],
                "height": processed.source_bbox_full_frame["height"],
            },
            "spatial_transform": {
                "scale_x": processed.spatial_transform.scale_x,
                "scale_y": processed.spatial_transform.scale_y,
                "pad_left": processed.spatial_transform.pad_left,
                "pad_top": processed.spatial_transform.pad_top,
                "output_width": processed.spatial_transform.output_width,
                "output_height": processed.spatial_transform.output_height,
            },
            "returned_frame_id": processed.frame_id,
            "cropped_bgr": bgr_save,
        })

        return processed

    def flush_log(self) -> list[dict[str, Any]]:
        log = list(self._log)
        self._log.clear()
        return log

    @property
    def cold_start_triggered(self) -> bool:
        """True if the most recent process_frame call triggered a cold-start condition."""
        return self._cold_start_triggered

class LoggingMotionDetection:
    """Wraps FakeMotionDetection; logs input current-frame image + output."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._log: list[dict[str, Any]] = []

    def detect(self, input: Any) -> MotionResult:
        call_index = len(self._log)
        # current_frame.image.data is H×W×1 GRAYSCALE uint8
        curr_data = input["current_frame"]["image"]["data"]
        curr_gray = curr_data[:, :, 0] if curr_data.ndim == 3 else curr_data
        curr_bgr = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2BGR)

        prev_data = input["previous_frame"]["image"]["data"]
        prev_gray = prev_data[:, :, 0] if prev_data.ndim == 3 else prev_data
        prev_bgr = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)

        # Temporal diagnostic: verify frame_ids differ after cold start
        print(
            f"[MOTION INPUT] current_frame_id={input['current_frame']['frame_id']!r}  "
            f"previous_frame_id={input['previous_frame']['frame_id']!r}  "
            f"current_ts_ms={input['current_frame']['timestamp_ms']}  "
            f"previous_ts_ms={input['previous_frame']['timestamp_ms']}  "
            f"module={type(self._delegate).__name__}"
        )

        result = self._delegate.detect(input)
        debug_info = {}
        if hasattr(self._delegate, "get_last_debug_info"):
            debug_info = _safe_dict(self._delegate.get_last_debug_info())

        debug_images = {}
        if hasattr(self._delegate, "get_last_debug_images"):
            debug_images = self._delegate.get_last_debug_images()

        self._log.append({
            "call_index": call_index,
            "input_bgr": curr_bgr,
            "previous_input_bgr": prev_bgr,
            "output": _safe_dict(result),
            "debug_info": debug_info,
            "debug_images": debug_images,
        })
        return result

    def get_input_contract(self) -> PipelineStageInputContract:
        return self._delegate.get_input_contract()

    def flush_log(self) -> list[dict[str, Any]]:
        log = list(self._log)
        self._log.clear()
        return log


class LoggingObjectDetection:
    """Wraps FakeObjectDetection; logs roi_image + output."""

    def __init__(self, delegate: FakeObjectDetection) -> None:
        self._delegate = delegate
        self._log: list[dict[str, Any]] = []

    def detect(self, input: Any) -> PersonDetectionResult:
        call_index = len(self._log)
        # roi_image.data is H×W×3 RGB uint8
        bgr_save = cv2.cvtColor(input["roi_image"]["data"], cv2.COLOR_RGB2BGR)

        result = self._delegate.detect(input)

        self._log.append({
            "call_index": call_index,
            "input_bgr": bgr_save,
            "output": _safe_dict(result),
        })
        return result

    def get_input_contract(self) -> PipelineStageInputContract:
        return self._delegate.get_input_contract()

    def flush_log(self) -> list[dict[str, Any]]:
        log = list(self._log)
        self._log.clear()
        return log


class LoggingFaceDetection:
    """Wraps FakeFaceDetection; logs roi_image + output."""

    def __init__(self, delegate: FakeFaceDetection) -> None:
        self._delegate = delegate
        self._log: list[dict[str, Any]] = []

    def detect_faces(self, input: Any) -> Any:
        call_index = len(self._log)
        bgr_save = cv2.cvtColor(input["roi_image"]["data"], cv2.COLOR_RGB2BGR)

        result = self._delegate.detect_faces(input)

        self._log.append({
            "call_index": call_index,
            "input_bgr": bgr_save,
            "output": _safe_dict(result),
        })
        return result

    def get_input_contract(self) -> PipelineStageInputContract:
        return self._delegate.get_input_contract()

    def flush_log(self) -> list[dict[str, Any]]:
        log = list(self._log)
        self._log.clear()
        return log


class LoggingFaceRecognition:
    """Wraps FakeFaceRecognition; logs face_roi_image + output."""

    def __init__(self, delegate: FakeFaceRecognition) -> None:
        self._delegate = delegate
        self._log: list[dict[str, Any]] = []

    def recognize(self, face_input: Any) -> Any:
        call_index = len(self._log)
        bgr_save = cv2.cvtColor(face_input["face_roi_image"]["data"], cv2.COLOR_RGB2BGR)

        result = self._delegate.recognize(face_input)

        self._log.append({
            "call_index": call_index,
            "input_bgr": bgr_save,
            "output": _safe_dict(result),
        })
        return result

    def get_input_contract(self) -> PipelineStageInputContract:
        return self._delegate.get_input_contract()

    def flush_log(self) -> list[dict[str, Any]]:
        log = list(self._log)
        self._log.clear()
        return log


# ---------------------------------------------------------------------------
# JSON safety helpers
# ---------------------------------------------------------------------------

def _safe_dict(obj: Any) -> Any:
    """Recursively convert TypedDict / numpy scalars to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _safe_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_dict(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return {"_ndarray_shape": list(obj.shape), "_dtype": str(obj.dtype)}
    return obj


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# YAML config helpers
# ---------------------------------------------------------------------------

def load_yaml_config(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_config_path(path_str: str, project_root: Path) -> Path:
    """Resolve a config path that may be absolute or project-root-relative."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root / p


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def load_frame_sets(input_dir: Path) -> list[FrameSet]:
    """
    Scan input_dir for per-video sub-directories.
    Prefer frames_manifest.json; fall back to raw image scan of frames/.
    Returns a sorted list of FrameSet objects.
    """
    if not input_dir.is_dir():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return []

    frame_sets: list[FrameSet] = []

    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue
        video_name = subdir.name

        manifest_path = subdir / "metadata" / "frames_manifest.json"
        frames_dir = subdir / "frames"

        entries: list[FrameEntry] = []
        frame_width = 0
        frame_height = 0

        # ------------------------------------------------------------------
        # Manifest-based loading (preferred)
        # ------------------------------------------------------------------
        if manifest_path.exists():
            try:
                with open(manifest_path, encoding="utf-8") as fh:
                    manifest = json.load(fh)

                frame_width = manifest.get("extracted_width", 0)
                frame_height = manifest.get("extracted_height", 0)

                for ef in manifest.get("extracted_frames", []):
                    if ef.get("skipped") or ef.get("read_failed"):
                        continue
                    rel = ef.get("image_path")
                    if not rel:
                        continue
                    abs_path = input_dir / rel
                    if not abs_path.exists():
                        print(f"  [WARN] Missing frame file: {abs_path}")
                        continue
                    entries.append(FrameEntry(
                        frame_id=ef.get("frame_id", f"{video_name}_frame_{ef.get('frame_index', 0):06d}"),
                        frame_index=ef.get("frame_index", 0),
                        timestamp_ms=ef.get("timestamp_ms"),
                        image_path=abs_path,
                    ))
                print(f"[INFO] {video_name}: {len(entries)} frames from manifest")

            except Exception as exc:
                print(f"  [WARN] Cannot read manifest for {video_name}: {exc} — falling back to raw scan")
                entries = []

        # ------------------------------------------------------------------
        # Fallback: raw image scan
        # ------------------------------------------------------------------
        if not entries and frames_dir.is_dir():
            raw_files = sorted(
                p for p in frames_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            for i, p in enumerate(raw_files):
                entries.append(FrameEntry(
                    frame_id=f"{video_name}_{p.stem}",
                    frame_index=i,
                    timestamp_ms=None,
                    image_path=p,
                ))
            print(f"[INFO] {video_name}: {len(entries)} frames from raw scan (no manifest)")

        if not entries:
            print(f"[INFO] {video_name}: no frames found — skipping")
            continue

        # If dimensions not in manifest, read first frame
        if frame_width == 0 or frame_height == 0:
            first_bgr = cv2.imread(str(entries[0].image_path))
            if first_bgr is not None:
                frame_height, frame_width = first_bgr.shape[:2]

        frame_sets.append(FrameSet(
            video_name=video_name,
            frames=entries,
            frame_width=frame_width,
            frame_height=frame_height,
        ))

    total = sum(len(fs.frames) for fs in frame_sets)
    print(f"[INFO] Found {len(frame_sets)} frame set(s), {total} total frame(s).")
    return frame_sets


# ---------------------------------------------------------------------------
# Debug component factory
# ---------------------------------------------------------------------------

@dataclass
class DebugComponents:
    rpm: RecognitionPipelineManager
    logging_ftl: LoggingFTL
    logging_motion: LoggingMotionDetection
    logging_obj: LoggingObjectDetection
    logging_face_det: LoggingFaceDetection
    logging_face_rec: LoggingFaceRecognition


def build_debug_components(
    frame_width: int,
    frame_height: int,
    person_json: Path,
    rpm_config: dict | None = None,
    real_object_detection: bool = False,
    real_face_detection: bool = False,
    real_face_recognition: bool = False,
    real_motion_detection: bool = False,
) -> DebugComponents:
    """
    Construct fakes with clearly inset bboxes so every annotation layer is
    visually distinct and not hidden at the image border.

      Layer        Coordinate space      Inset rule
      ----------   -------------------   -------------------------------------------
      Motion ROI   full-frame            3 % inset on every edge
      Person bbox  motion-ROI-local      25 % from left, 15 % from top, 50 % x 75 %
      Face bbox    person-crop-local     20 % from left, 5 % from top, 60 % x 50 %

    When real_object_detection=True, builds the real ObjectDetectionModule from
    rpm_config["models"]["object_detection_config"] and wraps it with
    LoggingObjectDetection.  When real_face_detection=True, builds the real
    FaceDetectionModule (backed by SCRFDFaceDetector) from
    rpm_config["models"]["face_detection_config"].  When real_motion_detection=True,
    builds the real MotionDetectionManager from
    rpm_config["models"]["motion_detection_config"].  All other modules remain STUB.
    """
    W = max(1, frame_width)
    H = max(1, frame_height)

    # Motion ROI: full-frame absolute coordinates, inset 3 % per edge
    mx = max(4, int(W * 0.03))
    my = max(4, int(H * 0.03))
    mw = W - 2 * mx
    mh = H - 2 * my
    motion_bbox: BoundingBox = {  # type: ignore[typeddict-item]
        "x": mx, "y": my, "width": mw, "height": mh,
    }

    # Person bbox: local to the motion-ROI crop (mw x mh image)
    px_l = max(2, int(mw * 0.25))
    py_l = max(2, int(mh * 0.15))
    pw   = max(4, int(mw * 0.50))
    ph   = max(4, int(mh * 0.75))
    person_bbox_local: BoundingBox = {  # type: ignore[typeddict-item]
        "x": px_l, "y": py_l, "width": pw, "height": ph,
    }

    # Face bbox: local to the person crop (pw x ph image)
    fx_l = max(2, int(pw * 0.20))
    fy_l = max(2, int(ph * 0.05))
    fw   = max(4, int(pw * 0.60))
    fh   = max(4, int(ph * 0.50))
    face_bbox_local: BoundingBox = {  # type: ignore[typeddict-item]
        "x": fx_l, "y": fy_l, "width": fw, "height": fh,
    }

    # Landmarks in person-crop-local space (same space as face_bbox_local)
    _fc_x = fx_l + fw // 2
    _fc_y = fy_l + fh // 2
    _fake_landmarks: FaceLandmarks = FaceLandmarks(
        left_eye=Point(x=_fc_x - fw // 6, y=_fc_y - fh // 8),
        right_eye=Point(x=_fc_x + fw // 6, y=_fc_y - fh // 8),
        nose=Point(x=_fc_x,                y=_fc_y),
        mouth_left=Point(x=_fc_x - fw // 8, y=_fc_y + fh // 6),
        mouth_right=Point(x=_fc_x + fw // 8, y=_fc_y + fh // 6),
    )

    # ------------------------------------------------------------------
    # Motion Detection: STUB (default) or REAL
    # ------------------------------------------------------------------
    _md_mode_label = "STUB"
    if real_motion_detection:
        assert rpm_config is not None, "rpm_config must be provided when real_motion_detection=True"
        from image_processing.motion_detection import (  # noqa: PLC0415
            MotionDetectionConfig,
            MotionDetectionManager,
        )
        _md_config_rel = rpm_config["models"]["motion_detection_config"]
        _md_config_path = resolve_config_path(_md_config_rel, _PROJECT_ROOT)
        print(f"[CONFIG] Real MD config: {_md_config_path}")
        _md_yaml = load_yaml_config(_md_config_path)
        _md_cfg = MotionDetectionConfig(
            motion_threshold=int(_md_yaml.get("motion_threshold", 50)),
            motion_fraction_threshold=float(_md_yaml.get("motion_fraction_threshold", 0.05)),
            min_bbox_area=int(_md_yaml.get("min_bbox_area", 5000)),
            blur_kernel_size=int(_md_yaml.get("blur_kernel_size", 0)),
            morph_open_iterations=int(_md_yaml.get("morph_open_iterations", 0)),
            morph_close_iterations=int(_md_yaml.get("morph_close_iterations", 0)),
            dilation_iterations=int(_md_yaml.get("dilation_iterations", 0)),
            min_aspect_ratio=float(_md_yaml.get("min_aspect_ratio", 0.0)),
            max_aspect_ratio=float(_md_yaml.get("max_aspect_ratio", 1000.0)),
            enable_global_motion_compensation=bool(_md_yaml.get("enable_global_motion_compensation", False)),
            global_motion_method=str(_md_yaml.get("global_motion_method", "gftt_lk_affine")),
            max_features=int(_md_yaml.get("max_features", 300)),
            min_feature_matches=int(_md_yaml.get("min_feature_matches", 20)),
            max_transform_shift=float(_md_yaml.get("max_transform_shift", 25.0)),
            global_motion_changed_ratio_threshold=float(_md_yaml.get("global_motion_changed_ratio_threshold", 0.30)),
            fallback_on_alignment_failure=bool(_md_yaml.get("fallback_on_alignment_failure", True)),
            enable_bbox_merging=bool(_md_yaml.get("enable_bbox_merging", False)),
            bbox_merge_iou_threshold=float(_md_yaml.get("bbox_merge_iou_threshold", 0.35)),
            bbox_merge_distance_threshold=float(_md_yaml.get("bbox_merge_distance_threshold", 12.0)),
            enable_temporal_persistence=bool(_md_yaml.get("enable_temporal_persistence", False)),
            min_persistence_frames=int(_md_yaml.get("min_persistence_frames", 2)),
            persistence_iou_threshold=float(_md_yaml.get("persistence_iou_threshold", 0.35)),
            max_history_frames=int(_md_yaml.get("max_history_frames", 8)),
        )
        print(f"[CONFIG] Real MD motion_threshold          : {_md_cfg.motion_threshold}")
        print(f"[CONFIG] Real MD motion_fraction_threshold : {_md_cfg.motion_fraction_threshold}")
        print(f"[CONFIG] Real MD min_bbox_area             : {_md_cfg.min_bbox_area}")
        print(f"[CONFIG] Real MD blur_kernel_size          : {_md_cfg.blur_kernel_size}")
        print(f"[CONFIG] Real MD morph_open_iterations     : {_md_cfg.morph_open_iterations}")
        print(f"[CONFIG] Real MD morph_close_iterations    : {_md_cfg.morph_close_iterations}")
        print(f"[CONFIG] Real MD dilation_iterations       : {_md_cfg.dilation_iterations}")
        print(f"[CONFIG] Real MD min_aspect_ratio          : {_md_cfg.min_aspect_ratio}")
        print(f"[CONFIG] Real MD max_aspect_ratio          : {_md_cfg.max_aspect_ratio}")
        print(f"[CONFIG] Real MD enable_global_motion_comp: {_md_cfg.enable_global_motion_compensation}")
        print(f"[CONFIG] Real MD global_motion_method      : {_md_cfg.global_motion_method}")
        print(f"[CONFIG] Real MD max_features              : {_md_cfg.max_features}")
        print(f"[CONFIG] Real MD min_feature_matches       : {_md_cfg.min_feature_matches}")
        print(f"[CONFIG] Real MD max_transform_shift       : {_md_cfg.max_transform_shift}")
        print(f"[CONFIG] Real MD global_changed_ratio_thr  : {_md_cfg.global_motion_changed_ratio_threshold}")
        print(f"[CONFIG] Real MD fallback_on_align_fail    : {_md_cfg.fallback_on_alignment_failure}")
        print(f"[CONFIG] Real MD enable_bbox_merging       : {_md_cfg.enable_bbox_merging}")
        print(f"[CONFIG] Real MD bbox_merge_iou_threshold  : {_md_cfg.bbox_merge_iou_threshold}")
        print(f"[CONFIG] Real MD bbox_merge_distance_thr   : {_md_cfg.bbox_merge_distance_threshold}")
        print(f"[CONFIG] Real MD enable_temporal_persist   : {_md_cfg.enable_temporal_persistence}")
        print(f"[CONFIG] Real MD min_persistence_frames    : {_md_cfg.min_persistence_frames}")
        print(f"[CONFIG] Real MD persistence_iou_threshold : {_md_cfg.persistence_iou_threshold}")
        print(f"[CONFIG] Real MD max_history_frames        : {_md_cfg.max_history_frames}")
        motion_module: Any = MotionDetectionManager(config=_md_cfg)
        _md_mode_label = f"REAL (config: {_md_config_path})"
    else:
        motion_module = FakeMotionDetection(
            result=MotionResult(detected=True, bboxes=[motion_bbox])  # type: ignore[call-arg]
        )

    # ------------------------------------------------------------------
    # Object Detection: STUB (default) or REAL
    # ------------------------------------------------------------------
    _od_mode_label = "STUB"
    _od_config_path_str = ""
    if real_object_detection:
        assert rpm_config is not None, "rpm_config must be provided when real_object_detection=True"
        from image_processing.object_detection.module import (  # noqa: PLC0415
            ObjectDetectionModule,
            PersonDetectionConfig,
        )
        _od_config_rel = rpm_config["models"]["object_detection_config"]
        _od_config_path = resolve_config_path(_od_config_rel, _PROJECT_ROOT)
        _od_config_path_str = str(_od_config_path)
        print(f"[CONFIG] Real OD config: {_od_config_path}")
        _od_yaml = load_yaml_config(_od_config_path)
        _model_path = str(resolve_config_path(_od_yaml["model_path"], _PROJECT_ROOT))
        _od_cfg = PersonDetectionConfig(
            model_path=_model_path,
            person_confidence_threshold=float(_od_yaml.get("confidence_threshold", 0.35)),
            nms_iou_threshold=float(_od_yaml.get("nms_threshold", 0.35)),
        )
        obj_det_module: Any = ObjectDetectionModule(config=_od_cfg)
        _od_mode_label = f"REAL (config: {_od_config_path})"
    else:
        obj_det_module = FakeObjectDetection(
            result=PersonDetectionResult(  # type: ignore[call-arg]
                frame_id="",
                person_detected=True,
                persons=[person_bbox_local],
            )
        )

    _fake_face_detection_output = FaceDetectionOutput(
        frame_id="", camera_id="", timestamp_ms=0,
        detections=[DetectedFace(face_bbox=face_bbox_local, landmarks=_fake_landmarks)],
    )

    # ------------------------------------------------------------------
    # Face Detection: STUB (default) or REAL
    # ------------------------------------------------------------------
    _fd_mode_label = "STUB"
    if real_face_detection:
        assert rpm_config is not None, "rpm_config must be provided when real_face_detection=True"
        from image_processing.face_detection import (  # noqa: PLC0415
            FaceDetectionConfig,
            FaceDetectionModule,
            SCRFDFaceDetector,
        )
        _fd_config_rel = rpm_config["models"]["face_detection_config"]
        _fd_config_path = resolve_config_path(_fd_config_rel, _PROJECT_ROOT)
        print(f"[CONFIG] Real FD config: {_fd_config_path}")
        _fd_yaml = load_yaml_config(_fd_config_path)
        # FaceDetectionConfig only supports confidence_threshold from this YAML.
        # model_path in face_detection.yaml is intentionally ignored — SCRFDFaceDetector
        # uses the project model-dir convention (models/face_detection/) instead.
        _fd_model_path_raw = _fd_yaml.get("model_path", "")
        if not _fd_model_path_raw or _fd_model_path_raw == "TODO":
            print(
                f"[CONFIG] face_detection.yaml model_path={_fd_model_path_raw!r} is ignored; "
                f"SCRFDFaceDetector uses project model dir."
            )
        _fd_cfg = FaceDetectionConfig(
            confidence_threshold=float(_fd_yaml.get("confidence_threshold", 0.5)),
        )
        _scrfd_model_dir = _PROJECT_ROOT / "models" / "face_detection"
        print(f"[CONFIG] Real FD SCRFD model_dir: {_scrfd_model_dir}")
        _scrfd = SCRFDFaceDetector(model_dir=_scrfd_model_dir)
        face_det_module: Any = FaceDetectionModule(config=_fd_cfg, detector_engine=_scrfd)
        _fd_mode_label = f"REAL (config: {_fd_config_path})"
    else:
        face_det_module = FakeFaceDetection(result=_fake_face_detection_output)

    # ------------------------------------------------------------------
    # Face Recognition: STUB (default) or REAL
    # ------------------------------------------------------------------
    _fr_mode_label = "STUB"
    if real_face_recognition:
        assert rpm_config is not None, "rpm_config must be provided when real_face_recognition=True"
        from image_processing.face_gallery_loader import (  # noqa: PLC0415
            FaceGalleryLoaderModule,
        )
        from image_processing.face_recognition import (  # noqa: PLC0415
            ArcFaceEmbeddingEngine,
            EnrolledIdentity,
            FaceRecognitionConfig,
            FaceRecognitionModule,
        )
        _fr_config_rel = rpm_config["models"]["face_recognition_config"]
        _fr_config_path = resolve_config_path(_fr_config_rel, _PROJECT_ROOT)
        print(f"[CONFIG] Real FR config: {_fr_config_path}")
        _fr_yaml = load_yaml_config(_fr_config_path)
        _fr_threshold = float(_fr_yaml.get("recognition_threshold", 0.4))

        _gallery_root_str = rpm_config["face_gallery"]["gallery_root_path"]
        _gallery_root_path = resolve_config_path(_gallery_root_str, _PROJECT_ROOT)
        print(f"[CONFIG] Real FR gallery_root_path: {_gallery_root_path}")

        _gallery_loader = FaceGalleryLoaderModule()
        _gallery_loader.load_gallery(str(_gallery_root_path))
        _all_embeddings = _gallery_loader.get_all_embeddings()
        _gallery_person_ids = sorted(_gallery_loader.get_person_ids())

        print(f"[CONFIG] Real FR gallery entries loaded: {len(_all_embeddings)}")
        print(f"[CONFIG] Real FR recognition_threshold : {_fr_threshold}")
        print(f"[CONFIG] Real FR gallery person_ids    : {_gallery_person_ids}")

        # Cross-check gallery person_ids against person_directory.json
        _pd_ids: set[str] = set()
        if person_json.exists():
            try:
                with open(person_json, encoding="utf-8") as _fh:
                    _pd_ids = set(json.load(_fh).keys())
            except Exception as _exc:
                print(f"[WARN] Could not read person_directory.json for cross-check: {_exc}")
        for _gid in _gallery_person_ids:
            if _gid not in _pd_ids:
                print(f"[WARN] Gallery person_id {_gid!r} is NOT in person_directory.json — will resolve as UNKNOWN")
        _missing_in_gallery = _pd_ids - set(_gallery_person_ids)
        if _missing_in_gallery:
            print(f"[INFO] PersonDirectory IDs not in gallery (expected): {sorted(_missing_in_gallery)}")

        _enrolled: list[EnrolledIdentity] = [
            {"person_id": e["person_id"], "embedding": e["embedding"]}
            for e in _all_embeddings
        ]
        _fr_cfg = FaceRecognitionConfig(recognition_threshold=_fr_threshold)
        _arcface = ArcFaceEmbeddingEngine()
        face_rec_module: Any = FaceRecognitionModule(
            config=_fr_cfg,
            embedding_engine=_arcface,
            gallery_entries=_enrolled,
        )
        _fr_mode_label = f"REAL (config: {_fr_config_path})"
    else:
        # Use the first person UUID from person_directory.json so the overlay
        # demonstrates real name resolution instead of a hardcoded fake label.
        _DEMO_PERSON_ID = "1f007fe2-6eaf-5148-a240-a449664cb6eb"
        face_rec_module = FakeFaceRecognition(
            result={
                "frame_id": "", "camera_id": "", "timestamp_ms": 0,
                "person_found": True, "person_id": _DEMO_PERSON_ID,
            }  # type: ignore[arg-type]
        )

    # Load PersonDirectory once at startup; never called again per frame.
    from image_processing.person_directory import PersonDirectory, PersonDirectoryConfig  # noqa: PLC0415
    _pd_config = PersonDirectoryConfig(
        json_file_path=str(person_json),
        is_optional=True,  # fall back gracefully when JSON is absent
    )
    person_dir = PersonDirectory(config=_pd_config)
    person_dir.load()

    log_motion = LoggingMotionDetection(motion_module)
    log_obj = LoggingObjectDetection(obj_det_module)
    log_face_det = LoggingFaceDetection(face_det_module)
    log_face_rec = LoggingFaceRecognition(face_rec_module)
    logging_ftl = LoggingFTL()

    # ------------------------------------------------------------------
    # Startup module-mode table
    # ------------------------------------------------------------------
    print("[MODULE MODE]")
    print(f"  Motion Detection  : {_md_mode_label}  [{type(motion_module).__name__}]")
    print(f"  Object Detection  : {_od_mode_label}  [{type(obj_det_module).__name__}]")
    print(f"  Face Detection    : {_fd_mode_label}  [{type(face_det_module).__name__}]")
    print(f"  Face Recognition  : {_fr_mode_label}  [{type(face_rec_module).__name__}]")
    print(f"  PersonDirectory   : REAL")
    print(f"  FTL               : REAL (FrameTransformationLayer via LoggingFTL)")

    pipeline_cfg = (rpm_config or {}).get("recognition_pipeline", {})
    max_motion_rois_per_frame = int(pipeline_cfg.get("max_motion_rois_per_frame", 8))
    max_person_rois_per_frame = int(pipeline_cfg.get("max_person_rois_per_frame", 16))
    max_face_rois_per_frame = int(pipeline_cfg.get("max_face_rois_per_frame", 32))
    print(f"  ROI Limits        : motion={max_motion_rois_per_frame} person={max_person_rois_per_frame} face={max_face_rois_per_frame}")

    rpm = RecognitionPipelineManager(
        ftl=logging_ftl,
        motion=log_motion,
        object_det=log_obj,
        face_det=log_face_det,
        face_rec=log_face_rec,
        person_dir=person_dir,
        max_motion_rois_per_frame=max_motion_rois_per_frame,
        max_person_rois_per_frame=max_person_rois_per_frame,
        max_face_rois_per_frame=max_face_rois_per_frame,
    )

    return DebugComponents(
        rpm=rpm,
        logging_ftl=logging_ftl,
        logging_motion=log_motion,
        logging_obj=log_obj,
        logging_face_det=log_face_det,
        logging_face_rec=log_face_rec,
    )


# ---------------------------------------------------------------------------
# Per-frame helpers
# ---------------------------------------------------------------------------

def bgr_to_frame_packet(
    bgr_frame: np.ndarray,
    frame_id: str,
    camera_id: str,
    timestamp_ms: int,
) -> FramePacket:
    """Convert BGR cv2 frame to FramePacket (RGB bytes)."""
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


def draw_final_stage_overlay(
    bgr_frame: np.ndarray,
    ftl_log: list[dict],
    motion_log: list[dict],
    obj_log: list[dict],
    face_det_log: list[dict],
    face_rec_log: list[dict],
    rpm_output: RecognitionPipelineOutput,
) -> np.ndarray:
    """Draw all pipeline stage annotations onto a copy of bgr_frame.

    Coordinate projection:
      - Motion bbox  : full-frame from motion_log (drawn as-is)
      - Person bbox  : full-frame from rpm_output.persons[*].person_bbox (drawn as-is)
      - Face bbox    : full-frame from rpm_output.persons[*].recognized_faces[*].face_bbox (drawn as-is)
      - Landmarks    : disabled — rpm_output does not expose projected landmarks
      - Recognition  : full-frame from rpm_output.persons[*].recognized_faces[*]
      - obj_log / face_det_log are used for [COORD AUDIT] debug prints only, not drawing
    """
    annotated = bgr_frame.copy()
    h, w = annotated.shape[:2]

    def _clamp_box(x: int, y: int, bw: int, bh: int) -> tuple[int, int, int, int]:
        x1 = max(0, min(x, w - 1))
        y1 = max(0, min(y, h - 1))
        x2 = max(x1 + 1, min(x + bw, w))
        y2 = max(y1 + 1, min(y + bh, h))
        return x1, y1, x2, y2

    def _clamp_pt(px: int, py: int) -> tuple[int, int]:
        return max(0, min(px, w - 1)), max(0, min(py, h - 1))

    n_motion = 0
    n_person = 0
    n_face = 0
    n_label = 0

    # ------------------------------------------------------------------
    # [COORD AUDIT] Pre-build FTL lookup keyed by region_bbox for spatial_transform access.
    # Later writes win — for the same region, CURRENT comes after PREVIOUS.
    # ------------------------------------------------------------------
    _ftl_by_bbox: dict[tuple, dict] = {}
    for _ftl_e in ftl_log:
        _ftl_key = (
            _ftl_e["region_bbox"]["x"], _ftl_e["region_bbox"]["y"],
            _ftl_e["region_bbox"]["width"], _ftl_e["region_bbox"]["height"],
        )
        _ftl_by_bbox[_ftl_key] = _ftl_e

    # Flat list of rpm_output persons for cursor-based comparison with OD detections
    _rpm_persons_flat = list(rpm_output.get("persons", []))
    _rpm_person_cursor = 0

    # ------------------------------------------------------------------
    # Motion ROI boxes — yellow, full-frame absolute coordinates
    # ------------------------------------------------------------------
    motion_bboxes: list[dict] = []
    for m_entry in motion_log:
        for bbox in m_entry.get("output", {}).get("bboxes", []):
            x1, y1, x2, y2 = _clamp_box(
                bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), _COLOR_MOTION, 2)
            cv2.putText(annotated, "MOTION ROI", (x1 + 4, y1 + 18),
                        _FONT, _FONT_SCALE, _COLOR_MOTION, _THICKNESS + 1, cv2.LINE_AA)
            n_motion += 1
            motion_bboxes.append(bbox)

            # [COORD AUDIT] Motion bbox is full-frame; look up OD FTL spatial metadata
            _ca_od_ftl_key = (bbox["x"], bbox["y"], bbox["width"], bbox["height"])
            _ca_od_ftl = _ftl_by_bbox.get(_ca_od_ftl_key, {})
            _ca_od_st = _ca_od_ftl.get("spatial_transform", {})
            _ca_od_src = _ca_od_ftl.get("source_bbox_full_frame", {})
            print(
                f"[COORD AUDIT] [MOTION ROI]  full-frame bbox="
                f"x={bbox['x']} y={bbox['y']} w={bbox['width']} h={bbox['height']}"
            )
            print(f"[COORD AUDIT] [OD FTL]      source_bbox_full_frame={_ca_od_src}")
            print(f"[COORD AUDIT] [OD FTL]      spatial_transform={_ca_od_st}")

    # ------------------------------------------------------------------
    # [COORD AUDIT] obj_log / face_det_log — debug prints only, not drawing.
    # OD bboxes are in 640×640 letterbox-local space (FTL crop), not full-frame.
    # FD bboxes are in person-crop-local space.
    # Final overlay draws from rpm_output which holds correctly projected full-frame coords.
    # ------------------------------------------------------------------
    face_det_cursor = 0
    for k, obj_entry in enumerate(obj_log):
        roi_x = motion_bboxes[k]["x"] if k < len(motion_bboxes) else 0
        roi_y = motion_bboxes[k]["y"] if k < len(motion_bboxes) else 0

        for p_idx, pb in enumerate(obj_entry.get("output", {}).get("persons", [])):
            # [COORD AUDIT] OD output bbox (OD-local / 640×640 letterbox space)
            _ca_rpm_p = _rpm_persons_flat[_rpm_person_cursor] if _rpm_person_cursor < len(_rpm_persons_flat) else None
            _ca_rpm_pb = _ca_rpm_p.get("person_bbox", {}) if _ca_rpm_p else {}
            print(
                f"[COORD AUDIT] [OD OUTPUT]   raw person bbox (OD-local/letterbox)="
                f"x={pb['x']} y={pb['y']} w={pb['width']} h={pb['height']}"
            )
            print(
                f"[COORD AUDIT] [VIZ PERSON]  old (wrong) offset would draw at "
                f"x={roi_x + pb['x']} y={roi_y + pb['y']} "
                f"w={pb['width']} h={pb['height']}  (motion_offset=({roi_x},{roi_y})+od_bbox)"
            )
            print(f"[COORD AUDIT] [RPM PERSON]  person_bbox FULL_FRAME={_ca_rpm_pb}")
            # FD FTL lookup: FD was called with region_bbox = projected person bbox from RPM
            if _ca_rpm_pb:
                _ca_fd_ftl_key = (
                    _ca_rpm_pb.get("x", 0), _ca_rpm_pb.get("y", 0),
                    _ca_rpm_pb.get("width", 0), _ca_rpm_pb.get("height", 0),
                )
                _ca_fd_ftl = _ftl_by_bbox.get(_ca_fd_ftl_key, {})
                _ca_fd_st = _ca_fd_ftl.get("spatial_transform", {})
                _ca_fd_src = _ca_fd_ftl.get("source_bbox_full_frame", {})
                print(f"[COORD AUDIT] [FD FTL]      source_bbox_full_frame={_ca_fd_src}")
                print(f"[COORD AUDIT] [FD FTL]      spatial_transform={_ca_fd_st}")

            # One face_det_log entry per person crop — audit prints only
            if face_det_cursor < len(face_det_log):
                fd_entry = face_det_log[face_det_cursor]
                face_det_cursor += 1

                for det in fd_entry.get("output", {}).get("detections", []):
                    fb = det.get("face_bbox", {})
                    if fb:
                        # [COORD AUDIT] FD output bbox (person-crop-local space)
                        print(
                            f"[COORD AUDIT] [FD OUTPUT]   raw face bbox (FD person-crop-local)="
                            f"x={fb['x']} y={fb['y']} w={fb['width']} h={fb['height']}"
                        )
                        # rpm_output face_bbox for recognized faces under this person
                        if _ca_rpm_p:
                            for _ca_rf in _ca_rpm_p.get("recognized_faces", []):
                                _ca_rpm_fb = _ca_rf.get("face_bbox", {})
                                print(f"[COORD AUDIT] [RPM FACE]    face_bbox FULL_FRAME={_ca_rpm_fb}")

            if _ca_rpm_p is not None:
                _rpm_person_cursor += 1

    # ------------------------------------------------------------------
    # Person boxes (green) — full-frame from rpm_output
    # Face boxes (cyan)   — full-frame from rpm_output
    # Landmarks           — disabled (rpm_output does not expose projected landmarks)
    # ------------------------------------------------------------------
    for p_idx, person in enumerate(rpm_output.get("persons", [])):
        pb = person.get("person_bbox", {})
        if pb:
            px1, py1, px2, py2 = _clamp_box(
                pb["x"], pb["y"], pb["width"], pb["height"]
            )
            cv2.rectangle(annotated, (px1, py1), (px2, py2), _COLOR_PERSON, 2)
            cv2.putText(annotated, f"person {p_idx}", (px1 + 4, py1 + 18),
                        _FONT, _FONT_SCALE, _COLOR_PERSON, _THICKNESS + 1, cv2.LINE_AA)
            n_person += 1

        for face in person.get("recognized_faces", []):
            fb = face.get("face_bbox", {})
            if fb:
                fx1, fy1, fx2, fy2 = _clamp_box(
                    fb["x"], fb["y"], fb["width"], fb["height"]
                )
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), _COLOR_FACE, 2)
                n_face += 1

    # ------------------------------------------------------------------
    # Recognition labels (magenta)
    # rpm_output.persons[*].recognized_faces[*].face_bbox is full-frame.
    # Label is drawn above the face bbox with a dark backing rect for legibility.
    # ------------------------------------------------------------------
    for person in rpm_output.get("persons", []):
        person_pb = person.get("person_bbox", {})
        for face in person.get("recognized_faces", []):
            label = face.get("person_name") or "UNKNOWN"
            # Prefer face_bbox for label position; fall back to person_bbox
            anchor_box = face.get("face_bbox") or person_pb
            if not anchor_box:
                continue
            lx = max(4, anchor_box["x"])
            label_y = max(22, anchor_box["y"] - 8)
            (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _THICKNESS + 1)
            # Dark backing rect so label is legible over any background colour
            cv2.rectangle(annotated,
                          (lx - 2, label_y - th - 4),
                          (lx + tw + 2, label_y + 2),
                          (0, 0, 0), -1)
            cv2.putText(annotated, label, (lx, label_y),
                        _FONT, _FONT_SCALE, _COLOR_RECOGNITION,
                        _THICKNESS + 1, cv2.LINE_AA)
            n_label += 1

    # ------------------------------------------------------------------
    # HUD overlay
    # ------------------------------------------------------------------
    hud = [
        f"frame:   {rpm_output.get('frame_id', '')}",
        f"ts:      {rpm_output.get('timestamp_ms', 0)} ms",
        f"motion:{n_motion}  person:{n_person}  face:{n_face}  label:{n_label}",
    ]
    for i, line in enumerate(hud):
        cv2.putText(annotated, line, (5, 14 + i * 14),
                    _FONT, _FONT_SCALE, _COLOR_HUD, _THICKNESS, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Fallback: no annotations drawn at all
    # ------------------------------------------------------------------
    if n_motion == 0 and n_person == 0 and n_face == 0 and n_label == 0:
        text = "NO DETECTIONS DRAWN"
        (tw, th), _ = cv2.getTextSize(text, _FONT, 0.7, 2)
        cx = max(0, (w - tw) // 2)
        cy = max(0, (h + th) // 2)
        cv2.putText(annotated, text, (cx, cy),
                    _FONT, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return annotated


def write_frame_debug_folder(
    frame_dir: Path,
    bgr_frame: np.ndarray,
    ftl_log: list[dict],
    motion_log: list[dict],
    obj_log: list[dict],
    face_det_log: list[dict],
    face_rec_log: list[dict],
    rpm_output: RecognitionPipelineOutput,
    cold_start: bool,
) -> None:
    """Write all debug files for a single frame into frame_dir."""
    frame_dir.mkdir(parents=True, exist_ok=True)

    # 00 — original frame
    cv2.imwrite(str(frame_dir / "00_original.jpg"), bgr_frame)

    # 01 — FTL get_frame crops
    for entry in ftl_log:
        sel = entry["temporal_selector"].lower()
        rb = entry["region_bbox"]
        label = f"{rb['x']}-{rb['y']}-{rb['width']}x{rb['height']}"
        fname = f"01_ftl_{entry['call_index']}_{sel}_{label}.jpg"
        cv2.imwrite(str(frame_dir / fname), entry["cropped_bgr"])

    # 02 — Motion output
    for entry in motion_log:
        cv2.imwrite(str(frame_dir / f"02_motion_input_{entry['call_index']}.jpg"), entry["input_bgr"])
        cv2.imwrite(str(frame_dir / f"02_motion_previous_{entry['call_index']}.jpg"), entry["previous_input_bgr"])
        for image_name, image_data in entry.get("debug_images", {}).items():
            if image_data is None:
                continue
            if image_data.ndim == 2:
                image_bgr = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
            else:
                gray_or_rgb = image_data[:, :, 0] if image_data.shape[2] == 1 else image_data
                image_bgr = cv2.cvtColor(gray_or_rgb, cv2.COLOR_GRAY2BGR) if image_data.shape[2] == 1 else cv2.cvtColor(gray_or_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_dir / f"02_motion_{image_name}_{entry['call_index']}.jpg"), image_bgr)
    motion_outputs = [e["output"] for e in motion_log]
    write_json(frame_dir / "02_motion_output.json", motion_outputs if motion_outputs else [])
    motion_debug = [e.get("debug_info", {}) for e in motion_log]
    write_json(frame_dir / "02_motion_debug.json", motion_debug if motion_debug else [])

    # 03 — Object detection
    for entry in obj_log:
        cv2.imwrite(str(frame_dir / f"03_obj_det_input_{entry['call_index']}.jpg"), entry["input_bgr"])
    obj_outputs = [e["output"] for e in obj_log]
    write_json(frame_dir / "03_obj_det_output.json", obj_outputs if obj_outputs else [])

    # 04 — Face detection
    for entry in face_det_log:
        cv2.imwrite(str(frame_dir / f"04_face_det_input_{entry['call_index']}.jpg"), entry["input_bgr"])
    face_det_outputs = [e["output"] for e in face_det_log]
    write_json(frame_dir / "04_face_det_output.json", face_det_outputs if face_det_outputs else [])

    # 05 — Face recognition (may be absent if no faces detected)
    for entry in face_rec_log:
        cv2.imwrite(str(frame_dir / f"05_face_rec_input_{entry['call_index']}.jpg"), entry["input_bgr"])
    face_rec_outputs = [e["output"] for e in face_rec_log]
    write_json(frame_dir / "05_face_rec_output.json", face_rec_outputs if face_rec_outputs else [])

    # 06 — Final annotated frame
    # Debug: dump raw log data so bbox values are visible in the console
    _motion_outputs = [e.get("output", {}) for e in motion_log]
    _obj_outputs    = [e.get("output", {}) for e in obj_log]
    _rpm_persons    = rpm_output.get("persons", [])
    print(f"    [DBG] motion_log outputs : {json.dumps(_safe_dict(_motion_outputs))}")
    print(f"    [DBG] obj_log outputs    : {json.dumps(_safe_dict(_obj_outputs))}")
    print(f"    [DBG] rpm_output persons : {json.dumps(_safe_dict(_rpm_persons))}")

    annotated = draw_final_stage_overlay(
        bgr_frame, ftl_log, motion_log, obj_log, face_det_log, face_rec_log, rpm_output
    )
    out_path = frame_dir / "06_final_annotated.jpg"
    cv2.imwrite(str(out_path), annotated)

    _n_motion = sum(len(e.get("output", {}).get("bboxes", [])) for e in motion_log)
    _n_person = sum(len(e.get("output", {}).get("persons", [])) for e in obj_log)
    _n_face   = sum(len(e.get("output", {}).get("detections", [])) for e in face_det_log)
    _n_label  = sum(len(p.get("recognized_faces", [])) for p in _rpm_persons)
    print(
        f"    [ANNOTATE] frame={rpm_output.get('frame_id', '')}  "
        f"motion={_n_motion}  person={_n_person}  face={_n_face}  label={_n_label}  "
        f"path={out_path}"
    )

    # metadata.json
    meta = {
        "frame_id": rpm_output.get("frame_id", ""),
        "timestamp_ms": rpm_output.get("timestamp_ms", None),
        "cold_start": cold_start,
        "rpm_output": _safe_dict(rpm_output),
        "ftl_calls": len(ftl_log),
        "motion_calls": len(motion_log),
        "obj_det_calls": len(obj_log),
        "face_det_calls": len(face_det_log),
        "face_rec_calls": len(face_rec_log),
        "motion_debug": motion_debug,
    }
    write_json(frame_dir / "metadata.json", meta)


# ---------------------------------------------------------------------------
# Per-frame extraction
# ---------------------------------------------------------------------------

def process_frame_entry(
    entry: FrameEntry,
    video_name: str,
    components: DebugComponents,
    video_output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Process a single extracted frame through the fake RPM.

    Returns a summary dict for inclusion in the global JSON.
    """
    frame_dir_name = f"frame_{entry.frame_index:06d}"
    frame_dir = video_output_dir / frame_dir_name

    # Skip existing unless --overwrite
    if not overwrite and frame_dir.exists() and (frame_dir / "metadata.json").exists():
        print(f"    [SKIP] {frame_dir_name} (exists; use --overwrite to redo)")
        return {
            "frame_id": entry.frame_id,
            "frame_index": entry.frame_index,
            "skipped": True,
            "success": False,
        }

    # Load frame image
    bgr = cv2.imread(str(entry.image_path))
    if bgr is None:
        print(f"    [ERROR] Cannot read image: {entry.image_path}")
        return {
            "frame_id": entry.frame_id,
            "frame_index": entry.frame_index,
            "error": "Cannot read image",
            "success": False,
        }

    timestamp_ms = entry.timestamp_ms if entry.timestamp_ms is not None else 0
    packet = bgr_to_frame_packet(
        bgr,
        frame_id=entry.frame_id,
        camera_id=video_name,
        timestamp_ms=timestamp_ms,
    )

    # Flush stale logs before this frame
    components.logging_ftl.flush_log()
    components.logging_motion.flush_log()
    components.logging_obj.flush_log()
    components.logging_face_det.flush_log()
    components.logging_face_rec.flush_log()

    # Run RPM
    cold_start = False
    error_msg: str | None = None
    rpm_output: RecognitionPipelineOutput | None = None

    try:
        rpm_output = components.rpm.process_frame(packet)
    except PreviousFrameNotAvailableError:
        cold_start = True
        # Still collect the FTL log (ingest was called; get_frame for PREVIOUS raised)
    except Exception as exc:
        error_msg = str(exc)
        print(f"    [ERROR] RPM raised: {exc}")

    # Detect cold start even when RPM handled it internally (does not propagate the error)
    if not cold_start and components.logging_ftl.cold_start_triggered:
        cold_start = True

    # Collect logs (after call; LoggingFTL log filled for all get_frame calls including CURRENT)
    ftl_log = components.logging_ftl.flush_log()
    motion_log = components.logging_motion.flush_log()
    obj_log = components.logging_obj.flush_log()
    face_det_log = components.logging_face_det.flush_log()
    face_rec_log = components.logging_face_rec.flush_log()

    if rpm_output is None:
        rpm_output = {  # type: ignore[typeddict-item]
            "frame_id": entry.frame_id,
            "camera_id": video_name,
            "timestamp_ms": timestamp_ms,
            "persons": [],
        }

    write_frame_debug_folder(
        frame_dir=frame_dir,
        bgr_frame=bgr,
        ftl_log=ftl_log,
        motion_log=motion_log,
        obj_log=obj_log,
        face_det_log=face_det_log,
        face_rec_log=face_rec_log,
        rpm_output=rpm_output,
        cold_start=cold_start,
    )

    persons = len(rpm_output.get("persons", []))
    rpm_metrics_getter = getattr(components.rpm, "get_last_frame_metrics", None)
    rpm_metrics: dict[str, Any] = {}
    if callable(rpm_metrics_getter):
        try:
            value = rpm_metrics_getter()
            if isinstance(value, dict):
                rpm_metrics = value
        except Exception:
            rpm_metrics = {}

    status = "cold_start" if cold_start else ("error" if error_msg else "ok")
    print(f"    [{status.upper():10s}] {frame_dir_name}  persons={persons}  "
          f"ftl_calls={len(ftl_log)}  motion={len(motion_log)}  "
          f"obj={len(obj_log)}  face_det={len(face_det_log)}")

    return {
        "frame_id": entry.frame_id,
        "frame_index": entry.frame_index,
        "cold_start": cold_start,
        "persons_in_output": persons,
        "ftl_calls": len(ftl_log),
        "motion_calls": len(motion_log),
        "obj_det_calls": len(obj_log),
        "face_det_calls": len(face_det_log),
        "face_rec_calls": len(face_rec_log),
        "rpm_metrics": _safe_dict(rpm_metrics),
        "error": error_msg,
        "success": error_msg is None,
        "debug_dir": str(frame_dir),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Validate and load YAML config
    # ------------------------------------------------------------------
    if args.real_object_detection and args.rpm_config is None:
        print("[ERROR] --real-object-detection requires --rpm-config.")
        sys.exit(1)

    if args.real_face_detection and args.rpm_config is None:
        print("[ERROR] --real-face-detection requires --rpm-config.")
        sys.exit(1)

    if args.real_face_recognition and args.rpm_config is None:
        print("[ERROR] --real-face-recognition requires --rpm-config.")
        sys.exit(1)

    if args.real_motion_detection and args.rpm_config is None:
        print("[ERROR] --real-motion-detection requires --rpm-config.")
        sys.exit(1)

    rpm_config: dict | None = None
    person_json: Path = args.person_json.resolve()

    if args.real_object_detection or args.real_face_detection or args.real_face_recognition or args.real_motion_detection:
        rpm_config_path = args.rpm_config.resolve()
        print(f"[INFO] Loading RPM config: {rpm_config_path}")
        rpm_config = load_yaml_config(rpm_config_path)
        # Override person_json from YAML when running in config-driven mode
        _pd_path_str = rpm_config.get("recognition_pipeline", {}).get("person_directory_json_path")
        if _pd_path_str:
            person_json = resolve_config_path(_pd_path_str, _PROJECT_ROOT)
            print(f"[INFO] PersonDirectory  : {person_json}  (from YAML)")

    input_dir: Path = args.input_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    print(f"[INFO] Input directory : {input_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Overwrite       : {args.overwrite}")
    print(f"[INFO] Real OD         : {args.real_object_detection}")
    print(f"[INFO] Real FD         : {args.real_face_detection}")
    print(f"[INFO] Real FR         : {args.real_face_recognition}")
    print(f"[INFO] Real MD         : {args.real_motion_detection}")

    frame_sets = load_frame_sets(input_dir)
    if not frame_sets:
        print("[INFO] No frame sets found. Nothing to do.")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Process each video's frames
    # ------------------------------------------------------------------
    global_per_video: list[dict[str, Any]] = []
    total_frames_processed = 0
    total_frames_skipped = 0
    total_frames_failed = 0

    for frame_set in frame_sets:
        video_name = frame_set.video_name
        print(f"\n{'=' * 60}")
        print(f"[VIDEO] {video_name}  "
              f"({len(frame_set.frames)} frames, "
              f"{frame_set.frame_width}×{frame_set.frame_height})")

        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        components = build_debug_components(
            frame_set.frame_width,
            frame_set.frame_height,
            person_json,
            rpm_config=rpm_config,
            real_object_detection=args.real_object_detection,
            real_face_detection=args.real_face_detection,
            real_face_recognition=args.real_face_recognition,
            real_motion_detection=args.real_motion_detection,
        )

        per_frame_results: list[dict[str, Any]] = []
        for entry in frame_set.frames:
            result = process_frame_entry(
                entry=entry,
                video_name=video_name,
                components=components,
                video_output_dir=video_output_dir,
                overwrite=args.overwrite,
            )
            per_frame_results.append(result)
            if result.get("skipped"):
                total_frames_skipped += 1
            elif result.get("success"):
                total_frames_processed += 1
            else:
                total_frames_failed += 1

        processed = sum(1 for r in per_frame_results if r.get("success"))
        skipped = sum(1 for r in per_frame_results if r.get("skipped"))
        failed = sum(1 for r in per_frame_results if not r.get("success") and not r.get("skipped"))

        print(f"  → {processed} processed, {skipped} skipped, {failed} failed")

        global_per_video.append({
            "video_name": video_name,
            "total_frames": len(frame_set.frames),
            "frames_processed": processed,
            "frames_skipped": skipped,
            "frames_failed": failed,
            "frame_width": frame_set.frame_width,
            "frame_height": frame_set.frame_height,
            "frames": per_frame_results,
        })

    # ------------------------------------------------------------------
    # Global summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"[DONE] Frame sets      : {len(frame_sets)}")
    print(f"[DONE] Frames processed: {total_frames_processed}")
    print(f"[DONE] Frames skipped  : {total_frames_skipped}")
    print(f"[DONE] Frames failed   : {total_frames_failed}")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_frame_sets": len(frame_sets),
        "total_frames_processed": total_frames_processed,
        "total_frames_skipped": total_frames_skipped,
        "total_frames_failed": total_frames_failed,
        "per_video": global_per_video,
    }
    summary_path = output_dir / "summary_all_frames.json"
    write_json(summary_path, summary)
    print(f"[DONE] Summary written → {summary_path}")


if __name__ == "__main__":
    main()
