"""
Run RecognitionPipelineManager sequentially on extracted frames.

Usage example:
python scripts/run_rpm_on_frames.py ^
  --frames-dir "tests/recognition_pipeline_manager/assets/extracted_video_frames/empty_scene" ^
  --output-dir "debug_outputs/empty_scene" ^
  --rpm-config "config/image_processing_service/local_debug.yaml" ^
  --real-object-detection ^
  --real-face-detection ^
  --real-face-recognition
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_TESTS_DIR = _PROJECT_ROOT / "tests"

for _p in [str(_SRC_DIR), str(_TESTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import cv2
except ImportError:
    print("[FATAL] opencv-python is not installed. Run: pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("[FATAL] numpy is not installed. Run: pip install numpy")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("[FATAL] pyyaml is not installed. Run: pip install pyyaml")
    sys.exit(1)

from image_processing.frame_transformation_layer import FrameTransformationLayer
from image_processing.frame_transformation_layer.contracts import FramePacket
from image_processing.recognition_pipeline_manager import (
    RecognitionPipelineManager,
    RecognitionPipelineOutput,
)
from image_processing.shared.contracts import FaceLandmarks, PipelineStageInputContract, Point

from recognition_pipeline_manager.fakes import (  # type: ignore[import]
    FakeFaceDetection,
    FakeFaceRecognition,
    FakeMotionDetection,
    FakeObjectDetection,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_THICKNESS = 1
_COLOR_MOTION = (0, 255, 255)       # yellow (BGR)
_COLOR_PERSON = (0, 220, 0)         # green (BGR)
_COLOR_FACE = (255, 255, 0)         # cyan (BGR)
_COLOR_LANDMARK = (0, 165, 255)     # orange (BGR)
_COLOR_RECOGNITION = (255, 0, 200)  # magenta (BGR)
_COLOR_HUD = (200, 200, 200)        # light-gray (BGR)


@dataclass
class RuntimeComponents:
    rpm: RecognitionPipelineManager
    motion: "LoggingMotionDetection"
    obj_det: "LoggingObjectDetection"
    face_det: "LoggingFaceDetection"
    face_rec: "LoggingFaceRecognition"


class LoggingMotionDetection:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._last_calls: list[dict[str, Any]] = []

    def detect(self, input: Any) -> Any:
        result = self._delegate.detect(input)
        self._last_calls.append(
            {
                "input": _safe_dict(input),
                "output": _safe_dict(result),
            }
        )
        return result

    def get_input_contract(self) -> PipelineStageInputContract:
        return self._delegate.get_input_contract()

    def get_backend_info(self) -> dict[str, Any]:
        """Return backend diagnostics from the wrapped motion module."""
        getter = getattr(self._delegate, "get_backend_info", None)
        info = getter() if callable(getter) else {}
        return info if isinstance(info, dict) else {}

    def flush_calls(self) -> list[dict[str, Any]]:
        items = list(self._last_calls)
        self._last_calls.clear()
        return items


class LoggingObjectDetection:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._last_calls: list[dict[str, Any]] = []

    def detect(self, input: Any) -> Any:
        result = self._delegate.detect(input)
        self._last_calls.append(
            {
                "input": _safe_dict(input),
                "output": _safe_dict(result),
            }
        )
        return result

    def get_input_contract(self) -> PipelineStageInputContract:
        return self._delegate.get_input_contract()

    def get_backend_info(self) -> dict[str, Any]:
        """Return backend diagnostics from the wrapped object detector."""
        getter = getattr(self._delegate, "get_backend_info", None)
        info = getter() if callable(getter) else {}
        return info if isinstance(info, dict) else {}

    def flush_calls(self) -> list[dict[str, Any]]:
        items = list(self._last_calls)
        self._last_calls.clear()
        return items


class LoggingFaceDetection:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._last_calls: list[dict[str, Any]] = []

    def detect_faces(self, input: Any) -> Any:
        result = self._delegate.detect_faces(input)
        self._last_calls.append(
            {
                "input": _safe_dict(input),
                "output": _safe_dict(result),
            }
        )
        return result

    def get_input_contract(self) -> PipelineStageInputContract:
        return self._delegate.get_input_contract()

    def get_backend_info(self) -> dict[str, Any]:
        """Return backend diagnostics from the wrapped face detector."""
        getter = getattr(self._delegate, "get_backend_info", None)
        info = getter() if callable(getter) else {}
        return info if isinstance(info, dict) else {}

    def flush_calls(self) -> list[dict[str, Any]]:
        items = list(self._last_calls)
        self._last_calls.clear()
        return items


class LoggingFaceRecognition:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._last_calls: list[dict[str, Any]] = []

    def recognize(self, input: Any) -> Any:
        result = self._delegate.recognize(input)
        self._last_calls.append(
            {
                "input": _safe_dict(input),
                "output": _safe_dict(result),
            }
        )
        return result

    def get_input_contract(self) -> PipelineStageInputContract:
        return self._delegate.get_input_contract()

    def get_backend_info(self) -> dict[str, Any]:
        """Return backend diagnostics from the wrapped face recognizer."""
        getter = getattr(self._delegate, "get_backend_info", None)
        info = getter() if callable(getter) else {}
        return info if isinstance(info, dict) else {}

    def flush_calls(self) -> list[dict[str, Any]]:
        items = list(self._last_calls)
        self._last_calls.clear()
        return items


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RecognitionPipelineManager sequentially on extracted image frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory containing extracted .jpg/.png frames.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where overlays/json/logs will be written.")
    parser.add_argument("--rpm-config", type=Path, required=True, help="Path to root RPM YAML config.")
    parser.add_argument("--camera-id", type=str, default="camera_01", help="Camera ID to use in generated FramePacket records.")

    parser.add_argument("--real-motion-detection", action="store_true", help="Use real MotionDetectionManager.")
    parser.add_argument("--real-object-detection", action="store_true", help="Use real ObjectDetectionModule.")
    parser.add_argument("--real-face-detection", action="store_true", help="Use real FaceDetectionModule.")
    parser.add_argument("--real-face-recognition", action="store_true", help="Use real FaceRecognitionModule.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output-dir before running this sequence.")
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_config_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root / p


def _safe_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _safe_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_dict(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return {
            "_ndarray_shape": list(obj.shape),
            "_dtype": str(obj.dtype),
        }
    return obj


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def bgr_to_frame_packet(bgr_frame: np.ndarray, frame_id: str, camera_id: str, timestamp_ms: int) -> FramePacket:
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


def _build_stub_detections(frame_width: int, frame_height: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], FaceLandmarks]:
    w = max(1, frame_width)
    h = max(1, frame_height)

    mx = max(4, int(w * 0.03))
    my = max(4, int(h * 0.03))
    mw = max(4, w - (2 * mx))
    mh = max(4, h - (2 * my))
    motion_bbox = {"x": mx, "y": my, "width": mw, "height": mh}

    px = max(2, int(mw * 0.25))
    py = max(2, int(mh * 0.15))
    pw = max(4, int(mw * 0.50))
    ph = max(4, int(mh * 0.75))
    person_bbox_local = {"x": px, "y": py, "width": pw, "height": ph}

    fx = max(2, int(pw * 0.20))
    fy = max(2, int(ph * 0.05))
    fw = max(4, int(pw * 0.60))
    fh = max(4, int(ph * 0.50))
    face_bbox_local = {"x": fx, "y": fy, "width": fw, "height": fh}

    cx = fx + fw // 2
    cy = fy + fh // 2
    landmarks = FaceLandmarks(
        left_eye=Point(x=cx - fw // 6, y=cy - fh // 8),
        right_eye=Point(x=cx + fw // 6, y=cy - fh // 8),
        nose=Point(x=cx, y=cy),
        mouth_left=Point(x=cx - fw // 8, y=cy + fh // 6),
        mouth_right=Point(x=cx + fw // 8, y=cy + fh // 6),
    )
    return motion_bbox, person_bbox_local, face_bbox_local, landmarks


def build_runtime_components(
    rpm_config: dict[str, Any],
    person_json: Path,
    frame_width: int,
    frame_height: int,
    real_motion_detection: bool,
    real_object_detection: bool,
    real_face_detection: bool,
    real_face_recognition: bool,
) -> RuntimeComponents:
    motion_bbox, person_bbox_local, face_bbox_local, fake_landmarks = _build_stub_detections(frame_width, frame_height)

    md_mode = "STUB"
    if real_motion_detection:
        from image_processing.motion_detection import MotionDetectionConfig, MotionDetectionManager

        md_config_rel = rpm_config["models"]["motion_detection_config"]
        md_config_path = resolve_config_path(md_config_rel, _PROJECT_ROOT)
        md_yaml = load_yaml_config(md_config_path)
        md_cfg = MotionDetectionConfig(
            motion_threshold=int(md_yaml.get("motion_threshold", 50)),
            motion_fraction_threshold=float(md_yaml.get("motion_fraction_threshold", 0.05)),
            min_bbox_area=int(md_yaml.get("min_bbox_area", 5000)),
            blur_kernel_size=int(md_yaml.get("blur_kernel_size", 0)),
            morph_open_iterations=int(md_yaml.get("morph_open_iterations", 0)),
            morph_close_iterations=int(md_yaml.get("morph_close_iterations", 0)),
            dilation_iterations=int(md_yaml.get("dilation_iterations", 0)),
            min_aspect_ratio=float(md_yaml.get("min_aspect_ratio", 0.0)),
            max_aspect_ratio=float(md_yaml.get("max_aspect_ratio", 1000.0)),
            enable_global_motion_compensation=bool(md_yaml.get("enable_global_motion_compensation", False)),
            global_motion_method=str(md_yaml.get("global_motion_method", "gftt_lk_affine")),
            max_features=int(md_yaml.get("max_features", 300)),
            min_feature_matches=int(md_yaml.get("min_feature_matches", 20)),
            max_transform_shift=float(md_yaml.get("max_transform_shift", 25.0)),
            global_motion_changed_ratio_threshold=float(md_yaml.get("global_motion_changed_ratio_threshold", 0.30)),
            fallback_on_alignment_failure=bool(md_yaml.get("fallback_on_alignment_failure", True)),
            enable_bbox_merging=bool(md_yaml.get("enable_bbox_merging", False)),
            bbox_merge_iou_threshold=float(md_yaml.get("bbox_merge_iou_threshold", 0.35)),
            bbox_merge_distance_threshold=float(md_yaml.get("bbox_merge_distance_threshold", 12.0)),
            enable_temporal_persistence=bool(md_yaml.get("enable_temporal_persistence", False)),
            min_persistence_frames=int(md_yaml.get("min_persistence_frames", 2)),
            persistence_iou_threshold=float(md_yaml.get("persistence_iou_threshold", 0.35)),
            max_history_frames=int(md_yaml.get("max_history_frames", 8)),
        )
        motion_module: Any = MotionDetectionManager(config=md_cfg)
        md_mode = f"REAL ({md_config_path})"
    else:
        from image_processing.motion_detection.module import MotionResult

        motion_module = FakeMotionDetection(result=MotionResult(detected=True, bboxes=[motion_bbox]))  # type: ignore[call-arg]

    od_mode = "STUB"
    if real_object_detection:
        from image_processing.object_detection.module import ObjectDetectionModule, PersonDetectionConfig

        od_config_rel = rpm_config["models"]["object_detection_config"]
        od_config_path = resolve_config_path(od_config_rel, _PROJECT_ROOT)
        od_yaml = load_yaml_config(od_config_path)
        model_path = str(resolve_config_path(od_yaml["model_path"], _PROJECT_ROOT))
        od_cfg = PersonDetectionConfig(
            model_path=model_path,
            person_confidence_threshold=float(od_yaml.get("confidence_threshold", 0.35)),
            nms_iou_threshold=float(od_yaml.get("nms_threshold", 0.35)),
        )
        obj_det_module: Any = ObjectDetectionModule(config=od_cfg)
        od_mode = f"REAL ({od_config_path})"
    else:
        from image_processing.object_detection.module import PersonDetectionResult

        obj_det_module = FakeObjectDetection(
            result=PersonDetectionResult(
                frame_id="",
                person_detected=True,
                persons=[person_bbox_local],
            )
        )

    fd_mode = "STUB"
    if real_face_detection:
        from image_processing.face_detection import FaceDetectionConfig, FaceDetectionModule, SCRFDFaceDetector

        fd_config_rel = rpm_config["models"]["face_detection_config"]
        fd_config_path = resolve_config_path(fd_config_rel, _PROJECT_ROOT)
        fd_yaml = load_yaml_config(fd_config_path)
        fd_cfg = FaceDetectionConfig(confidence_threshold=float(fd_yaml.get("confidence_threshold", 0.5)))
        scrfd_model_dir = _PROJECT_ROOT / "models" / "face_detection"
        scrfd = SCRFDFaceDetector(model_dir=scrfd_model_dir)
        face_det_module: Any = FaceDetectionModule(config=fd_cfg, detector_engine=scrfd)
        fd_mode = f"REAL ({fd_config_path})"
    else:
        from image_processing.face_detection.module import DetectedFace, FaceDetectionOutput

        fake_face_output = FaceDetectionOutput(
            frame_id="",
            camera_id="",
            timestamp_ms=0,
            detections=[DetectedFace(face_bbox=face_bbox_local, landmarks=fake_landmarks)],
        )
        face_det_module = FakeFaceDetection(result=fake_face_output)

    fr_mode = "STUB"
    if real_face_recognition:
        from image_processing.face_gallery_loader import FaceGalleryLoaderModule
        from image_processing.face_recognition import (
            ArcFaceEmbeddingEngine,
            EnrolledIdentity,
            FaceRecognitionConfig,
            FaceRecognitionModule,
        )

        fr_config_rel = rpm_config["models"]["face_recognition_config"]
        fr_config_path = resolve_config_path(fr_config_rel, _PROJECT_ROOT)
        fr_yaml = load_yaml_config(fr_config_path)
        fr_threshold = float(fr_yaml.get("recognition_threshold", 0.4))

        gallery_root_str = rpm_config["face_gallery"]["gallery_root_path"]
        gallery_root_path = resolve_config_path(gallery_root_str, _PROJECT_ROOT)

        gallery_loader = FaceGalleryLoaderModule()
        gallery_loader.load_gallery(str(gallery_root_path))
        all_embeddings = gallery_loader.get_all_embeddings()

        enrolled: list[EnrolledIdentity] = [
            {"person_id": e["person_id"], "embedding": e["embedding"]}
            for e in all_embeddings
        ]
        fr_cfg = FaceRecognitionConfig(recognition_threshold=fr_threshold)
        arcface = ArcFaceEmbeddingEngine()
        face_rec_module: Any = FaceRecognitionModule(
            config=fr_cfg,
            embedding_engine=arcface,
            gallery_entries=enrolled,
        )
        fr_mode = f"REAL ({fr_config_path})"
    else:
        demo_person_id = "1f007fe2-6eaf-5148-a240-a449664cb6eb"
        face_rec_module = FakeFaceRecognition(
            result={
                "frame_id": "",
                "camera_id": "",
                "timestamp_ms": 0,
                "person_found": True,
                "person_id": demo_person_id,
            }
        )

    from image_processing.person_directory import PersonDirectory, PersonDirectoryConfig

    person_dir = PersonDirectory(
        config=PersonDirectoryConfig(
            json_file_path=str(person_json),
            is_optional=True,
        )
    )
    person_dir.load()

    log_motion = LoggingMotionDetection(motion_module)
    log_obj = LoggingObjectDetection(obj_det_module)
    log_face_det = LoggingFaceDetection(face_det_module)
    log_face_rec = LoggingFaceRecognition(face_rec_module)

    # IMPORTANT: real FTL only, no capturing/fake FTL wrappers.
    real_ftl = FrameTransformationLayer()

    pipeline_cfg = rpm_config.get("recognition_pipeline", {})
    max_motion_rois = int(pipeline_cfg.get("max_motion_rois_per_frame", 8))
    max_person_rois = int(pipeline_cfg.get("max_person_rois_per_frame", 16))
    max_face_rois = int(pipeline_cfg.get("max_face_rois_per_frame", 32))

    rpm = RecognitionPipelineManager(
        ftl=real_ftl,
        motion=log_motion,
        object_det=log_obj,
        face_det=log_face_det,
        face_rec=log_face_rec,
        person_dir=person_dir,
        max_motion_rois_per_frame=max_motion_rois,
        max_person_rois_per_frame=max_person_rois,
        max_face_rois_per_frame=max_face_rois,
    )

    print("[MODULE MODE]")
    print(f"  Motion Detection  : {md_mode}  [{type(motion_module).__name__}]")
    print(f"  Object Detection  : {od_mode}  [{type(obj_det_module).__name__}]")
    print(f"  Face Detection    : {fd_mode}  [{type(face_det_module).__name__}]")
    print(f"  Face Recognition  : {fr_mode}  [{type(face_rec_module).__name__}]")
    print("  PersonDirectory   : REAL")
    print("  FTL               : REAL (FrameTransformationLayer)")
    print(f"  ROI Limits        : motion={max_motion_rois} person={max_person_rois} face={max_face_rois}")

    return RuntimeComponents(
        rpm=rpm,
        motion=log_motion,
        obj_det=log_obj,
        face_det=log_face_det,
        face_rec=log_face_rec,
    )


def _collect_frame_files(frames_dir: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def _count_persons_and_faces(rpm_output: RecognitionPipelineOutput) -> tuple[int, int]:
    persons = rpm_output.get("persons", [])
    person_count = len(persons)
    face_count = 0
    for person in persons:
        face_count += len(person.get("recognized_faces", []))
    return person_count, face_count


def _map_landmarks_to_full_frame(
    face_det_calls: list[dict[str, Any]],
    rpm_output: RecognitionPipelineOutput,
) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    rpm_persons = rpm_output.get("persons", [])

    for person_idx, person in enumerate(rpm_persons):
        if person_idx >= len(face_det_calls):
            break

        fd_output = face_det_calls[person_idx].get("output", {})
        detections = fd_output.get("detections", [])
        recognized = person.get("recognized_faces", [])

        for face_idx, det in enumerate(detections):
            if face_idx >= len(recognized):
                continue

            local_bbox = det.get("face_bbox") or {}
            full_bbox = recognized[face_idx].get("face_bbox") or {}
            landmarks = det.get("landmarks") or {}

            if not local_bbox or not full_bbox or not landmarks:
                continue

            lw = max(1, int(local_bbox.get("width", 0)))
            lh = max(1, int(local_bbox.get("height", 0)))
            fw = max(1, int(full_bbox.get("width", 0)))
            fh = max(1, int(full_bbox.get("height", 0)))

            for name in ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]:
                p = landmarks.get(name)
                if not isinstance(p, dict):
                    continue

                px = int(p.get("x", 0))
                py = int(p.get("y", 0))
                rel_x = (px - int(local_bbox.get("x", 0))) / float(lw)
                rel_y = (py - int(local_bbox.get("y", 0))) / float(lh)

                full_x = int(round(int(full_bbox.get("x", 0)) + rel_x * fw))
                full_y = int(round(int(full_bbox.get("y", 0)) + rel_y * fh))
                pts.append((full_x, full_y))

    return pts


def _draw_overlay(
    bgr_frame: np.ndarray,
    rpm_output: RecognitionPipelineOutput,
    motion_calls: list[dict[str, Any]],
    face_det_calls: list[dict[str, Any]],
    camera_id: str,
) -> np.ndarray:
    img = bgr_frame.copy()
    h, w = img.shape[:2]

    def clamp_box(x: int, y: int, bw: int, bh: int) -> tuple[int, int, int, int]:
        x1 = max(0, min(x, w - 1))
        y1 = max(0, min(y, h - 1))
        x2 = max(x1 + 1, min(x + bw, w))
        y2 = max(y1 + 1, min(y + bh, h))
        return x1, y1, x2, y2

    motion_count = 0
    for call in motion_calls:
        bboxes = call.get("output", {}).get("bboxes", [])
        for bbox in bboxes:
            x1, y1, x2, y2 = clamp_box(
                int(bbox.get("x", 0)),
                int(bbox.get("y", 0)),
                int(bbox.get("width", 0)),
                int(bbox.get("height", 0)),
            )
            cv2.rectangle(img, (x1, y1), (x2, y2), _COLOR_MOTION, 2)
            motion_count += 1

    person_count = 0
    face_count = 0
    label_count = 0
    for person in rpm_output.get("persons", []):
        pb = person.get("person_bbox", {})
        if pb:
            x1, y1, x2, y2 = clamp_box(
                int(pb.get("x", 0)),
                int(pb.get("y", 0)),
                int(pb.get("width", 0)),
                int(pb.get("height", 0)),
            )
            cv2.rectangle(img, (x1, y1), (x2, y2), _COLOR_PERSON, 2)
            person_count += 1

        for rf in person.get("recognized_faces", []):
            fb = rf.get("face_bbox", {})
            if fb:
                x1, y1, x2, y2 = clamp_box(
                    int(fb.get("x", 0)),
                    int(fb.get("y", 0)),
                    int(fb.get("width", 0)),
                    int(fb.get("height", 0)),
                )
                cv2.rectangle(img, (x1, y1), (x2, y2), _COLOR_FACE, 2)
                face_count += 1

            label = rf.get("person_name")
            if isinstance(label, str) and label.strip():
                anchor = rf.get("face_bbox") or pb
                if anchor:
                    lx = max(4, int(anchor.get("x", 0)))
                    ly = max(20, int(anchor.get("y", 0)) - 6)
                    cv2.putText(
                        img,
                        label,
                        (lx, ly),
                        _FONT,
                        _FONT_SCALE,
                        _COLOR_RECOGNITION,
                        _THICKNESS + 1,
                        cv2.LINE_AA,
                    )
                    label_count += 1

    for px, py in _map_landmarks_to_full_frame(face_det_calls, rpm_output):
        px = max(0, min(px, w - 1))
        py = max(0, min(py, h - 1))
        cv2.circle(img, (px, py), 2, _COLOR_LANDMARK, -1)

    hud_lines = [
        f"frame_id: {rpm_output.get('frame_id', '')}",
        f"timestamp_ms: {rpm_output.get('timestamp_ms', 0)}",
        f"camera_id: {camera_id}",
        f"motion:{motion_count} persons:{person_count} faces:{face_count} labels:{label_count}",
    ]
    for idx, line in enumerate(hud_lines):
        cv2.putText(
            img,
            line,
            (6, 16 + idx * 15),
            _FONT,
            _FONT_SCALE,
            _COLOR_HUD,
            _THICKNESS,
            cv2.LINE_AA,
        )

    return img


def run_sequence(
    frames_dir: Path | str,
    output_dir: Path | str,
    rpm_config: Path | str | dict[str, Any],
    camera_id: str,
    real_motion_detection: bool,
    real_object_detection: bool,
    real_face_detection: bool,
    real_face_recognition: bool,
    overwrite: bool = False,
) -> dict[str, Any]:
    frames_dir_path = Path(frames_dir).resolve()
    output_dir_path = Path(output_dir).resolve()

    if not frames_dir_path.is_dir():
        raise ValueError(f"frames directory not found: {frames_dir_path}")

    if isinstance(rpm_config, dict):
        rpm_config_data: dict[str, Any] = rpm_config
    else:
        rpm_config_path = Path(rpm_config).resolve()
        if not rpm_config_path.exists():
            raise FileNotFoundError(f"rpm-config file not found: {rpm_config_path}")
        rpm_config_data = load_yaml_config(rpm_config_path)

    person_json = _PROJECT_ROOT / "data" / "person_directory.json"
    pd_path_str = rpm_config_data.get("recognition_pipeline", {}).get("person_directory_json_path")
    if pd_path_str:
        person_json = resolve_config_path(str(pd_path_str), _PROJECT_ROOT)

    frame_files = _collect_frame_files(frames_dir_path)
    if not frame_files:
        raise ValueError(f"no .jpg/.jpeg/.png frames found in: {frames_dir_path}")

    sample = cv2.imread(str(frame_files[0]))
    if sample is None:
        raise ValueError(f"cannot read first frame: {frame_files[0]}")
    frame_h, frame_w = sample.shape[:2]

    if overwrite and output_dir_path.exists():
        shutil.rmtree(output_dir_path)

    overlays_dir = output_dir_path / "overlays"
    json_dir = output_dir_path / "json"
    logs_dir = output_dir_path / "logs"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    components = build_runtime_components(
        rpm_config=rpm_config_data,
        person_json=person_json,
        frame_width=frame_w,
        frame_height=frame_h,
        real_motion_detection=bool(real_motion_detection),
        real_object_detection=bool(real_object_detection),
        real_face_detection=bool(real_face_detection),
        real_face_recognition=bool(real_face_recognition),
    )

    total = len(frame_files)
    processed = 0
    failures = 0
    total_persons = 0
    total_recognized_faces = 0
    per_frame_summary: list[dict[str, Any]] = []

    print(f"[RUN] frames_dir={frames_dir_path}")
    print(f"[RUN] output_dir={output_dir_path}")
    print(f"[RUN] total_frames={total}")

    for idx, frame_path in enumerate(frame_files, start=1):
        frame_id = f"frame_{idx:06d}"
        timestamp_ms = (idx - 1) * 33

        components.motion.flush_calls()
        components.obj_det.flush_calls()
        components.face_det.flush_calls()
        components.face_rec.flush_calls()

        frame_start = time.perf_counter()
        success = True
        error_text: str | None = None
        rpm_output: RecognitionPipelineOutput = {
            "frame_id": frame_id,
            "camera_id": camera_id,
            "timestamp_ms": timestamp_ms,
            "persons": [],
        }

        try:
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                raise ValueError(f"cv2.imread returned None for {frame_path}")

            packet = bgr_to_frame_packet(
                bgr_frame=bgr,
                frame_id=frame_id,
                camera_id=camera_id,
                timestamp_ms=timestamp_ms,
            )

            process_start = time.perf_counter()
            rpm_output = components.rpm.process_frame(packet)
            process_ms = (time.perf_counter() - process_start) * 1000.0

            motion_calls = components.motion.flush_calls()
            obj_calls = components.obj_det.flush_calls()
            face_det_calls = components.face_det.flush_calls()
            face_rec_calls = components.face_rec.flush_calls()

            overlay = _draw_overlay(
                bgr_frame=bgr,
                rpm_output=rpm_output,
                motion_calls=motion_calls,
                face_det_calls=face_det_calls,
                camera_id=camera_id,
            )
            overlay_path = overlays_dir / f"{frame_id}.jpg"
            cv2.imwrite(str(overlay_path), overlay)

            metrics_getter = getattr(components.rpm, "get_last_frame_metrics", None)
            rpm_metrics: dict[str, Any] = {}
            if callable(metrics_getter):
                try:
                    maybe = metrics_getter()
                    if isinstance(maybe, dict):
                        rpm_metrics = maybe
                except Exception:
                    rpm_metrics = {}

            frame_total_ms = (time.perf_counter() - frame_start) * 1000.0
            frame_json = {
                "source_frame_path": str(frame_path),
                "frame_id": frame_id,
                "camera_id": camera_id,
                "timestamp_ms": timestamp_ms,
                "success": True,
                "error": None,
                "rpm_output": _safe_dict(rpm_output),
                "raw_module_outputs": {
                    "motion": [c.get("output", {}) for c in motion_calls],
                    "object_detection": [c.get("output", {}) for c in obj_calls],
                    "face_detection": [c.get("output", {}) for c in face_det_calls],
                    "face_recognition": [c.get("output", {}) for c in face_rec_calls],
                },
                "timing": {
                    "process_frame_ms": round(process_ms, 3),
                    "frame_total_ms": round(frame_total_ms, 3),
                },
                "debug": {
                    "rpm_metrics": _safe_dict(rpm_metrics),
                    "module_call_counts": {
                        "motion": len(motion_calls),
                        "object_detection": len(obj_calls),
                        "face_detection": len(face_det_calls),
                        "face_recognition": len(face_rec_calls),
                    },
                },
                "artifacts": {
                    "overlay_path": str(overlay_path),
                },
            }
            write_json(json_dir / f"{frame_id}.json", frame_json)

            person_count, face_count = _count_persons_and_faces(rpm_output)
            total_persons += person_count
            total_recognized_faces += face_count
            processed += 1

            print(
                f"[FRAME {idx}/{total}] SUCCESS persons={person_count} faces={face_count} "
                f"file={frame_path.name}"
            )

            per_frame_summary.append(
                {
                    "frame_index": idx,
                    "frame_id": frame_id,
                    "success": True,
                    "persons": person_count,
                    "recognized_faces": face_count,
                    "json_path": str(json_dir / f"{frame_id}.json"),
                    "overlay_path": str(overlay_path),
                }
            )

        except Exception as exc:
            success = False
            error_text = str(exc)
            failures += 1

            motion_calls = components.motion.flush_calls()
            obj_calls = components.obj_det.flush_calls()
            face_det_calls = components.face_det.flush_calls()
            face_rec_calls = components.face_rec.flush_calls()

            frame_total_ms = (time.perf_counter() - frame_start) * 1000.0
            error_json = {
                "source_frame_path": str(frame_path),
                "frame_id": frame_id,
                "camera_id": camera_id,
                "timestamp_ms": timestamp_ms,
                "success": False,
                "error": error_text,
                "rpm_output": _safe_dict(rpm_output),
                "raw_module_outputs": {
                    "motion": [c.get("output", {}) for c in motion_calls],
                    "object_detection": [c.get("output", {}) for c in obj_calls],
                    "face_detection": [c.get("output", {}) for c in face_det_calls],
                    "face_recognition": [c.get("output", {}) for c in face_rec_calls],
                },
                "timing": {
                    "frame_total_ms": round(frame_total_ms, 3),
                },
                "debug": {
                    "module_call_counts": {
                        "motion": len(motion_calls),
                        "object_detection": len(obj_calls),
                        "face_detection": len(face_det_calls),
                        "face_recognition": len(face_rec_calls),
                    },
                },
            }
            write_json(json_dir / f"{frame_id}.json", error_json)

            print(f"[FRAME {idx}/{total}] FAILURE persons=0 faces=0 file={frame_path.name} error={error_text}")

            per_frame_summary.append(
                {
                    "frame_index": idx,
                    "frame_id": frame_id,
                    "success": False,
                    "error": error_text,
                    "json_path": str(json_dir / f"{frame_id}.json"),
                }
            )

        if not success:
            continue

    summary = {
        "frames_dir": str(frames_dir_path),
        "output_dir": str(output_dir_path),
        "camera_id": camera_id,
        "total_frames": total,
        "total_frames_processed": processed,
        "failures_count": failures,
        "total_persons_detected": total_persons,
        "total_recognized_faces": total_recognized_faces,
        "real_flags": {
            "real_motion_detection": bool(real_motion_detection),
            "real_object_detection": bool(real_object_detection),
            "real_face_detection": bool(real_face_detection),
            "real_face_recognition": bool(real_face_recognition),
        },
        "frames": per_frame_summary,
    }

    write_json(logs_dir / "run_summary.json", summary)

    print("[FINAL SUMMARY]")
    print(f"  total frames processed: {processed}")
    print(f"  failures count: {failures}")
    print(f"  total persons detected: {total_persons}")
    print(f"  total recognized faces: {total_recognized_faces}")
    print(f"  summary file: {logs_dir / 'run_summary.json'}")

    return summary


def main() -> None:
    args = _parse_args()
    try:
        run_sequence(
            frames_dir=args.frames_dir,
            output_dir=args.output_dir,
            rpm_config=args.rpm_config,
            camera_id=args.camera_id,
            real_motion_detection=bool(args.real_motion_detection),
            real_object_detection=bool(args.real_object_detection),
            real_face_detection=bool(args.real_face_detection),
            real_face_recognition=bool(args.real_face_recognition),
            overwrite=bool(args.overwrite),
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
