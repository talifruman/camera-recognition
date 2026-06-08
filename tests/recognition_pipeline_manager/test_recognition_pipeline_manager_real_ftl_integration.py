"""RPM integration tests for per-frame cache behavior with real FTL.

These tests validate that RPM's per-frame cache semantics hold when using the
implemented FrameTransformationLayer and real ProcessedFrame objects.
"""

from __future__ import annotations

from dataclasses import asdict
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
_TESTS_DIR = Path(__file__).resolve().parent
for _p in (_SRC_DIR, _TESTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from image_processing.face_detection.module import DetectedFace, FaceDetectionOutput
from image_processing.face_recognition.module import FaceRecognitionOutput
from image_processing.frame_transformation_layer import FrameTransformationLayer
from image_processing.frame_transformation_layer.contracts import (
    FramePacket,
    FrameTemporalSelector,
)
from image_processing.motion_detection.module import MotionResult
from image_processing.object_detection.module import PersonDetectionResult
from image_processing.person_directory.module import PersonDirectoryOutput
from image_processing.recognition_pipeline_manager import RecognitionPipelineManager
from image_processing.shared.contracts import BoundingBox, FaceLandmarks, Point

from fakes import (  # type: ignore[import-not-found]
    FakeFaceDetection,
    FakeFaceRecognition,
    FakeMotionDetection,
    FakeObjectDetection,
    FakePersonDirectory,
)


_FRAME_W = 64
_FRAME_H = 48


def _make_packet(
    *,
    frame_id: str,
    camera_id: str = "cam-real-ftl",
    timestamp_ms: int = 1000,
    width: int = _FRAME_W,
    height: int = _FRAME_H,
    fill_value: int = 0,
) -> FramePacket:
    image = np.full((height, width, 3), fill_value=fill_value, dtype=np.uint8)
    return FramePacket(
        frame_id=frame_id,
        camera_id=camera_id,
        timestamp_ms=timestamp_ms,
        width=width,
        height=height,
        pixel_format="RGB",
        layout="HWC",
        dtype="uint8",
        value_range="[0,255]",
        num_color_channels=3,
        bits_per_channel=8,
        packing="tightly_packed",
        image_bytes=image.tobytes(),
    )


def _landmarks_at(x: int, y: int) -> FaceLandmarks:
    pt = Point(x=x, y=y)
    return FaceLandmarks(
        left_eye=pt,
        right_eye=pt,
        nose=pt,
        mouth_left=pt,
        mouth_right=pt,
    )


class CountingDelegatingFTL:
    """Delegates to real FTL and records each get_frame call + return metadata."""

    def __init__(self, delegate: FrameTransformationLayer) -> None:
        self._delegate = delegate
        self.get_frame_calls: list[dict[str, Any]] = []

    def ingest_frame(self, frame_packet: FramePacket) -> None:
        self._delegate.ingest_frame(frame_packet)

    def get_frame(
        self,
        camera_id: str,
        temporal_selector: FrameTemporalSelector,
        region_bbox: BoundingBox,
        output_type: Any,
        geometry_spec: Any,
    ) -> Any:
        record: dict[str, Any] = {
            "camera_id": camera_id,
            "temporal_selector": temporal_selector,
            "region_bbox": dict(region_bbox),
            "output_type": output_type,
            "geometry_spec": dict(geometry_spec),
            "processed_frame_id": None,
            "source_bbox_full_frame": None,
            "spatial_transform": None,
            "raised": None,
        }
        try:
            processed = self._delegate.get_frame(
                camera_id,
                temporal_selector,
                region_bbox,
                output_type,
                geometry_spec,
            )
        except Exception as exc:
            record["raised"] = type(exc).__name__
            self.get_frame_calls.append(record)
            raise

        record["processed_frame_id"] = processed.frame_id
        record["source_bbox_full_frame"] = dict(processed.source_bbox_full_frame)
        record["spatial_transform"] = asdict(processed.spatial_transform)
        self.get_frame_calls.append(record)
        return processed


def _normalize_key(call: dict[str, Any]) -> tuple[Any, ...]:
    geometry_spec = call["geometry_spec"]
    resize_policy = geometry_spec["resize_policy"]
    output_type = call["output_type"]
    region_bbox = call["region_bbox"]
    return (
        call["camera_id"],
        call["processed_frame_id"],
        call["temporal_selector"].value,
        int(region_bbox["x"]),
        int(region_bbox["y"]),
        int(region_bbox["width"]),
        int(region_bbox["height"]),
        output_type.value if hasattr(output_type, "value") else str(output_type),
        int(geometry_spec["width"]),
        int(geometry_spec["height"]),
        resize_policy.value if hasattr(resize_policy, "value") else str(resize_policy),
    )


def _build_rpm_for_duplicate_requests(ftl: CountingDelegatingFTL) -> RecognitionPipelineManager:
    motion_bbox = BoundingBox(x=4, y=3, width=20, height=16)
    person_bbox_local = BoundingBox(x=2, y=1, width=8, height=10)
    face_bbox_local = BoundingBox(x=1, y=2, width=4, height=4)

    motion = FakeMotionDetection(
        result=MotionResult(
            detected=True,
            bboxes=[motion_bbox, motion_bbox],
        )
    )
    object_det = FakeObjectDetection(
        result=PersonDetectionResult(
            frame_id="",
            person_detected=True,
            persons=[person_bbox_local, person_bbox_local],
        )
    )
    face_det = FakeFaceDetection(
        result=FaceDetectionOutput(
            frame_id="",
            camera_id="",
            timestamp_ms=0,
            detections=[
                DetectedFace(
                    face_bbox=face_bbox_local,
                    landmarks=_landmarks_at(2, 3),
                ),
                DetectedFace(
                    face_bbox=face_bbox_local,
                    landmarks=_landmarks_at(2, 3),
                ),
            ],
        )
    )
    face_rec = FakeFaceRecognition(
        result=FaceRecognitionOutput(
            frame_id="",
            camera_id="",
            timestamp_ms=0,
            person_found=True,
            person_id="person-42",
        )
    )
    person_dir = FakePersonDirectory(
        lookup_map={
            "person-42": PersonDirectoryOutput(
                person_id="person-42",
                person_name="Ada",
                found=True,
            )
        }
    )

    return RecognitionPipelineManager(
        ftl=ftl,
        motion=motion,
        object_det=object_det,
        face_det=face_det,
        face_rec=face_rec,
        person_dir=person_dir,
    )


def test_real_ftl_cache_reuses_processed_frames_for_duplicate_requests() -> None:
    real_ftl = FrameTransformationLayer()
    counting_ftl = CountingDelegatingFTL(real_ftl)
    rpm = _build_rpm_for_duplicate_requests(counting_ftl)

    first = _make_packet(frame_id="frame-1", timestamp_ms=1000, fill_value=10)
    second = _make_packet(frame_id="frame-2", timestamp_ms=2000, fill_value=20)

    first_out = rpm.process_frame(first)
    assert first_out["persons"] == []

    out = rpm.process_frame(second)

    assert out["frame_id"] == "frame-2"
    assert out["camera_id"] == "cam-real-ftl"
    assert out["timestamp_ms"] == 2000
    assert len(out["persons"]) == 4
    for person in out["persons"]:
        assert person["person_bbox"] == BoundingBox(x=6, y=4, width=8, height=10)
        assert len(person["recognized_faces"]) == 2
        for recognized in person["recognized_faces"]:
            assert recognized["face_bbox"] == BoundingBox(x=7, y=6, width=4, height=4)
            assert recognized["person_id"] == "person-42"
            assert recognized["person_name"] == "Ada"
            assert recognized["found"] is True

    calls = counting_ftl.get_frame_calls
    assert len(calls) == 7  # 2 calls on frame-1 + 5 unique calls on frame-2

    second_frame_calls = calls[2:]
    assert len(second_frame_calls) == 5

    unique_keys = {_normalize_key(c) for c in second_frame_calls}
    assert len(unique_keys) == 5

    full_frame_current = [
        c
        for c in second_frame_calls
        if c["temporal_selector"] is FrameTemporalSelector.CURRENT
        and c["region_bbox"] == BoundingBox(x=0, y=0, width=_FRAME_W, height=_FRAME_H)
    ]
    full_frame_previous = [
        c
        for c in second_frame_calls
        if c["temporal_selector"] is FrameTemporalSelector.PREVIOUS
        and c["region_bbox"] == BoundingBox(x=0, y=0, width=_FRAME_W, height=_FRAME_H)
    ]
    od_crop_calls = [
        c for c in second_frame_calls if c["region_bbox"] == BoundingBox(x=4, y=3, width=20, height=16)
    ]
    fd_crop_calls = [
        c for c in second_frame_calls if c["region_bbox"] == BoundingBox(x=6, y=4, width=8, height=10)
    ]
    fr_crop_calls = [
        c for c in second_frame_calls if c["region_bbox"] == BoundingBox(x=7, y=6, width=4, height=4)
    ]

    assert len(full_frame_current) == 1
    assert len(full_frame_previous) == 1
    assert len(od_crop_calls) == 1
    assert len(fd_crop_calls) == 1
    assert len(fr_crop_calls) == 1

    assert full_frame_current[0]["processed_frame_id"] == "frame-2"
    assert full_frame_previous[0]["processed_frame_id"] == "frame-1"

    for call in second_frame_calls:
        req_bbox = call["region_bbox"]
        assert call["source_bbox_full_frame"] == req_bbox
        st = call["spatial_transform"]
        assert st["scale_x"] == 1.0
        assert st["scale_y"] == 1.0
        assert st["pad_left"] == 0
        assert st["pad_top"] == 0
        assert st["output_width"] == req_bbox["width"]
        assert st["output_height"] == req_bbox["height"]


def test_real_ftl_calls_are_bounded_by_motion_roi_limit() -> None:
    real_ftl = FrameTransformationLayer()
    counting_ftl = CountingDelegatingFTL(real_ftl)

    noisy_bboxes = [
        BoundingBox(x=2 + (i * 2), y=3 + (i % 6), width=12, height=10)
        for i in range(20)
    ]
    motion = FakeMotionDetection(
        result=MotionResult(detected=True, bboxes=noisy_bboxes)
    )
    object_det = FakeObjectDetection(
        result=PersonDetectionResult(frame_id="", person_detected=False, persons=[])
    )

    rpm = RecognitionPipelineManager(
        ftl=counting_ftl,
        motion=motion,
        object_det=object_det,
        face_det=FakeFaceDetection(),
        face_rec=FakeFaceRecognition(),
        person_dir=FakePersonDirectory(),
        max_motion_rois_per_frame=4,
        max_person_rois_per_frame=16,
        max_face_rois_per_frame=16,
    )

    first = _make_packet(frame_id="frame-a", timestamp_ms=1000, fill_value=5)
    second = _make_packet(frame_id="frame-b", timestamp_ms=2000, fill_value=7)
    assert rpm.process_frame(first)["persons"] == []

    out = rpm.process_frame(second)
    metrics = rpm.get_last_frame_metrics()

    assert out["persons"] == []
    assert len(counting_ftl.get_frame_calls) == 8  # 2 from frame-a cold-start + 6 from frame-b
    assert metrics["total_ftl_calls_per_frame"] == 6
    assert metrics["ftl_calls_by_stage"]["motion_detection"] == 2
    assert metrics["ftl_calls_by_stage"]["object_detection"] == 4
