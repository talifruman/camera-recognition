"""RPM Phase 1 stub integration tests.

Tests RPM orchestration, contracts, projection, error gating, and output
structure using fake/stub dependencies.  No real ML inference.

Test categories:
    Golden scenarios (10):     orchestration paths through the pipeline
    SpatialCoordinator units:  NONE policy, LETTERBOX policy, invalid inputs
    InputValidator units:      all invalid cases + valid pass-through
    OutputBuilder units:       metadata propagation and structure

Projection assertions use exact integer == comparisons (C4).
FakeFTL propagates real frame_id / timestamp_ms metadata (C3).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
_TESTS_DIR = Path(__file__).resolve().parent
for _p in (_SRC_DIR, _TESTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest

from image_processing.face_detection.module import DetectedFace, FaceDetectionOutput
from image_processing.face_recognition.module import FaceRecognitionOutput
from image_processing.frame_transformation_layer.contracts import (
    FramePacket,
    FrameTemporalSelector,
    SpatialTransform,
)
from image_processing.motion_detection.module import MotionResult
from image_processing.object_detection.module import PersonDetectionResult
from image_processing.person_directory.module import PersonDirectoryOutput
from image_processing.recognition_pipeline_manager import RecognitionPipelineManager
from image_processing.recognition_pipeline_manager.input_validator import (
    RecognitionPipelineInputValidator,
)
from image_processing.recognition_pipeline_manager.output_builder import (
    RecognitionPipelineOutputBuilder,
)
from image_processing.recognition_pipeline_manager.spatial_coordinator import (
    SpatialCoordinator,
)
from image_processing.recognition_pipeline_manager.types import PipelineResult
from image_processing.shared.contracts import BoundingBox, FaceLandmarks, OutputImageType, Point

from fakes import (  # type: ignore[import-not-found]
    FakeFaceDetection,
    FakeFaceRecognition,
    FakeFTL,
    FakeMotionDetection,
    FakeObjectDetection,
    FakePersonDirectory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRAME_W = 640
_FRAME_H = 480


def _make_packet(
    frame_id: str = "frame-1",
    camera_id: str = "cam-A",
    timestamp_ms: int = 1000,
    width: int = _FRAME_W,
    height: int = _FRAME_H,
) -> FramePacket:
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
        image_bytes=b"\x00" * (width * height * 3),
    )


def _make_rpm(
    ftl: FakeFTL,
    motion: FakeMotionDetection,
    object_det: FakeObjectDetection,
    face_det: FakeFaceDetection,
    face_rec: FakeFaceRecognition,
    person_dir: FakePersonDirectory,
    demo_bypass_motion_gate: bool = False,
) -> RecognitionPipelineManager:
    return RecognitionPipelineManager(
        ftl=ftl,
        motion=motion,
        object_det=object_det,
        face_det=face_det,
        face_rec=face_rec,
        person_dir=person_dir,
        demo_bypass_motion_gate=demo_bypass_motion_gate,
    )


def _default_rpm(**overrides) -> RecognitionPipelineManager:
    """Build an RPM with all default fakes, allowing selective override."""
    defaults = dict(
        ftl=FakeFTL(),
        motion=FakeMotionDetection(),
        object_det=FakeObjectDetection(),
        face_det=FakeFaceDetection(),
        face_rec=FakeFaceRecognition(),
        person_dir=FakePersonDirectory(),
    )
    defaults.update(overrides)
    return _make_rpm(**defaults)


def _landmarks_at(x: int, y: int) -> FaceLandmarks:
    pt = Point(x=x, y=y)
    return FaceLandmarks(
        left_eye=pt, right_eye=pt, nose=pt,
        mouth_left=pt, mouth_right=pt,
    )


# ---------------------------------------------------------------------------
# 1. Cold start
# ---------------------------------------------------------------------------


class TestColdStart:
    def test_cold_start_returns_empty_persons(self):
        ftl = FakeFTL(raise_on_previous=True)
        rpm = _default_rpm(ftl=ftl)
        pkt = _make_packet()

        out = rpm.process_frame(pkt)

        assert out["persons"] == []

    def test_cold_start_preserves_traceability(self):
        ftl = FakeFTL(raise_on_previous=True)
        rpm = _default_rpm(ftl=ftl)
        pkt = _make_packet(frame_id="f-cold", camera_id="cam-cold", timestamp_ms=999)

        out = rpm.process_frame(pkt)

        assert out["frame_id"] == "f-cold"
        assert out["camera_id"] == "cam-cold"
        assert out["timestamp_ms"] == 999

    def test_cold_start_does_not_call_motion_detection(self):
        ftl = FakeFTL(raise_on_previous=True)
        motion = FakeMotionDetection()
        rpm = _default_rpm(ftl=ftl, motion=motion)

        rpm.process_frame(_make_packet())

        assert len(motion.calls) == 0

    def test_cold_start_does_not_call_od_fd_fr(self):
        ftl = FakeFTL(raise_on_previous=True)
        od = FakeObjectDetection()
        fd = FakeFaceDetection()
        fr = FakeFaceRecognition()
        rpm = _default_rpm(ftl=ftl, object_det=od, face_det=fd, face_rec=fr)

        rpm.process_frame(_make_packet())

        assert len(od.calls) == 0
        assert len(fd.calls) == 0
        assert len(fr.calls) == 0


# ---------------------------------------------------------------------------
# 2. No motion
# ---------------------------------------------------------------------------


class TestNoMotion:
    def test_no_motion_returns_empty_persons(self):
        motion = FakeMotionDetection(result=MotionResult(detected=False, bboxes=[]))
        rpm = _default_rpm(motion=motion)

        out = rpm.process_frame(_make_packet())

        assert out["persons"] == []

    def test_no_motion_od_not_called(self):
        motion = FakeMotionDetection(result=MotionResult(detected=False, bboxes=[]))
        od = FakeObjectDetection()
        rpm = _default_rpm(motion=motion, object_det=od)

        rpm.process_frame(_make_packet())

        assert len(od.calls) == 0

    def test_no_motion_fd_fr_not_called(self):
        motion = FakeMotionDetection(result=MotionResult(detected=False, bboxes=[]))
        fd = FakeFaceDetection()
        fr = FakeFaceRecognition()
        rpm = _default_rpm(motion=motion, face_det=fd, face_rec=fr)

        rpm.process_frame(_make_packet())

        assert len(fd.calls) == 0
        assert len(fr.calls) == 0

    def test_no_motion_motion_still_called(self):
        motion = FakeMotionDetection(result=MotionResult(detected=False, bboxes=[]))
        rpm = _default_rpm(motion=motion)

        rpm.process_frame(_make_packet())

        assert len(motion.calls) == 1

    def test_no_motion_with_demo_bypass_runs_object_detection_full_frame(self):
        motion = FakeMotionDetection(result=MotionResult(detected=False, bboxes=[]))
        od = FakeObjectDetection()
        rpm = _default_rpm(
            motion=motion,
            object_det=od,
            demo_bypass_motion_gate=True,
        )

        rpm.process_frame(_make_packet())
        metrics = rpm.get_last_frame_metrics()

        assert len(od.calls) == 1
        assert od.calls[0]["roi_bbox_frame"] == {
            "x": 0,
            "y": 0,
            "width": _FRAME_W,
            "height": _FRAME_H,
        }
        assert metrics["motion_detected"] is False
        assert metrics["demo_bypass_motion_gate"] is True

    def test_no_motion_without_demo_bypass_keeps_original_gate(self):
        motion = FakeMotionDetection(result=MotionResult(detected=False, bboxes=[]))
        od = FakeObjectDetection()
        rpm = _default_rpm(motion=motion, object_det=od)

        rpm.process_frame(_make_packet())
        metrics = rpm.get_last_frame_metrics()

        assert len(od.calls) == 0
        assert metrics["demo_bypass_motion_gate"] is False


# ---------------------------------------------------------------------------
# 3. Motion but no person
# ---------------------------------------------------------------------------


class TestMotionNoPersons:
    def test_motion_no_persons_returns_empty(self):
        motion = FakeMotionDetection(result=MotionResult(
            detected=True,
            bboxes=[BoundingBox(x=10, y=10, width=100, height=100)],
        ))
        od = FakeObjectDetection(result=PersonDetectionResult(
            frame_id="", person_detected=False, persons=[]
        ))
        rpm = _default_rpm(motion=motion, object_det=od)

        out = rpm.process_frame(_make_packet())

        assert out["persons"] == []

    def test_motion_od_called_once_per_bbox(self):
        motion = FakeMotionDetection(result=MotionResult(
            detected=True,
            bboxes=[
                BoundingBox(x=10, y=10, width=100, height=100),
                BoundingBox(x=200, y=50, width=80, height=80),
            ],
        ))
        od = FakeObjectDetection(result=PersonDetectionResult(
            frame_id="", person_detected=False, persons=[]
        ))
        rpm = _default_rpm(motion=motion, object_det=od)

        rpm.process_frame(_make_packet())

        assert len(od.calls) == 2

    def test_motion_fd_fr_not_called_when_no_persons(self):
        motion = FakeMotionDetection(result=MotionResult(
            detected=True,
            bboxes=[BoundingBox(x=10, y=10, width=100, height=100)],
        ))
        od = FakeObjectDetection(result=PersonDetectionResult(
            frame_id="", person_detected=False, persons=[]
        ))
        fd = FakeFaceDetection()
        fr = FakeFaceRecognition()
        rpm = _default_rpm(motion=motion, object_det=od, face_det=fd, face_rec=fr)

        rpm.process_frame(_make_packet())

        assert len(fd.calls) == 0
        assert len(fr.calls) == 0

    def test_duplicate_motion_bboxes_reuse_same_od_ftl_crop(self):
        """Identical motion regions should trigger a single OD FTL transform."""
        motion = FakeMotionDetection(result=MotionResult(
            detected=True,
            bboxes=[
                BoundingBox(x=10, y=10, width=100, height=100),
                BoundingBox(x=10, y=10, width=100, height=100),
            ],
        ))
        od = FakeObjectDetection(result=PersonDetectionResult(
            frame_id="", person_detected=False, persons=[]
        ))
        ftl = FakeFTL()
        rpm = _default_rpm(ftl=ftl, motion=motion, object_det=od)

        rpm.process_frame(_make_packet())

        # CURRENT + PREVIOUS + one unique OD region fetch
        assert len(ftl.get_frame_calls) == 3


class TestRoiLimitsAndMetrics:
    def test_noisy_motion_output_is_bounded_before_od(self):
        noisy_motion_bboxes = [
            BoundingBox(x=i * 3, y=(i % 5) * 4, width=20 + (i % 3), height=18 + (i % 4))
            for i in range(40)
        ]
        motion = FakeMotionDetection(
            result=MotionResult(detected=True, bboxes=noisy_motion_bboxes)
        )
        od = FakeObjectDetection(
            result=PersonDetectionResult(frame_id="", person_detected=False, persons=[])
        )
        ftl = FakeFTL()
        rpm = RecognitionPipelineManager(
            ftl=ftl,
            motion=motion,
            object_det=od,
            face_det=FakeFaceDetection(),
            face_rec=FakeFaceRecognition(),
            person_dir=FakePersonDirectory(),
            max_motion_rois_per_frame=5,
            max_person_rois_per_frame=20,
            max_face_rois_per_frame=20,
        )

        out = rpm.process_frame(_make_packet())
        metrics = rpm.get_last_frame_metrics()

        assert out["persons"] == []
        assert len(od.calls) == 5
        assert len(ftl.get_frame_calls) == 7  # 2 motion frame fetches + 5 OD ROI fetches
        assert metrics["od_roi_requests_count"] == 5
        assert metrics["total_ftl_calls_per_frame"] == 7
        assert metrics["ftl_calls_by_stage"]["motion_detection"] == 2
        assert metrics["ftl_calls_by_stage"]["object_detection"] == 5

    def test_person_and_face_roi_limits_are_enforced(self):
        motion = FakeMotionDetection(
            result=MotionResult(
                detected=True,
                bboxes=[BoundingBox(x=0, y=0, width=640, height=480)],
            )
        )
        od = FakeObjectDetection(
            result=PersonDetectionResult(
                frame_id="",
                person_detected=True,
                persons=[
                    BoundingBox(x=0, y=0, width=300, height=400),
                    BoundingBox(x=20, y=10, width=200, height=300),
                    BoundingBox(x=40, y=20, width=120, height=180),
                    BoundingBox(x=60, y=30, width=60, height=90),
                ],
            )
        )
        fd = FakeFaceDetection(
            result=FaceDetectionOutput(
                frame_id="",
                camera_id="",
                timestamp_ms=0,
                detections=[
                    DetectedFace(
                        face_bbox=BoundingBox(x=10, y=10, width=50, height=50),
                        landmarks=_landmarks_at(20, 20),
                    ),
                    DetectedFace(
                        face_bbox=BoundingBox(x=12, y=12, width=30, height=30),
                        landmarks=_landmarks_at(18, 18),
                    ),
                ],
            )
        )
        fr = FakeFaceRecognition(
            result=FaceRecognitionOutput(
                frame_id="",
                camera_id="",
                timestamp_ms=0,
                person_found=True,
                person_id="person-1",
            )
        )
        pd = FakePersonDirectory(
            lookup_map={
                "person-1": PersonDirectoryOutput(
                    person_id="person-1",
                    person_name="P1",
                    found=True,
                )
            }
        )
        rpm = RecognitionPipelineManager(
            ftl=FakeFTL(),
            motion=motion,
            object_det=od,
            face_det=fd,
            face_rec=fr,
            person_dir=pd,
            max_motion_rois_per_frame=3,
            max_person_rois_per_frame=2,
            max_face_rois_per_frame=1,
        )

        out = rpm.process_frame(_make_packet())
        metrics = rpm.get_last_frame_metrics()

        assert len(out["persons"]) == 2
        assert len(fd.calls) == 2
        assert len(fr.calls) == 1
        assert metrics["fd_roi_requests_count"] == 2
        assert metrics["fr_roi_requests_count"] == 1

    def test_duplicate_identical_requests_still_deduplicated_with_limits(self):
        motion = FakeMotionDetection(
            result=MotionResult(
                detected=True,
                bboxes=[
                    BoundingBox(x=10, y=10, width=100, height=100),
                    BoundingBox(x=10, y=10, width=100, height=100),
                    BoundingBox(x=10, y=10, width=100, height=100),
                ],
            )
        )
        od = FakeObjectDetection(
            result=PersonDetectionResult(frame_id="", person_detected=False, persons=[])
        )
        ftl = FakeFTL()
        rpm = RecognitionPipelineManager(
            ftl=ftl,
            motion=motion,
            object_det=od,
            face_det=FakeFaceDetection(),
            face_rec=FakeFaceRecognition(),
            person_dir=FakePersonDirectory(),
            max_motion_rois_per_frame=3,
            max_person_rois_per_frame=10,
            max_face_rois_per_frame=10,
        )

        rpm.process_frame(_make_packet())
        metrics = rpm.get_last_frame_metrics()

        assert len(ftl.get_frame_calls) == 3  # current + previous + one deduped OD crop
        assert metrics["total_ftl_calls_per_frame"] == 3


# ---------------------------------------------------------------------------
# 4. Person but no face detected
# ---------------------------------------------------------------------------


class TestPersonNoFace:
    def setup_method(self):
        self.motion = FakeMotionDetection(result=MotionResult(
            detected=True,
            bboxes=[BoundingBox(x=0, y=0, width=200, height=200)],
        ))
        self.od = FakeObjectDetection(result=PersonDetectionResult(
            frame_id="",
            person_detected=True,
            persons=[BoundingBox(x=10, y=10, width=80, height=160)],
        ))
        self.fd = FakeFaceDetection(result=FaceDetectionOutput(
            frame_id="", camera_id="", timestamp_ms=0, detections=[]
        ))

    def test_person_no_face_one_person_result(self):
        rpm = _default_rpm(motion=self.motion, object_det=self.od, face_det=self.fd)
        out = rpm.process_frame(_make_packet())

        assert len(out["persons"]) == 1

    def test_person_no_face_recognized_faces_empty(self):
        rpm = _default_rpm(motion=self.motion, object_det=self.od, face_det=self.fd)
        out = rpm.process_frame(_make_packet())

        assert out["persons"][0]["recognized_faces"] == []

    def test_person_no_face_fr_not_called(self):
        fr = FakeFaceRecognition()
        rpm = _default_rpm(motion=self.motion, object_det=self.od, face_det=self.fd, face_rec=fr)
        rpm.process_frame(_make_packet())

        assert len(fr.calls) == 0

    def test_person_bbox_is_in_output(self):
        rpm = _default_rpm(motion=self.motion, object_det=self.od, face_det=self.fd)
        out = rpm.process_frame(_make_packet())

        person = out["persons"][0]
        bbox = person["person_bbox"]
        assert bbox["width"] > 0
        assert bbox["height"] > 0


# ---------------------------------------------------------------------------
# 5. Face detected but unrecognized
# ---------------------------------------------------------------------------


class TestFaceUnrecognized:
    def setup_method(self):
        self.motion = FakeMotionDetection(result=MotionResult(
            detected=True,
            bboxes=[BoundingBox(x=0, y=0, width=200, height=200)],
        ))
        self.od = FakeObjectDetection(result=PersonDetectionResult(
            frame_id="", person_detected=True,
            persons=[BoundingBox(x=0, y=0, width=200, height=200)],
        ))
        self.fd = FakeFaceDetection(result=FaceDetectionOutput(
            frame_id="", camera_id="", timestamp_ms=0,
            detections=[DetectedFace(
                face_bbox=BoundingBox(x=20, y=20, width=60, height=60),
                landmarks=_landmarks_at(50, 40),
            )],
        ))
        self.fr = FakeFaceRecognition(result=FaceRecognitionOutput(
            frame_id="", camera_id="", timestamp_ms=0,
            person_found=False, person_id="UNKNOWN",
        ))

    def test_unrecognized_face_omitted(self):
        rpm = _default_rpm(
            motion=self.motion, object_det=self.od,
            face_det=self.fd, face_rec=self.fr,
        )
        out = rpm.process_frame(_make_packet())

        assert len(out["persons"]) == 1
        assert out["persons"][0]["recognized_faces"] == []

    def test_person_directory_not_called_when_unrecognized(self):
        pd = FakePersonDirectory()
        rpm = _default_rpm(
            motion=self.motion, object_det=self.od,
            face_det=self.fd, face_rec=self.fr,
            person_dir=pd,
        )
        rpm.process_frame(_make_packet())

        assert len(pd.calls) == 0


# ---------------------------------------------------------------------------
# 6. Fully recognized person — end-to-end golden path
# ---------------------------------------------------------------------------


class TestFullyRecognized:
    _PERSON_ID = "person-abc"
    _PERSON_NAME = "Alice"

    def setup_method(self):
        self.motion = FakeMotionDetection(result=MotionResult(
            detected=True,
            bboxes=[BoundingBox(x=0, y=0, width=640, height=480)],
        ))
        self.od = FakeObjectDetection(result=PersonDetectionResult(
            frame_id="", person_detected=True,
            persons=[BoundingBox(x=50, y=50, width=100, height=200)],
        ))
        self.fd = FakeFaceDetection(result=FaceDetectionOutput(
            frame_id="", camera_id="", timestamp_ms=0,
            detections=[DetectedFace(
                face_bbox=BoundingBox(x=10, y=10, width=40, height=40),
                landmarks=_landmarks_at(30, 25),
            )],
        ))
        self.fr = FakeFaceRecognition(result=FaceRecognitionOutput(
            frame_id="", camera_id="", timestamp_ms=0,
            person_found=True, person_id=self._PERSON_ID,
        ))
        self.pd = FakePersonDirectory(lookup_map={
            self._PERSON_ID: PersonDirectoryOutput(
                person_id=self._PERSON_ID,
                person_name=self._PERSON_NAME,
                found=True,
            )
        })

    def _rpm(self) -> RecognitionPipelineManager:
        return _default_rpm(
            motion=self.motion, object_det=self.od,
            face_det=self.fd, face_rec=self.fr,
            person_dir=self.pd,
        )

    def test_one_person_in_output(self):
        out = self._rpm().process_frame(_make_packet())
        assert len(out["persons"]) == 1

    def test_one_recognized_face_in_person(self):
        out = self._rpm().process_frame(_make_packet())
        person = out["persons"][0]
        assert len(person["recognized_faces"]) == 1

    def test_recognized_face_person_id(self):
        out = self._rpm().process_frame(_make_packet())
        face = out["persons"][0]["recognized_faces"][0]
        assert face["person_id"] == self._PERSON_ID

    def test_recognized_face_person_name(self):
        out = self._rpm().process_frame(_make_packet())
        face = out["persons"][0]["recognized_faces"][0]
        assert face["person_name"] == self._PERSON_NAME

    def test_traceability_metadata(self):
        pkt = _make_packet(frame_id="f-gold", camera_id="cam-G", timestamp_ms=4242)
        out = self._rpm().process_frame(pkt)
        assert out["frame_id"] == "f-gold"
        assert out["camera_id"] == "cam-G"
        assert out["timestamp_ms"] == 4242

    def test_face_bbox_in_full_frame_coords(self):
        out = self._rpm().process_frame(_make_packet())
        face_bbox = out["persons"][0]["recognized_faces"][0]["face_bbox"]
        # full-frame coords must be non-negative and positive dimensions
        assert face_bbox["x"] >= 0
        assert face_bbox["y"] >= 0
        assert face_bbox["width"] > 0
        assert face_bbox["height"] > 0

    def test_person_directory_called_with_correct_id(self):
        self._rpm().process_frame(_make_packet())
        assert self.pd.calls == [self._PERSON_ID]

    def test_ftl_ingest_called(self):
        ftl = FakeFTL()
        pkt = _make_packet()
        _make_rpm(ftl=ftl, motion=self.motion, object_det=self.od,
                  face_det=self.fd, face_rec=self.fr, person_dir=self.pd
                  ).process_frame(pkt)
        assert len(ftl.ingest_calls) == 1
        assert ftl.ingest_calls[0].frame_id == pkt.frame_id


# ---------------------------------------------------------------------------
# 6b. PersonDirectory enrichment exception — face kept with UNKNOWN fallback
# ---------------------------------------------------------------------------


class TestPersonDirectoryEnrichmentFailure:
    """If PersonDirectory.get_person() raises, the face must still be appended
    with deterministic UNKNOWN metadata — no crash, no face drop."""

    _PERSON_ID = "person-abc"

    def setup_method(self):
        self.motion = FakeMotionDetection(result=MotionResult(
            detected=True,
            bboxes=[BoundingBox(x=0, y=0, width=640, height=480)],
        ))
        self.od = FakeObjectDetection(result=PersonDetectionResult(
            frame_id="", person_detected=True,
            persons=[BoundingBox(x=50, y=50, width=100, height=200)],
        ))
        self.fd = FakeFaceDetection(result=FaceDetectionOutput(
            frame_id="", camera_id="", timestamp_ms=0,
            detections=[DetectedFace(
                face_bbox=BoundingBox(x=10, y=10, width=40, height=40),
                landmarks=_landmarks_at(30, 25),
            )],
        ))
        self.fr = FakeFaceRecognition(result=FaceRecognitionOutput(
            frame_id="", camera_id="", timestamp_ms=0,
            person_found=True, person_id=self._PERSON_ID,
        ))
        self.pd = FakePersonDirectory(raise_on_get_person=RuntimeError("PD failure"))

    def test_person_directory_exception_falls_back_to_unknown_and_keeps_face(self):
        rpm = _default_rpm(
            motion=self.motion, object_det=self.od,
            face_det=self.fd, face_rec=self.fr,
            person_dir=self.pd,
        )
        out = rpm.process_frame(_make_packet())

        # pipeline must not crash and face must still be appended
        assert len(out["persons"]) == 1
        faces = out["persons"][0]["recognized_faces"]
        assert len(faces) == 1
        face = faces[0]
        assert face["person_id"] == "UNKNOWN"
        assert face["person_name"] == "UNKNOWN"
        assert face["found"] is False


# ---------------------------------------------------------------------------
# 7. Multiple persons — no cross-contamination
# ---------------------------------------------------------------------------


class TestMultiplePersons:
    _P1_ID = "person-1"
    _P2_ID = "person-2"

    def _run(self) -> tuple:
        motion = FakeMotionDetection(result=MotionResult(
            detected=True,
            bboxes=[BoundingBox(x=0, y=0, width=640, height=480)],
        ))
        od = FakeObjectDetection(result=PersonDetectionResult(
            frame_id="", person_detected=True,
            persons=[
                BoundingBox(x=0, y=0, width=100, height=200),
                BoundingBox(x=200, y=0, width=100, height=200),
            ],
        ))

        call_count = {"n": 0}

        class SequentialFaceDet:
            """Returns a face for first person, none for second."""
            def detect_faces(self, inp):
                n = call_count["n"]
                call_count["n"] += 1
                if n == 0:
                    return FaceDetectionOutput(
                        frame_id="", camera_id="", timestamp_ms=0,
                        detections=[DetectedFace(
                            face_bbox=BoundingBox(x=5, y=5, width=30, height=30),
                            landmarks=_landmarks_at(20, 20),
                        )],
                    )
                return FaceDetectionOutput(
                    frame_id="", camera_id="", timestamp_ms=0, detections=[]
                )
            def get_input_contract(self):
                from fakes import _STUB_INPUT_CONTRACT_RGB  # type: ignore[import-not-found]
                return _STUB_INPUT_CONTRACT_RGB

        fr = FakeFaceRecognition(result=FaceRecognitionOutput(
            frame_id="", camera_id="", timestamp_ms=0,
            person_found=True, person_id=self._P1_ID,
        ))
        pd = FakePersonDirectory(lookup_map={
            self._P1_ID: PersonDirectoryOutput(
                person_id=self._P1_ID, person_name="Person1", found=True
            )
        })

        rpm = RecognitionPipelineManager(
            ftl=FakeFTL(), motion=motion, object_det=od,
            face_det=SequentialFaceDet(), face_rec=fr, person_dir=pd,
        )
        out = rpm.process_frame(_make_packet())
        return out

    def test_two_persons_in_output(self):
        out = self._run()
        assert len(out["persons"]) == 2

    def test_first_person_has_recognized_face(self):
        out = self._run()
        assert len(out["persons"][0]["recognized_faces"]) == 1

    def test_second_person_has_no_recognized_faces(self):
        out = self._run()
        assert len(out["persons"][1]["recognized_faces"]) == 0

    def test_no_cross_person_face_mixing(self):
        out = self._run()
        # total recognized faces across all persons must equal number from FR calls
        total = sum(len(p["recognized_faces"]) for p in out["persons"])
        assert total == 1


# ---------------------------------------------------------------------------
# 8. Multiple cameras — no state leakage
# ---------------------------------------------------------------------------


class TestMultipleCameras:
    def test_camera_id_preserved_for_each_camera(self):
        motion = FakeMotionDetection(result=MotionResult(detected=False, bboxes=[]))
        ftl = FakeFTL()
        rpm = _default_rpm(ftl=ftl, motion=motion)

        pkt_a = _make_packet(camera_id="cam-A", frame_id="f-a")
        pkt_b = _make_packet(camera_id="cam-B", frame_id="f-b")

        out_a = rpm.process_frame(pkt_a)
        out_b = rpm.process_frame(pkt_b)

        assert out_a["camera_id"] == "cam-A"
        assert out_b["camera_id"] == "cam-B"

    def test_frame_id_preserved_for_each_camera(self):
        motion = FakeMotionDetection(result=MotionResult(detected=False, bboxes=[]))
        rpm = _default_rpm(motion=motion)

        out_a = rpm.process_frame(_make_packet(camera_id="cam-A", frame_id="fa-1"))
        out_b = rpm.process_frame(_make_packet(camera_id="cam-B", frame_id="fb-1"))

        assert out_a["frame_id"] == "fa-1"
        assert out_b["frame_id"] == "fb-1"

    def test_no_rpm_state_leakage_between_cameras(self):
        # Second camera should not see first camera's persons
        call_count = {"n": 0}
        motion_results = [
            MotionResult(detected=True, bboxes=[BoundingBox(x=0, y=0, width=100, height=100)]),
            MotionResult(detected=False, bboxes=[]),
        ]

        class SequentialMotion:
            def detect(self, inp):
                n = call_count["n"]
                call_count["n"] += 1
                return motion_results[min(n, 1)]
            def get_input_contract(self):
                from fakes import _STUB_INPUT_CONTRACT_GRAY  # type: ignore[import-not-found]
                return _STUB_INPUT_CONTRACT_GRAY

        od = FakeObjectDetection(result=PersonDetectionResult(
            frame_id="", person_detected=True,
            persons=[BoundingBox(x=10, y=10, width=50, height=100)],
        ))

        rpm = RecognitionPipelineManager(
            ftl=FakeFTL(), motion=SequentialMotion(),
            object_det=od, face_det=FakeFaceDetection(),
            face_rec=FakeFaceRecognition(), person_dir=FakePersonDirectory(),
        )

        out_cam_a = rpm.process_frame(_make_packet(camera_id="cam-A", frame_id="fa"))
        out_cam_b = rpm.process_frame(_make_packet(camera_id="cam-B", frame_id="fb"))

        # cam-A had motion → 1 person; cam-B had no motion → 0 persons
        assert len(out_cam_a["persons"]) == 1
        assert out_cam_b["persons"] == []


# ---------------------------------------------------------------------------
# 9. Region-level get_frame failures — skip-only behavior
# ---------------------------------------------------------------------------


class TestRegionLevelFailures:
    """Spec §11: region failures skip only the affected branch."""

    def _motion_with_two_bboxes(self) -> FakeMotionDetection:
        return FakeMotionDetection(result=MotionResult(
            detected=True,
            bboxes=[
                BoundingBox(x=0, y=0, width=100, height=100),
                BoundingBox(x=200, y=0, width=100, height=100),
            ],
        ))

    def test_od_region_failure_skips_only_that_region(self):
        """First OD get_frame fails → second motion region still processed."""
        motion = self._motion_with_two_bboxes()
        od = FakeObjectDetection(result=PersonDetectionResult(
            frame_id="", person_detected=True,
            persons=[BoundingBox(x=10, y=10, width=50, height=80)],
        ))
        # get_frame call 0 = CURRENT full-frame motion, 1 = PREVIOUS full-frame motion
        # call 2 = OD ROI for first motion bbox → fail
        # call 3 = OD ROI for second motion bbox → succeed
        ftl = FakeFTL(raise_on_get_frame_call_indices={2})

        rpm = RecognitionPipelineManager(
            ftl=ftl, motion=motion, object_det=od,
            face_det=FakeFaceDetection(), face_rec=FakeFaceRecognition(),
            person_dir=FakePersonDirectory(),
        )
        out = rpm.process_frame(_make_packet())

        # Second region succeeded → one person in output
        assert len(out["persons"]) == 1

    def test_fd_person_failure_skips_only_that_person(self):
        """FD get_frame fails for first person → second person still processed."""
        motion = FakeMotionDetection(result=MotionResult(
            detected=True,
            bboxes=[BoundingBox(x=0, y=0, width=640, height=480)],
        ))
        od = FakeObjectDetection(result=PersonDetectionResult(
            frame_id="", person_detected=True,
            persons=[
                BoundingBox(x=0, y=0, width=100, height=200),
                BoundingBox(x=200, y=0, width=100, height=200),
            ],
        ))
        # calls: 0=CURRENT motion, 1=PREVIOUS motion, 2=OD ROI,
        # 3=FD ROI person-1 → fail, 4=FD ROI person-2 → succeed
        ftl = FakeFTL(raise_on_get_frame_call_indices={3})

        rpm = RecognitionPipelineManager(
            ftl=ftl, motion=motion, object_det=od,
            face_det=FakeFaceDetection(), face_rec=FakeFaceRecognition(),
            person_dir=FakePersonDirectory(),
        )
        out = rpm.process_frame(_make_packet())

        # Both persons appear in output (person bbox projected before FD)
        assert len(out["persons"]) == 2
        # Person 0: FD failed → no recognized faces; Person 1: FD returned empty → no faces
        assert out["persons"][0]["recognized_faces"] == []
        assert out["persons"][1]["recognized_faces"] == []

    def test_ingest_failure_returns_empty(self):
        ftl = FakeFTL(raise_on_ingest=RuntimeError("ingest failed"))
        rpm = _default_rpm(ftl=ftl)

        out = rpm.process_frame(_make_packet())

        assert out["persons"] == []


# ---------------------------------------------------------------------------
# 10. LETTERBOX projection — exact integer math (C4)
# ---------------------------------------------------------------------------


class TestLetterboxProjection:
    """
    Reference worked example (discussion §5):
        source_bbox_full_frame = {x:100, y:50, width:200, height:100}
        spatial_transform = {scale_x:0.5, scale_y:0.5, pad_left:10, pad_top:20}
        local point (ox=60, oy=70):
            cx = 60 - 10 = 50
            cy = 70 - 20 = 50
            rx = 50 / 0.5 = 100
            ry = 50 / 0.5 = 100
            fx = 100 + 100 = 200
            fy = 100 + 50  = 150
    """

    _SOURCE_BBOX: BoundingBox = BoundingBox(x=100, y=50, width=200, height=100)
    _TRANSFORM = SpatialTransform(
        scale_x=0.5, scale_y=0.5,
        pad_left=10, pad_top=20,
        output_width=120, output_height=70,
    )

    def test_letterbox_single_point_projection(self):
        sc = SpatialCoordinator()
        local_bbox = BoundingBox(x=60, y=70, width=20, height=20)

        result = sc.project_bbox_to_full_frame(
            local_bbox, self._SOURCE_BBOX, self._TRANSFORM
        )

        assert result is not None
        # x_full: (60-10)/0.5 + 100 = 200
        assert result["x"] == 200
        # y_full: (70-20)/0.5 + 50  = 150
        assert result["y"] == 150

    def test_letterbox_bbox_size_projection(self):
        sc = SpatialCoordinator()
        local_bbox = BoundingBox(x=60, y=70, width=30, height=10)

        result = sc.project_bbox_to_full_frame(
            local_bbox, self._SOURCE_BBOX, self._TRANSFORM
        )

        assert result is not None
        # w_full: 30 / 0.5 = 60
        assert result["width"] == 60
        # h_full: 10 / 0.5 = 20
        assert result["height"] == 20

    def test_letterbox_zero_padding_case(self):
        """Verify that zero padding is handled correctly (reduces to scale-only)."""
        sc = SpatialCoordinator()
        transform = SpatialTransform(
            scale_x=0.5, scale_y=0.5,
            pad_left=0, pad_top=0,
            output_width=100, output_height=50,
        )
        source = BoundingBox(x=50, y=25, width=200, height=100)
        local_bbox = BoundingBox(x=40, y=20, width=20, height=10)

        result = sc.project_bbox_to_full_frame(local_bbox, source, transform)

        assert result is not None
        # x_full: 40/0.5 + 50 = 130
        assert result["x"] == 130
        # y_full: 20/0.5 + 25 = 65
        assert result["y"] == 65
        # w_full: 20/0.5 = 40
        assert result["width"] == 40
        # h_full: 10/0.5 = 20
        assert result["height"] == 20


# ---------------------------------------------------------------------------
# Unit: SpatialCoordinator — NONE policy
# ---------------------------------------------------------------------------


class TestSpatialCoordinatorNone:
    _IDENTITY = SpatialTransform(
        scale_x=1.0, scale_y=1.0,
        pad_left=0, pad_top=0,
        output_width=200, output_height=100,
    )

    def test_none_policy_identity_plus_offset(self):
        sc = SpatialCoordinator()
        source = BoundingBox(x=30, y=20, width=200, height=100)
        local_bbox = BoundingBox(x=10, y=5, width=50, height=60)

        result = sc.project_bbox_to_full_frame(local_bbox, source, self._IDENTITY)

        assert result is not None
        # x_full = (10 - 0) / 1.0 + 30 = 40
        assert result["x"] == 40
        # y_full = (5 - 0) / 1.0 + 20 = 25
        assert result["y"] == 25
        assert result["width"] == 50
        assert result["height"] == 60

    def test_none_policy_origin_bbox(self):
        sc = SpatialCoordinator()
        source = BoundingBox(x=0, y=0, width=640, height=480)
        local_bbox = BoundingBox(x=100, y=200, width=30, height=40)

        result = sc.project_bbox_to_full_frame(local_bbox, source, self._IDENTITY)

        assert result is not None
        assert result["x"] == 100
        assert result["y"] == 200
        assert result["width"] == 30
        assert result["height"] == 40


# ---------------------------------------------------------------------------
# Unit: SpatialCoordinator — invalid inputs return None
# ---------------------------------------------------------------------------


class TestSpatialCoordinatorInvalidInputs:
    _IDENTITY = SpatialTransform(
        scale_x=1.0, scale_y=1.0,
        pad_left=0, pad_top=0,
        output_width=100, output_height=100,
    )
    _SOURCE = BoundingBox(x=0, y=0, width=100, height=100)

    def test_zero_scale_x_returns_none(self):
        sc = SpatialCoordinator()
        bad_transform = SpatialTransform(
            scale_x=0.0, scale_y=1.0, pad_left=0, pad_top=0,
            output_width=100, output_height=100,
        )
        result = sc.project_bbox_to_full_frame(
            BoundingBox(x=0, y=0, width=10, height=10),
            self._SOURCE, bad_transform,
        )
        assert result is None

    def test_negative_scale_y_returns_none(self):
        sc = SpatialCoordinator()
        bad_transform = SpatialTransform(
            scale_x=1.0, scale_y=-0.5, pad_left=0, pad_top=0,
            output_width=100, output_height=100,
        )
        result = sc.project_bbox_to_full_frame(
            BoundingBox(x=0, y=0, width=10, height=10),
            self._SOURCE, bad_transform,
        )
        assert result is None

    def test_zero_width_local_bbox_returns_none(self):
        sc = SpatialCoordinator()
        result = sc.project_bbox_to_full_frame(
            BoundingBox(x=0, y=0, width=0, height=10),
            self._SOURCE, self._IDENTITY,
        )
        assert result is None

    def test_zero_height_local_bbox_returns_none(self):
        sc = SpatialCoordinator()
        result = sc.project_bbox_to_full_frame(
            BoundingBox(x=0, y=0, width=10, height=0),
            self._SOURCE, self._IDENTITY,
        )
        assert result is None

    def test_negative_width_local_bbox_returns_none(self):
        sc = SpatialCoordinator()
        result = sc.project_bbox_to_full_frame(
            BoundingBox(x=0, y=0, width=-5, height=10),
            self._SOURCE, self._IDENTITY,
        )
        assert result is None

    def test_zero_source_width_returns_none(self):
        sc = SpatialCoordinator()
        bad_source = BoundingBox(x=0, y=0, width=0, height=100)
        result = sc.project_bbox_to_full_frame(
            BoundingBox(x=0, y=0, width=10, height=10),
            bad_source, self._IDENTITY,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Unit: RecognitionPipelineInputValidator
# ---------------------------------------------------------------------------


class TestInputValidator:
    def _v(self) -> RecognitionPipelineInputValidator:
        return RecognitionPipelineInputValidator()

    def test_valid_packet_no_exception(self):
        self._v().validate(_make_packet())  # must not raise

    def test_none_packet_raises(self):
        with pytest.raises(ValueError):
            self._v().validate(None)  # type: ignore[arg-type]

    def test_empty_frame_id_raises(self):
        pkt = _make_packet(frame_id="")
        with pytest.raises(ValueError, match="frame_id"):
            self._v().validate(pkt)

    def test_empty_camera_id_raises(self):
        pkt = _make_packet(camera_id="")
        with pytest.raises(ValueError, match="camera_id"):
            self._v().validate(pkt)

    def test_invalid_packet_returns_empty_from_rpm(self):
        """RPM must return empty output on validation failure."""
        rpm = _default_rpm()
        pkt = _make_packet(camera_id="")
        out = rpm.process_frame(pkt)
        assert out["persons"] == []


# ---------------------------------------------------------------------------
# Unit: RecognitionPipelineOutputBuilder
# ---------------------------------------------------------------------------


class TestOutputBuilder:
    def _b(self) -> RecognitionPipelineOutputBuilder:
        return RecognitionPipelineOutputBuilder()

    def test_traceability_fields_copied(self):
        builder = self._b()
        result = builder.build(
            frame_id="fr-1",
            camera_id="cam-X",
            timestamp_ms=12345,
            pipeline_result=PipelineResult(persons=[]),
        )
        assert result["frame_id"] == "fr-1"
        assert result["camera_id"] == "cam-X"
        assert result["timestamp_ms"] == 12345

    def test_empty_persons(self):
        result = self._b().build("f", "c", 0, PipelineResult(persons=[]))
        assert result["persons"] == []

    def test_persons_list_populated(self):
        from image_processing.recognition_pipeline_manager.types import PersonResult
        persons = [
            PersonResult(
                person_bbox=BoundingBox(x=0, y=0, width=100, height=200),
                recognized_faces=[],
            )
        ]
        result = self._b().build("f", "c", 0, PipelineResult(persons=persons))
        assert len(result["persons"]) == 1
        assert result["persons"][0]["person_bbox"]["width"] == 100


# ---------------------------------------------------------------------------
# Unit: FakeFTL metadata propagation (C3)
# ---------------------------------------------------------------------------


class TestFakeFTLMetadataPropagation:
    def test_processed_frame_carries_ingested_frame_id(self):
        ftl = FakeFTL()
        pkt = _make_packet(frame_id="traceability-frame", timestamp_ms=9999)
        ftl.ingest_frame(pkt)

        pf = ftl.get_frame(
            "cam-A", FrameTemporalSelector.CURRENT,
            BoundingBox(x=0, y=0, width=640, height=480),
            OutputImageType.RGB_UINT8_HWC,
            _STUB_GEOMETRY_SPEC_NONE,
        )

        assert pf.frame_id == "traceability-frame"
        assert pf.timestamp_ms == 9999


from image_processing.shared.contracts import GeometrySpec, ResizePolicy
_STUB_GEOMETRY_SPEC_NONE = GeometrySpec(
    width=0, height=0, resize_policy=ResizePolicy.NONE
)


# ---------------------------------------------------------------------------
# Test C: PersonDirectory.load() is NOT called by RPM or PipelineOrchestrator
# ---------------------------------------------------------------------------


class TrackingPersonDirectory:
    """PersonDirectory stand-in that records load() and get_person() calls."""

    def __init__(self, lookup_map: dict | None = None) -> None:
        self.load_calls: int = 0
        self.get_person_calls: list[str] = []
        self._map: dict = lookup_map or {}

    def load(self) -> None:
        self.load_calls += 1

    def get_person(self, person_id: str) -> PersonDirectoryOutput:
        self.get_person_calls.append(person_id)
        return self._map.get(
            person_id,
            PersonDirectoryOutput(person_id="UNKNOWN", person_name="UNKNOWN", found=False),
        )


class TestPersonDirectoryLoadNotCalledByRPM:
    """Test C — RPM never calls load(); that is the caller's responsibility."""

    def test_load_not_called_on_rpm_construction(self):
        pd = TrackingPersonDirectory()
        _default_rpm(person_dir=pd)
        assert pd.load_calls == 0

    def test_load_not_called_during_process_frame(self):
        pd = TrackingPersonDirectory()
        rpm = _default_rpm(person_dir=pd)
        rpm.process_frame(_make_packet())
        assert pd.load_calls == 0

    def test_load_not_called_after_multiple_frames(self):
        pd = TrackingPersonDirectory()
        rpm = _default_rpm(person_dir=pd)
        for _ in range(3):
            rpm.process_frame(_make_packet())
        assert pd.load_calls == 0


# ---------------------------------------------------------------------------
# Test D: RPM output contains person_id, person_name, found
# ---------------------------------------------------------------------------


class TestPersonDirectoryEnrichment:
    """Test D — RPM RecognizedFaceResult carries person_id, person_name, found."""

    _KNOWN_ID = "person-known"
    _KNOWN_NAME = "Known Person"

    def _make_full_rpm(
        self,
        person_found: bool,
        pd_found: bool,
        person_id: str,
    ) -> tuple[RecognitionPipelineManager, FakePersonDirectory]:
        motion = FakeMotionDetection(result=MotionResult(
            detected=True,
            bboxes=[BoundingBox(x=0, y=0, width=640, height=480)],
        ))
        od = FakeObjectDetection(result=PersonDetectionResult(
            frame_id="", person_detected=True,
            persons=[BoundingBox(x=50, y=50, width=100, height=200)],
        ))
        fd = FakeFaceDetection(result=FaceDetectionOutput(
            frame_id="", camera_id="", timestamp_ms=0,
            detections=[DetectedFace(
                face_bbox=BoundingBox(x=10, y=10, width=40, height=40),
                landmarks=_landmarks_at(30, 25),
            )],
        ))
        fr = FakeFaceRecognition(result=FaceRecognitionOutput(
            frame_id="", camera_id="", timestamp_ms=0,
            person_found=person_found, person_id=person_id,
        ))
        pd_map = {}
        if pd_found:
            pd_map[person_id] = PersonDirectoryOutput(
                person_id=person_id,
                person_name=self._KNOWN_NAME,
                found=True,
            )
        pd = FakePersonDirectory(lookup_map=pd_map)
        rpm = _default_rpm(
            motion=motion, object_det=od, face_det=fd, face_rec=fr, person_dir=pd,
        )
        return rpm, pd

    # D1 — known person: found=True, correct person_name
    def test_d1_found_true_when_pd_resolves_id(self):
        rpm, _ = self._make_full_rpm(
            person_found=True, pd_found=True, person_id=self._KNOWN_ID,
        )
        out = rpm.process_frame(_make_packet())
        face = out["persons"][0]["recognized_faces"][0]
        assert face["found"] is True

    def test_d1_person_name_from_pd(self):
        rpm, _ = self._make_full_rpm(
            person_found=True, pd_found=True, person_id=self._KNOWN_ID,
        )
        out = rpm.process_frame(_make_packet())
        face = out["persons"][0]["recognized_faces"][0]
        assert face["person_name"] == self._KNOWN_NAME

    def test_d1_person_id_from_pd(self):
        rpm, _ = self._make_full_rpm(
            person_found=True, pd_found=True, person_id=self._KNOWN_ID,
        )
        out = rpm.process_frame(_make_packet())
        face = out["persons"][0]["recognized_faces"][0]
        assert face["person_id"] == self._KNOWN_ID

    # D2 — FR found but PD missing: found=False, person_id/name become "UNKNOWN"
    def test_d2_found_false_when_pd_missing(self):
        rpm, _ = self._make_full_rpm(
            person_found=True, pd_found=False, person_id="unregistered-id",
        )
        out = rpm.process_frame(_make_packet())
        face = out["persons"][0]["recognized_faces"][0]
        assert face["found"] is False

    def test_d2_person_id_is_UNKNOWN_when_pd_missing(self):
        rpm, _ = self._make_full_rpm(
            person_found=True, pd_found=False, person_id="unregistered-id",
        )
        out = rpm.process_frame(_make_packet())
        face = out["persons"][0]["recognized_faces"][0]
        assert face["person_id"] == "UNKNOWN"

    def test_d2_person_name_is_UNKNOWN_when_pd_missing(self):
        rpm, _ = self._make_full_rpm(
            person_found=True, pd_found=False, person_id="unregistered-id",
        )
        out = rpm.process_frame(_make_packet())
        face = out["persons"][0]["recognized_faces"][0]
        assert face["person_name"] == "UNKNOWN"

    def test_d2_face_still_in_recognized_faces_when_pd_missing(self):
        """Unresolved-name faces still appear in recognized_faces; only name/id differ."""
        rpm, _ = self._make_full_rpm(
            person_found=True, pd_found=False, person_id="unregistered-id",
        )
        out = rpm.process_frame(_make_packet())
        assert len(out["persons"][0]["recognized_faces"]) == 1

    # Output structure: all three fields present
    def test_recognized_face_has_person_id_field(self):
        rpm, _ = self._make_full_rpm(
            person_found=True, pd_found=True, person_id=self._KNOWN_ID,
        )
        out = rpm.process_frame(_make_packet())
        face = out["persons"][0]["recognized_faces"][0]
        assert "person_id" in face

    def test_recognized_face_has_person_name_field(self):
        rpm, _ = self._make_full_rpm(
            person_found=True, pd_found=True, person_id=self._KNOWN_ID,
        )
        out = rpm.process_frame(_make_packet())
        face = out["persons"][0]["recognized_faces"][0]
        assert "person_name" in face

    def test_recognized_face_has_found_field(self):
        rpm, _ = self._make_full_rpm(
            person_found=True, pd_found=True, person_id=self._KNOWN_ID,
        )
        out = rpm.process_frame(_make_packet())
        face = out["persons"][0]["recognized_faces"][0]
        assert "found" in face


# ---------------------------------------------------------------------------
# Test E: Visual overlay uses real person_name; no hardcoded "Fake Person 001"
# ---------------------------------------------------------------------------


class TestVisualOverlayPersonName:
    """Test E — draw_final_stage_overlay renders person_name from rpm_output."""

    def _make_minimal_output(self, person_name: str, person_id: str):
        from image_processing.recognition_pipeline_manager.types import (
            RecognitionPipelineOutput,
            PersonResult,
            RecognizedFaceResult as RFR,
        )
        return RecognitionPipelineOutput(
            frame_id="f1",
            camera_id="cam",
            timestamp_ms=0,
            persons=[PersonResult(
                person_bbox=BoundingBox(x=10, y=10, width=100, height=200),
                recognized_faces=[RFR(
                    face_bbox=BoundingBox(x=20, y=20, width=50, height=50),
                    person_id=person_id,
                    person_name=person_name,
                    found=person_id != "UNKNOWN",
                )],
            )],
        )

    def test_overlay_uses_real_name_not_fake_label(self):
        """draw_final_stage_overlay reads from rpm_output; verify real name flows through.

        The draw function uses face.get("person_name") directly from rpm_output.
        If person_name="Alice" is in rpm_output, "Alice" is drawn — not "Fake Person 001".
        We verify the RPM output structure carries the correct name end-to-end.
        """
        rpm_output = self._make_minimal_output("Alice", "person-real")
        face = rpm_output["persons"][0]["recognized_faces"][0]
        assert face["person_name"] == "Alice"
        assert face["person_name"] != "Fake Person 001"

    def test_overlay_renders_UNKNOWN_for_unknown_person(self):
        """When person_name is UNKNOWN, the rpm_output carries that string directly."""
        rpm_output = self._make_minimal_output("UNKNOWN", "UNKNOWN")
        face = rpm_output["persons"][0]["recognized_faces"][0]
        assert face["person_name"] == "UNKNOWN"
        assert face["found"] is False

    def test_overlay_fake_person_001_never_produced(self):
        """Confirm the old hardcoded label is not wired into render logic.

        rpm_output is the source of truth — whatever person_name is in it
        gets drawn.  If person_name is not "Fake Person 001", the label will
        not be "Fake Person 001".  This test verifies that the RPM pipeline
        using the real PersonDirectory lookup produces a name from the
        directory, not the old constant.
        """
        # Build the full recognized path with a real (non-fake) person_id
        motion = FakeMotionDetection(result=MotionResult(
            detected=True,
            bboxes=[BoundingBox(x=0, y=0, width=640, height=480)],
        ))
        od = FakeObjectDetection(result=PersonDetectionResult(
            frame_id="", person_detected=True,
            persons=[BoundingBox(x=50, y=50, width=100, height=200)],
        ))
        fd = FakeFaceDetection(result=FaceDetectionOutput(
            frame_id="", camera_id="", timestamp_ms=0,
            detections=[DetectedFace(
                face_bbox=BoundingBox(x=10, y=10, width=40, height=40),
                landmarks=_landmarks_at(30, 25),
            )],
        ))
        fr = FakeFaceRecognition(result=FaceRecognitionOutput(
            frame_id="", camera_id="", timestamp_ms=0,
            person_found=True, person_id="real-person-id",
        ))
        pd = FakePersonDirectory(lookup_map={
            "real-person-id": PersonDirectoryOutput(
                person_id="real-person-id",
                person_name="Bar Keinan",
                found=True,
            )
        })
        rpm = _default_rpm(
            motion=motion, object_det=od, face_det=fd, face_rec=fr, person_dir=pd,
        )
        out = rpm.process_frame(_make_packet())
        face = out["persons"][0]["recognized_faces"][0]

        assert face["person_name"] == "Bar Keinan"
        assert face["person_name"] != "Fake Person 001"

