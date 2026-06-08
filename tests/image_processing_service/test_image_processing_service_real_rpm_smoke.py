"""Real-RPM runtime smoke test for IPS with fake/replay frames."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
_TESTS_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_SRC_DIR, _TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from frame_ingestion_gateway.replay_helpers import (  # type: ignore[import-not-found]
    build_replay_message,
)
from image_processing.frame_ingestion_gateway import (  # type: ignore[import-not-found]
    InMemoryFrameIngressTransport,
)
from image_processing.image_processing_service import (  # type: ignore[import-not-found]
    ImageProcessingService,
)
from image_processing_service.ips_yaml_config_helpers import (  # type: ignore[import-not-found]
    build_ips_config_from_yaml,
)
from scripts.run_rpm_on_frames import build_runtime_components, load_yaml_config


class RecordingResultHandler:
    """Capture outputs emitted by IPS workers for smoke assertions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: list[tuple[str, str, dict[str, Any]]] = []

    def handle(self, camera_id: str, frame_packet, output: dict[str, Any]) -> None:
        """Record one processing output from a worker lane."""
        with self._lock:
            self.records.append((camera_id, frame_packet.frame_id, output))


def _wait_until(predicate, timeout_s: float = 20.0) -> bool:
    """Poll until a predicate is true or timeout is reached."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.integration
def test_real_rpm_runtime_smoke_pipeline_path() -> None:
    rpm_config_path = _PROJECT_ROOT / "config" / "image_processing_service" / "local_debug.yaml"
    ips_config_path = _PROJECT_ROOT / "config" / "image_processing_service" / "image_processing_service.yaml"
    person_json = _PROJECT_ROOT / "data" / "person_directory.json"
    rpm_config = load_yaml_config(rpm_config_path)

    components = build_runtime_components(
        rpm_config=rpm_config,
        person_json=person_json,
        frame_width=320,
        frame_height=240,
        real_motion_detection=True,
        real_object_detection=True,
        real_face_detection=True,
        real_face_recognition=True,
    )

    transport = InMemoryFrameIngressTransport()
    result_handler = RecordingResultHandler()
    service = ImageProcessingService(
        rpm=components.rpm,
        transport=transport,
        result_handler=result_handler,
    )
    service.configure(
        build_ips_config_from_yaml(
            camera_ids=["cam-real"],
            config_path=ips_config_path,
        )
    )

    service.start()
    try:
        now_ms = int(time.time() * 1000)
        message = build_replay_message(
            "frame_000008.jpg",
            frame_id="real-rpm-0",
            camera_id="cam-real",
            timestamp_ms=now_ms,
        )
        transport.inject_message(message)

        assert _wait_until(
            lambda: service._metrics.get_camera_counter("cam-real", "processed_per_camera") >= 1,
            timeout_s=25.0,
        )
        assert _wait_until(lambda: len(result_handler.records) >= 1, timeout_s=25.0)

        health = service.health()
        assert health.state == "RUNNING"
        assert health.worker_error_total == 0
        assert health.active_processing_workers == 1
        assert health.total_queued_frames == 0
        assert service._metrics.get_camera_counter("cam-real", "accepted_per_camera") >= 1
        assert service._metrics.get_camera_counter("cam-real", "dequeued_per_camera") >= 1
        assert service._metrics.get_camera_counter("cam-real", "processed_per_camera") >= 1
        assert service._runtime.gateway.health().frames_in_total >= 1
        assert service._runtime.gateway.health().frames_published_total >= 1

        frame_metrics: dict[str, Any] = components.rpm.get_last_frame_metrics()
        assert frame_metrics["total_ftl_calls_per_frame"] >= 1
        _, processed_frame_id, output = result_handler.records[0]
        assert processed_frame_id == "real-rpm-0"
        assert "persons" in output
        assert isinstance(output["persons"], list)
    finally:
        service.stop(drain=True)

    final_health = service.health()
    assert final_health.state in {"STOPPED", "DEGRADED"}
    assert final_health.active_processing_workers == 0
