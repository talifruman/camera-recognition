from __future__ import annotations

from typing import Any, TypedDict


class BoundingBox(TypedDict):
    x: int
    y: int
    width: int
    height: int


class FrameMetadata(TypedDict):
    camera_id: str
    frame_id: int
    width: int
    height: int


class PersonDetectionResult(TypedDict):
    frame_id: int
    person_detected: bool
    persons: list[BoundingBox]


def _require_metadata(metadata: FrameMetadata) -> None:
    if not metadata.get("camera_id"):
        raise ValueError("camera_id is required")
    if "frame_id" not in metadata:
        raise ValueError("frame_id is required")
    if metadata["width"] <= 0 or metadata["height"] <= 0:
        raise ValueError("width and height must be positive")


def _build_box(width: int, height: int, x_ratio: float) -> BoundingBox:
    box_width = max(1, int(width * 0.22))
    box_height = max(1, int(height * 0.48))
    x = min(max(0, int(width * x_ratio)), max(0, width - box_width))
    y = min(max(0, int(height * 0.18)), max(0, height - box_height))
    return {
        "x": x,
        "y": y,
        "width": box_width,
        "height": box_height,
    }


def process(model_ready_input: Any, metadata: FrameMetadata) -> PersonDetectionResult:
    if model_ready_input is None:
        raise ValueError("model_ready_input is required")

    _require_metadata(metadata)

    persons = [
        _build_box(metadata["width"], metadata["height"], 0.14),
        _build_box(metadata["width"], metadata["height"], 0.64),
    ]

    return {
        "frame_id": metadata["frame_id"],
        "person_detected": True,
        "persons": persons,
    }