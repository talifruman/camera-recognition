from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np


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


@dataclass(slots=True)
class PersonDetectionConfig:
    model_path: str = "yolo11m.pt"
    person_confidence_threshold: float = 0.35
    nms_iou_threshold: float = 0.35
    inference_backend: str = "ultralytics"
    enable_bbox_validation: bool = True
    max_detections: int = 100
    optional_frame_skip: int = 0


@dataclass(slots=True)
class RawDetection:
    label: str
    confidence: float
    bbox: BoundingBox


class InputValidator:
    def validate_input(
        self,
        model_ready_input: Any,
        metadata: FrameMetadata,
        config: PersonDetectionConfig,
    ) -> None:
        del config

        if not metadata.get("camera_id"):
            raise ValueError("camera_id is required")
        if "frame_id" not in metadata:
            raise ValueError("frame_id is required")
        if metadata["width"] <= 0 or metadata["height"] <= 0:
            raise ValueError("width and height must be positive")
        if not isinstance(model_ready_input, np.ndarray):
            raise TypeError("model_ready_input must be a numpy.ndarray")
        if model_ready_input.ndim not in (2, 3):
            raise ValueError("model_ready_input must be a 2D or 3D numpy array")
        if model_ready_input.shape[0] != metadata["height"] or model_ready_input.shape[1] != metadata["width"]:
            raise ValueError("metadata width and height must match model_ready_input shape")


class UltralyticsInferenceEngine:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._loaded_model_path: str | None = None

    def _load_model(self, config: PersonDetectionConfig):
        if self._model is not None and self._loaded_model_path == config.model_path:
            return self._model

        try:
            YOLO = importlib.import_module("ultralytics").YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for real object detection. Install it in the workspace Python environment."
            ) from exc

        model_path = config.model_path
        if Path(model_path).suffix and Path(model_path).parent != Path(".") and not Path(model_path).exists():
            raise RuntimeError(f"Configured model_path does not exist: {model_path}")

        try:
            self._model = YOLO(model_path)
            self._loaded_model_path = model_path
        except Exception as exc:
            raise RuntimeError(f"Failed to load or download YOLO model '{model_path}': {exc}") from exc

        return self._model

    def infer(self, model_ready_input: np.ndarray, config: PersonDetectionConfig):
        model = self._load_model(config)

        try:
            return model.predict(
                source=model_ready_input,
                verbose=False,
                conf=0.001,
                iou=config.nms_iou_threshold,
                max_det=config.max_detections,
            )
        except Exception as exc:
            raise RuntimeError(f"Object detection inference failed: {exc}") from exc


class Postprocessor:
    def decode_and_nms(self, inference_output: Any, config: PersonDetectionConfig) -> list[RawDetection]:
        del config

        if isinstance(inference_output, list) and all(isinstance(item, RawDetection) for item in inference_output):
            return list(inference_output)

        raw_detections: list[RawDetection] = []
        for result in inference_output:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy_values = boxes.xyxy.cpu().tolist()
            confidence_values = boxes.conf.cpu().tolist()
            class_values = boxes.cls.cpu().tolist()
            names = getattr(result, "names", {})

            for xyxy, confidence, class_id in zip(xyxy_values, confidence_values, class_values):
                label = self._resolve_label(names, int(class_id))
                x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
                raw_detections.append(
                    RawDetection(
                        label=label,
                        confidence=float(confidence),
                        bbox={
                            "x": x1,
                            "y": y1,
                            "width": max(0, x2 - x1),
                            "height": max(0, y2 - y1),
                        },
                    )
                )

        return raw_detections

    def _resolve_label(self, names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)


class PersonFilteringLayer:
    def filter_persons(
        self,
        detections: list[RawDetection],
        config: PersonDetectionConfig,
    ) -> list[RawDetection]:
        filtered = [
            detection
            for detection in detections
            if detection.label == "person"
            and detection.confidence >= config.person_confidence_threshold
        ]
        return filtered[: config.max_detections]

    def validate_boxes(
        self,
        detections: list[RawDetection],
        metadata: FrameMetadata,
        config: PersonDetectionConfig,
    ) -> list[RawDetection]:
        if not config.enable_bbox_validation:
            return detections

        valid_detections: list[RawDetection] = []
        frame_width = metadata["width"]
        frame_height = metadata["height"]

        for detection in detections:
            bbox = detection.bbox
            x1 = max(0, bbox["x"])
            y1 = max(0, bbox["y"])
            x2 = min(frame_width, bbox["x"] + bbox["width"])
            y2 = min(frame_height, bbox["y"] + bbox["height"])
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue

            valid_detections.append(
                RawDetection(
                    label=detection.label,
                    confidence=detection.confidence,
                    bbox={"x": x1, "y": y1, "width": width, "height": height},
                )
            )

        return valid_detections


class ResultBuilder:
    def build(self, frame_id: int, detections: list[RawDetection]) -> PersonDetectionResult:
        persons = [detection.bbox for detection in detections]
        return {
            "frame_id": frame_id,
            "person_detected": len(persons) > 0,
            "persons": persons,
        }


class ObjectDetectionModule:
    def __init__(
        self,
        config: PersonDetectionConfig | None = None,
        inference_engine: Any | None = None,
    ) -> None:
        self._config = config or PersonDetectionConfig()
        self._input_validator = InputValidator()
        self._inference_engine = inference_engine or UltralyticsInferenceEngine()
        self._postprocessor = Postprocessor()
        self._person_filtering = PersonFilteringLayer()
        self._result_builder = ResultBuilder()

    def process(self, model_ready_input: Any, metadata: FrameMetadata) -> PersonDetectionResult:
        self._input_validator.validate_input(model_ready_input, metadata, self._config)
        inference_output = self._inference_engine.infer(model_ready_input, self._config)
        raw_detections = self._postprocessor.decode_and_nms(inference_output, self._config)
        person_detections = self._person_filtering.filter_persons(raw_detections, self._config)
        valid_detections = self._person_filtering.validate_boxes(
            person_detections,
            metadata,
            self._config,
        )
        return self._result_builder.build(metadata["frame_id"], valid_detections)


_DEFAULT_MODULE: ObjectDetectionModule | None = None


def process(model_ready_input: Any, metadata: FrameMetadata) -> PersonDetectionResult:
    global _DEFAULT_MODULE

    if _DEFAULT_MODULE is None:
        _DEFAULT_MODULE = ObjectDetectionModule()

    return _DEFAULT_MODULE.process(model_ready_input, metadata)