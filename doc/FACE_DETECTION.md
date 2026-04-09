# Face Detection Module Specification

## 1. Purpose

The Face Detection module is responsible only for detecting faces inside an already prepared person ROI image.

The module does not crop images, does not detect persons, and does not perform face recognition or identity matching. Its responsibility starts after upstream processing has already produced person-specific ROI images and ends when the module returns accepted face locations in full-frame coordinates.

The module does not describe or manage any external system responsibilities. External components such as the upstream person ROI preparation pipeline are treated as black boxes.

## 2. Input

### 2.1 Input Responsibility Boundary

The module receives pre-cropped person images. These person ROIs are produced outside the module.

The preparation done outside the module includes:

- cropping each person ROI from the original frame
- color conversion
- layout conversion
- dtype conversion
- normalization

The representation produced outside the module matches the configured detector input contract exactly. The Face Detection module does not perform cropping, color conversion, layout conversion, dtype conversion, normalization, or any other image preprocessing.

The module processes exactly one person ROI per invocation. If multiple people exist in the source frame, upstream IPS is responsible for splitting the frame into individual person ROIs and invoking the module separately for each one. The module is stateless per invocation.

### 2.2 Input Structure

```text
struct BoundingBox {
    int32 x;
    int32 y;
    int32 width;
    int32 height;
}

struct FaceDetectionInput {
    uint64 frame_id;
    string camera_id;
    uint64 timestamp_ms;
    Image roi_image;
    BoundingBox roi_bbox_frame;
}
```

- `frame_id` identifies the source frame for traceability.
- `camera_id` identifies the source camera.
- `timestamp_ms` is the capture timestamp in milliseconds.
- `roi_image` is the prepared person ROI image.
- `roi_bbox_frame` is the bounding box of that ROI in the original frame coordinate system.

### 2.3 ROI Image Contract

The `roi_image` must already match the configured detector input contract before inference starts.

Default SCRFD-oriented contract:

- color format: `RGB`
- layout: `HWC`
- dtype: `uint8`
- value range: `[0, 255]`

The exact detector contract is configuration-defined, but the module expects the ROI to be consistent with that contract. The module validates the incoming ROI image shape, layout compatibility, dtype, and range assumptions before sending it to the detector engine.

The ROI image therefore arrives fully prepared. The Face Detection module does not perform any image preprocessing. Any minimal runtime-specific adaptation required for inference, such as wrapping the validated image into the backend's tensor type or adding a batch dimension, is handled internally by `FaceDetectorEngine` and does not modify the image data.

## 3. Configuration

### 3.1 Configuration Structure

```text
struct DetectorInputContract {
    string color_format;
    string layout;
    string dtype;
    string value_range;
}

struct FaceDetectionConfig {
    float32 confidence_threshold;
    DetectorInputContract detector_input_contract;
}
```

- `confidence_threshold` is the minimum confidence score a raw detection must meet to be accepted as a valid face detection.
- `detector_input_contract` defines the expected properties of the incoming ROI image, including color format, layout, dtype, and value range.

### 3.2 Configuration Loading Behavior

Configuration is loaded exactly once during module initialization. `FaceDetectionModule` reads the configuration source and injects the relevant values into internal subcomponents:

- `confidence_threshold` is injected into `FaceDetectionPostprocessor` at construction time.
- `detector_input_contract` is injected into `FaceDetectionInputValidator` at construction time.

Once loaded, configuration is immutable for the entire module lifecycle. It is reused unchanged across all invocations. Configuration is not reloaded per call and is not part of `FaceDetectionInput` or any other per-call runtime input.

## 4. AI Model Usage

The module depends on `FaceDetectorEngine`, an internal detector abstraction, not on any specific model implementation. This keeps the public API model-agnostic.

SCRFD is the default implementation of `FaceDetectorEngine`. It can be replaced by another face detector without changing `FaceDetectionInput`, `FaceDetectionOutput`, or the calling code.

After validation, the module sends the validated `roi_image` directly to the configured `FaceDetectorEngine`.

The engine returns raw face detections in ROI coordinates. A raw detection may contain:

- a face bounding box relative to the ROI image
- a confidence score produced by the detector
- facial landmarks relative to the ROI image

All raw outputs, confidence scores, rejected detections, and backend-specific artifacts remain strictly internal to the module and are never exposed through the public API.

## 5. Internal Pipeline

The Face Detection module is not a single flat processing unit. It is an orchestrated module composed of internal subcomponents with separate responsibilities.

### 5.1 FaceDetectionModule

`FaceDetectionModule` is the orchestration layer only. It owns no detection logic.

Its responsibilities are:

- receive `FaceDetectionInput`
- invoke internal subcomponents in the correct order
- process the single ROI provided in the input
- pass projected detections to `FaceDetectionOutputBuilder`

During initialization, `FaceDetectionModule` is responsible for loading the module configuration and wiring each internal subcomponent with its required settings, including injecting the `confidence_threshold` into `FaceDetectionPostprocessor`.

`FaceDetectionModule` must not embed validation, inference, postprocessing/acceptance, coordinate projection, or output assembly logic directly. Each of those responsibilities belongs to a dedicated internal component.

### 5.2 FaceDetectionInputValidator

`FaceDetectionInputValidator` is responsible only for input validation. It validates the input fields and the ROI before inference begins.

Validation rules:

- `frame_id` must exist
- `camera_id` must exist and be non-empty
- `timestamp_ms` must exist
- `roi_image` must exist
- `roi_bbox_frame` must exist
- `roi_bbox_frame.width` and `roi_bbox_frame.height` must be greater than zero
- `roi_image` must match the configured detector input contract

`FaceDetectionInputValidator` does not perform any image preprocessing and makes no acceptance decisions.

### 5.3 FaceDetectorEngine

`FaceDetectorEngine` is the internal detector runtime abstraction used by the module to run AI inference.

Its responsibilities are:

- accept the validated ROI image directly from the orchestrator
- run detector inference
- return raw face detections in ROI coordinates, including bounding boxes, confidence scores, and landmarks

`FaceDetectorEngine` does not apply confidence thresholds, filter detections, or project coordinates. It returns raw output only.

`SCRFDFaceDetector` is the current default implementation of `FaceDetectorEngine`. The module depends on the `FaceDetectorEngine` abstraction, not on SCRFD directly, so a different detector can be substituted in the future without changing `FaceDetectionInput`, `FaceDetectionOutput`, or any other part of the public API.

`FaceDetectorEngine` may return model-specific raw landmarks whose format, count, or naming vary by detector. These raw landmark formats are strictly internal and are never exposed through the public API. Any `FaceDetectorEngine` implementation must return landmarks that can be mapped to the canonical 5-point `FaceLandmarks` representation (`left_eye`, `right_eye`, `nose`, `mouth_left`, `mouth_right`). If a detector does not natively provide the canonical landmarks, an internal mapping or approximation must be applied to produce the required canonical representation.

### 5.4 FaceDetectionPostprocessor

`FaceDetectionPostprocessor` is responsible for interpreting raw detector output and deciding which detections are accepted.

Its responsibilities are:

- interpret raw face detections produced by the engine
- apply `confidence_threshold` to discard low-confidence detections
- discard malformed or invalid boxes
- apply detector-specific suppression or refinement rules if required
- determine which detections are accepted and passed to coordinate projection

The `confidence_threshold` is read from the module configuration file exactly once during module initialization. It is injected into `FaceDetectionPostprocessor` at construction time and stored as immutable internal state. It is reused unchanged for every invocation. The `confidence_threshold` is not part of `FaceDetectionInput` or any other per-call runtime input.

This component is the only place inside the module that decides whether a raw detector result becomes an accepted face detection. `FaceDetectionPostprocessor` does not project coordinates and does not construct output structures.

### 5.5 FaceCoordinateProjector

`FaceCoordinateProjector` is responsible for the geometric transformation that converts accepted detections from ROI-local coordinates into full-frame coordinates.

Its responsibilities are:

- offset each accepted face bounding box by `roi_bbox_frame.x` and `roi_bbox_frame.y` to produce `face_bbox_frame` in frame coordinates
- map landmarks from ROI space into full-frame space using the same ROI offset
- ensure all coordinates passed to `FaceDetectionOutputBuilder` are expressed in frame coordinates

`FaceCoordinateProjector` receives only detections already accepted by `FaceDetectionPostprocessor`. It does not apply acceptance logic.

### 5.6 FaceDetectionOutputBuilder

`FaceDetectionOutputBuilder` is responsible for constructing the final `FaceDetectionOutput` from already projected detections and preserved input metadata.

Its responsibilities are:

- create `DetectedFace` entries from projected detections
- map projected raw landmarks into the canonical `FaceLandmarks` structure
- assemble the final `FaceDetectionOutput`
- copy `frame_id`, `camera_id`, and `timestamp_ms` from the input for traceability

`FaceDetectionOutputBuilder` acts as the internal adaptation layer that decouples model-specific landmark formats from the public contract. It receives whatever landmark format the detector produced (after coordinate projection by `FaceCoordinateProjector`) and produces the canonical 5-point `FaceLandmarks` output. Model-specific landmark formats do not escape this component.

`FaceDetectionOutputBuilder` receives only frame-space coordinates from `FaceCoordinateProjector`. It does not project coordinates and does not apply acceptance logic.

### 5.7 End-to-End Processing Flow

For one invocation of `detect_faces`, the internal pipeline follows this order:

**validation → inference → postprocessing/acceptance → coordinate projection → output construction**

1. `FaceDetectionModule` receives `FaceDetectionInput`.
2. `FaceDetectionModule` calls `FaceDetectionInputValidator` to validate the input.
3. `FaceDetectionModule` sends the validated `roi_image` to `FaceDetectorEngine`.
4. `FaceDetectorEngine` runs inference and returns raw detections in ROI coordinates.
5. `FaceDetectionModule` sends the raw detections to `FaceDetectionPostprocessor`.
6. `FaceDetectionPostprocessor` applies acceptance rules and returns only accepted detections. Rejected detections are discarded.
7. `FaceDetectionModule` sends the accepted detections and `roi_bbox_frame` to `FaceCoordinateProjector`.
8. `FaceCoordinateProjector` returns projected detections in full-frame coordinates.
9. `FaceDetectionModule` calls `FaceDetectionOutputBuilder` with frame metadata and projected detections.
10. `FaceDetectionOutputBuilder` constructs and returns the final `FaceDetectionOutput`.

All raw model outputs, confidence scores, rejected detections, and intermediate backend-specific artifacts remain internal to the module.

## 6. Internal Data Structures

All intermediate representations below are strictly internal and never exposed through the public API.

- **`RawFaceDetections`** — Raw bounding boxes, confidence scores, and model-specific landmarks in ROI-local coordinates. Produced by `FaceDetectorEngine`, consumed by `FaceDetectionPostprocessor`.
- **`AcceptedDetections`** — Detections that passed confidence threshold and validation checks, still in ROI-local coordinates. Produced by `FaceDetectionPostprocessor`, consumed by `FaceCoordinateProjector`.
- **`ProjectedDetections`** — Accepted detections with all coordinates transformed to full-frame space. Produced by `FaceCoordinateProjector`, consumed by `FaceDetectionOutputBuilder`.

## 7. Output

### 7.1 Output Structure

```text
struct Point {
    int32 x;
    int32 y;
}

struct FaceLandmarks {
    Point left_eye;
    Point right_eye;
    Point nose;
    Point mouth_left;
    Point mouth_right;
}

struct DetectedFace {
    BoundingBox face_bbox_frame;
    FaceLandmarks landmarks;
}

struct FaceDetectionOutput {
    uint64 frame_id;
    string camera_id;
    uint64 timestamp_ms;
    vector<DetectedFace> detections;
}
```

### 7.2 Output Semantics

- `frame_id`, `camera_id`, and `timestamp_ms` are copied from the input for traceability.
- `detections` contains only accepted face detections.
- `face_bbox_frame` is always expressed in full-frame coordinates, not ROI coordinates.
- `landmarks` are always present in every `DetectedFace`. The module exposes a canonical 5-point landmark representation: `left_eye`, `right_eye`, `nose`, `mouth_left`, and `mouth_right`.

The `FaceLandmarks` structure is a fixed, model-agnostic public contract. It does not change based on the underlying detector. Model-specific landmark formats are never exposed outside the module. The public API remains stable even if the detector engine is replaced.

The output must not expose detector confidence, raw inference tensors, rejection reasons, model-specific landmark formats, or any recognition-related data.

## 8. Acceptance Logic

`FaceDetectionPostprocessor` is responsible for deciding whether a raw detector result is accepted as a valid face detection.

Acceptance is internal and configuration-driven. Typical rules include:

- minimum confidence threshold loaded from module configuration during initialization
- optional suppression of duplicate overlapping face boxes
- bounding box sanity checks
- landmark validation

Detector confidence is strictly internal. It is used by `FaceDetectionPostprocessor` when applying acceptance rules but is never returned in `FaceDetectionOutput`.

Only accepted detections proceed to coordinate projection. If no detections survive postprocessing, the module returns an empty `detections` list.

## 9. Module Interface

### 9.1 Public API

The public module contract is:

```text
detect_faces(input: FaceDetectionInput) -> FaceDetectionOutput
```

The public API must not expose detector confidence, raw inference tensors, rejection reasons, internal state, or any model-specific data. The contract is stable regardless of which detector engine is configured.

### 9.2 Detector Abstraction

The detector backend is abstracted behind an internal interface:

```text
interface FaceDetectorEngine {
    detect(prepared_roi: PreparedROI) -> RawFaceDetections
}
```

Default implementation:

```text
class SCRFDFaceDetector implements FaceDetectorEngine
```

The Face Detection module depends on `FaceDetectorEngine`, not on SCRFD directly. `SCRFDFaceDetector` is the default implementation of that interface. Another detector engine can replace it in the future without changing the public API or the responsibilities of the surrounding internal components.

## 10. Error Handling

All errors are handled internally. The caller never receives exceptions or partial output. In every failure scenario, the module returns a structurally valid `FaceDetectionOutput`.

- **Invalid input** (missing fields, zero-size ROI, contract mismatch) → return valid output with empty `detections`
- **Inference failure** (engine runtime error) → catch internally, return valid output with empty `detections`
- **Empty detection result** (zero detections or all rejected) → return valid output with empty `detections`; normal operation

## 11. Lifecycle

### 11.1 Initialization

- Load `FaceDetectionConfig` from the configuration source
- Wire subcomponents: `FaceDetectionInputValidator` (with `detector_input_contract`), `FaceDetectorEngine` (with configured backend + model loading), `FaceDetectionPostprocessor` (with `confidence_threshold`), `FaceCoordinateProjector`, `FaceDetectionOutputBuilder`

### 11.2 Per-Invocation

**validation → inference → postprocessing → coordinate projection → output construction**

Stateless per invocation. No state carries between calls. One person ROI per call.

## 12. Metrics / Observability

- `inference_time_ms` — `FaceDetectorEngine.detect()` duration
- `validation_time_ms` — `FaceDetectionInputValidator.validate()` duration
- `accepted_faces_count` — accepted detections per invocation
- `rejected_detections_count` — detections discarded by `FaceDetectionPostprocessor` per invocation

Metrics are internal and operational. Not part of the public API.

## 13. Class Diagram

```mermaid
classDiagram
    class FaceDetectionModule {
        +detect_faces(input: FaceDetectionInput) FaceDetectionOutput
    }

    class FaceDetectionInputValidator {
        +validate(input: FaceDetectionInput) void
    }

    class FaceDetectorEngine {
        <<interface>>
        +detect(prepared_roi: PreparedROI) RawFaceDetections
    }

    class SCRFDFaceDetector {
        +detect(prepared_roi: PreparedROI) RawFaceDetections
    }

    class FaceDetectionPostprocessor {
        +accept(raw_detections) AcceptedDetections
    }

    class FaceCoordinateProjector {
        +project(detections, roi_bbox_frame) ProjectedDetections
    }

    class FaceDetectionOutputBuilder {
        +build(frame_id, camera_id, timestamp_ms, detections) FaceDetectionOutput
    }

    class FaceDetectionInput {
        +frame_id: uint64
        +camera_id: string
        +timestamp_ms: uint64
        +roi_image: Image
        +roi_bbox_frame: BoundingBox
    }

    class FaceDetectionOutput {
        +frame_id: uint64
        +camera_id: string
        +timestamp_ms: uint64
        +detections: DetectedFace[]
    }

    class DetectedFace {
        +face_bbox_frame: BoundingBox
        +landmarks: FaceLandmarks
    }

    class FaceLandmarks {
        +left_eye: Point
        +right_eye: Point
        +nose: Point
        +mouth_left: Point
        +mouth_right: Point
    }

    class BoundingBox {
        +x: int32
        +y: int32
        +width: int32
        +height: int32
    }

    FaceDetectionModule --> FaceDetectionInputValidator : orchestrates
    FaceDetectionModule --> FaceDetectorEngine : orchestrates
    FaceDetectionModule --> FaceDetectionPostprocessor : orchestrates
    FaceDetectionModule --> FaceCoordinateProjector : orchestrates
    FaceDetectionModule --> FaceDetectionOutputBuilder : orchestrates
    SCRFDFaceDetector ..|> FaceDetectorEngine : implements
    FaceDetectionModule --> FaceDetectionInput : consumes
    FaceDetectionModule --> FaceDetectionOutput : returns
    FaceDetectionOutput --> DetectedFace : contains
    DetectedFace --> BoundingBox : uses
    DetectedFace --> FaceLandmarks : uses
```

## 14. Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant FaceDetectionModule
    participant FaceDetectionInputValidator
    participant FaceDetectorEngine
    participant FaceDetectionPostprocessor
    participant FaceCoordinateProjector
    participant FaceDetectionOutputBuilder

    Caller->>FaceDetectionModule: detect_faces(input)
    FaceDetectionModule->>FaceDetectionInputValidator: validate(input)
    FaceDetectionInputValidator-->>FaceDetectionModule: input valid
    FaceDetectionModule->>FaceDetectorEngine: detect(roi_image)
    FaceDetectorEngine->>FaceDetectorEngine: run inference
    FaceDetectorEngine-->>FaceDetectionModule: raw face detections
    FaceDetectionModule->>FaceDetectionPostprocessor: filter(raw detections)
    FaceDetectionPostprocessor-->>FaceDetectionModule: accepted detections
    FaceDetectionModule->>FaceCoordinateProjector: project(accepted detections, roi_bbox_frame)
    FaceCoordinateProjector-->>FaceDetectionModule: projected detections
    FaceDetectionModule->>FaceDetectionOutputBuilder: build(frame metadata, projected detections)
    FaceDetectionOutputBuilder-->>FaceDetectionModule: FaceDetectionOutput
    FaceDetectionModule-->>Caller: FaceDetectionOutput
```

## 15. Constraints

The Face Detection module must not:

- perform image preprocessing (cropping, color conversion, layout conversion, dtype conversion, normalization)
- perform face recognition
- perform identity matching
- detect persons
- manage state across frames
- expose raw detector outputs externally
- expose raw detector confidence externally

The module is limited to face detection inside a prepared person ROI and to returning accepted face locations in full-frame coordinates.
