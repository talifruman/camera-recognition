# Shared Pipeline Contracts

This document defines the shared public contract types used across the image-processing pipeline.

All types defined here are the single authoritative definition. No module spec should duplicate these definitions inline. Each module spec should reference this document when using these types.

---

## 1. BoundingBox

`BoundingBox` is the single shared bounding-box structure used throughout the pipeline.

```text
struct BoundingBox {
    int x;       // top-left x in pixels
    int y;       // top-left y in pixels
    int width;   // box width in pixels
    int height;  // box height in pixels
}
```

### Rules

- Pixel coordinates. Origin is the top-left corner of the relevant image.
- `width` must be positive (`> 0`).
- `height` must be positive (`> 0`).
- `x` and `y` may be zero but must not be negative in normal pipeline use.

### Coordinate Space

`BoundingBox` does not encode its own coordinate space. The coordinate space must always be stated by context:

- **ROI-local BoundingBox** — coordinates are relative to the top-left corner of the ROI image.
- **Full-frame BoundingBox** — coordinates are relative to the top-left corner of the full camera frame.

Every field, function parameter, and return value that uses `BoundingBox` must explicitly state which coordinate space applies. Ambiguous usage is a spec error.

### Replacing `CanonicalBoundingBox`

`BoundingBox` replaces any prior usage of `CanonicalBoundingBox`. The two types are structurally identical: `{x, y, width, height}` in pixels, top-left origin. All module specs should use `BoundingBox`.

---

## 2. ResizePolicy

`ResizePolicy` defines the spatial resize behavior applied during image preparation.

```text
enum ResizePolicy {
    NONE,
    STRETCH,
    LETTERBOX,
    PRESERVE_ASPECT_RATIO
}
```

### Values

| Value | Behavior |
|-------|----------|
| `NONE` | No resize is applied. The image is returned at its natural cropped size. `width` and `height` in `GeometrySpec` are ignored. |
| `STRETCH` | The image is resized to exactly `(width, height)`. Aspect ratio is not preserved. |
| `LETTERBOX` | The image is resized to fit within `(width, height)` while preserving aspect ratio. Padding is added on the shorter axis to fill the target canvas. Padding color is implementation-defined (typically black). |
| `PRESERVE_ASPECT_RATIO` | The image is resized to fit within `(width, height)` while preserving aspect ratio. No padding is added; the output may be smaller than `(width, height)` on one axis. |

---

## 3. GeometrySpec

`GeometrySpec` defines the required spatial transformation applied during frame preparation.

```text
struct GeometrySpec {
    int          width;          // target width in pixels
    int          height;         // target height in pixels
    ResizePolicy resize_policy;  // spatial transformation behavior
}
```

### Rules

- If `resize_policy` is `NONE`, `width` and `height` are ignored. Implementations may omit or zero them.
- If `resize_policy` is `STRETCH` or `LETTERBOX` or `PRESERVE_ASPECT_RATIO`, `width` and `height` must be positive (`> 0`).
- `GeometrySpec` is consumed by the Frame Transformation Layer to prepare the output image geometry for a given `get_frame` call.
- Each pipeline stage declares its required `GeometrySpec` through `get_input_contract()`.

### FTL Extension Note

The Frame Transformation Layer may extend the `LETTERBOX` behavior with an optional padding color parameter. This is an FTL implementation detail and is not part of the shared `GeometrySpec` definition.

---

## 4. OutputImageType

`OutputImageType` is the shared enum describing the image pixel representation required by a pipeline stage. Each value uniquely identifies a combination of color format, layout, dtype, and value range.

```text
enum OutputImageType {
    GRAYSCALE_UINT8_HWC,
    RGB_UINT8_HWC,
    RGB_FLOAT32_HWC_NORMALIZED_0_TO_1,
    RGB_FLOAT32_HWC_NORMALIZED_MINUS1_TO_1
}
```

### Contract per Value

| Value | Color Format | Layout | dtype | Value Range | Channel Order | Shape Convention |
|-------|-------------|--------|-------|-------------|---------------|-----------------|
| `GRAYSCALE_UINT8_HWC` | Grayscale | HWC | uint8 | [0, 255] | Single channel (no channel dim, or last dim = 1) | (H, W) or (H, W, 1) |
| `RGB_UINT8_HWC` | RGB | HWC | uint8 | [0, 255] | R, G, B | (H, W, 3) |
| `RGB_FLOAT32_HWC_NORMALIZED_0_TO_1` | RGB | HWC | float32 | [0.0, 1.0] | R, G, B | (H, W, 3) |
| `RGB_FLOAT32_HWC_NORMALIZED_MINUS1_TO_1` | RGB | HWC | float32 | [-1.0, 1.0] | R, G, B | (H, W, 3) |

### Rules

- The Frame Transformation Layer maps each `OutputImageType` to exactly one hardcoded pixel-format conversion contract. No caller-supplied conversion parameters are accepted.
- `OutputImageType` defines pixel representation only. Geometry (resize, letterbox) is defined by `GeometrySpec`.
- Each pipeline stage must declare the `OutputImageType` it requires through `get_input_contract()`.

---

## 5. PipelineStageInputContract

`PipelineStageInputContract` is the shared structure returned by each pipeline stage through its `get_input_contract()` method.

```text
struct PipelineStageInputContract {
    OutputImageType output_image_type;
    GeometrySpec    geometry_spec;
}
```

### Purpose

- The `RecognitionPipelineManager` queries this contract during initialization for every configured pipeline stage.
- The `Frame Transformation Layer` uses the `output_image_type` and `geometry_spec` from this contract when preparing the model-ready image for each `get_frame()` call.
- The pipeline stage itself does not perform frame pulling, cropping, coordinate projection, or full-frame spatial mapping. All image preparation is owned by the Frame Transformation Layer.

### Rules

- Every pipeline stage interface (`ObjectDetectionInterface`, `FaceDetectionInterface`, `FaceRecognitionInterface`, `MotionDetectionInterface`) must expose:
  ```text
  get_input_contract() -> PipelineStageInputContract
  ```
- Pipeline stages must not perform frame pulling, cropping, coordinate projection, or full-frame spatial mapping. These responsibilities belong to `PipelineOrchestrator` and the Frame Transformation Layer.
- The `get_input_contract()` return value must be stable across invocations. It is queried once at initialization, not per frame.
- **Coordinate-space ownership**: Pipeline stages return detections relative to the coordinate space of the image they received, unless their own module spec explicitly defines otherwise. Full-frame projection is the responsibility of `RPM` / `PipelineOrchestrator`-level coordination logic. Shared contracts themselves do not assume that stages project to full-frame.

---

## 6. Image

`Image` is the single canonical public image type shared across the entire pipeline. All public APIs must use `Image` to carry pixel data. Raw `np.ndarray` must not be exposed directly in any public contract.

```text
struct Image {
    data:         np.ndarray  // the actual pixel buffer
    width:        uint32      // logical image width in pixels
    height:       uint32      // logical image height in pixels
    color_format: string      // pixel color representation
    layout:       string      // memory layout of the pixel buffer
    dtype:        string      // element data type of the pixel buffer
    value_range:  string      // nominal value range of pixel elements
}
```

### Field Definitions

| Field | Allowed values | Description |
|-------|----------------|-------------|
| `data` | any `np.ndarray` | The actual pixel buffer. Must not be `None`. |
| `width` | `> 0` | Logical image width in pixels. |
| `height` | `> 0` | Logical image height in pixels. |
| `color_format` | `RGB`, `BGR`, `GRAY` | Pixel color representation. |
| `layout` | `HWC`, `CHW` | Memory layout of the pixel buffer. |
| `dtype` | `uint8`, `float32` | Element data type of the pixel buffer. |
| `value_range` | `[0,255]`, `[0,1]`, `[-1,1]` | Nominal value range of pixel elements. |

### OutputImageType to Image Field Mapping

Each `OutputImageType` value (§4) maps to exactly one valid combination of `Image` metadata fields:

| OutputImageType | color_format | layout | dtype | value_range |
|----------------|--------------|--------|-------|-------------|
| `GRAYSCALE_UINT8_HWC` | `GRAY` | `HWC` | `uint8` | `[0,255]` |
| `RGB_UINT8_HWC` | `RGB` | `HWC` | `uint8` | `[0,255]` |
| `RGB_FLOAT32_HWC_NORMALIZED_0_TO_1` | `RGB` | `HWC` | `float32` | `[0,1]` |
| `RGB_FLOAT32_HWC_NORMALIZED_MINUS1_TO_1` | `RGB` | `HWC` | `float32` | `[-1,1]` |

### Rules

- `Image` is the single shared public image type. There must not be multiple public image wrapper types across the pipeline.
- `Image.data` must not be `None` and must contain valid pixel data.
- `Image.width` and `Image.height` are explicit contract fields — they are not derived from `Image.data.shape` alone.
- Public APIs must NOT expose raw `np.ndarray` directly. All image payloads crossing a public module boundary must be carried in an `Image` struct.
- Validators MAY compare `image.width` and `image.height` against `image.data.shape` for consistency verification.
- `Image.data.shape`, `Image.color_format`, `Image.layout`, `Image.dtype`, and `Image.value_range` must all be mutually consistent and must match the declared `OutputImageType` for the pipeline stage. An `Image` whose metadata fields are inconsistent or do not match the declared `OutputImageType` is invalid input.
- `BaseImage` is not a public or shared contract type. It is an FTL-internal concept only.
- `FrameBuffer` is not a public or shared contract type. It is an FTL-internal processing type only.

---

## 7. Point

```text
struct Point {
    int x;
    int y;
}
```

### Rules

- Pixel coordinates. The origin is the top-left corner of the relevant image.
- `Point` does not encode its own coordinate space. The coordinate space must always be documented by context — by the owning field, function parameter, or return-value description.
- Ambiguous usage of `Point` without a documented coordinate space is a spec error.

---

## 8. FaceLandmarks

`FaceLandmarks` is the canonical 5-point landmark structure used throughout the pipeline.

```text
struct FaceLandmarks {
    Point left_eye;
    Point right_eye;
    Point nose;
    Point mouth_left;
    Point mouth_right;
}
```

### Rules

- All five landmark fields are mandatory. A `FaceLandmarks` value with any field absent is invalid.
- `FaceLandmarks` does not encode its own coordinate space. The coordinate space must be documented by the owning field or context (for example, the `DetectedFace.landmarks` field in `FaceDetectionOutput` documents that landmarks are ROI-local relative to `roi_image`).
- Model-specific landmark formats must be mapped to this canonical representation internally before being exposed through any public API. Model-specific formats must not appear in any public output structure.
- `Point` fields inside `FaceLandmarks` follow the same coordinate-space rules as standalone `Point` values — coordinate space is inherited from the context of the `FaceLandmarks` value, not from the struct itself.

---

## 9. Coordinate Space Terminology

The following coordinate-space labels are used consistently across all module specs.

| Label | Meaning |
|-------|--------|
| `FULL_FRAME` | Coordinates are relative to the top-left corner of the full camera frame. |
| `ROI_LOCAL` | Coordinates are relative to the top-left corner of a specific ROI image that was supplied to a pipeline stage as input. |
| `CROP_LOCAL` | Coordinates are relative to the top-left corner of a specific cropped sub-region derived from a larger image. |

### Rules

- `BoundingBox`, `Point`, and `FaceLandmarks` do **not** store coordinate-space metadata internally. None of these structs carry a coordinate-space tag.
- Every field, function parameter, and return value that uses `BoundingBox`, `Point`, or `FaceLandmarks` must explicitly document which coordinate space applies.
- Ambiguous coordinate-space usage — any usage without an explicit coordinate-space declaration — is a spec error.
- Pipeline stages return detections in the coordinate space of the image they received as input, unless their own module spec explicitly defines otherwise.
- Full-frame projection (converting from `ROI_LOCAL` to `FULL_FRAME`) is the responsibility of `RPM` / `PipelineOrchestrator`-level coordination logic. Individual pipeline stages are not responsible for full-frame projection unless their module spec explicitly assigns that responsibility.

---

## 10. Cross-Module Usage Rules

1. Use `BoundingBox` everywhere. Do not define a separate box type in any module spec.
2. Always state the coordinate space when using `BoundingBox`, `Point`, or `FaceLandmarks` (ROI-local or full-frame). See §9 for terminology.
3. `OutputImageType` and `GeometrySpec` are pipeline contracts — they travel from each stage's `get_input_contract()` through RPM to the FTL's `get_frame()`.
4. `ResizePolicy` is part of `GeometrySpec`. Do not define a separate geometry enum in any module spec.
5. `PipelineStageInputContract` is the single initialization-time negotiation mechanism between RPM and each stage.
6. Use `Image` (as defined in §6) everywhere images are referenced in public APIs. Do not expose raw `np.ndarray` in any public module contract. Do not define a separate image wrapper type in any module spec unless it is documented as an internal-only type.
7. Use `Point` as defined in §7. Do not define a separate coordinate type in any module spec.
8. Use `FaceLandmarks` as defined in §8. Do not define a separate landmark struct in any module spec.
9. Full-frame projection is owned by `RPM` / `PipelineOrchestrator`. Individual pipeline stage specs must not claim full-frame projection as a stage responsibility unless explicitly justified.
