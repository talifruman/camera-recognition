# Shared Pipeline Contracts

This document defines the shared public contract types used across the image-processing pipeline.

All types defined here are the single authoritative definition. No module spec should duplicate these definitions inline. Each module spec should reference this document when using these types.

Runtime ownership model (overview):

- Runtime lifecycle ownership is defined by module specifications, not by shared contract types.
- Image Processing Service is the only top-level runtime lifecycle owner.
- Only Image Processing Service creates, owns, starts, stops, and supervises runtime threads.
- Frame Ingestion Gateway owns ingestion logic only; it does not own runtime lifecycle or runtime threads.
- RecognitionPipelineManager owns processing logic only; it does not own queues, queue pulling, runtime threads, or lifecycle management.
- Image Processing Service is the sole owner of service-wide DEGRADED/ERROR escalation.
- FramePacketSink is a publication boundary contract only; it does not imply queue ownership.
- Canonical enqueue rejection codes are expected to remain consistent across Gateway and IPS module specs.

---

## 1. BoundingBox

BoundingBox is the single shared bounding-box structure used throughout the pipeline.

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
- width must be positive (> 0).
- height must be positive (> 0).
- x and y may be zero but must not be negative in normal pipeline use.

### Coordinate Space

BoundingBox does not encode its own coordinate space. The coordinate space must always be stated by context:

- ROI-local BoundingBox: coordinates are relative to the top-left corner of the ROI image.
- Full-frame BoundingBox: coordinates are relative to the top-left corner of the full camera frame.

Every field, function parameter, and return value that uses BoundingBox must explicitly state which coordinate space applies. Ambiguous usage is a spec error.

### Replacing CanonicalBoundingBox

BoundingBox replaces any prior usage of CanonicalBoundingBox. The two types are structurally identical: {x, y, width, height} in pixels, top-left origin. All module specs should use BoundingBox.

---

## 2. ResizePolicy

ResizePolicy defines the spatial resize behavior applied during image preparation.

```text
enum ResizePolicy {
    NONE,
    LETTERBOX
}
```

### Values

| Value | Behavior |
|-------|----------|
| NONE | No resize is applied. The image is returned at its natural cropped size. width and height in GeometrySpec are ignored. Placeholder values 0/0 are valid for stage contracts. |
| LETTERBOX | The image is resized to fit within (width, height) while preserving aspect ratio. Padding is added on the shorter axis to fill the target canvas. Padding color is implementation-defined (typically black). |

---

## 3. GeometrySpec

GeometrySpec defines the required spatial transformation applied during frame preparation.

```text
struct GeometrySpec {
    int          width;          // target width in pixels
    int          height;         // target height in pixels
    ResizePolicy resize_policy;  // spatial transformation behavior
}
```

### Rules

- If resize_policy is NONE, width and height are ignored. width = 0 and height = 0 are valid placeholders. Implementations may also omit these values.
- If resize_policy is LETTERBOX, width and height must be positive (> 0).
- GeometrySpec is consumed by the Frame Transformation Layer to prepare the output image geometry for a given get_frame call.
- Each pipeline stage declares its required GeometrySpec through get_input_contract().

### FTL Extension Note

The Frame Transformation Layer may extend the LETTERBOX behavior with an optional padding color parameter. This is an FTL implementation detail and is not part of the shared GeometrySpec definition.

---

## 4. OutputImageType

OutputImageType is the shared enum describing the image pixel representation required by a pipeline stage. Each value uniquely identifies a combination of color format, layout, dtype, and value range.

```text
enum OutputImageType {
    GRAYSCALE_UINT8_HWC,
    RGB_UINT8_HWC
}
```

### Contract per Value

| Value | Color Format | Layout | dtype | Value Range | Channel Order | Shape Convention |
|-------|-------------|--------|-------|-------------|---------------|------------------|
| GRAYSCALE_UINT8_HWC | Grayscale | HWC | uint8 | [0, 255] | Single channel (no channel dim, or last dim = 1) | (H, W) or (H, W, 1) |
| RGB_UINT8_HWC | RGB | HWC | uint8 | [0, 255] | R, G, B | (H, W, 3) |

### Rules

- The Frame Transformation Layer maps each OutputImageType to exactly one hardcoded pixel-format conversion contract. No caller-supplied conversion parameters are accepted.
- OutputImageType defines pixel representation only. Geometry (resize, letterbox) is defined by GeometrySpec.
- Each pipeline stage must declare the OutputImageType it requires through get_input_contract().
- Shared OutputImageType values describe pipeline-level image formats only. Model-specific normalization must be performed inside the model-owning module.
- Face Recognition does not request normalized images from FTL. ArcFace normalization to [-1,1] is internal to ArcFaceEmbeddingEngine.

---

## 5. PipelineStageInputContract

PipelineStageInputContract is the shared structure returned by each pipeline stage through its get_input_contract() method.

```text
struct PipelineStageInputContract {
    OutputImageType output_image_type;
    GeometrySpec    geometry_spec;
}
```

### Purpose

- The RecognitionPipelineManager queries this contract during initialization for every configured pipeline stage.
- The Frame Transformation Layer uses the output_image_type and geometry_spec from this contract when preparing the model-ready image for each get_frame() call.
- The pipeline stage itself does not perform frame pulling, cropping, coordinate projection, or full-frame spatial mapping. All image preparation is owned by the Frame Transformation Layer.

### Rules

- Every pipeline stage interface (ObjectDetectionInterface, FaceDetectionInterface, FaceRecognitionInterface, MotionDetectionInterface) must expose:
  ```text
  get_input_contract() -> PipelineStageInputContract
  ```
- Pipeline stages must not perform frame pulling, cropping, coordinate projection, or full-frame spatial mapping. These responsibilities belong to PipelineOrchestrator and the Frame Transformation Layer.
- The get_input_contract() return value must be stable across invocations. It is queried once at initialization, not per frame.
- Coordinate-space ownership: Pipeline stages return detections relative to the coordinate space of the image they received, unless their own module spec explicitly defines otherwise. Full-frame projection is the responsibility of RPM / PipelineOrchestrator-level coordination logic. Shared contracts themselves do not assume that stages project to full-frame.

---

## 6. Image

Image is the single canonical public image type shared across the entire pipeline for processed image representations. Raw ndarray values must not be exposed directly in any public contract.

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
| data | any np.ndarray | The actual pixel buffer. Must not be None. |
| width | > 0 | Logical image width in pixels. |
| height | > 0 | Logical image height in pixels. |
| color_format | RGB, BGR, GRAY | Pixel color representation. |
| layout | HWC, CHW | Memory layout of the pixel buffer. |
| dtype | uint8, float32 | Element data type of the pixel buffer. |
| value_range | [0,255] | Nominal value range of pixel elements. |

### OutputImageType to Image Field Mapping

Each OutputImageType value (Section 4) maps to exactly one valid combination of Image metadata fields:

| OutputImageType | color_format | layout | dtype | value_range |
|----------------|--------------|--------|-------|-------------|
| GRAYSCALE_UINT8_HWC | GRAY | HWC | uint8 | [0,255] |
| RGB_UINT8_HWC | RGB | HWC | uint8 | [0,255] |

### Rules

- Image is a shared public image type for processed image contracts.
- Image.data must not be None and must contain valid pixel data.
- Image.width and Image.height are explicit contract fields and are not derived from Image.data.shape alone.
- Public APIs must not expose raw ndarray directly. All processed image payloads crossing a public module boundary must be carried in an Image struct.
- Validators may compare image.width and image.height against image.data.shape for consistency verification.
- Image.data.shape, Image.color_format, Image.layout, Image.dtype, and Image.value_range must be mutually consistent and must match the declared OutputImageType for the pipeline stage.
- Image.data must be treated as read-only by downstream consumers unless a module contract explicitly states otherwise.
- Consumers must not mutate Image.data in-place across module boundaries.
- Whether an implementation returns a defensive copy or a shared reference is an implementation detail.
- BaseImage is not a public or shared contract type. It is FTL-internal only.
- FrameBuffer is not a public or shared contract type. It is FTL-internal only.

### Image.data Ownership and Mutability Rationale

- The read-only consumer rule avoids forcing defensive copies at every module boundary.
- Keeping copy/reference behavior as an implementation detail preserves performance flexibility.
- Treating shared image buffers as immutable by contract keeps pipeline behavior safe and predictable.

---

## 7. FramePacket

FramePacket is the single canonical raw frame container at the ingestion boundary.

```text
struct FramePacket {
    string frame_id;
    string camera_id;
    uint64 timestamp_ms;
    int32  width;
    int32  height;
    string pixel_format;        // must be "RGB"
    string layout;              // must be "HWC"
    string dtype;               // must be "uint8"
    string value_range;         // must be "[0,255]"
    int32  num_color_channels;  // must be 3
    int32  bits_per_channel;    // must be 8
    string packing;             // must be "tightly_packed"
    bytes  image_bytes;         // raw RGB pixels, HWC order, tightly packed, no stride or row padding
}
```

### Canonical Input Rules

- pixel_format = RGB
- layout = HWC
- dtype = uint8
- value_range = [0,255]
- num_color_channels = 3
- bits_per_channel = 8
- packing = tightly_packed with no stride or row padding
- image_bytes contains raw unencoded RGB pixels

### FramePacket Clarifications

- FramePacket is immutable after construction.
- FramePacket.image_bytes ownership is immutable after construction.
- FramePacket must not contain transport metadata.
- FramePacket must not contain source_format, source_layout, codec hints, validation status, or rejection reasons.
- pixel_format describes the raw pixel layout of image_bytes.
- FramePacket.image_bytes is not replaced with Image.
- Other shared image abstractions may exist for internal processed image representations, but FramePacket remains the canonical raw frame container.
- Queue and worker boundaries must treat FramePacket as an immutable shared reference.
- Any stage requiring derived buffers must allocate a derived representation instead of mutating FramePacket or image_bytes.

---

## 8. FramePacketSink

FramePacketSink is the shared publication boundary for accepted FramePacket objects.

```text
interface FramePacketSink {
    EnqueueResult enqueue(frame_packet: FramePacket)
}

struct EnqueueResult {
    bool accepted;
    string reason_code; // optional, for internal metric classification only
}
```

### Rules

- Implementations must treat frame_packet as immutable.
- enqueue does not expose queue internals or scheduling internals.
- reason_code is optional and used only for internal classification.
- Queue enqueue and dequeue operations must not mutate frame_packet fields or image_bytes.
- Immutability requirements apply across queue boundaries and worker boundaries.
- In deployed runtime, FramePacketSink implementation ownership belongs to Image Processing Service.
- Frame Ingestion Gateway publishes to FramePacketSink but does not own sink runtime lifecycle, queue internals, or queue scheduling.

---

## 9. Point

```text
struct Point {
    int x;
    int y;
}
```

### Rules

- Pixel coordinates. The origin is the top-left corner of the relevant image.
- Point does not encode its own coordinate space. The coordinate space must always be documented by context.
- Ambiguous usage of Point without a documented coordinate space is a spec error.

---

## 10. FaceLandmarks

FaceLandmarks is the canonical 5-point landmark structure used throughout the pipeline.

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

- All five landmark fields are mandatory.
- FaceLandmarks does not encode its own coordinate space. The coordinate space must be documented by the owning field or context.
- Model-specific landmark formats must be mapped to this canonical representation internally before exposure through any public API.
- Point fields inside FaceLandmarks follow the same coordinate-space rules as standalone Point values.

---

## 11. Coordinate Space Terminology

The following coordinate-space labels are used consistently across all module specs.

| Label | Meaning |
|-------|--------|
| FULL_FRAME | Coordinates are relative to the top-left corner of the full camera frame. |
| ROI_LOCAL | Coordinates are relative to the top-left corner of a specific ROI image supplied to a pipeline stage. |
| CROP_LOCAL | Coordinates are relative to the top-left corner of a specific cropped sub-region derived from a larger image. |

### Rules

- BoundingBox, Point, and FaceLandmarks do not store coordinate-space metadata internally.
- Every field, function parameter, and return value that uses BoundingBox, Point, or FaceLandmarks must explicitly document which coordinate space applies.
- Ambiguous coordinate-space usage is a spec error.
- Pipeline stages return detections in the coordinate space of the image they received as input, unless their module spec explicitly defines otherwise.
- Full-frame projection is the responsibility of RPM / PipelineOrchestrator-level coordination logic.

---

## 12. Cross-Module Usage Rules

1. Use BoundingBox everywhere. Do not define a separate box type in any module spec.
2. Always state the coordinate space when using BoundingBox, Point, or FaceLandmarks (ROI-local or full-frame). See Section 11.
3. OutputImageType and GeometrySpec are pipeline contracts that travel from each stage get_input_contract() through RPM to FTL get_frame().
4. ResizePolicy is part of GeometrySpec. Do not define a separate geometry enum in any module spec.
5. PipelineStageInputContract is the single initialization-time negotiation mechanism between RPM and each stage.
6. Use Image (Section 6) for processed image public contracts. Do not expose raw ndarray in public contracts.
7. Use FramePacket (Section 7) as the canonical ingestion-boundary raw frame container. Do not redefine FramePacket in any module spec.
8. Publish accepted ingestion-boundary frames only through FramePacketSink (Section 8).
9. Use Point as defined in Section 9. Do not define a separate coordinate type in any module spec.
10. Use FaceLandmarks as defined in Section 10. Do not define a separate landmark struct in any module spec.
11. Full-frame projection is owned by RPM / PipelineOrchestrator unless explicitly justified otherwise.
