# Frame Transformation Layer

## Purpose

The Frame Transformation Layer is an internal component of the Image Processing Service (IPS) responsible for converting an immutable input `FramePacket` into algorithm-specific working representations.

Its role is limited to data preparation. It does not ingest frames, connect to cameras, execute detection or recognition algorithms, or make pipeline control decisions.

## Architectural Role

- Receives canonical `FramePacket` objects from within IPS.
- Interprets frame metadata required to understand the payload.
- Decodes encoded payloads when necessary.
- Builds deterministic working representations required by downstream algorithms.
- Caches decoded images and derived representations for the lifetime of a frame.

**Key Principle:**
The Frame Transformation Layer treats `FramePacket` as immutable input and returns derived representations separately. It does not modify or extend the `FramePacket` payload.

## Responsibility Boundary

### In Scope

- Payload interpretation using `FramePacket` metadata.
- Decode of encoded image payloads when required.
- Conversion of base image data into algorithm-specific representations.
- Per-frame caching of decoded images and final representations.
- Deterministic preprocessing such as resize, letterbox, grayscale conversion, normalization, cropping, alignment, and optional blur.

### Out of Scope

- Camera connectivity.
- Frame ingestion over gRPC, RTSP, USB, file replay, or any other transport.
- `FramePacket` creation.
- Pipeline ordering or routing decisions.
- Motion, object, face detection, face embedding, or identity recognition.
- Event generation, storage, or downstream delivery.

## Position in IPS

The Frame Transformation Layer is a standalone internal IPS component used by processing modules on demand.

It is not part of the Frame Adapter.

It is not a pipeline stage that pushes frames downstream.

It is a pull-based preparation layer invoked by modules that need a specific representation.

## Input Contract

The layer receives a `FramePacket` as input.

The `FramePacket` is assumed to contain:

- `frame_id`
- `camera_id`
- `timestamp_ms`
- `width`
- `height`
- `pixel_format`
- `num_color_channels`
- `bits_per_pixel`
- `encoding` (optional when payload is already raw)
- image payload bytes
- metadata map with any source-specific context required by upstream contracts

### Input Rules

- The Frame Transformation Layer does not modify the `FramePacket`.
- The layer treats `FramePacket` as immutable input.
- The layer relies on metadata to determine how the payload must be interpreted.
- The layer may reject frames with incomplete or inconsistent metadata.

## Output Contract

The layer returns algorithm-specific working representations.

These representations are not stored inside `FramePacket`.

Each representation is returned independently to the caller that requested it.

### Output Rules

- Each algorithm consumes one well-defined representation contract.
- Representation contracts are versioned and explicit.
- Returned representations are derived artifacts, not canonical frame state.
- Callers must not assume that one representation can be substituted for another.

## Architectural Decisions

The following decisions are normative:

1. Transformations are computed lazily, only when a caller requests a specific representation.
2. Results are cached per frame using `frame_id` as the cache key.
3. Each algorithm consumes a single well-defined representation.
4. `FramePacket` is treated as immutable input.
5. The Frame Transformation Layer is stateless across frames except for a per-frame cache whose lifetime matches the frame lifecycle.
6. The public interface uses dedicated methods for each representation; it does not expose a generic string-based selector API.

## Internal Architecture

The layer consists of four internal components.

### 1. Metadata Interpreter

Responsibilities:

- Read `FramePacket` metadata.
- Validate required payload interpretation fields.
- Determine whether the payload is raw or encoded.
- Determine expected dimensions, channel layout, and pixel semantics.
- Provide normalized decode instructions to downstream internal components.

The Metadata Interpreter does not decode payload bytes and does not build final algorithm representations.

### 2. Payload Decoder

Responsibilities:

- Decode encoded payloads when necessary.
- Convert raw payload bytes into a base image representation when metadata indicates the payload is already decoded.
- Produce a deterministic base image form suitable for subsequent conversions.
- Reuse previously decoded base images from cache when available.

The Payload Decoder does not perform algorithm-specific normalization or cropping beyond what is required to materialize the base image.

### 3. Representation Builder

Responsibilities:

- Convert the decoded base image into required downstream representations.
- Apply deterministic preprocessing required by the target representation.
- Perform operations such as grayscale conversion, color conversion, resizing, letterboxing, normalization, cropping, alignment, and optional blur.
- Enforce the exact versioned representation contract expected by each algorithm.

The Representation Builder does not make pipeline decisions and does not execute inference.

### 4. Transformation Cache

Responsibilities:

- Store cached artifacts per `frame_id`.
- Cache decoded base images.
- Cache final derived representations.
- Prevent duplicate decoding and duplicate transformation work for the same frame.
- Release cached entries when the frame lifecycle ends.

The cache is not global across frames and is not intended for long-lived reuse across unrelated packets.

## Public Interface

The public interface exposes dedicated methods only.

No generic API of the form `get_representation(frame_packet, representation_name)` is permitted.

### Interface Methods

#### `get_motion_frame(frame_packet)`

- Purpose: Return `motion_frame_v1` for motion analysis.
- Behavior:
  1. Check the per-frame cache for `motion_frame_v1`.
  2. If present, return the cached representation.
  3. If absent, ensure the base image is available, decoding if necessary.
  4. Build `motion_frame_v1`.
  5. Store the result in the per-frame cache.
  6. Return the representation.

#### `get_object_tensor(frame_packet)`

- Purpose: Return `object_tensor_v1` for object detection.
- Behavior:
  1. Check the per-frame cache for `object_tensor_v1`.
  2. If present, return the cached representation.
  3. If absent, ensure the base image is available, decoding if necessary.
  4. Build `object_tensor_v1`.
  5. Store the result in the per-frame cache.
  6. Return the representation.

#### `get_face_detect_frame(frame_packet)`

- Purpose: Return `face_detect_frame_v1` for face detection.
- Behavior:
  1. Check the per-frame cache for `face_detect_frame_v1`.
  2. If present, return the cached representation.
  3. If absent, ensure the base image is available, decoding if necessary.
  4. Build `face_detect_frame_v1`.
  5. Store the result in the per-frame cache.
  6. Return the representation.

#### `get_face_embed_tensor(frame_packet, face_bbox)`

- Purpose: Return `face_embed_tensor_v1` for face embedding generation.
- Inputs:
  - `frame_packet`
  - `face_bbox` identifying the face crop region to embed
- Behavior:
  1. Validate `face_bbox` against the input frame bounds.
  2. Check the per-frame cache for the requested `face_embed_tensor_v1` keyed by frame and crop identity.
  3. If present, return the cached representation.
  4. If absent, ensure the base image is available, decoding if necessary.
  5. Build `face_embed_tensor_v1` using the specified face crop.
  6. Store the result in the per-frame cache.
  7. Return the representation.

## Representation Definitions

Representation contracts are internal but normative. Each named representation defines the exact working form consumed by one algorithm category.

### `motion_frame_v1`

- Grayscale.
- 8-bit.
- Resized to motion-processing dimensions, for example `640x360`.
- Optional blur may be applied when configured by the motion consumer contract.
- Intended consumer: motion analysis only.

### `object_tensor_v1`

- RGB.
- Resized or letterboxed to the detector input size.
- Normalized according to the object detector contract.
- Returned in tensor format.
- Intended consumer: object detection only.

### `face_detect_frame_v1`

- RGB.
- Full-frame or detector-size representation.
- Minimal preprocessing only.
- Preserves as much original facial detail as possible while matching detector input requirements.
- Intended consumer: face detection only.

### `face_embed_tensor_v1`

- Aligned face crop.
- RGB.
- Resized to the embedding input size, for example `112x112`.
- Normalized tensor.
- Intended consumer: face embedding generation only.

## Processing Flow

The processing flow is deterministic and request-driven.

1. A module requests a specific representation from the Frame Transformation Layer.
2. The layer checks the per-frame cache for the requested representation.
3. If the representation is absent, the layer ensures a decoded base image is available.
4. If the payload is encoded and no decoded base image exists in cache, the layer decodes the payload.
5. The Representation Builder constructs the requested representation from the base image.
6. The layer stores the decoded base image and/or final representation in the per-frame cache.
7. The layer returns the requested representation to the caller.

## Caching Strategy

Caching is frame-scoped and mandatory for efficiency.

### Cache Rules

- Cache key scope is per `frame_id`.
- The cache stores decoded base images.
- The cache stores final derived representations.
- Cache lifetime matches the frame lifecycle.
- No global cross-frame cache is used.
- Cache entries are disposable internal artifacts and not part of the external IPS contract.

### Cache Intent

- Avoid duplicate decode work when multiple modules need the same frame.
- Avoid rebuilding the same representation multiple times within a frame lifecycle.
- Reuse intermediate results whenever a downstream representation can be built from a previously computed base artifact.

## Error Handling

The Frame Transformation Layer must fail in a controlled and explicit way.

### Error Cases

#### Payload cannot be decoded

- The layer returns a controlled decode error.
- No partially decoded artifact is cached as a valid representation.
- Callers receive a failure that clearly indicates the requested representation could not be produced.

#### Metadata is invalid

- The layer returns a controlled validation error.
- Invalid metadata includes missing dimensions, inconsistent channel information, unsupported encoding declarations, or payload-size mismatches when those can be determined.
- The layer does not attempt speculative reconstruction when the contract is ambiguous.

#### Unsupported pixel format

- The layer returns a controlled unsupported-format error.
- Unsupported formats are not silently coerced unless an explicit deterministic interpretation rule exists within the layer contract.

### Error Policy

- Errors must be surfaced in a form the caller can handle deterministically.
- Failures in one requested representation do not imply mutation or corruption of the source `FramePacket`.
- The layer may expose fallback behavior only when that fallback is explicitly defined by the representation contract; otherwise it returns a controlled error.

## Performance Considerations

The Frame Transformation Layer exists to centralize preprocessing while preserving efficiency.

### Performance Principles

- Use lazy evaluation so work happens only when a representation is requested.
- Decode at most once per frame whenever possible.
- Minimize memory copies during decode and transformation.
- Reuse intermediate results across multiple representation requests within the same frame lifecycle.
- Avoid rebuilding derived tensors or image buffers that already exist in cache.
- Keep cross-frame state out of the layer to avoid stale-cache behavior and unnecessary memory growth.

## Integration Notes

The Frame Transformation Layer uses a pull-based integration model.

- Processing modules call transformation methods directly.
- The layer does not push representations into modules.
- The layer does not publish data autonomously.
- The layer does not decide whether a module should execute.
- The Pipeline Orchestrator retains responsibility for pipeline order and execution decisions.

## Boundaries

Responsibility separation is strict.

- Frame Adapter handles ingestion and canonical `FramePacket` creation.
- Frame Transformation Layer handles payload interpretation, decoding, and algorithm-specific data preparation.
- Algorithms handle detection, recognition, scoring, and result production.

There is no responsibility overlap among these three concerns.

## Design Goals

- Separation of concerns.
- Algorithm simplicity through stable working inputs.
- Format abstraction so algorithms do not need to understand raw ingress payload diversity.
- Performance efficiency through lazy evaluation and per-frame caching.
- Maintainability through explicit contracts, versioned representations, and isolated responsibilities.

## Implementation Notes

- The layer should remain implementation-library agnostic at the architecture level.
- Representation versions should change only when the consumer contract changes.
- The layer should prefer deterministic, reproducible preprocessing over implicit heuristics.
- Internal observability may record decode failures, cache hit rates, and representation build latency, but such telemetry is outside the public representation contract.

## Summary

The Frame Transformation Layer is a standalone internal IPS component that converts immutable `FramePacket` input into cached, algorithm-specific working representations on demand. Its contract is intentionally narrow: interpret metadata, decode payloads when necessary, build deterministic representations, and return them through dedicated methods without mutating `FramePacket`, executing algorithms, or making pipeline decisions.