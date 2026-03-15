adapter.start()

# Frame Adapter

## Purpose
The Frame Adapter is a robust, source-agnostic component that ingests frames from external sources (e.g., RTSP cameras, video files), decodes and normalizes them, attaches canonical metadata, and emits standardized `FramePacket` objects into the Image Processing Service (IPS) pipeline.

## Architectural Role
The Frame Adapter is responsible for:
- Receiving frames from a configured source (RTSP, file, etc.).
- Decoding or extracting raw pixel buffers as needed.
- Populating the canonical `FramePacket` fields (timestamp, camera_id, resolution, format, pixel data).
- Attaching metadata (source URI, codec, sequence ids) and pipeline control flags.
- Emitting metrics/diagnostics: frames_in, frames_dropped, reconnects, ingest_latency_ms.
- Respecting backpressure (bounded handoff queue) and applying a configurable drop policy.

**Key Principle:**
The Frame Adapter normalizes all heterogeneous camera or file inputs to a single, in-process data model (`FramePacket`). It does not perform algorithm-specific image transformations; those are handled by the Frame Transformation Layer.

## Supported Input Types
- RTSP camera streams (live cameras)
- Video file sources (testing/offline)

## Base Interface: `FrameAdapter`
All concrete adapters implement the following conceptual interface:
- `start()` — Non-blocking. Allocates resources and starts capture/processing threads or async loops. Returns quickly; errors are reported via status/metrics or exceptions during initialization.
- `stop()` — Graceful shutdown: stop capture loop, drain handoff queue (configurable), release decoders and network handles.
- `configure(config: FrameAdapterConfig)` — Provide adapter-specific configuration (URI, camera_id, queue sizes, decoder options).
- `register_consumer(callable)` — Supply a callback or pipeline ingest function to receive `FramePacket` objects. Alternatively, adapters may push into a shared `PipelineOrchestrator.ingest_frame()` API.
- `health() -> AdapterHealth` — Optional health/status report for monitoring.

**Concurrency Model:**
Single capture thread (or async task) performs I/O and decoding, then enqueues `FramePacket` objects to a bounded in-memory handoff queue. A short-lived worker drains the queue and calls the registered consumer. This decouples network/decoder jitter from pipeline processing.

**Queue Sizing:**
Configurable `max_queue_size`. Backpressure policy: `drop_oldest` (recommended) or `drop_newest`.

## Example Implementations

### RTSPFrameAdapter
- Connects to an RTSP URI, reads encoded frames, decodes to raw pixels, and emits `FramePacket` at the stream rate.
- Uses robust decoding stack (GStreamer, FFmpeg/PyAV). For prototyping, OpenCV `cv2.VideoCapture(rtsp_uri)` is acceptable.
- Parses stream metadata (fps, resolution, codec). Uses PTS (presentation timestamp) as authoritative timestamp when available.
- Timestamping fallback: PTS from stream > camera-supplied timestamp > local receive time.
- Prefer hardware-accelerated decoders where available.
- Normalizes decoder output to BGR (default for downstream modules).
- Handles reconnects with exponential backoff. Emits health/error events on fatal error.
- Applies drop policy if handoff queue is full.

**Configuration Example:**
```yaml
adapter_type: rtsp
camera_id: cam-123
uri: rtsp://10.0.0.5/stream
decoder: ffmpeg
max_queue_size: 8
drop_policy: drop_oldest
reconnect:
  max_retries: 0    # 0 == infinite
  base_backoff_ms: 500
  max_backoff_ms: 10000
```

### VideoFileFrameAdapter
- Reads video frames from disk for testing and offline analysis, producing deterministic timestamps and optional looped playback.
- Uses OpenCV `VideoCapture(file)` or PyAV/FFmpeg for better timestamp fidelity.
- Derives per-frame timestamp as: file_start_epoch_ms + round(frame_index * (1000 / fps)).
- Supports playback options: `loop`, `playback_rate`, `start_time`, `end_time`.

**Configuration Example:**
```yaml
adapter_type: file
camera_id: cam-test
path: /data/test_clip.mp4
loop: false
playback_rate: 1.0
max_queue_size: 16
```

## Frame Object: `FramePacket`
This is the canonical in-process frame model used by the IPS pipeline.

```python
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass(frozen=True)
class FramePacket:
    frame_id: str           # UUID v4 string
    camera_id: str
    timestamp_ms: int      # epoch milliseconds
    width: int
    height: int
    format: str            # e.g., 'BGR', 'RGB', 'JPEG'
    image_bytes: bytes     # raw BGR bytes (row-major) OR encoded bytes if chosen
    metadata: Dict[str, str]
    pipeline_flags: Dict[str, bool]
    # In-process implementations MAY also carry a numpy.ndarray for speed, but the serializable contract uses bytes.
```

**Adapter Conversion Rules:**
- Decoded frames should be normalized to `format='BGR'` and `image_bytes` containing contiguous row-major BGR pixel bytes.
- Populate `width`/`height` from decoder output; validate these match the byte length.
- `frame_id` must be unique (UUID). `camera_id` comes from adapter config.
- `metadata` should include `source_uri`, `codec`, `stream_seq` (optional), and original timestamp source.

## Communication and Integration

**Camera Service:**
- May host its own adapters and push normalized frames to IPS over gRPC; or IPS may run adapters directly (pull from RTSP).
- In-process: adapter directly calls `PipelineOrchestrator.ingest_frame(frame_packet)`.
- Remote: Camera Service streams `Frame` protobuf messages; a gRPC client adapter decodes/normalizes them to `FramePacket`.

**Image Processing Pipeline:**
- The adapter hands frames to the pipeline via a small, stable API: `ingest_frame(frame: FramePacket) -> bool` (return indicates accepted/dropped). A shared, bounded queue decouples producer and consumer.

**Initialization:**
1. Configuration loader reads camera-specific adapter configs (URI, camera_id, adapter_type).
2. `FrameAdapterFactory.create(adapter_config)` instantiates the requested adapter class.
3. The adapter registers the pipeline consumer callback (e.g., `PipelineOrchestrator.ingest_frame`).
4. Call `adapter.start()` which spins the capture and decode loop and returns immediately.
5. Monitoring registers adapter health metrics; metrics exported (Prometheus, logs).

**Example Registration:**
```python
adapter = FrameAdapterFactory.create(config)
adapter.register_consumer(pipeline.ingest_frame)
adapter.start()
```

## Implementation Notes
- Prototyping: OpenCV (cv2.VideoCapture)
- Production decoding: PyAV (FFmpeg bindings) or GStreamer (for complex pipelines and hardware acceleration).
- Unit tests: use `VideoFileFrameAdapter` with short synthetic clips; assert `FramePacket` fields and timestamps.
- Integration tests: `rtsp-simple-server` as local RTSP source to validate reconnect and backpressure behavior.
- Metrics: `frames_in_total`, `frames_dropped_total`, `ingest_latency_ms`, `adapter_reconnects_total`.
- Security: Secure RTSP via SRTP/HTTPS tunneling where supported.
- Always use bounded queues and cap frames retained in-memory.
- Validate and limit resolution to configured maximums to prevent OOM from malicious or misconfigured cameras.

## Verification Checklist
- [ ] `VideoFileFrameAdapter` unit test exists and passes.
- [ ] `RTSPFrameAdapter` reconnect behavior validated against local RTSP server.
- [ ] Metrics are emitted and visible in dev environment.
- [ ] Drop policy tested under simulated slow pipeline (sleeping consumer).

---

# Frame Transformation Layer (Integration Overview)

## Purpose
The Frame Transformation Layer is an internal component that sits between the Frame Adapter and all downstream algorithm modules. It is responsible for converting heterogeneous camera-native formats into the exact working representations required by each algorithm.

## Responsibilities
- Receives raw frames (`FramePacket`) from the gRPC Client Frame Adapter.
- Accepts heterogeneous image formats (BGR, RGB, grayscale, infrared, different bit depths, compressed, etc.).
- Inspects canonical metadata (width, height, pixel_format, channels, bits_per_pixel, etc.).
- Decodes payloads if necessary.
- Converts the image into the exact working representation required by downstream algorithms.
- Provides module-specific image views so algorithms do not need to support every camera-native format directly.

**Architectural Rule:**
Downstream algorithms must not be required to support all camera-native formats. The Frame Transformation Layer provides normalized, algorithm-specific inputs.

## Algorithm Input Format Specifications

### Motion Detection
- Input: Grayscale (single-channel intensity)
- Color Format: Grayscale (luma channel)
- Bit Depth: 8 bits per pixel
- Resolution: Downscaled to 640×360 (or configurable)
- Preprocessing: Optional Gaussian blur or denoising before motion comparison

### Object Detection
- Input: RGB image as tensor
- Color Format: RGB
- Bit Depth: 8 bits per channel
- Resolution: Resized to model input size (e.g., 640×640)
- Preprocessing: Resize, normalization (e.g., [0,1] or mean/std), optional letterboxing

### Face Detection
- Input: RGB or BGR image
- Color Format: RGB or BGR (as required by the model)
- Bit Depth: 8 bits per channel
- Resolution: Full frame or region of interest (ROI), typically not upscaled
- Preprocessing: Resize to model input size if required

### Face Recognition
- Input: Aligned face crop as tensor
- Color Format: RGB
- Bit Depth: 8 bits per channel
- Resolution: Model-specific (e.g., 112×112)
- Preprocessing: Face alignment, crop, normalization (e.g., [0,1] or mean/std)

## Frame Transformation Flow
1. **Receive raw frame** from the gRPC Client Frame Adapter as a `FramePacket` (metadata + payload).
2. **Inspect metadata and source format** (width, height, pixel_format, channels, bits_per_pixel, compression, etc.).
3. **Decode payload** if compressed or encoded.
4. **Generate algorithm-specific representations**:
   - Grayscale, resized, blurred for Motion Detection
   - RGB tensor, resized, normalized for Object Detection
   - RGB/BGR, full frame or ROI for Face Detection
   - Aligned, cropped, normalized RGB tensor for Face Recognition
5. **Provide those representations** to pipeline modules, ensuring each receives only its required format.

## Conceptual Data Flow Diagram

Camera
    ↓
gRPC Client Frame Adapter
    ↓
FramePacket (metadata + payload)
    ↓
**Frame Transformation Layer**
    ↓
- Motion Detection (grayscale, resized, blurred)
- Object Detection (RGB tensor, resized, normalized)
- Face Detection (RGB/BGR, full frame or ROI)
- Face Recognition (aligned face crop, normalized tensor)

## Design Goals
- Support all camera-native formats without burdening algorithms.
- Algorithms operate on stable, normalized internal representations.
- Image conversions are performed once per frame, only as needed.
- Each algorithm explicitly declares its required input format.
- Separation of concerns improves maintainability and simplifies algorithm development.

---

## Appendix: Minimal FramePacket Conversion Example (Python)
```python
import time, uuid
def decoded_frame_to_framepacket(camera_id, decoded_ndarray, src_meta):
    h, w = decoded_ndarray.shape[:2]
    frame_id = str(uuid.uuid4())
    timestamp_ms = src_meta.get('pts_ms') or int(time.time() * 1000)
    # Convert to contiguous BGR bytes
    image_bytes = decoded_ndarray.tobytes()
    return FramePacket(
        frame_id=frame_id,
        camera_id=camera_id,
        timestamp_ms=timestamp_ms,
        width=w,
        height=h,
        format='BGR',
        image_bytes=image_bytes,
        metadata=src_meta,
        pipeline_flags={'skip_remaining': False}
    )
```

---

Document created by architecture working notes; implementors should adapt low-level APIs for chosen language and runtime (Python/C++/Go). Submit PRs for code + unit tests that exercise the behavior described here.
