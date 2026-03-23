# Frame Adapter

## Purpose
The Frame Adapter in IPS is a gRPC client adapter. It receives frames from the Camera Service gRPC stream, validates ingress fields, and builds canonical `FramePacket` objects for IPS.

IPS does not connect directly to cameras. RTSP/USB/file connectivity and camera-side stream handling belong to Camera Service.

## Architectural Role
The Frame Adapter is composed of two focused classes wired together by a thin coordinator:

**`FrameReceiver`** - ingress and parsing only:
- Receives frame messages from Camera Service over gRPC stream.
- Parses ingress transport messages and validates required fields.
- Populates canonical `FramePacket` fields (`frame_id`, `camera_id`, `timestamp_ms`, `width`, `height`, `pixel_format`, `num_color_channels`, `bits_per_pixel`, `image_bytes`, `metadata`).
- Emits metrics: `frames_in_total`, `adapter_reconnects_total`, `ingest_latency_ms`.
- Calls a single callback with each new `FramePacket` (`FrameStore.store_or_replace`).

**`FrameStore`** - latest-frame-per-camera storage only:
- Owns in-memory storage with exactly one slot per `camera_id`.
- Exposes `store_or_replace(frame_packet)` as callback target for `FrameReceiver`.
- Replaces any older frame currently stored for the same camera with the newest frame.
- Exposes `get_next_frame()` to return one frame in round-robin camera order.
- Does not invoke downstream consumers directly.
- Emits metrics: `frames_overwritten_total`, `active_camera_slots`, `frames_served_total`.

**`GrpcClientFrameAdapter`** - thin coordinator (public-facing):
- Instantiates and wires `FrameReceiver` and `FrameStore`.
- Exposes unchanged adapter lifecycle API (`configure`, `start`, `stop`, `health`).
- Exposes `get_next_frame()` by delegating to `FrameStore`.

**Key Principle:**
Camera Service sends raw frame payload only. Frame Adapter does not decode or transcode ingress payloads. It always keeps only the newest frame per camera.

**Responsibility Boundary:**
`FrameReceiver` handles ingress parsing and packet creation. `FrameStore` handles latest-frame storage and retrieval only. Neither class performs downstream image processing.

## Supported Ingress Contract
- gRPC streaming messages from Camera Service only.
- Required ingress fields: `frame_id`, `camera_id`, `timestamp_ms`, `width`, `height`, `pixel_format`, `num_color_channels`, `bits_per_pixel`, raw payload bytes.
- Encoded/compressed payload ingress is out of scope for this contract.

## Base Interface: `FrameAdapter`
All concrete adapters implement the following conceptual interface:
- `start()` - Non-blocking. Allocates resources and starts gRPC receive loop.
- `stop()` - Graceful shutdown: stop receive loop, stop storage access, close stream and network handles.
- `configure(config: FrameAdapterConfig)` - Provide adapter configuration (service endpoint, stream name, filters, storage and retry options).
- `get_next_frame() -> FramePacket | None` - Return next frame in round-robin camera order, or `None` if no frame is available.
- `health() -> AdapterHealth` - Optional health/status report for monitoring.

**Concurrency Model:**
`FrameReceiver` runs a single gRPC receive thread (or async task). For each message it builds a `FramePacket` and calls `FrameStore.store_or_replace(packet)`.

`FrameStore` is passive. It stores one latest frame per camera key and exposes `get_next_frame()` for pull-based retrieval. Each call to `get_next_frame()` returns the next available camera slot in round-robin order using the fixed configured camera list order: camera 1, camera 2, ..., camera N, then back to camera 1.

**Round-Robin Retrieval Rule:**
- Camera traversal order is fixed by configured camera list order.
- If a camera slot has no new frame, skip it and continue to next camera.
- The internal cursor advances across calls and resumes from the next camera after each returned frame.
- Returning a frame marks that camera slot as consumed until a new frame arrives.
- If all configured camera slots are empty, `get_next_frame()` returns `None` immediately.

**Storage Sizing:**
Storage capacity is proportional to active camera count: one frame slot per camera. With 5 active cameras, maximum retained frames is 5.

**Lifecycle Ordering:**
- `start()`: start `FrameStore`, then `FrameReceiver`.
- `stop()`: stop `FrameReceiver`, then `FrameStore`.

## IPS Core Implementation

### FrameReceiver
Pure ingress component. No storage or consumer logic.

- Connects to Camera Service gRPC streaming endpoint and receives frame messages.
- Parses envelope and metadata (`frame_id`, `camera_id`, `timestamp_ms`, `width`, `height`, `pixel_format`, `num_color_channels`, `bits_per_pixel`).
- Creates `FramePacket` from raw ingress payload without altering representation.
- Handles stream reconnect with exponential backoff.
- Calls `FrameStore.store_or_replace` for each frame.

**Interface:**
- `set_on_frame(callback)`
- `start()` / `stop()`
- `health() -> ReceiverHealth`

**Metrics owned:** `frames_in_total`, `ingest_latency_ms`, `adapter_reconnects_total`.

**Configuration fields owned:**
```yaml
camera_service_endpoint: camera-service:50051
stream_name: frames
reconnect:
  max_retries: 0    # 0 == infinite
  base_backoff_ms: 500
  max_backoff_ms: 10000
```

---

### FrameStore
Pure latest-frame storage and retrieval. No gRPC parsing.

- Exposes `store_or_replace(frame_packet)` entry point.
- Stores exactly one latest frame per camera key (`camera_id`).
- Overwrites the previously stored frame for the same camera on every new arrival.
- Exposes `get_next_frame()` for round-robin pull retrieval.
- Does not execute downstream callbacks.

**Interface:**
- `store_or_replace(frame_packet: FramePacket)`
- `get_next_frame() -> FramePacket | None`
- `start()` / `stop()`
- `health() -> StoreHealth`

**Metrics owned:** `frames_overwritten_total`, `active_camera_slots`, `frames_served_total`.

**Configuration fields owned:**
```yaml
max_camera_slots: 128
camera_order_source: configured_list
```

---

### GrpcClientFrameAdapter (IPS coordinator)
Thin coordinator. Creates and wires `FrameReceiver` + `FrameStore`.

- `configure(config)` forwards relevant fields to `FrameReceiver` and `FrameStore`.
- `start()` calls `FrameStore.start()` then `FrameReceiver.start()` and wires `FrameStore.store_or_replace` as receiver callback.
- `get_next_frame()` delegates to `FrameStore.get_next_frame()`.
- `stop()` calls `FrameReceiver.stop()` then `FrameStore.stop()`.
- `health()` aggregates `ReceiverHealth` + `StoreHealth` into `AdapterHealth`.

**Full Configuration Example:**
```yaml
adapter_type: grpc_client
camera_service_endpoint: camera-service:50051
stream_name: frames
max_camera_slots: 128
camera_order_source: configured_list
reconnect:
  max_retries: 0    # 0 == infinite
  base_backoff_ms: 500
  max_backoff_ms: 10000
```

## Frame Object: `FramePacket`
Canonical in-process frame model used by IPS pipeline.

```python
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class FramePacket:
    frame_id: str           # provided by Camera Service ingress
    camera_id: str
    timestamp_ms: int       # epoch milliseconds
    width: int
    height: int
    pixel_format: str       # e.g., 'RGB', 'BGR', 'GRAY8', 'YUV420'
    num_color_channels: int
    bits_per_pixel: int
    image_bytes: bytes      # raw ingress payload bytes only
    metadata: Dict[str, Any]
```

**Mutability Strategy:**
- `FramePacket` carries ingest/core fields produced by adapter.
- Raw payload fields are immutable by policy after creation.

**Adapter Conversion Rules:**
- Preserve ingress representation; do not decode or transcode in Frame Adapter.
- Populate `width`, `height`, `pixel_format`, `num_color_channels`, `bits_per_pixel` from ingress metadata.
- Validate metadata consistency against payload expectations when possible.
- `frame_id` and `camera_id` come from Camera Service ingress and are mandatory.

## Communication and Integration

- Frame Adapter receives gRPC frame messages from Camera Service.
- FrameStore stores latest available frame per camera.
- IPS pipeline (or coordinator loop) calls `get_next_frame()` to pull frames in round-robin order.

**Initialization:**
1. Config loader reads adapter config (`camera_service_endpoint`, `stream_name`, storage/retry options).
2. `FrameAdapterFactory.create(adapter_config)` instantiates gRPC client adapter.
3. Call `adapter.start()` to start receiver and store.
4. Processing loop pulls frames:

```python
adapter.start()

while running:
    frame = adapter.get_next_frame()
    if frame is None:
        continue
    process_frame(frame)
```

## Implementation Notes
- Camera ingestion inside IPS is gRPC-only.
- Ingress payload is raw-only; encoded/compressed ingress is rejected.
- Stream reconnect uses bounded exponential backoff (`FrameReceiver`).
- `FrameStore` intentionally overwrites stale frames for same camera.
- Retrieval order is round-robin across cameras using fixed configured camera list order.
- Unit tests for `FrameStore`: overwrite semantics, round-robin order, per-camera isolation, max slot bounds.

## Verification Checklist

**`FrameReceiver`**
- [ ] Unit tests pass: gRPC parsing, `FramePacket` mapping, timestamp handling.
- [ ] Reconnect with exponential backoff validated.
- [ ] Ingress mapping validated (`frame_id`, `camera_id`, `pixel_format`, `num_color_channels`, `bits_per_pixel`).

**`FrameStore`**
- [ ] One-slot-per-camera behavior validated.
- [ ] With 5 active cameras, retained frames never exceed 5.
- [ ] New frame for camera replaces prior stored frame for same camera.
- [ ] `get_next_frame()` returns frames in round-robin fixed configured camera order.
- [ ] Overwrite on camera A does not affect camera B.
- [ ] `FrameStore` does not invoke downstream consumers; retrieval is pull-only via `get_next_frame()`.

**`GrpcClientFrameAdapter`**
- [ ] Integration test: end-to-end frame flow via pull-based `get_next_frame()`.

## Design Goals
- Clear separation: `FrameReceiver` owns ingress; `FrameStore` owns latest-frame storage and retrieval.
- Always prefer newest frame per camera over backlog processing.
- Bound memory by camera count (one frame per camera).
- Round-robin fairness across cameras in retrieval order.

---

## Appendix: Minimal FramePacket Conversion Example (Python)
```python
import time


def ingress_to_framepacket(payload_bytes, src_meta):
    return FramePacket(
        frame_id=src_meta["frame_id"],
        camera_id=src_meta["camera_id"],
        timestamp_ms=src_meta.get("pts_ms") or int(time.time() * 1000),
        width=src_meta["width"],
        height=src_meta["height"],
        pixel_format=src_meta["pixel_format"],
        num_color_channels=src_meta["num_color_channels"],
        bits_per_pixel=src_meta["bits_per_pixel"],
        image_bytes=payload_bytes,
        metadata=src_meta,
    )
```
