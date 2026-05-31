# Camera-regogintion

## Current Architecture Snapshot

The system currently follows an event-driven pipeline for camera monitoring on a single-host MVP baseline.

- Camera Service publishes normalized frames.
- Image processing service performs motion, object, face detection, and recognition.
- Frame Buffer Service maintains pre/post-roll frame history.
- Event Service creates canonical events and drives downstream actions.
- Media Service (merged Clip Recording + Storage boundary) fetches frame ranges, builds MP4 clips, and persists media artifacts.
- Telegram Notification Service sends event alerts after clip readiness.

## Event-Triggered Clip Flow

1. Image processing service emits person or motion-derived event signals.
2. Event Service publishes event.created.
3. Media Service reads event.created, requests frame window from Frame Buffer Service, muxes MP4, persists clip, and emits event.clip.ready.
4. Telegram Notification Service consumes event.clip.ready and sends rich notification payloads.

See [doc/system.md](doc/system.md) for the full architecture, sequence diagrams, and API surface.

## IPS Single-Camera Visual Replay

Place the replay video at:

- `tests/visual_assets/image_processing_service/videos/single_camera_test.mp4`
- `tests/image_processing_service/videos/single_camera_test.mp4` (supported fallback path)

Run the replay runner:

- `c:/Users/talif/Desktop/Camera-regogintion/.venv/Scripts/python.exe tests/image_processing_service/visual_replay_single_camera.py`

Optional replay flags:

- `--replay-fps <fps>` to pace frame injection (for example `--replay-fps 5`)
- `--quality-mode` to slow replay ingress and relax IPS freshness limits for completeness-first runs
- `--side-by-side` to write left-original/right-overlay output video
- `--sample-frame-interval <N>` to write `sample_frames/*.png` every N frames

Run the integration test:

- `python.exe -m pytest tests/image_processing_service/test_image_processing_service_visual_replay_single_camera.py -q`

Replay outputs are written to:

- `tests/results/image_processing_service_visual/single_camera/output_overlay.mp4`
- `tests/results/image_processing_service_visual/single_camera/summary.json`
- `tests/results/image_processing_service_visual/single_camera/metrics.csv`
- `tests/results/image_processing_service_visual/single_camera/output_side_by_side.mp4` (when `--side-by-side` is enabled)
- `tests/results/image_processing_service_visual/single_camera/sample_frames/*.png` (when sample frame interval is enabled)
- `tests/results/image_processing_service_visual/single_camera_quality_mode/` when `--quality-mode` is enabled and no custom output directory is provided

Visual checks to perform:

- Detection alignment between frame content and rendered person/face boxes
- Runtime HUD progression across frame index, queue depth, and latency values
- Explicit status rendering for no detections, stale dropped, skipped, or dropped frames
- Side-by-side correspondence between original timing and overlay timing