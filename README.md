# Camera-regogintion

## Current Architecture Snapshot

The system currently follows an event-driven pipeline for camera monitoring on a single-host MVP baseline.

- Camera Service publishes normalized frames.
- Unified Vision Service performs motion, object, face detection, and recognition.
- Frame Buffer Service maintains pre/post-roll frame history.
- Event Service creates canonical events and drives downstream actions.
- Media Service (merged Clip Recording + Storage boundary) fetches frame ranges, builds MP4 clips, and persists media artifacts.
- Telegram Notification Service sends event alerts after clip readiness.

## Event-Triggered Clip Flow

1. Unified Vision emits person or motion-derived event signals.
2. Event Service publishes event.created.
3. Media Service reads event.created, requests frame window from Frame Buffer Service, muxes MP4, persists clip, and emits event.clip.ready.
4. Telegram Notification Service consumes event.clip.ready and sends rich notification payloads.

See [doc/system.md](doc/system.md) for the full architecture, sequence diagrams, and API surface.