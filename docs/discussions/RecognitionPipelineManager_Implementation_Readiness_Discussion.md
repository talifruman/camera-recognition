PRE-IMPLEMENTATION REVIEW REQUIRED

# RecognitionPipelineManager Implementation Readiness Discussion

Review date: 2026-05-12
Scope type: Pre-implementation compatibility and readiness verification
Target: RecognitionPipelineManager (RPM) implementation has not started in src

---

# 1. Overview

## Purpose of this review
This review verifies whether RPM can be implemented safely against the actual current repository state, not against assumptions. The focus is implementation-readiness at module boundaries and orchestration seams.

## Goal
Confirm that current module APIs, contracts, temporal semantics, projection semantics, and failure behavior are sufficiently aligned so RPM can be implemented without hidden integration risk.

## Scope covered
- Frame Transformation Layer (FTL)
- Motion Detection
- Object Detection
- Face Detection
- Face Recognition
- PersonDirectory
- Shared contracts and shared typed structures

## Source of truth used
Primary evidence:
- Module specs under doc/image_processing_service
- Runtime implementations under src/image_processing
- Existing tests under tests

Secondary corroborative evidence:
- Existing compatibility discussions under docs/discussions

---

# 2. Current RPM Responsibilities

Based on doc/image_processing_service/RecognitionPipelineManager.md, RPM (through PipelineOrchestrator) is intended to own:

- ingest orchestration:
  - call FTL ingest_frame for each FramePacket before stage execution
- FTL usage:
  - call get_frame for CURRENT/PREVIOUS and ROI crops per stage input contracts
- temporal frame selection:
  - use PREVIOUS + CURRENT for Motion Detection
  - use CURRENT for all downstream stages
- pipeline sequencing:
  - Motion Detection -> Object Detection -> Face Detection -> Face Recognition
- ROI propagation:
  - pass motion ROI to object stage
  - pass projected person ROI to face detection
  - pass projected face ROI to face recognition
- projection to full-frame coordinates:
  - project ObjectDetection ROI-local person bboxes
  - project FaceDetection ROI-local face bboxes
- recognition aggregation:
  - append recognized faces under each person entry
- PersonDirectory enrichment:
  - person_id from FR -> PersonDirectory lookup -> person_name enrichment
- output building:
  - build final RecognitionPipelineOutput with full-frame bboxes only
- per-camera orchestration expectations:
  - no internal per-camera mutable state in RPM; rely on FTL camera isolation
- error handling:
  - empty/partial output behavior as documented per stage failure type
- event generation assumptions:
  - no explicit event system contract identified in current implementation artifacts

---

# 3. Module-by-Module API Compatibility Review

## 3.1 RPM ↔ FTL

### Public APIs reviewed
Implementation:
- FrameTransformationLayer.ingest_frame(frame_packet: FramePacket) -> None
- FrameTransformationLayer.get_frame(camera_id, temporal_selector, region_bbox, output_type, geometry_spec) -> ProcessedFrame

Core types:
- FrameTemporalSelector: CURRENT, PREVIOUS
- ProcessedFrame: frame_id, timestamp_ms, image, source_bbox_full_frame, spatial_transform
- SpatialTransform: scale_x, scale_y, pad_left, pad_top, output_width, output_height

### Expected RPM usage
- ingest incoming FramePacket before any stage execution
- get PREVIOUS and CURRENT full-frame processed images for motion stage
- get CURRENT cropped ROI images for object, face detection, and face recognition
- use source_bbox_full_frame + spatial_transform for projection back to full-frame

### Input contract verification
Verified:
- camera_id string routing by per-camera storage
- region_bbox must be positive and in-bounds
- output_type must be valid OutputImageType
- geometry_spec validated

Resolved alignment:
- ResizePolicy.NONE stage contracts return width=0 and height=0 placeholders by design.
- FTL geometry validation now accepts NONE placeholders (`0`/`0`) and omittable values.
- Direct RPM pass-through of stage `get_input_contract()` GeometrySpec is now compatible.

### Output contract verification
Verified:
- image is shared Image TypedDict-like structure with metadata
- source_bbox_full_frame always returned
- spatial_transform always returned
- timestamp_ms and frame_id preserved from stored frame

### Runtime assumptions
- FTL FrameStore uses RLock and per-camera state
- PREVIOUS exists only after two successful ingests per camera
- failed ingest does not rotate CURRENT/PREVIOUS
- get_frame performs on-demand crop/convert; no derived-image caching

### Mismatches / undefined behavior
No active blocker remains in this seam after contract alignment.

### Existing tests/evidence
- tests/frame_transformation_layer/test_frame_transformation_layer_module.py
  - temporal rotation and cold start behavior
  - ResizePolicy behavior coverage
  - multi-camera isolation tests

---

## 3.2 RPM ↔ Motion Detection

### Public APIs reviewed
Implementation:
- MotionDetectionManager.detect(input: MotionDetectionInput) -> MotionResult
- MotionDetectionManager.get_input_contract() -> PipelineStageInputContract

Input/output contracts:
- MotionDetectionInput: current_frame, previous_frame (MotionInputFrame)
- MotionResult: detected, bboxes

### Expected RPM usage
- build MotionDetectionInput from FTL CURRENT and PREVIOUS ProcessedFrame.image
- pass shared Image payload for both frames
- use returned bboxes as full-frame crop regions in current invocation flow

### Input contract verification
Verified:
- strict image contract: GRAY, HWC, uint8, [0,255]
- camera_id equality enforced across frames
- previous.timestamp <= current.timestamp enforced
- equal dimensions enforced

### Output contract verification
Verified:
- bboxes are relative to current_frame.image
- no confidence/extras exposed
- returns empty output on invalid input or exceptions

### Runtime assumptions
- module is stateless across calls
- caller supplies PREVIOUS/CURRENT ordering and consistency
- no internal frame history

### Mismatches / undefined behavior
No active API mismatch with RPM contracts beyond FTL NONE geometry mismatch affecting upstream get_frame.

### Existing tests/evidence
- tests/motion_detection/test_motion_detection_module.py
  - contract validation
  - timestamp ordering
  - deterministic behavior
  - get_input_contract values

---

## 3.3 RPM ↔ Object Detection

### Public APIs reviewed
Implementation:
- ObjectDetectionModule.detect(input: ObjectDetectionInput) -> PersonDetectionResult
- ObjectDetectionModule.get_input_contract() -> PipelineStageInputContract

Input/output contracts:
- ObjectDetectionInput: frame_id, camera_id, timestamp_ms, roi_image (Image), roi_bbox_frame (BoundingBox)
- PersonDetectionResult: frame_id, person_detected, persons

### Expected RPM usage
- for each motion bbox, call FTL.get_frame using OD input contract
- construct ObjectDetectionInput with roi_image from ProcessedFrame.image
- pass roi_bbox_frame as the source motion ROI in full-frame coords
- project output persons from ROI-local to full-frame using FTL spatial metadata

### Input contract verification
Verified:
- roi_image requires RGB/HWC/uint8/[0,255]
- roi_bbox_frame required and validated
- timestamp/camera/frame required

### Output contract verification
Verified:
- persons are ROI-local relative to roi_image
- no projection to full-frame in module output
- no confidence exposed

### Runtime assumptions
- stateless inference behavior
- upstream preprocesses image via FTL
- projection is external responsibility (RPM)

### Mismatches / undefined behavior
No active OD API mismatch found.

### Existing tests/evidence
- tests/object_detection/test_real_object_detection.py
  - ROI-local coordinate behavior
  - contract validation and get_input_contract checks

---

## 3.4 RPM ↔ Face Detection

### Public APIs reviewed
Implementation:
- FaceDetectionModule.detect_faces(face_input: FaceDetectionInput) -> FaceDetectionOutput
- FaceDetectionModule.get_input_contract() -> PipelineStageInputContract

Input/output contracts:
- FaceDetectionInput: frame_id, camera_id, timestamp_ms, roi_image (Image)
- FaceDetectionOutput: frame_id, camera_id, timestamp_ms, detections
- DetectedFace: face_bbox, landmarks

### Expected RPM usage
- for each projected person full-frame bbox, request ROI image from FTL
- pass FaceDetectionInput with roi_image and trace metadata
- for each DetectedFace.face_bbox (ROI-local), project to full-frame before public output and FR crop request
- re-express landmarks to face-crop-local before FR input

### Input contract verification
Verified:
- validator expects Image struct metadata and shape consistency
- detector engine receives roi_image data array

### Output contract verification
Verified:
- face_bbox is ROI-local to person roi_image
- landmarks are ROI-local to person roi_image
- no confidence exposed

### Runtime assumptions
- stateless detection orchestration
- upstream provides preprocessed ROI image
- downstream handles full-frame projection

### Mismatches / undefined behavior
No active API mismatch found.

### Existing tests/evidence
- tests/face_detection/test_face_detection_module.py
  - ROI-local bbox and landmark assertions
  - contract validation and deterministic behavior

---

## 3.5 RPM ↔ Face Recognition

### Public APIs reviewed
Implementation:
- FaceRecognitionModule.recognize(face_input: FaceRecognitionInput) -> FaceRecognitionOutput
- FaceRecognitionModule.get_input_contract() -> PipelineStageInputContract

Input/output contracts:
- FaceRecognitionInput: frame_id, camera_id, timestamp_ms, face_roi_image, landmarks
- FaceRecognitionOutput: frame_id, camera_id, timestamp_ms, person_found, person_id

### Expected RPM usage
- obtain face ROI image from FTL using projected face full-frame bbox
- pass landmarks transformed to face-crop-local coordinates
- call recognize per detected face
- if person_found true, enrich person_name using PersonDirectory

### Input contract verification
Verified:
- expects RGB/HWC/uint8/[0,255] image contract
- landmark keys and in-bounds coordinates validated against face ROI

### Output contract verification
Verified:
- returns person_found/person_id only
- unknown represented by person_found false and person_id UNKNOWN
- no person_name in FR output

### Runtime assumptions
- stateless per call except immutable enrolled gallery cache
- no per-frame mutable state

### Mismatches / undefined behavior
No active FR API mismatch found.

### Existing tests/evidence
- tests/face_recognition/test_face_recognition_module.py
  - output contract and threshold behavior
  - deterministic behavior
  - get_input_contract checks

---

## 3.6 RPM ↔ PersonDirectory

### Public APIs reviewed
Implementation:
- PersonDirectory.load() -> None
- PersonDirectory.get_person(person_id: str) -> PersonDirectoryOutput

Types:
- PersonDirectoryOutput: person_id, person_name, found
- PersonDirectoryConfig: json_file_path, is_optional

### Expected RPM usage
- initialize and load directory before any process_frame invocation
- call get_person(person_id) when FR returns person_found true
- enrich output person_name from lookup

### Input contract verification
Verified:
- get_person accepts person_id string
- load uses configured JSON file path and optional behavior

### Output contract verification
Verified:
- unknown lookup returns canonical UNKNOWN output in implementation
- lookup does not raise for unknown IDs

### Runtime assumptions
- startup load sequencing is crucial
- read path should be in-memory only

### Mismatches / undefined behavior
No active blocker remains in this seam after pre-load semantics alignment.

### Existing tests/evidence
- tests/person_directory/test_person_directory.py
  - load/validation/unknown output behavior
  - explicit pre-load fail-safe tests assert UNKNOWN/found=false and no exception

---

# 4. FTL Temporal & Projection Readiness

## CURRENT / PREVIOUS semantics
Verified in implementation and tests:
- CURRENT available after first successful ingest
- PREVIOUS unavailable until second successful ingest
- failed ingest does not rotate state
- per-camera CURRENT/PREVIOUS isolation exists

## Cold-start behavior
Documented and test-backed:
- PREVIOUS request before second ingest raises PreviousFrameNotAvailableError
- RPM spec expects cold-start early return with empty output

## Unknown camera behavior
- get_frame on unknown camera raises FrameNotFoundError

## get_frame contract readiness
- Returns ProcessedFrame with image, source_bbox_full_frame, spatial_transform
- Supports crop/convert geometry
- Uses temporal selector per camera

## source_bbox_full_frame semantics
- Represents crop rectangle in full-frame coordinate space
- Used as projection offset anchor

## spatial_transform semantics
- Encodes scaling and padding used in output geometry
- Required for inverse projection when geometry is non-identity

## ResizePolicy behavior verification
- NONE: identity transform expected
- LETTERBOX: uniform scaling + padding

## Exact inverse projection math RPM must implement
For an output-local point (ox, oy):
1. cx = ox - pad_left
2. cy = oy - pad_top
3. rx = cx / scale_x
4. ry = cy / scale_y
5. fx = rx + source_bbox_full_frame.x
6. fy = ry + source_bbox_full_frame.y

For bbox projection, apply the same transform to x/y and size terms using the same scale handling.

## Required SpatialCoordinator behavior
- consume local bbox + source_bbox_full_frame + spatial_transform
- apply inverse geometry first, then crop-origin translation
- reject invalid scale/padding metadata for non-identity geometry as integration error

## Already implemented/tested vs RPM work still required
Already implemented/tested:
- FTL returns required spatial metadata
- resize policy transforms and temporal behavior covered at module level

RPM must still implement:
- SpatialCoordinator logic for all policy cases
- consistent projection for Object and Face Detection outputs
- landmark re-expression from person-ROI-local to face-crop-local
- orchestration integration tests proving end-to-end projection correctness

---

# 5. Coordinate Space Consistency Review

## Coordinate-space glossary used here
- FULL_FRAME: coordinates relative to original camera frame
- ROI_LOCAL: coordinates relative to supplied stage ROI image
- CROP_LOCAL: coordinates relative to a narrower crop generated from a parent ROI

## Stage-by-stage mapping

### Motion Detection
- Input space: full-frame image in current design (RPM requests full frame)
- Output space: bbox relative to current_frame.image
- Projection requirement: none when current_frame.image is full-frame; if future ROI use is introduced, projection would be required

### Object Detection
- Input space: motion ROI image (ROI_LOCAL)
- Output space: person bboxes in motion ROI_LOCAL
- Projection requirement: required to full-frame via source_bbox_full_frame + spatial_transform

### Face Detection
- Input space: person ROI image (ROI_LOCAL relative to person crop)
- Output space:
  - face_bbox ROI_LOCAL relative to person ROI image
  - landmarks ROI_LOCAL relative to person ROI image
- Projection requirement:
  - face bbox projection to FULL_FRAME required
  - landmarks re-expression to face CROP_LOCAL required for FR input

### Face Recognition
- Input space:
  - face_roi_image in CROP_LOCAL
  - landmarks in CROP_LOCAL relative to face ROI image
- Output space:
  - identity only (person_found, person_id)
- Projection requirement: none in this stage

## Worked projection example (LETTERBOX)
Given:
- source_bbox_full_frame = {x: 100, y: 50, width: 200, height: 100}
- spatial_transform = {scale_x: 0.5, scale_y: 0.5, pad_left: 10, pad_top: 20}
- local point in transformed output: (ox, oy) = (60, 70)

Compute:
- cx = 60 - 10 = 50
- cy = 70 - 20 = 50
- rx = 50 / 0.5 = 100
- ry = 50 / 0.5 = 100
- fx = 100 + 100 = 200
- fy = 100 + 50 = 150

Projected full-frame point = (200, 150)

---

# 6. Shared Contract Verification

Authoritative contracts reviewed:
- Image
- BoundingBox
- GeometrySpec
- ResizePolicy
- OutputImageType
- SpatialTransform
- FaceLandmarks
- Point

## Classification of observed issues

1) Duplicate public definitions
- Status: Resolved / acceptable internal detail
- Notes: internal helper dataclasses in some modules are acceptable if public boundary uses shared shapes

2) Internal-only helper types
- Status: Acceptable internal detail
- Notes: internal dataclasses and helper records do not currently violate public API boundary intent

3) Stale contracts
- Status: Resolved
- Notes: PersonDirectory pre-load behavior now consistently documented as fail-safe UNKNOWN output

4) Incompatible shapes
- Status: Resolved
- Notes: GeometrySpec NONE width/height placeholder handling aligned between stage contracts, shared contracts, and FTL validator

5) Value range representation drift (spacing)
- Status: Future cleanup only
- Notes: [0, 255] vs [0,255] normalization is currently handled in face detection validator

---

# 7. PersonDirectory / Identity Enrichment Review

## Verification outcomes

- RPM does not need FaceGalleryLoader at runtime orchestration path:
  - Verified. FR receives enrolled identities at initialization; RPM runtime path is FR + PersonDirectory lookup only.

- FaceRecognition returns person_id only:
  - Verified. FR output contains person_found and person_id; no person_name.

- PersonDirectory enriches person_name:
  - Verified in design intent and implementation API.

- UNKNOWN semantics:
  - Verified in implementation: UNKNOWN output for unknown IDs.

- startup loading requirements:
  - Spec intends load-before-lookup.
  - Implementation does not enforce not-initialized exception.

- initialization ordering:
  - Must be explicit in RPM bootstrapping.

- lookup failure behavior:
  - returns found false with UNKNOWN values; no exception for unknown IDs.

## Exact intended RPM flow
- FaceRecognitionOutput.person_id (when person_found true)
- PersonDirectory.get_person(person_id)
- Enrich RecognizedFaceResult.person_name using PersonDirectoryOutput.person_name

---

# 8. Error Handling & Failure Semantics

## Classification matrix

- invalid input behavior:
  - Type: Empty output / fail-safe in stage modules
  - Notes: most modules catch and return empty/no-match outputs

- FTL ingest failure:
  - Type: Integration error at orchestration layer; RPM expected to short-circuit to empty output

- missing PREVIOUS frame:
  - Type: Recoverable cold-start path; empty output

- invalid bbox requests:
  - Type: Integration error from FTL crop validation

- unsupported ResizePolicy:
  - Type: Integration error (configuration/runtime contract failure)

- detector failures:
  - Type: Usually recoverable in-module with empty output (module dependent)

- recognizer failures:
  - Type: recoverable no-match output (person_found false)

- PersonDirectory lookup failures:
  - Type: warning/empty enrichment path via UNKNOWN output

- empty/no-match behavior:
  - Type: expected normal branch

- concurrency edge cases:
  - Type: mostly untested at orchestration level; hardening gap

---

# 9. Concurrency & Camera Isolation Review

## Reviewed assumptions
- per-camera CURRENT/PREVIOUS isolation in FTL
- FTL locking assumptions via RLock
- same-camera sequential processing expected externally in RPM spec
- multi-camera parallel safety documented as dependent on FTL concurrency safety
- concurrent ingest/get_frame behavior partially covered in tests

## Verified guarantees
- multi-camera state isolation in FTL tests
- no cross-camera contamination tests present

## Untested assumptions
- high-contention stress patterns (same camera ingest/get interleaving under load)

---

# RPM Stub Implementation Plan — Phase 1 Results

Review date: 2026-05-13
Scope: Phase 1 stub implementation — real RPM orchestration with fake/stub dependencies

## Implemented real RPM components

All production RPM source files created under `src/image_processing/recognition_pipeline_manager/`:

| File | Responsibility |
|---|---|
| `contracts.py` (FTL) | Extracted 6 stable FTL public types so RPM never imports from `module.py` |
| `interfaces.py` | `@runtime_checkable Protocol` definitions for all 6 injectable dependencies |
| `types.py` | `RecognitionPipelineOutput`, `PersonResult`, `RecognizedFaceResult` (public); `PipelineResult` (internal) |
| `spatial_coordinator.py` | Projects ROI-local bounding boxes to full-frame using NONE / LETTERBOX transforms |
| `input_validator.py` | Validates `FramePacket` fields before entering the pipeline |
| `output_builder.py` | Assembles final `RecognitionPipelineOutput` from pipeline accumulator |
| `pipeline_orchestrator.py` | Executes the four-stage pipeline (Motion → OD → FD → FR) for one frame |
| `recognition_pipeline_manager.py` | Public entry point — composes validator, orchestrator, and output builder |
| `__init__.py` | Package re-exports (public API + interfaces only; no internal types) |

Key properties enforced:
- `Image.data` never mutated (C1)
- Stage input contracts cached once at `PipelineOrchestrator.__init__` (C2)
- Traceability fields (`frame_id`, `camera_id`, `timestamp_ms`) propagated unchanged (C3)
- All errors caught per spec §11 — always returns structurally valid output

## Fake/stub dependencies

Test doubles in `tests/recognition_pipeline_manager/`:

| Fake | Wraps | Configurable behaviours |
|---|---|---|
| `FakeFTL` | `FrameTransformationLayerInterface` | `raise_on_ingest`, `raise_on_previous` (cold start), per-call-index `get_frame` failures, per-call-index `SpatialTransform` overrides |
| `FakeMotionDetection` | `MotionDetectionInterface` | `result: MotionResult` |
| `FakeObjectDetection` | `ObjectDetectionInterface` | `result: PersonDetectionResult` |
| `FakeFaceDetection` | `FaceDetectionInterface` | `result: FaceDetectionOutput` |
| `FakeFaceRecognition` | `FaceRecognitionInterface` | `result: FaceRecognitionOutput` |
| `FakePersonDirectory` | `PersonDirectoryInterface` | `lookup_map: dict[str, PersonDirectoryOutput]` |

All fakes record every call with full arguments. `FakeFTL` propagates `frame_id`/`timestamp_ms` from the last ingested `FramePacket` into returned `ProcessedFrame` instances (C3).

Test results: **55 tests passed, 0 failed** (`tests/recognition_pipeline_manager/`).

## Remaining Phase 2 integration work

The following tasks are deferred to Phase 2 when real ML backends are available:

- Replace `FakeFTL` with a real `FrameTransformationLayer` instance wired to actual camera ingestion
- Replace `FakeMotionDetection` / `FakeObjectDetection` / `FakeFaceDetection` / `FakeFaceRecognition` with real module implementations
- Replace `FakePersonDirectory` with a real gallery-backed `PersonDirectory`
- Integration tests with real images to verify projection accuracy end-to-end
- Performance / latency profiling under concurrent camera workloads
- full pipeline multi-camera concurrency behavior at orchestration level
- orchestration-level race/failure handling (RPM not implemented yet)

## Required future tests
- orchestrator-level multi-camera concurrency tests with deterministic stubs
- contention stress around rapid ingest and downstream get_frame calls

---

# 10. Required RPM Stub/Test Infrastructure

## Recommended pre-RPM test doubles
- FakeFTL implementing ingest_frame and get_frame with deterministic ProcessedFrame returns
- FakeMotionDetection returning configurable motion/no-motion outcomes
- FakeObjectDetection returning deterministic ROI-local person boxes
- FakeFaceDetection returning deterministic ROI-local face boxes and landmarks
- FakeFaceRecognition returning deterministic person_found/person_id outcomes
- FakePersonDirectory returning deterministic lookup results and UNKNOWN cases

## Why stub-based tests are required before real-module integration
- isolate orchestration correctness from ML variability
- validate sequencing, branching, and projection math deterministically
- enforce full-frame output contract correctness early
- detect integration regressions faster than model-backed end-to-end tests

---

# 11. Golden Scenario Definitions

## 1. Cold start
- Expected stage flow: ingest -> PREVIOUS unavailable -> short-circuit
- Expected output: persons empty
- Required assertions: no downstream stage invocation; deterministic empty output

## 2. No motion
- Expected stage flow: motion returns detected false -> short-circuit
- Expected output: persons empty
- Required assertions: object/face/recognition not called

## 3. Motion with no person
- Expected stage flow: motion yes -> object no persons
- Expected output: persons empty
- Required assertions: face detection/recognition not called

## 4. Person with no face
- Expected stage flow: motion yes -> object persons -> face detection empty
- Expected output: persons present with recognized_faces empty
- Required assertions: recognition not called for that person

## 5. Face unrecognized
- Expected stage flow: face detected -> FR person_found false
- Expected output: face omitted from recognized_faces
- Required assertions: no placeholder recognized face entry

## 6. Fully recognized person
- Expected stage flow: face detected -> FR recognized -> PersonDirectory enrichment
- Expected output: recognized_faces entry with full-frame face bbox, person_id, person_name
- Required assertions: enrichment call exactly once per recognized face

## 7. Multiple persons
- Expected stage flow: per-person loop with independent face pipelines
- Expected output: multiple PersonResult entries, each with own recognized_faces
- Required assertions: no cross-person mixing

## 8. Multiple cameras
- Expected stage flow: independent per-camera invocations
- Expected output: camera-isolated results
- Required assertions: no cross-camera state leakage

## 9. Invalid ROI
- Expected stage flow: FTL get_frame/crop error on affected branch
- Expected output: partial/empty depending on error stage and policy
- Required assertions: controlled handling, no crash

## 10. LETTERBOX projection case
- Expected stage flow: ROI-local detections projected via spatial_transform with padding removal
- Expected output: accurate full-frame bbox placement
- Required assertions: numeric projection equality against expected coordinates

---

# 12. Resolved Contract Alignment Items

## B1 Resolution: GeometrySpec NONE width/height semantics
- Final decision implemented:
  - For `ResizePolicy.NONE`, width and height are ignored by FTL geometry validation.
  - `width=0` and `height=0` are valid placeholders.
  - RPM must pass stage contracts through directly; no normalization or patching required.
- Evidence:
  - `src/image_processing/frame_transformation_layer/module.py` now accepts `NONE` width/height values when integer `>= 0`.
  - `tests/frame_transformation_layer/test_frame_transformation_layer_module.py` includes:
    - NONE placeholder success test (`width=0`, `height=0`)
    - cross-module compatibility tests using real `MotionDetectionManager.get_input_contract()` and `FaceRecognitionModule.get_input_contract()` outputs
    - identity `spatial_transform` assertions for NONE.

## B2 Resolution: PersonDirectory pre-load semantics
- Final decision implemented:
  - Keep current fail-safe behavior.
  - `PersonDirectory.get_person()` remains fail-safe and returns UNKNOWN/found=false before or without load.
  - Before/without load, return canonical UNKNOWN output with `found=false`.
- Evidence:
  - `tests/person_directory/test_person_directory.py` now includes explicit pre-load and failed-load fail-safe tests asserting UNKNOWN/found=false with no exception.
  - `doc/image_processing_service/PersonDirectory.md` now documents startup load as recommended and pre-load lookup as fail-safe.

## Image.data ownership and mutability semantics finalized
- Contract:
  - Consumers treat `Image.data` / `ProcessedFrame.image.data` as read-only.
  - Mutation across module boundaries is forbidden.
  - Whether FTL returns copies or references is an implementation detail and not a contract guarantee.
- Rationale:
  - Avoids mandatory defensive-copy guarantees.
  - Preserves performance flexibility in FTL and downstream stages.
  - Keeps pipeline behavior safe and predictable at module boundaries.
- Evidence:
  - `doc/image_processing_service/shared_contracts.md`
  - `doc/image_processing_service/frame_transformation_layer.md`
  - `doc/image_processing_service/RecognitionPipelineManager.md`

---

# 13. Non-Blocking Hardening Gaps

These are not blockers for starting RPM implementation.

- Missing orchestration-level end-to-end tests (RPM not implemented yet)
- Concurrency stress coverage beyond current FTL isolation tests
- Projection fuzz tests across random boxes and supported ResizePolicy variants (NONE and LETTERBOX)
- Performance profiling under dense multi-person scenes
- Future orchestration tests once RPM implementation begins

---

# 14. Final Go/No-Go Decision

## Can RPM implementation begin now?
YES — READY FOR RPM IMPLEMENTATION

## What blockers remain?
No API/contract blockers remain.

## What must be implemented/fixed before first RPM coding?
No additional contract-alignment fixes are required.

## What can be deferred until RPM v2/hardening?
- concurrency stress expansion
- projection fuzzing/performance hardening
- future orchestration integration tests

## Is the API surface stable enough to freeze?
Yes. The RPM-facing API surface and runtime contract semantics are stable enough to begin implementation.

---

## Appendix A: Modules Reviewed

- Frame Transformation Layer
- Motion Detection
- Object Detection
- Face Detection
- Face Recognition
- PersonDirectory
- FaceGalleryLoader (startup dependency boundary check)
- Shared contracts

## Appendix B: Major mismatch categories found

- API mismatch:
  - GeometrySpec NONE semantics mismatch across module contracts and FTL
- Architecture mismatch:
  - none active beyond contract semantics issue above
- Missing implementation:
  - RPM itself (expected and in-scope for upcoming work)
- Missing test coverage:
  - orchestration-level and projection integration tests
- Production hardening gap:
  - concurrency stress and projection fuzz coverage
