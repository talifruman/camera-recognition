# RecognitionPipelineManager ↔ Motion Detection Compatibility Review

---

## 1. Reviewed Files

| File | Type | Purpose |
|------|------|---------|
| `doc/image_processing_service/RecognitionPipelineManager.md` | Specification | RPM design, interfaces, orchestration flow |
| `doc/image_processing_service/motion_detection.md` | Specification | Motion Detection module design and contracts |
| `src/image_processing/motion_detection/module.py` | Source | Motion Detection implementation (~450 lines) |
| `src/image_processing/motion_detection/__init__.py` | Source | Public exports |
| `tests/motion_detection/test_motion_detection_module.py` | Tests | Unit tests covering validation, pipeline, determinism |
| `tests/motion_detection/run_visual_test.py` | Tests | Visual integration test runner |

**Not yet implemented:** No source code exists for `RecognitionPipelineManager`, `PipelineOrchestrator`, `FrameTransformationLayerInterface`, or `SpatialCoordinator`. Analysis of RPM is documentation-only.

---

## 2. Current Integration Flow

### How RPM Invokes Motion Detection

**Spec-defined flow (RPM §8.5.1 and §8.8):**

1. External caller invokes `RecognitionPipelineManager.process_frame(frame_packet: FramePacket)`.
2. `RecognitionPipelineManager` validates `frame_packet` via `RecognitionPipelineInputValidator`.
3. `RecognitionPipelineManager` calls `orchestrator.execute(frame_packet)`.
4. `PipelineOrchestrator` calls `FrameTransformationLayerInterface.ingest_frame(frame_packet)` → void.
5. `PipelineOrchestrator` calls `FTL.get_frame(camera_id, CURRENT, full_frame_bbox, motion_contract.output_type, motion_contract.geometry_spec)` → `current_processed: ProcessedFrame`.
6. `PipelineOrchestrator` calls `FTL.get_frame(camera_id, PREVIOUS, full_frame_bbox, motion_contract.output_type, motion_contract.geometry_spec)` → `previous_processed: ProcessedFrame`. If `PreviousFrameNotAvailableError` → return empty result (cold start).
7. `PipelineOrchestrator` constructs `MotionDetectionInput` and calls `MotionDetectionInterface.detect(MotionDetectionInput)` → `MotionResult`.

### Data Passed to Motion Detection

`MotionDetectionInput` as constructed by RPM (`PipelineOrchestrator`):

```
MotionDetectionInput {
    current_frame: FramePacket {
        frame_id:     frame_packet.frame_id,
        camera_id:    frame_packet.camera_id,
        timestamp_ms: current_processed.timestamp_ms,
        image:        current_processed.image      ← GRAY HWC uint8, full frame
    },
    previous_frame: FramePacket {
        frame_id:     previous_processed.frame_id, ← sourced from FTL ProcessedFrame
        camera_id:    frame_packet.camera_id,
        timestamp_ms: previous_processed.timestamp_ms,
        image:        previous_processed.image     ← GRAY HWC uint8, full frame
    }
}
```

### Internal Components Responsible

| Component | Responsibility |
|-----------|----------------|
| `PipelineOrchestrator` | Retrieves frames from FTL, constructs `MotionDetectionInput`, calls `MotionDetectionInterface.detect()` |
| `FrameTransformationLayerInterface` | Stores frames, maintains CURRENT/PREVIOUS per `camera_id`, converts image to required format/geometry |
| `MotionDetectionInterface` | Abstraction RPM depends on; expects `detect()` and `get_input_contract()` |
| `MotionDetectionManager` | Actual implementation; exposes `process()` — does NOT implement `MotionDetectionInterface` |

### Assumptions on Each Side

**RPM assumptions about Motion Detection:**
- Motion Detection implements `MotionDetectionInterface` with methods `detect(MotionDetectionInput) -> MotionResult` and `get_input_contract() -> PipelineStageInputContract`
- `get_input_contract()` returns an `OutputImageType` of GRAY HWC uint8 so the FTL delivers correctly formatted frames
- `MotionResult.bboxes` coordinates are relative to `current_frame.image` (the full-frame image supplied by RPM)
- On any failure, `MotionResult.detected = false` and `bboxes = []`

**Motion Detection assumptions about its caller:**
- Caller supplies both `current_frame` and `previous_frame` as fully populated `FramePacket` objects
- Images are already GRAY, HWC, uint8 — preprocessing was done upstream
- Frames are from the same camera (`camera_id` equal in both)
- `previous_frame.timestamp_ms ≤ current_frame.timestamp_ms`
- Frame dimensions of current and previous are identical

### Frame Source and Pairing Summary

| Aspect | Value |
|--------|-------|
| Current frame source | `FTL.get_frame(camera_id, CURRENT, full_frame_bbox, ...)` |
| Previous frame source | `FTL.get_frame(camera_id, PREVIOUS, full_frame_bbox, ...)` |
| Frame pairing logic | FTL's internal per-camera CURRENT/PREVIOUS slots; RPM never stores frame references |
| Image source | `ProcessedFrame.image` returned by FTL |
| Coordinate space | Full-frame; `MotionResult.bboxes` are relative to `current_frame.image` (full frame) |
| `frame_id` handling | `current_frame.frame_id = frame_packet.frame_id`; `previous_frame.frame_id = previous_processed.frame_id` |
| `camera_id` handling | `frame_packet.camera_id` propagated to both frames |
| Image contract | GRAY, HWC, uint8 — enforced by FTL using `motion_contract.output_type` from `get_input_contract()` |
| ROI/full-frame | Both frames are full-frame images; bboxes returned are regions within that full frame |
| Cold start | Handled entirely by RPM before calling MD — FTL raises `PreviousFrameNotAvailableError`; RPM returns empty result immediately; MD is never invoked |

---

## 3. Compatibility Verification Checklist

### 3.1 API + Method Contracts

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| Public method name | `MotionDetectionInterface.detect(input: MotionDetectionInput) -> MotionResult` | `MotionDetectionManager.process(motion_input: MotionDetectionInput) -> MotionResult` | ❌ NO | Rename `process` → `detect` in `MotionDetectionManager` and in spec |
| `get_input_contract()` method | `MotionDetectionInterface.get_input_contract() -> PipelineStageInputContract` (required) | Not defined anywhere in `MotionDetectionManager` | ❌ NO | Add `get_input_contract()` to `MotionDetectionManager` |
| Return type | `MotionResult` | `MotionResult` | ✅ YES | None |
| Sync/async | Synchronous | Synchronous | ✅ YES | None |
| Input type | `MotionDetectionInput` | `MotionDetectionInput` | ✅ YES | None |
| Batch/single | Single invocation | Single invocation | ✅ YES | None |
| Parameter name | `input` (interface pseudocode) | `motion_input` (code) | ✅ YES (no impact in Python) | None |

### 3.2 Input Structures

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| `current_frame` field name | `current_frame` | `current_frame` | ✅ YES | None |
| `previous_frame` field name | `previous_frame` | `previous_frame` | ✅ YES | None |
| `current_frame` type | `FramePacket` | `FramePacket` | ✅ YES | None |
| `previous_frame` type | `FramePacket` | `FramePacket` | ✅ YES | None |
| `FramePacket.frame_id` type | `string` | `str` | ✅ YES | None |
| `FramePacket.camera_id` type | `string` | `str` | ✅ YES | None |
| `FramePacket.timestamp_ms` type | `uint64` (spec) | `int` (Python annotation) | ⚠️ MINOR | Align code annotation with spec intent |
| `FramePacket.image` type | `Image` / `ProcessedFrame.image` | `Any` (numpy.ndarray at runtime) | ✅ YES | None |
| Missing fields in `MotionDetectionInput` | None | None | ✅ YES | None |
| Extra fields | None | None | ✅ YES | None |
| Nullable expectations | Both frames must be non-null | Enforced by `InputValidator` | ✅ YES | None |

### 3.3 `frame_id` Verification

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| `current_frame.frame_id` value | `frame_packet.frame_id` | Must exist and be non-null | ✅ YES | None |
| `previous_frame.frame_id` value | `previous_processed.frame_id` (from FTL `ProcessedFrame`) | Must exist and be non-null | ⚠️ AMBIGUITY | Verify `ProcessedFrame` carries `frame_id`; document dependency |
| `frame_id` type | `string` | `str` | ✅ YES | None |
| `frame_id` semantics | Unique identifier; preserved for traceability | Preserved for traceability | ✅ YES | None |
| `frame_id` propagation | Not forwarded to `MotionResult`; MD uses it for traceability only | Same | ✅ YES | None |

### 3.4 `camera_id` Verification

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| `camera_id` source | `frame_packet.camera_id` for both frames | Must exist, non-empty, equal in both frames | ✅ YES | None |
| `camera_id` type | `string` | `str` | ✅ YES | None |
| Cross-frame equality | Both frames share `frame_packet.camera_id` | `InputValidator` requires `current_frame.camera_id == previous_frame.camera_id` | ✅ YES | None |
| `camera_id` propagation to output | Not in `MotionResult`; MD uses it for traceability | Same | ✅ YES | None |

### 3.5 Previous Frame Handling

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| Who owns previous frame retrieval | RPM `PipelineOrchestrator` via FTL `PREVIOUS` slot | Caller provides `previous_frame` in input | ✅ YES (aligned) | None |
| Previous frame availability | FTL raises `PreviousFrameNotAvailableError`; RPM catches before calling MD | Not MD's concern; receives fully populated input | ✅ YES | None |
| Cold start behavior | RPM returns empty result before calling MD; MD never receives a cold-start input | `InputValidator` rejects null `previous_frame`; returns `detected=False` | ✅ YES (never reaches MD) | None |
| Missing previous frame behavior | RPM returns empty `RecognitionPipelineOutput`; MD not called | `detected=False` returned if `previous_frame` is null or invalid | ✅ YES | None |
| First-frame behavior | Cold start — FTL has no PREVIOUS slot populated; RPM handles entirely | Not MD's responsibility | ✅ YES | None |

### 3.6 Image Contract

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| Color format | Declared via `get_input_contract()` → `OutputImageType` | `GRAY` | ❌ NO (indirectly) | `get_input_contract()` must be implemented to declare GRAY |
| Layout | Declared via `get_input_contract()` → `OutputImageType` | `HWC` | ❌ NO (indirectly) | Same |
| Dtype | Declared via `get_input_contract()` → `OutputImageType` | `uint8` | ❌ NO (indirectly) | Same |
| Value range | `[0, 255]` implied by uint8 GRAY | `[0, 255]` | ✅ YES | None |
| Resize/geometry | Declared via `get_input_contract()` → `GeometrySpec`; FTL handles | Not explicitly defined in MD | ⚠️ AMBIGUITY | `get_input_contract()` must define `GeometrySpec` for full-frame retrieval |
| Preprocessing ownership | FTL performs all preprocessing | Caller delivers pre-processed frames; MD does no preprocessing | ✅ YES (aligned) | None |
| Dimension matching | Both frames from same `get_frame` geometry → same size | `InputValidator` requires identical dimensions | ✅ YES | None |

### 3.7 Coordinate Systems

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| `MotionResult.bboxes` coordinate space | Relative to `current_frame.image` (full frame) | Relative to `current_frame.image` (image used for measurement) | ✅ YES | None |
| Bboxes used as crop regions | `motion_region_bbox` passed to FTL for object detection crops | MD does not define downstream use | ✅ YES | None |
| ROI-local vs full-frame | Motion bboxes are full-frame because RPM supplies full-frame image | MD produces bboxes relative to input image (full-frame when RPM calls it) | ✅ YES | None |
| Projection by `SpatialCoordinator` | NOT applied to motion bboxes; used directly as crop coords | No projection defined or expected | ✅ YES | None |
| Origin | Top-left corner of `current_frame.image` | Top-left corner of `current_frame.image` | ✅ YES | None |
| Axis alignment | Axis-aligned rectangle | Axis-aligned rectangle | ✅ YES | None |
| Coordinates within image bounds | Guaranteed by `MotionDetectionAlgorithm` output | `InputValidator` enforces positive dimensions; algorithm clamps to image bounds | ✅ YES | None |

### 3.8 Output Structures

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| `MotionResult.detected` name | `detected` | `detected` | ✅ YES | None |
| `MotionResult.bboxes` name | `bboxes` | `bboxes` | ✅ YES | None |
| `MotionResult.detected` type | `bool` | `bool` | ✅ YES | None |
| `MotionResult.bboxes` type | `vector<BoundingBox>` (RPM pseudocode) | `list[BoundingBox]` | ✅ YES | None |
| `BoundingBox` vs `CanonicalBoundingBox` | RPM uses `CanonicalBoundingBox` for its own outputs; `BoundingBox` for FTL `region_bbox` | `BoundingBox` | ⚠️ MEDIUM | No shared type; structurally identical but named differently; no canonical source |
| `BoundingBox` fields | `x: int32, y: int32, width: int32, height: int32` | `x: int, y: int, width: int, height: int` | ✅ YES (structurally) | None |
| Empty output when no motion | `bboxes = []` when `detected = false` | `bboxes = []` when `detected = False` | ✅ YES | None |
| Extra fields in `MotionResult` | None expected | None present — `{detected, bboxes}` only | ✅ YES | None |

### 3.9 Configuration Expectations

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| Threshold parameters | Not defined by RPM; queried via `get_input_contract()` | `motion_threshold`, `motion_fraction_threshold`, `min_bbox_area` in `MotionDetectionConfig` | ✅ YES | None (MD owns its config) |
| Runtime config injection | None — config immutable after init | None — immutable after init | ✅ YES | None |
| Per-call config | Not allowed | Not allowed | ✅ YES | None |
| Duplicated configuration responsibilities | None — each module owns its own | None | ✅ YES | None |

### 3.10 Error Handling

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| Motion Detection stage failure | Catch internally; return `RecognitionPipelineOutput{persons=[]}` | Catch internally; return `MotionResult{detected=False, bboxes=[]}` | ✅ YES | None |
| Invalid frame handling | Cold start handled before calling MD | `InputValidator` returns `detected=False` for invalid input | ✅ YES | None |
| Missing previous frame | RPM catches `PreviousFrameNotAvailableError` before MD is called | `detected=False` if `previous_frame` is null or invalid | ✅ YES | None |
| Unknown camera handling | Per-camera FTL state is independent per `camera_id` | Not MD's concern — receives populated input | ✅ YES | None |
| Validation behavior | Both modules validate their direct inputs | `InputValidator` enforces 15+ constraints | ✅ YES | None |

### 3.11 Diagrams

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| Sequence diagram — method called on MD | `Orch->>Motion: detect(MotionDetectionInput)` | `Caller->>Manager: process(input)` | ❌ NO | MD sequence diagram must reflect `detect()` after rename |
| Class diagram — method name | `MotionDetectionInterface.detect()` | `MotionDetectionManager.process()` | ❌ NO | Rename in code; update MD class diagram |
| Class diagram — `get_input_contract()` | Present on all stage interfaces | Not present in MD class diagram | ❌ NO | Add `get_input_contract()` to MD class diagram after implementation |
| Data flow diagram — consistency | Consistent with sequence | Consistent internally | ✅ YES | None |
| Cold start path | Shown in RPM sequence before MD invocation | Not applicable to MD | ✅ YES | None |

---

## 4. Detected Mismatches

### Mismatch 1 — Method Name: `detect` vs `process`

**Files involved:**
- `doc/image_processing_service/RecognitionPipelineManager.md` — §6.1 `MotionDetectionInterface`, §8.5.1 invocation, §8.8 end-to-end flow, §18 compliance checklist
- `src/image_processing/motion_detection/module.py` — `MotionDetectionManager.process()`
- `src/image_processing/motion_detection/__init__.py` — exports `MotionDetectionManager`
- `doc/image_processing_service/motion_detection.md` — §4 Public API, §8.1, §8.6, §14 class diagram, §15 sequence diagram

**Exact structures involved:**
- RPM: `MotionDetectionInterface.detect(input: MotionDetectionInput) -> MotionResult`
- MD code: `def process(self, motion_input: MotionDetectionInput) -> MotionResult`
- MD spec §4: `MotionResult process(input: MotionDetectionInput)`

**Why this is problematic:** RPM calls `.detect()` on the injected `MotionDetectionInterface`. The actual `MotionDetectionManager` has no `detect()` method. At runtime, wiring `MotionDetectionManager` as the `MotionDetectionInterface` implementation results in an immediate `AttributeError`. Integration is completely broken without a rename or adapter.

**Which module should change:** Motion Detection — rename `process()` to `detect()` in both the implementation class and the specification.

**Recommended fix:**
- Rename `process` → `detect` in `MotionDetectionManager` in `module.py`
- Update motion_detection spec §4, §8.1, §8.6
- Update §14 class diagram: `MotionDetectionManager.detect(input: MotionDetectionInput) -> MotionResult`
- Update §15 sequence diagram: `Caller->>Manager: detect(input)`
- Update all test helpers that call `.process()` → `.detect()`

**Impact level:** 🔴 Critical

---

### Mismatch 2 — Missing `get_input_contract()` Method

**Files involved:**
- `doc/image_processing_service/RecognitionPipelineManager.md` — §6.1 `MotionDetectionInterface`, §9.1 Configuration, §13.1 Initialization
- `src/image_processing/motion_detection/module.py` — entire `MotionDetectionManager` class
- `doc/image_processing_service/motion_detection.md` — §4 Public API, §8.1, §14 class diagram

**Exact structures involved:**
- RPM requires: `MotionDetectionInterface.get_input_contract() -> PipelineStageInputContract`
- MD code: no such method exists anywhere in `MotionDetectionManager`
- MD spec: no mention of `get_input_contract()`, `PipelineStageInputContract`, `OutputImageType`, or `GeometrySpec`

**Why this is problematic:** RPM queries every pipeline stage's `get_input_contract()` at initialization to determine the `OutputImageType` and `GeometrySpec` to pass to the FTL when requesting processed frames. If `MotionDetectionManager` does not implement this method, RPM initialization fails before any frame is processed. The FTL cannot deliver correctly formatted GRAY HWC uint8 frames to Motion Detection without this contract being declared.

**Which module should change:** Motion Detection — add `get_input_contract()` method.

**Recommended fix:**
- Define `PipelineStageInputContract` type (in a shared module) with `output_type: OutputImageType` and `geometry_spec: GeometrySpec`
- Add `get_input_contract()` to `MotionDetectionManager`:
  ```python
  def get_input_contract(self) -> PipelineStageInputContract:
      return PipelineStageInputContract(
          output_type=OutputImageType.GRAY_HWC_UINT8,
          geometry_spec=GeometrySpec.FULL_FRAME,
      )
  ```
- Update MD spec §4 and §8.1 to document this method
- Add `get_input_contract()` to §14 class diagram
- Add `PipelineStageInputContract` to exports or shared types

**Impact level:** 🔴 Critical

---

### Mismatch 3 — `BoundingBox` vs `CanonicalBoundingBox` — No Shared Type

**Files involved:**
- `doc/image_processing_service/RecognitionPipelineManager.md` — §3.1 output structs (uses `CanonicalBoundingBox`), §14 class diagram (`CanonicalBoundingBox`), FTL interface signature (`region_bbox: BoundingBox`)
- `doc/image_processing_service/motion_detection.md` — §3.1 (uses `BoundingBox`)
- `src/image_processing/motion_detection/module.py` — `class BoundingBox(TypedDict): x: int; y: int; width: int; height: int`
- `src/image_processing/motion_detection/__init__.py` — exports `BoundingBox`

**Exact structures involved:**
- MD: `class BoundingBox(TypedDict): x: int; y: int; width: int; height: int`
- RPM PersonResult: `person_bbox: CanonicalBoundingBox`
- RPM FTL interface: `get_frame(..., region_bbox: BoundingBox, ...)` (uses `BoundingBox` for the motion region crop)

**Why this is problematic:** `MotionResult.bboxes` produces `list[BoundingBox]`. RPM uses each entry as the `motion_region_bbox` argument to `FTL.get_frame(...)`. Two structurally identical but differently named types exist in the same integration path with no shared definition. RPM's output types use `CanonicalBoundingBox` while the FTL interface uses `BoundingBox`. This inconsistency is a maintenance hazard: if one type gains a field (e.g., a confidence score or `frame_relative: bool`), the other will silently diverge.

**Which module should change:** Both — establish a shared `BoundingBox` / `CanonicalBoundingBox` type definition and align naming across all specs and code.

**Recommended fix:**
- Create a shared types module (e.g., `src/image_processing/shared_types.py`) containing a single `BoundingBox` (or `CanonicalBoundingBox`) definition
- All modules import from it rather than defining their own
- Align RPM spec to use one consistent name throughout

**Impact level:** 🟡 Medium

---

### Mismatch 4 — No Formal `MotionDetectionInterface` Implementation

**Files involved:**
- `doc/image_processing_service/RecognitionPipelineManager.md` — §6.1 defines `MotionDetectionInterface`
- `src/image_processing/motion_detection/module.py` — `MotionDetectionManager` is a plain Python class with no Protocol inheritance, no base class, no structural subtyping declaration

**Exact structures involved:**
- RPM defines `MotionDetectionInterface` with `detect()` and `get_input_contract()` as an interface
- `MotionDetectionManager` is a standalone class; no relationship to `MotionDetectionInterface` is declared or enforced

**Why this is problematic:** There is no compile-time or runtime verification that `MotionDetectionManager` satisfies `MotionDetectionInterface`. If the interface adds a method (e.g., `reset()` or `health_check()`), `MotionDetectionManager` will silently be non-compliant until a runtime call fails. The existing `MotionDetectionAlgorithm` Protocol in the same module demonstrates the team already uses structural subtyping — this pattern is not applied to the public module interface itself.

**Which module should change:** Motion Detection — define a `MotionDetectionInterface` Protocol (or inherit from one in a shared location) and verify `MotionDetectionManager` satisfies it.

**Recommended fix:**
- Define `MotionDetectionInterface` as a `@runtime_checkable Protocol` in a shared module or within MD's module
- Ensure `MotionDetectionManager` structurally satisfies it after the rename (Mismatch 1) and `get_input_contract()` addition (Mismatch 2)
- Add `assert isinstance(manager, MotionDetectionInterface)` assertion in tests

**Impact level:** 🟡 Medium

---

### Mismatch 5 — `FramePacket` Defined in Motion Detection — No Shared Source

**Files involved:**
- `src/image_processing/motion_detection/module.py` — defines `class FramePacket(TypedDict): ...`
- `src/image_processing/motion_detection/__init__.py` — exports `FramePacket`
- `doc/image_processing_service/RecognitionPipelineManager.md` — uses `FramePacket` throughout; provides no Python definition

**Exact structures involved:**
- MD code: `class FramePacket(TypedDict): frame_id: str; camera_id: str; timestamp_ms: int; image: Any`
- RPM spec: `struct FramePacket { string frame_id; string camera_id; uint64 timestamp_ms; Image image; }` (pseudocode only)

**Why this is problematic:** RPM constructs `FramePacket` objects for `MotionDetectionInput.current_frame` and `previous_frame`. The only Python `FramePacket` definition lives in the Motion Detection module. When implementing RPM, the developer must either import `FramePacket` from `motion_detection` — creating an unintended cross-module dependency — or redefine it independently, creating drift risk. Neither is appropriate for a shared pipeline type.

**Which module should change:** Both — `FramePacket` should be moved to a shared types module.

**Recommended fix:**
- Create `src/image_processing/shared_types.py` with `FramePacket` and `BoundingBox`
- All modules import from the shared module
- Remove the `FramePacket` definition from `motion_detection/module.py`

**Impact level:** 🟡 Medium

---

### Mismatch 6 — `timestamp_ms` Type Annotation Inconsistency

**Files involved:**
- `src/image_processing/motion_detection/module.py` — `FramePacket.timestamp_ms: int`
- `doc/image_processing_service/motion_detection.md` — `timestamp_ms: uint64`
- `doc/image_processing_service/RecognitionPipelineManager.md` — `timestamp_ms: uint64`

**Exact structures involved:**
- MD code: `timestamp_ms: int`
- Both specs: `timestamp_ms: uint64`

**Why this is problematic:** While Python's `int` handles arbitrary positive values, the annotation diverges from the spec. The `InputValidator` checks temporal ordering but does not validate `timestamp_ms >= 0`. A negative timestamp would pass validation, violating the `uint64` semantic contract implied by both specs.

**Which module should change:** Motion Detection — align annotation intent and add non-negative validation.

**Recommended fix:**
- Add `timestamp_ms >= 0` check to `InputValidator._validate_frame()`
- Consider using a type alias `Timestamp = int` with a docstring noting `uint64` semantics

**Impact level:** 🔵 Minor

---

### Mismatch 7 — Stale Method Name in Motion Detection Diagrams

**Files involved:**
- `doc/image_processing_service/motion_detection.md` — §14 class diagram, §15 sequence diagram

**Exact locations:**
- §14 class diagram: `MotionDetectionManager { +process(input: MotionDetectionInput) MotionResult }`
- §15 sequence diagram: `Caller->>Manager: process(input)`

**Why this is problematic:** After the required rename (Mismatch 1 fix), both diagrams will show stale method names. A developer reading the MD spec would see `process()` while the code exposes `detect()`, creating confusion and breaking the spec-as-source-of-truth principle.

**Which module should change:** Motion Detection documentation — update diagrams atomically with the code rename.

**Recommended fix:**
- Update §14 class diagram: `MotionDetectionManager { +detect(input: MotionDetectionInput) MotionResult }`
- Update §15 sequence diagram: `Caller->>Manager: detect(input)`
- Apply changes at the same time as the code rename to avoid a window of inconsistency

**Impact level:** 🔵 Minor

---

### Ambiguity 1 — `ProcessedFrame.frame_id` Assumption

**Files involved:**
- `doc/image_processing_service/RecognitionPipelineManager.md` — §8.5.1 constructs `previous_frame.frame_id = previous_processed.frame_id`
- `src/image_processing/motion_detection/module.py` — `InputValidator` requires `previous_frame.frame_id` must exist and be non-null

**Description:**
RPM's `MotionDetectionInput` construction sets `previous_frame.frame_id = previous_processed.frame_id`, implying the `ProcessedFrame` type (a FTL return type) carries a `frame_id` field. If `ProcessedFrame` does not include `frame_id`, RPM cannot satisfy Motion Detection's `InputValidator` requirement that `previous_frame.frame_id` is present and non-null. The failure would be silent — Motion Detection returns `detected=False` with no propagated error to the RPM caller. The FTL module specification was not in scope for this review.

**Required action:** Verify the `ProcessedFrame` definition in the FTL specification includes `frame_id`. Document this dependency explicitly in both RPM and FTL specs.

**Impact level:** 🟡 Medium (pending FTL verification)

---

## 5. Recommended Resolution Plan

Steps are ordered by severity and blocking dependency.

1. **Define shared types: `PipelineStageInputContract`, `OutputImageType`, `GeometrySpec`** — These are prerequisites for step 3. RPM cannot call `get_input_contract()` without these types being available for both modules to use. Candidate location: `src/image_processing/shared_types.py` or an RPM-scoped types module.

2. **Rename `MotionDetectionManager.process()` → `detect()`** — Update `module.py`, update `__init__.py` if needed, update all test files that call `.process()` → `.detect()`. This directly unblocks runtime integration. (Resolves Mismatch 1.)

3. **Implement `MotionDetectionManager.get_input_contract()`** — Returns `PipelineStageInputContract` specifying GRAY HWC uint8 and full-frame `GeometrySpec`. Update MD spec §4, §8.1, and §14 class diagram. (Resolves Mismatch 2.)

4. **Verify `ProcessedFrame.frame_id` in FTL specification** — Confirm the FTL's `ProcessedFrame` type includes a `frame_id` field. If absent, update FTL spec and implementation, or update RPM's construction logic to supply a synthetic `frame_id` for `previous_frame`. (Resolves Ambiguity 1.)

5. **Create a shared types module** — Move `FramePacket` and `BoundingBox` (canonically named) to `src/image_processing/shared_types.py`. Update all modules to import from it. Remove duplicate definitions. (Resolves Mismatches 3 and 5.)

6. **Define `MotionDetectionInterface` as a formal Protocol** — Add to the shared types module or RPM scope. Verify `MotionDetectionManager` structurally satisfies it after steps 2 and 3. Add compliance assertion to tests. (Resolves Mismatch 4.)

7. **Add non-negative `timestamp_ms` validation to `InputValidator`** — Add `timestamp_ms >= 0` check to align with `uint64` semantic intent. (Resolves Mismatch 6.)

8. **Update Motion Detection diagrams** — After code changes in steps 2 and 3, update §14 class diagram and §15 sequence diagram in `motion_detection.md` atomically. (Resolves Mismatch 7.)

---

## 6. Final Compatibility Status

### Overall Status

**🔴 Not Compatible — High Risk Integration**

### Compatibility Summary

| Dimension | Status |
|-----------|--------|
| Public API method name | ❌ Broken — `detect` vs `process` |
| Interface completeness | ❌ Missing `get_input_contract()` |
| Input structure fields | ✅ Compatible |
| Output structure | ✅ Compatible |
| Image format contract | ❌ Indirectly broken — requires `get_input_contract()` to declare format |
| Coordinate system | ✅ Compatible |
| Previous frame ownership | ✅ Aligned |
| Cold start handling | ✅ Compatible |
| Error handling | ✅ Compatible |
| `frame_id` propagation | ⚠️ Ambiguous — depends on unverified FTL `ProcessedFrame` |
| `camera_id` propagation | ✅ Compatible |
| Type definitions | ⚠️ Fragmented — no shared source of truth for `FramePacket` / `BoundingBox` |
| Diagrams | ❌ Stale after required rename |
| Formal interface contract | ❌ `MotionDetectionManager` not declared as implementing any interface |

### Implementation Readiness

**Not ready for integration.** Two critical blockers prevent any successful call path from RPM to Motion Detection.

### Blocking Issues

1. `MotionDetectionManager` has no `detect()` method — results in `AttributeError` at first invocation
2. `MotionDetectionManager` has no `get_input_contract()` method — results in `AttributeError` during RPM initialization, before any frame is processed
3. `ProcessedFrame.frame_id` field not confirmed — potential silent `detected=False` return from Motion Detection on every invocation

### Safety Assessment

**Integration is currently not safe.** Wiring `MotionDetectionManager` as the `MotionDetectionInterface` implementation will fail immediately: RPM initialization calls `get_input_contract()` (Blocker 2) and the first `detect()` call fails (Blocker 1). Resolving steps 1–3 of the resolution plan is the minimum required before integration testing can begin. Steps 4–8 are required before production readiness can be claimed.
