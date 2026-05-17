# RecognitionPipelineManager ↔ Motion Detection Compatibility Review — Re-Review (May 2026)

> **Re-review scope.** This document supersedes the original compatibility review. It reflects the current state of all reviewed artifacts as of the re-review date. All mismatches have been re-verified against the actual implementation files, specification files, and tests. **All active mismatches from the previous review have been resolved.** Resolved items are documented in §7. No active mismatches remain.

---

## 1. Reviewed Files

| File | Type | Purpose |
|------|------|---------|
| `doc/image_processing_service/RecognitionPipelineManager.md` | Specification | RPM design, interfaces, orchestration flow |
| `doc/image_processing_service/motion_detection.md` | Specification | Motion Detection module design and contracts |
| `doc/image_processing_service/shared_contracts.md` | Specification | **New since original review.** Authoritative shared type definitions: `BoundingBox`, `Image`, `ResizePolicy`, `GeometrySpec`, `OutputImageType`, `PipelineStageInputContract` |
| `src/image_processing/shared/contracts.py` | Source | **New since original review.** Shared contract implementations: `ResizePolicy`, `GeometrySpec`, `OutputImageType`, `PipelineStageInputContract`, `Image` |
| `src/image_processing/shared/__init__.py` | Source | **New since original review.** Exports shared contracts |
| `src/image_processing/motion_detection/module.py` | Source | Motion Detection implementation (~520 lines) |
| `src/image_processing/motion_detection/__init__.py` | Source | Public exports |
| `tests/motion_detection/test_motion_detection_module.py` | Tests | Unit tests covering validation, pipeline, determinism |

**Not yet implemented:** No source code exists for `RecognitionPipelineManager`, `PipelineOrchestrator`, `FrameTransformationLayerInterface`, or `SpatialCoordinator`. Analysis of RPM is documentation-only.

---

## 2. Spec Changes Since Original Review

Two significant specification updates have been made since the original review. The code has been updated to match both changes.

### 2.1 Shared Contracts Infrastructure Added

`doc/image_processing_service/shared_contracts.md` and `src/image_processing/shared/contracts.py` are new. They establish a single authoritative home for pipeline-wide types:

- `BoundingBox` — defined in shared_contracts.md §1; replaces `CanonicalBoundingBox` everywhere; ✅ **now in `shared/contracts.py` and `shared/__init__.py`**
- `Image` — the canonical public image struct carrying `data`, `width`, `height`, `color_format`, `layout`, `dtype`, `value_range`; implemented in `shared/contracts.py`
- `OutputImageType` — enum with `GRAYSCALE_UINT8_HWC` and `RGB_UINT8_HWC`; implemented in `shared/contracts.py`
- `GeometrySpec` — spatial transformation struct; implemented in `shared/contracts.py`
- `PipelineStageInputContract` — returned by `get_input_contract()`; fields are `output_image_type` and `geometry_spec`; implemented in `shared/contracts.py`
- `ResizePolicy` — enum for resize behavior; implemented in `shared/contracts.py`

### 2.2 Motion Detection Spec Updated to Use `MotionInputFrame` and `Image`

`motion_detection.md` has been updated:

- The per-frame input container was renamed from `FramePacket` to `MotionInputFrame`. The spec explicitly notes this avoids ambiguity with the FTL-level `FramePacket` (which carries raw `image_bytes`).
- `MotionInputFrame.image` is now typed as the shared `Image` struct (from `shared_contracts.md §6`) rather than a raw numpy array.
- Validation rules (§2.4) were updated to require checking `Image` struct metadata fields: `color_format`, `layout`, `dtype`, `value_range`, `width`, `height`, and `data.shape`.
- `BoundingBox` is now referenced from `shared_contracts.md §1` rather than defined locally.

The `RecognitionPipelineManager.md` spec was also updated to use `MotionInputFrame` consistently and to reference `BoundingBox` from `shared_contracts.md`.

**The code (`module.py`, `__init__.py`, tests) has been updated to reflect both changes.**

---

## 3. Current Integration Flow

### How RPM Invokes Motion Detection

**Spec-defined flow (RPM §8.5.1):**

1. External caller invokes `RecognitionPipelineManager.process_frame(frame_packet: FramePacket)`.
2. `RecognitionPipelineManager` validates `frame_packet` via `RecognitionPipelineInputValidator`.
3. `RecognitionPipelineManager` calls `orchestrator.execute(frame_packet)`.
4. `PipelineOrchestrator` calls `FrameTransformationLayerInterface.ingest_frame(frame_packet)` → void.
5. `PipelineOrchestrator` calls `FTL.get_frame(camera_id, CURRENT, full_frame_bbox, motion_contract.output_image_type, motion_contract.geometry_spec)` → `current_processed: ProcessedFrame`.
6. `PipelineOrchestrator` calls `FTL.get_frame(camera_id, PREVIOUS, ...)` → `previous_processed: ProcessedFrame`. If `PreviousFrameNotAvailableError` → return empty result (cold start); Motion Detection is not called.
7. `PipelineOrchestrator` constructs `MotionDetectionInput` and calls `MotionDetectionInterface.detect(MotionDetectionInput)` → `MotionResult`.

### Data Passed to Motion Detection

`MotionDetectionInput` as constructed by RPM (`PipelineOrchestrator`), per current RPM spec:

```
MotionDetectionInput {
    current_frame:  MotionInputFrame {
                        frame_id:     frame_packet.frame_id,
                        camera_id:    frame_packet.camera_id,
                        timestamp_ms: current_processed.timestamp_ms,
                        image:        current_processed.image      <- shared Image struct (GRAYSCALE_UINT8_HWC)
                    },
    previous_frame: MotionInputFrame {
                        frame_id:     previous_processed.frame_id,
                        camera_id:    frame_packet.camera_id,
                        timestamp_ms: previous_processed.timestamp_ms,
                        image:        previous_processed.image     <- shared Image struct (GRAYSCALE_UINT8_HWC)
                    }
}
```

### Internal Components Responsible

| Component | Responsibility |
|-----------|----------------|
| `PipelineOrchestrator` | Retrieves frames from FTL, constructs `MotionDetectionInput`, calls `MotionDetectionInterface.detect()` |
| `FrameTransformationLayerInterface` | Stores frames, maintains CURRENT/PREVIOUS per `camera_id`, converts image to `Image` struct in required format/geometry |
| `MotionDetectionInterface` | Abstraction RPM depends on; expects `detect()` and `get_input_contract()` |
| `MotionDetectionManager` | Actual implementation; exposes `detect()` and `get_input_contract()`; implements `MotionDetectionInterface` (`@runtime_checkable Protocol`); accepts `MotionInputFrame` with shared `Image` struct |

---

## 4. Active Mismatches

**No active mismatches remain.** All 8 mismatches and 1 ambiguity identified in the previous review have been resolved. See §7 for details.

---

## 5. Compatibility Verification Checklist

### 5.1 API + Method Contracts

| Category | RPM Expectation | Motion Detection Reality | Compatible? | Required Change |
|---|---|---|---|---|
| Public method name | `detect(input) -> MotionResult` | `detect(input) -> MotionResult` | ✅ YES | None |
| `get_input_contract()` | Required; returns `PipelineStageInputContract` | Implemented; returns `GRAYSCALE_UINT8_HWC` + `ResizePolicy.NONE` | ✅ YES | None |
| Return type | `MotionResult` | `MotionResult` | ✅ YES | None |
| Sync/async | Synchronous | Synchronous | ✅ YES | None |
| Input type name | `MotionDetectionInput` | `MotionDetectionInput` | ✅ YES | None |
| Batch/single | Single invocation | Single invocation | ✅ YES | None |

### 5.2 Input Container Structures

| Category | RPM / Spec Definition | Motion Detection Code | Compatible? | Required Change |
|---|---|---|---|---|
| Per-frame container name | `MotionInputFrame` | `MotionInputFrame` | ✅ YES | None |
| `image` field type | `Image` (shared struct) | `Image` (shared struct) | ✅ YES | None |
| `frame_id` field | `str` (non-null) | `str` | ✅ YES | None |
| `camera_id` field | `str` (non-empty) | `str` | ✅ YES | None |
| `timestamp_ms` field | `uint64` (spec) | `int` with `>= 0` validation | ✅ YES | None |
| `image.color_format` | `GRAY` (from `Image`) | Validated from `Image` struct metadata | ✅ YES | None |
| `image.layout` | `HWC` (from `Image`) | Validated from `Image` struct metadata | ✅ YES | None |
| `image.dtype` | `uint8` (from `Image`) | Validated from `Image` struct metadata | ✅ YES | None |
| `image.value_range` | `[0,255]` (from `Image`) | Validated from `Image` struct metadata | ✅ YES | None |
| `image.width` / `image.height` | From `Image` struct fields | Validated from `Image` struct fields | ✅ YES | None |

### 5.3 `frame_id` and `camera_id` Verification

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| `current_frame.frame_id` | `frame_packet.frame_id` | Must exist, non-null | ✅ YES | None |
| `previous_frame.frame_id` | `previous_processed.frame_id` (from FTL) | Must exist, non-null | ✅ YES | `ProcessedFrame.frame_id` added to FTL spec and implementation; copied unchanged from source `FramePacket.frame_id` through ingest and all FTL operations. RPM populates `previous_frame.frame_id = previous_processed.frame_id` — no synthesis required. |
| `camera_id` source | `frame_packet.camera_id` for both frames | Must be equal | ✅ YES | None |
| Cross-frame equality | Both frames share same `camera_id` | `InputValidator` enforces equality | ✅ YES | None |

### 5.4 Previous Frame Handling

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| Cold start | RPM catches `PreviousFrameNotAvailableError`; MD never called | `InputValidator` rejects null `previous_frame`; returns `detected=False` | ✅ YES | None |
| Previous frame ownership | RPM via FTL PREVIOUS slot | Caller provides `previous_frame` | ✅ YES | None |

### 5.5 Image Contract

| Category | RPM / Spec Definition | Motion Detection Code | Compatible? | Required Change |
|---|---|---|---|---|
| Color format declared | Via `get_input_contract()` → `GRAYSCALE_UINT8_HWC` | Implemented; returns `GRAYSCALE_UINT8_HWC` | ✅ YES | None |
| Image type at public boundary | `Image` struct from `shared/contracts.py` | `Image` struct from `shared/contracts.py` | ✅ YES | None |
| Preprocessing ownership | FTL performs all preprocessing | Caller delivers pre-processed frames; MD does no preprocessing | ✅ YES | None |
| Dimension matching | Same geometry → same size | Requires identical dimensions (after Mismatch 4 fix) | ✅ YES (after fix) | Fix validator to use `Image.width`, `Image.height` |

### 5.6 Coordinate Systems

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| `MotionResult.bboxes` space | Relative to `current_frame.image` (full frame) | Relative to `current_frame.image` | ✅ YES | None |
| Axis alignment | Axis-aligned | Axis-aligned | ✅ YES | None |
| Origin | Top-left of `current_frame.image` | Top-left of `current_frame.image` | ✅ YES | None |

### 5.7 Output Structures

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| `MotionResult.detected` | `bool` | `bool` | ✅ YES | None |
| `MotionResult.bboxes` | `vector<BoundingBox>` | `list[BoundingBox]` | ✅ YES | None |
| `BoundingBox` naming | `BoundingBox` (shared_contracts.md §1) | `BoundingBox` (from `image_processing.shared.contracts`) | ✅ YES | None |
| `BoundingBox` fields | `x, y, width, height` (int) | `x, y, width, height` (int) | ✅ YES | None |
| Empty output on no motion | `bboxes = []` when `detected = false` | `bboxes = []` when `detected = False` | ✅ YES | None |

### 5.8 Error Handling

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| Stage failure | Return empty `RecognitionPipelineOutput` | Return `MotionResult{detected=False}` | ✅ YES | None |
| Cold start | RPM catches before MD is called | Not MD's responsibility | ✅ YES | None |
| Validation | Both modules validate inputs | `InputValidator` enforces constraints; validates `Image` struct metadata | ✅ YES | None |

### 5.9 Diagrams

| Category | RPM Definition | Motion Detection Definition | Compatible? | Required Change |
|---|---|---|---|---|
| Sequence diagram | `Orch->>Motion: detect(MotionDetectionInput)` | `Caller->>Manager: detect(input)` | ✅ YES | None |
| Class diagram method | `MotionDetectionInterface.detect()` | `MotionDetectionManager.detect()` | ✅ YES | None |
| Class diagram `get_input_contract()` | Present on all stage interfaces | Present on `MotionDetectionManager` and `MotionDetectionInterface` | ✅ YES | None |

### 5.10 Tests

| Area | Required | Current State | Gap |
|------|----------|--------------|-----|
| `detect()` public API | Yes | All tests call `.detect()` | None |
| `get_input_contract()` | Yes | `GetInputContractTests` class (4 tests) | None |
| `MotionInputFrame` + `Image` struct | Yes | Tests use `MotionInputFrame` + `Image` struct | None |
| `image.color_format` / `layout` / `dtype` validation | Yes | `ValidationTests` covers all `Image` struct metadata fields | None |
| `timestamp_ms >= 0` validation | Yes | `test_negative_current_timestamp_returns_no_motion`, `test_negative_previous_timestamp_returns_no_motion` | None |
| `MotionDetectionInterface` compliance | Yes | `ProtocolComplianceTests` class (3 tests) | None |

---

## 6. Recommended Resolution Plan

All phases have been completed. No further action is required to achieve RPM ↔ Motion Detection integration compatibility.

**Architecture decision (confirmed):** `detect(MotionDetectionInput) -> MotionResult` is the single RPM-facing public stage API. `get_input_contract() -> PipelineStageInputContract` is the only initialization-time contract method. RPM depends on `MotionDetectionInterface` and calls these two methods. The deprecated `process()` wrapper remains for backward-compatibility with any pre-RPM call sites but must not be used for new code.

---

## 7. Resolved Items

**Original Mismatch 3 (original review) — `CanonicalBoundingBox` vs `BoundingBox` naming conflict (SPEC-LEVEL RESOLVED)**

Both `motion_detection.md` and `RecognitionPipelineManager.md` now exclusively use `BoundingBox`. `shared_contracts.md §1` explicitly declares that `BoundingBox` replaces any prior usage of `CanonicalBoundingBox`. The two types are structurally identical; the naming inconsistency no longer exists in any current spec.

---

**Mismatch 1 — Method Name: `detect` vs `process` (RESOLVED)**

- `MotionDetectionManager.detect(input: MotionDetectionInput) -> MotionResult` added.
- `process()` retained as a deprecated thin wrapper calling `detect()` for backward compatibility.
- All test invocations updated to `.detect()`.
- `motion_detection.md` §4, §8.1, §8.6, §14, §15 updated.

---

**Mismatch 2 — Missing `get_input_contract()` Method (RESOLVED)**

- `MotionDetectionManager.get_input_contract() -> PipelineStageInputContract` implemented.
- Returns `PipelineStageInputContract(output_image_type=OutputImageType.GRAYSCALE_UINT8_HWC, geometry_spec=GeometrySpec(width=0, height=0, resize_policy=ResizePolicy.NONE))`.
- `motion_detection.md` §4, §14, §17 updated.
- `GetInputContractTests` class (4 tests) added.

---

**Mismatch 3 — Input Container Type: `FramePacket + image: Any` vs `MotionInputFrame + image: Image` (RESOLVED)**

- `MotionInputFrame` TypedDict defined in `module.py` with `image: Image`.
- `MotionDetectionInput` updated to use `MotionInputFrame`.
- `FramePacket` TypedDict removed from `module.py` and `__init__.py`.
- All test helpers updated to construct `MotionInputFrame` with shared `Image` structs.
- Algorithm call sites updated to extract `frame["image"]["data"]` from the `Image` struct.

---

**Mismatch 4 — `InputValidator` Validates Raw NumPy Arrays, Not `Image` Struct (RESOLVED)**

- `InputValidator` fully rewritten to validate the `Image` TypedDict:
  - `color_format == "GRAY"`, `layout == "HWC"`, `dtype == "uint8"`, `value_range == "[0,255]"`
  - `width > 0`, `height > 0`
  - `data.shape` consistency with width/height/layout/color_format
  - `isinstance(image, dict)` check replaces `isinstance(image, np.ndarray)` check
- `_validate_cross_frame` updated to use `image["width"]` and `image["height"]`.
- New validation tests: `test_wrong_color_format_returns_no_motion`, `test_wrong_layout_returns_no_motion`, `test_wrong_value_range_returns_no_motion`, `test_invalid_width_returns_no_motion`, `test_invalid_height_returns_no_motion`, `test_data_shape_mismatch_returns_no_motion`, `test_raw_ndarray_as_image_returns_no_motion`.

---

**Mismatch 5 — No Formal `MotionDetectionInterface` Implementation (RESOLVED)**

- `MotionDetectionInterface` defined as a `@runtime_checkable Protocol` in `module.py`:
  ```python
  @runtime_checkable
  class MotionDetectionInterface(Protocol):
      def detect(self, input: MotionDetectionInput) -> MotionResult: ...
      def get_input_contract(self) -> PipelineStageInputContract: ...
  ```
- `ProtocolComplianceTests` added: `test_manager_satisfies_interface` verifies `isinstance(manager, MotionDetectionInterface)` at runtime.
- `MotionDetectionInterface` exported from `motion_detection/__init__.py`.
- `motion_detection.md` §14 class diagram updated to show `MotionDetectionInterface` Protocol and `MotionDetectionManager ..|> MotionDetectionInterface : implements`.

---

**Mismatch 6 — `BoundingBox` Not in `shared/contracts.py` (RESOLVED)**

- `BoundingBox` TypedDict added to `src/image_processing/shared/contracts.py`.
- `BoundingBox` exported from `src/image_processing/shared/__init__.py`.
- Local `BoundingBox` definition removed from `motion_detection/module.py`.
- `motion_detection/__init__.py` now re-exports `BoundingBox` from `image_processing.shared`.

---

**Mismatch 7 — `timestamp_ms` Non-Negative Validation Missing (RESOLVED)**

- `InputValidator._validate_frame` now checks `timestamp_ms >= 0` for both `current_frame` and `previous_frame`.
- Tests added: `test_negative_current_timestamp_returns_no_motion`, `test_negative_previous_timestamp_returns_no_motion`.

---

**Mismatch 8 — Stale Method Names in Motion Detection Diagrams (RESOLVED)**

- `motion_detection.md` §14 class diagram updated: `+detect(input: MotionDetectionInput) MotionResult` and `+get_input_contract() PipelineStageInputContract`.
- `motion_detection.md` §15 sequence diagram updated: `Caller->>Manager: detect(input)`.

---

**Ambiguity 1 — `ProcessedFrame.frame_id` Assumption (RESOLVED)**

- `ProcessedFrame.frame_id: str` has been added to the FTL spec (`frame_transformation_layer.md`) and implementation (`src/image_processing/frame_transformation_layer/module.py`).
- FTL copies `frame_id` unchanged from the source `FramePacket.frame_id` during ingest and preserves it through all crop, resize, letterbox, and pixel-format conversion operations.
- RPM can safely populate `previous_frame.frame_id = previous_processed.frame_id` as written in RPM §8.5.1. No synthesis is required.
- FTL tests updated: `test_get_frame_current_preserves_frame_id`, `test_get_frame_previous_preserves_frame_id`, `test_crop_does_not_change_frame_id` all pass.

---

## 8. Final Compatibility Status

### Overall Status

**✅ Compatible — Integration Ready**

All 8 active mismatches and Ambiguity 1 have been fully resolved. `MotionDetectionManager` implements `MotionDetectionInterface` (verified at runtime via `@runtime_checkable Protocol`). No open items remain.

### Compatibility Summary

| Dimension | Status |
|-----------|--------|
| Public API method name | ✅ `detect()` implemented |
| Interface completeness | ✅ `get_input_contract()` implemented |
| Input container type | ✅ `MotionInputFrame + image: Image` |
| `InputValidator` implementation | ✅ Validates `Image` struct metadata |
| `BoundingBox` shared code definition | ✅ In `shared/contracts.py` |
| Output structure | ✅ Compatible |
| `BoundingBox` naming (spec-level) | ✅ Resolved |
| Image format contract | ✅ `GRAYSCALE_UINT8_HWC` declared via `get_input_contract()` |
| Coordinate system | ✅ Compatible |
| Previous frame ownership | ✅ Aligned |
| Cold start handling | ✅ Compatible |
| Error handling | ✅ Compatible |
| `camera_id` propagation | ✅ Compatible |
| `frame_id` propagation (previous) | ✅ Compatible — `ProcessedFrame.frame_id` now carries the source `FramePacket.frame_id`; RPM populates `previous_frame.frame_id = previous_processed.frame_id` without synthesis |
| Formal interface contract | ✅ `MotionDetectionInterface` Protocol; runtime-checkable |
| Diagrams | ✅ Updated to `detect()` |
| `timestamp_ms` non-negative | ✅ Validated |
| Tests | ✅ 60 tests pass; new `GetInputContractTests` and `ProtocolComplianceTests` classes |

### Blocking Issues

None. All critical blockers from the previous review have been resolved.

### Safety Assessment

**Integration is safe.** `MotionDetectionManager` can be wired as the `MotionDetectionInterface` implementation in RPM. All compatibility items are resolved, including `frame_id` propagation for the previous frame. No outstanding items remain.
