# RecognitionPipelineManager ↔ Face Detection Compatibility Review

**Review Date:** 2026-05-10 (re-review — post-fix validation)
**Status:** COMPATIBLE — all previously reported mismatches resolved; 28/28 tests passing
**Reviewed By:** Static analysis of specs and implementation

---

## 1. Reviewed Files

| # | File | Role |
|---|------|------|
| 1 | `doc/image_processing_service/RecognitionPipelineManager.md` | RecognitionPipelineManager (RPM) full module specification (§1–18) |
| 2 | `doc/image_processing_service/face_detection.md` | Face Detection module full specification (§1–15) |
| 3 | `doc/image_processing_service/shared_contracts.md` | Shared pipeline contract types (BoundingBox, Image, PipelineStageInputContract, OutputImageType, GeometrySpec, etc.) |
| 4 | `doc/image_processing_service/face_recognition.md` | Face Recognition module specification (§1–4) |
| 5 | `src/image_processing/face_detection/module.py` | Face Detection implementation — TypedDicts, validator, postprocessor, output builder, module orchestrator |
| 6 | `src/image_processing/face_detection/__init__.py` | Face Detection public API exports |
| 7 | `tests/face_detection/test_face_detection_module.py` | Face Detection unit and contract tests — 28 tests, all passing |

> **Note:** RecognitionPipelineManager has **no source code implementation** yet. The RPM side of this review is spec-only. The Face Detection module is fully implemented and is the authoritative truth for that side.

---

## 2. Current Integration Flow

### 2.1 How RPM Invokes Face Detection

**Component responsible:** `PipelineOrchestrator` (internal RPM component defined in §8.3).

RPM calls Face Detection in step 11 of the end-to-end flow (§8.8):

1. For each `PersonResult` in `PipelineResult.persons`, `PipelineOrchestrator` calls:
   ```
   FrameTransformationLayerInterface.get_frame(
       camera_id,
       CURRENT,
       PersonResult.person_bbox,           ← full-frame person bbox
       face_detection_contract.output_type,
       face_detection_contract.geometry_spec
   ) → ProcessedFrame
   ```
2. `PipelineOrchestrator` constructs `FaceDetectionInput` from the returned `ProcessedFrame`:
   ```
   FaceDetectionInput {
       frame_id:     frame_packet.frame_id,
       camera_id:    frame_packet.camera_id,
       timestamp_ms: frame_packet.timestamp_ms,
       roi_image:    ProcessedFrame.image      ← Image struct from shared_contracts.md §6
   }
   ```
3. `PipelineOrchestrator` calls `FaceDetectionInterface.detect_faces(FaceDetectionInput)`.
4. `PipelineOrchestrator` receives `FaceDetectionOutput` and iterates `FaceDetectionOutput.detections`. For each `DetectedFace`, it calls `SpatialCoordinator.project_bbox_to_full_frame(DetectedFace.face_bbox, ProcessedFrame.source_bbox_full_frame)` to project the ROI-local face bbox to full-frame coordinates.

### 2.2 What Data Is Passed

| Field | Source | Value |
|-------|--------|-------|
| `frame_id` | `frame_packet.frame_id` | `str` — original frame identifier |
| `camera_id` | `frame_packet.camera_id` | `str` — source camera identifier |
| `timestamp_ms` | `frame_packet.timestamp_ms` | `int` — capture timestamp (non-negative, milliseconds since epoch) |
| `roi_image` | `ProcessedFrame.image` | **`Image` struct** (shared_contracts.md §6) — person ROI prepared by FTL |

`FaceDetectionInput` does not carry `roi_bbox_frame` or any spatial metadata. This is consistent across both specs and the implementation. The person-ROI position in full-frame coordinates is maintained by `PipelineOrchestrator` and used only for coordinate projection after face detection returns.

### 2.3 Person Bbox Source

`PersonResult.person_bbox` is a full-frame projected bounding box produced by `SpatialCoordinator` after Object Detection. It is used as the crop region passed to the Frame Transformation Layer to generate the person ROI image.

### 2.4 ROI Image Source

`ProcessedFrame.image` — the FTL returns a `ProcessedFrame` containing an `Image` struct (carrying `data`, `width`, `height`, `color_format`, `layout`, `dtype`, `value_range` as defined in shared_contracts.md §6). The FTL applies all color conversion, layout conversion, dtype conversion, normalization, and resizing before returning the image. RPM sets `FaceDetectionInput.roi_image = ProcessedFrame.image`, which is therefore an `Image` struct.

### 2.5 Whether Face Detection Receives Person ROI or Full Frame

Face Detection receives only the **person ROI image** — not the full frame. This is consistent across both specs and the implementation.

### 2.6 `roi_bbox_frame` Handling

Neither spec includes `roi_bbox_frame` in `FaceDetectionInput`. The Face Detection spec §2.2 explicitly states: *"FaceDetectionInput does not carry `roi_bbox_frame` or any spatial metadata."* The RPM spec §8.5.3 confirms. The implementation has no `roi_bbox_frame` field. Tests confirm `roi_bbox_frame` is not required (`test_no_roi_bbox_frame_required`).

### 2.7 Coordinate Space of Input Image

The input `roi_image` is in **person-ROI-local coordinate space** (the person ROI is the origin). All pixel coordinates within the image are relative to the top-left corner of the cropped person region.

### 2.8 Coordinate Space of Returned Face Bbox

- **Face Detection spec §7.2 and implementation:** `face_bbox` is expressed in **ROI-local coordinates relative to `roi_image`** (person-ROI-local). Full-frame projection is the responsibility of RPM / `PipelineOrchestrator`.
- **RPM (§8.5.3, §8.8 step 11–12):** Correctly expects ROI-local face bboxes. After receiving `FaceDetectionOutput`, calls `SpatialCoordinator.project_bbox_to_full_frame(DetectedFace.face_bbox, ProcessedFrame.source_bbox_full_frame)` to project each face bbox to full-frame coordinates.
- **Consistent** across both specs and the implementation.

### 2.9 Landmark Handling

- Face Detection outputs landmarks in **person-ROI-local coordinates relative to `roi_image`** (spec §7.2 and implementation).
- RPM §8.5.4 explicitly re-expresses landmarks in face-crop-local coordinates before constructing `FaceRecognitionInput`: for each landmark `lm`, `face_local_lm.x = lm.x - DetectedFace.face_bbox.x` and `face_local_lm.y = lm.y - DetectedFace.face_bbox.y`.
- RPM §8.8 step 12 documents the same transformation.
- `FaceRecognitionInput.landmarks = face_crop_local_landmarks` — landmarks arrive at Face Recognition in the correct coordinate space.
- Face Recognition spec §2.2 requires landmarks that are *"ROI-local relative to `face_roi_image`"* — i.e., face-crop-local. RPM now correctly produces face-crop-local landmarks before invoking Face Recognition.
- **Consistent** across all specs.

### 2.10 `frame_id` and `camera_id` Handling

Both sides treat `frame_id` and `camera_id` as `str` values used for traceability, passed unchanged from input to output.

### 2.11 Image Contract

| Property | RPM Expectation | Face Detection Spec | Implementation |
|----------|----------------|---------------------|----------------|
| Image type | `Image` struct (shared_contracts.md §6) via `ProcessedFrame.image` | `Image` struct in `FaceDetectionInput.roi_image` | `Image` TypedDict; validator checks `isinstance(roi_image, dict)`, validates all metadata fields ✅ |
| Color format | Delegated to `face_detection_contract.output_type` | RGB (default) | `DetectorInputContract.color_format = "RGB"` ✅ |
| Layout | Delegated to stage contract | HWC | `DetectorInputContract.layout = "HWC"` ✅ |
| dtype | Delegated to stage contract | uint8 | `DetectorInputContract.dtype = "uint8"` ✅ |
| Value range | Delegated to stage contract | [0, 255] | `DetectorInputContract.value_range = "[0, 255]"` (whitespace-normalized in comparison) ✅ |
| Preprocessing ownership | FTL preprocesses; Face Detection validates | Upstream preprocesses; module validates | Consistent ✅ |
| ndarray extraction | RPM provides Image struct; engine receives `Image.data` | Module extracts `roi_image.data` before calling engine | `face_input["roi_image"]["data"]` passed to `FaceDetectorEngine.detect()` ✅ |

### 2.12 `get_input_contract()` Method

`FaceDetectionModule.get_input_contract()` is implemented, documented in spec §9.1, and shown in class diagram §13. It returns `PipelineStageInputContract(output_image_type=OutputImageType.RGB_UINT8_HWC, geometry_spec=self._config.geometry_spec)`. RPM §6.1 and §18 correctly require it on every pipeline stage interface. Consistent across spec and implementation.

---

## 3. Compatibility Verification Checklist

| Category | RPM Definition | Face Detection Definition | Compatible? |
|----------|---------------|--------------------------|-------------|
| **API — method name** | `FaceDetectionInterface.detect_faces(input)` | `FaceDetectionModule.detect_faces(face_input)` | ✅ Yes |
| **API — return type** | `FaceDetectionOutput` | `FaceDetectionOutput` | ✅ Yes |
| **API — sync/async** | Synchronous | Synchronous | ✅ Yes |
| **API — `get_input_contract()`** | Required (§6.1); queried at initialization | Implemented; documented in spec §9.1 and class diagram §13 | ✅ Yes |
| **Input — `frame_id`** | `str`, present | `str`, required | ✅ Yes |
| **Input — `camera_id`** | `str`, present | `str`, required, non-empty | ✅ Yes |
| **Input — `timestamp_ms`** | `int` (uint64 in pseudo-struct notation; non-negative) | `int`, required, non-negative (noted in spec §2.2) | ✅ Yes |
| **Input — `roi_image` type** | `Image` struct from `ProcessedFrame.image` (shared_contracts.md §6) | `Image` struct; validator accepts and validates all `Image` struct fields | ✅ Yes |
| **Input — `roi_bbox_frame`** | Not provided; confirmed excluded per §8.5.3 | Not present in spec or implementation | ✅ Yes |
| **Input — extra fields** | None added | None extra expected | ✅ Yes |
| **ROI — input is cropped person ROI** | Yes (from FTL) | Yes | ✅ Yes |
| **ROI — resizing before calling Face Detection** | FTL performs resize per stage contract | Expects pre-resized image matching contract | ✅ Yes |
| **Image — color format** | RGB_UINT8_HWC | RGB, HWC, uint8, [0,255] | ✅ Yes |
| **Image — metadata validation** | FTL preprocesses; Face Detection validates metadata | Validator checks `color_format`, `layout`, `dtype`, `value_range` against `DetectorInputContract` | ✅ Yes |
| **Image — ndarray extraction for engine** | RPM supplies Image struct; engine receives ndarray | Module extracts `roi_image["data"]` before calling `FaceDetectorEngine.detect()` | ✅ Yes |
| **Image — preprocessing ownership** | FTL preprocesses; Face Detection validates | Upstream preprocesses; module validates | ✅ Yes |
| **Coordinate — face bbox output space** | RPM expects ROI-local; projects via `SpatialCoordinator` | ROI-local (spec §7.2 and implementation) | ✅ Yes |
| **Coordinate — no double projection** | Single `SpatialCoordinator` call after Face Detection | No internal projection in Face Detection | ✅ Yes |
| **Coordinate — landmark space (into FaceRecognition)** | RPM subtracts face bbox origin per §8.5.4 and §8.8 step 12; passes face-crop-local landmarks | Person-ROI-local output; transformation is RPM's responsibility | ✅ Yes |
| **`SpatialCoordinator` role** | Projects person and face bboxes to full-frame | Not applicable to Face Detection internals | ✅ Yes |
| **Landmark — required or optional** | Always present in `DetectedFace` | Always present per spec §7.2 | ✅ Yes |
| **Landmark — count** | 5-point canonical | 5-point canonical | ✅ Yes |
| **Landmark — structure** | `FaceLandmarks` (shared_contracts.md §8) | `FaceLandmarks` TypedDict | ✅ Yes |
| **Output — detection container field name** | `FaceDetectionOutput.detections` (§8.5.3, §8.8, §10, §18) | `detections: list[DetectedFace]` | ✅ Yes |
| **Output — face bbox field name** | `DetectedFace.face_bbox` (§8.8 step 12) | `DetectedFace.face_bbox` | ✅ Yes |
| **Output — landmarks field** | `DetectedFace.landmarks` | `DetectedFace.landmarks: FaceLandmarks` | ✅ Yes |
| **Output — confidence exposure** | Not exposed | Confidence is strictly internal | ✅ Yes |
| **Output — empty output handling** | `detections = []` | Returns `detections = []` on any error | ✅ Yes |
| **Output — multiple faces handling** | Iterates `FaceDetectionOutput.detections` | Returns all accepted faces | ✅ Yes |
| **Config — confidence threshold ownership** | Not referenced by RPM; internal | `FaceDetectionPostprocessor` owns it; init-time | ✅ Yes |
| **Config — `geometry_spec` in `FaceDetectionConfig`** | Not referenced directly | Documented in spec §3.1; present in implementation | ✅ Yes |
| **`FaceDetectorEngine.detect()` signature** | N/A (internal interface) | Spec §9.2: `detect(roi_image: np.ndarray)`; impl: `detect(roi_image: np.ndarray)` — module extracts `Image.data` before calling | ✅ Yes |
| **BoundingBox naming** | `BoundingBox` (shared_contracts.md §1) | `BoundingBox` | ✅ Yes |
| **Error handling** | Returns valid output with empty `persons` | Returns valid output with empty `detections` | ✅ Yes |
| **Diagrams — Face Detection class diagram §13** | N/A | Shows `+get_input_contract() PipelineStageInputContract` and `+detect(roi_image: np.ndarray) RawFaceDetections` | ✅ Yes |
| **Diagrams — Face Detection sequence diagram §14** | N/A | Generic `Caller` participant — architecturally correct | ✅ Yes |

---

## 4. Active Mismatches

**None.** All previously reported mismatches have been resolved. See §5 for the complete resolution history.

---

## 5. Resolved Mismatches

All items below were active in a prior review and are now confirmed resolved.

| # | Old Mismatch | Impact Was | Resolution |
|---|-------------|------------|------------|
| 1 | `roi_image` type: implementation required `np.ndarray`; spec and RPM supplied `Image` struct — validator's `isinstance(roi_image, np.ndarray)` check raised `TypeError` silently converted to empty detections on every RPM call | Critical | **Resolved.** `FaceDetectionInput.roi_image` is now declared as `Image` (TypedDict). `FaceDetectionInputValidator.validate()` checks `isinstance(roi_image, dict)` and validates `data`, `width`, `height`, `color_format`, `layout`, `dtype`, and `value_range` against `DetectorInputContract`. `FaceDetectionModule._detect_faces_internal()` passes `face_input["roi_image"]["data"]` to `FaceDetectorEngine.detect()`. |
| 2 | `FaceDetectorEngine.detect()` spec used `detect(prepared_roi: Image)`; implementation used `detect(roi_image: np.ndarray)` — spec could not be used as integration guide for new engine implementations | Medium | **Resolved.** Spec §9.2 now documents `detect(roi_image: np.ndarray) -> RawFaceDetections` with an explicit note: *"FaceDetectionModule validates `FaceDetectionInput.roi_image` as an `Image` struct and extracts `roi_image.data` before passing it to `FaceDetectorEngine.detect()`. The engine interface is kept free of `Image` struct dependency."* Class diagram §13 shows `+detect(roi_image: np.ndarray) RawFaceDetections`. Spec and implementation are aligned. |
| 3 | Landmark coordinate space: Face Detection outputs person-ROI-local landmarks; Face Recognition requires face-crop-local; RPM passed `DetectedFace.landmarks` to `FaceRecognitionInput.landmarks` unchanged — silent face alignment failure | Medium | **Resolved.** RPM §8.5.4 now explicitly re-expresses landmarks in face-crop-local coordinates before constructing `FaceRecognitionInput`: *"for each landmark `lm`, `face_local_lm.x = lm.x - DetectedFace.face_bbox.x` and `face_local_lm.y = lm.y - DetectedFace.face_bbox.y`."* RPM §8.8 step 12 documents the same transformation. `FaceRecognitionInput.landmarks = face_crop_local_landmarks`. |
| 4 | Stale `face_bboxes` reference in RPM §8.5.3 prose — developer reading §8.5.3 in isolation would encounter a `KeyError` at runtime | Medium | **Resolved.** Zero `face_bboxes` references remain in any spec file. RPM §8.5.3 now correctly reads: *"`FaceDetectionOutput.detections` carries zero or more `DetectedFace` entries; each `DetectedFace.face_bbox` is in ROI-local coordinates relative to the person ROI image."* |
| 5 | `FaceDetectionConfig.geometry_spec` present in implementation; absent from spec §3.1 — configuration contract was incomplete | Minor | **Resolved.** Spec §3.1 now documents `geometry_spec: GeometrySpec` with description: *"defines the target spatial dimensions and resize policy expected by the configured detector engine. It is returned to RPM via `get_input_contract()` so the Frame Transformation Layer can prepare the model-ready ROI image at the correct size."* |
| 6 | `get_input_contract()` not documented in Face Detection spec §9.1 or class diagram §13 — documentation-only gap | Minor | **Resolved.** Spec §9.1 now documents `get_input_contract() -> PipelineStageInputContract` with description of its role. Class diagram §13 now shows `+get_input_contract() PipelineStageInputContract`. Implementation is present and returns correctly. |
| 7 | `timestamp_ms` declared as `uint64` in pseudo-structs vs Python `int` in implementation — notation difference only | Minor | **Resolved.** Spec §2.2 now includes the note: *"In the Python implementation it is represented as `int` and must be non-negative (milliseconds since epoch)."* |
| 8 | `roi_bbox_frame` missing from RPM-constructed `FaceDetectionInput` | Prior session | **Resolved** (carried over). `roi_bbox_frame` has been removed from `FaceDetectionInput` in both specs and the implementation. Neither spec references it. |
| 9 | Coordinate system ownership conflict — double projection risk | Prior session | **Resolved** (carried over). Face Detection outputs ROI-local face bboxes and landmarks. RPM correctly applies `SpatialCoordinator.project_bbox_to_full_frame` after receiving output. No double projection occurs. |
| 10 | `CanonicalBoundingBox` (RPM) vs `BoundingBox` (Face Detection) naming conflict | Prior session | **Resolved** (carried over). `shared_contracts.md §1` establishes `BoundingBox` as the single canonical type. All specs use `BoundingBox`. |
| 11 | Stale `PreparedROI` type in Face Detection spec §9.2 and class diagram §13 | Prior session | **Resolved** (carried over). Spec §9.2 now uses `np.ndarray` with the `Image.data` extraction note. No `PreparedROI` type remains anywhere. |

---

## 6. Test Coverage

As of this re-review, `tests/face_detection/test_face_detection_module.py` contains 28 tests — all passing.

| Test Class | Scenarios Covered |
|-----------|------------------|
| `FaceDetectionModuleOutputContractTests` | Output keys, metadata preservation, face bbox fields, landmark fields and types |
| `RoiLocalOutputTests` | ROI-local coordinate verification, raw coordinate equality, bounds checks |
| `DeterminismTests` | Same input produces identical output across repeated calls |
| `ErrorHandlingTests` | Missing camera_id, raw ndarray rejected, None roi_image |
| `PostprocessorFilteringTests` | Low-confidence rejection, above-threshold acceptance |
| `StubEngineTests` | Detection count, landmark count, ROI-local coordinates, determinism |
| `ImageContractValidationTests` | Valid Image struct accepted; missing `data` rejected; wrong `color_format`, `layout`, `dtype`, `value_range` each rejected individually; zero `width`/`height` rejected; `roi_bbox_frame` not required; engine receives raw `np.ndarray` |

**Minor test observation (non-blocking):** `ErrorHandlingTests.test_wrong_dtype_roi_returns_empty_detections` passes a raw `np.ndarray` for `roi_image` rather than an `Image` struct with `dtype="float32"`. The validator rejects it at the `isinstance(roi_image, dict)` check — not at the dtype-metadata check — so the test exercises the ndarray-rejection path rather than the Image-struct dtype-validation path. The result (empty detections) is still correct. The proper dtype metadata validation path is correctly covered by `ImageContractValidationTests.test_rejects_wrong_dtype`. No functional impact.

---

## 7. Final Compatibility Status

### Overall Status: COMPATIBLE

### Summary Table

| Dimension | Status |
|-----------|--------|
| API method name | ✅ Compatible |
| Input — `roi_bbox_frame` absence | ✅ Compatible |
| Input — `roi_image` type | ✅ Compatible (`Image` struct; validator accepts and validates) |
| Input — `roi_image` metadata validation | ✅ Compatible (`color_format`, `layout`, `dtype`, `value_range` validated against `DetectorInputContract`) |
| Image — ndarray extraction for engine | ✅ Compatible (`roi_image["data"]` passed to engine) |
| Coordinate system — face bbox ROI-local output | ✅ Compatible |
| No double projection | ✅ Compatible |
| Landmark coordinate space (Face Detection → RPM → Face Recognition) | ✅ Compatible (RPM transforms to face-crop-local per §8.5.4 and §8.8 step 12) |
| Stage interface — `get_input_contract()` | ✅ Compatible (implemented, spec-documented, diagram-documented) |
| Config — `geometry_spec` | ✅ Compatible (implemented and spec-documented) |
| Error handling | ✅ Compatible |
| BoundingBox type | ✅ Compatible |
| Preprocessing ownership | ✅ Compatible |
| Confidence exposure | ✅ Compatible |
| Traceability metadata | ✅ Compatible |
| Field names — output structure | ✅ Compatible |
| `FaceDetectorEngine.detect()` signature | ✅ Compatible (spec §9.2 and implementation both use `np.ndarray`; `Image.data` extraction documented) |
| `timestamp_ms` semantics | ✅ Compatible (non-negative note in spec §2.2) |

### Implementation Readiness

**Ready for integration.** The Face Detection module is fully implemented and aligned with the shared `Image` contract. All public API boundaries, validation rules, coordinate space contracts, and pipeline stage interface requirements are consistent across the RPM spec, Face Detection spec, and Face Detection implementation.

RecognitionPipelineManager has no source code yet — it is spec-only. When implemented, it may proceed using the RPM spec as written, with no contract gaps relative to the Face Detection module.

### Blocking Issues

None.

### Whether Integration Is Currently Safe

**Integration is safe.** Connecting RPM (as currently specified) to the existing Face Detection implementation will produce correct behavior:

- `FaceDetectionInput.roi_image = ProcessedFrame.image` (an `Image` struct) passes the validator's `isinstance(roi_image, dict)` check and all metadata field checks.
- `FaceDetectorEngine.detect()` receives the validated `roi_image["data"]` (raw ndarray) — no type mismatch at the engine boundary.
- Landmarks output from Face Detection (person-ROI-local) are correctly transformed to face-crop-local by RPM before constructing `FaceRecognitionInput` — no silent alignment failure.
- All field names (`detections`, `face_bbox`, `landmarks`) are consistent across specs and implementation.
