# RecognitionPipelineManager ↔ Object Detection Compatibility Review

**Date:** 2026-05-10
**Scope:** Interface and integration compatibility between `RecognitionPipelineManager` and the `ObjectDetectionModule`.
**Purpose:** Re-review the current real state of the implementation, OD spec, RPM spec, and shared contracts. Verify which previously identified mismatches have been resolved and whether new mismatches were introduced.

---

## 1. Reviewed Files

| File | Role |
|---|---|
| `doc/image_processing_service/RecognitionPipelineManager.md` | RPM specification (§1–18) |
| `doc/image_processing_service/object_detection.md` | Object Detection specification (§1–18) |
| `doc/image_processing_service/shared_contracts.md` | Shared pipeline contract types (§1–10) |
| `src/image_processing/object_detection/module.py` | Object Detection concrete implementation |
| `src/image_processing/object_detection/__init__.py` | Object Detection public API exports |
| `src/image_processing/shared/contracts.py` | Shared contract types implemented in Python |

---

## 2. Previously Identified Mismatches — Resolution Status

All eleven mismatches from the previous review have been resolved. The table below documents what was fixed and where the evidence is in the current codebase.

| # | Previous Mismatch | Current Status | Evidence |
|---|---|---|---|
| 1 | `detect` vs `process` method name | ✅ Resolved | `ObjectDetectionModule.detect(input: ObjectDetectionInput) → PersonDetectionResult` exists in `module.py` |
| 2 | Missing `get_input_contract()` method | ✅ Resolved | `ObjectDetectionModule.get_input_contract()` implemented; returns `{ output_image_type: RGB_UINT8_HWC, geometry_spec: { width: 640, height: 640, resize_policy: LETTERBOX } }` |
| 3 | Split `(model_ready_input, metadata)` args vs single `ObjectDetectionInput` | ✅ Resolved | `ObjectDetectionInput` TypedDict with 7 fields defined; `detect()` accepts it as a single argument and maps fields internally |
| 4 | `timestamp_ms` absent from `FrameMetadata` and not validated | ✅ Resolved | `FrameMetadata` includes `timestamp_ms: int`; `InputValidator` checks presence and `isinstance(int)` |
| 5 | `roi_bbox_frame` absent from `FrameMetadata` and not validated | ✅ Resolved | `FrameMetadata` includes `roi_bbox_frame: BoundingBox`; `InputValidator` checks presence and positive dimensions |
| 6 | OD spec §7.4 sequence diagram shows `process()` not `detect()` | ✅ Resolved | OD spec §7.4 now shows `detect(ObjectDetectionInput)` |
| 7 | OD spec §7.2 ambiguous "original frame resolution" wording | ✅ Resolved | OD spec §7.2 now reads "Maps decoded detection boxes back to `roi_image` coordinate space (ROI-local coordinates, relative to the input ROI image)" |
| 8 | Stale `run_stage` method in OD spec §7.3 class diagram | ✅ Resolved | OD spec §7.3 class diagram updated; `detect(input: ObjectDetectionInput)` is shown on `ObjectDetectionModule` |
| 9 | `CanonicalBoundingBox` (RPM) vs `BoundingBox` (OD) naming inconsistency | ✅ Resolved | `shared_contracts.md §1` explicitly retires `CanonicalBoundingBox`; both RPM spec and OD spec now use `BoundingBox` uniformly |
| 10 | `input_representation` vs `model_ready_input` parameter name in OD spec diagrams | ✅ Resolved | OD spec §7.3 and §7.4 diagrams now use `model_ready_input` |
| 11 | OD spec §5.2 "preserved for spatial context" misleading wording for `roi_bbox_frame` | ✅ Resolved | OD spec §5.2 now reads: "received for validation purposes; `roi_bbox_frame` is not returned in `PersonDetectionResult` — coordinate projection is performed by the caller using spatial metadata from the Frame Transformation Layer" |

---

## 3. Current Integration Flow

### How RPM Invokes Object Detection

`RecognitionPipelineManager` does not invoke `ObjectDetectionModule` directly. It invokes `ObjectDetectionInterface` — an abstract protocol boundary that `ObjectDetectionModule` must satisfy. Invocation is owned exclusively by `PipelineOrchestrator`, the internal component that manages stage sequencing inside RPM.

#### Step-by-step per motion region (RPM spec §8.5.2, §8.8):

1. `PipelineOrchestrator` receives a non-empty `MotionResult.bboxes` from the motion detection stage.
2. For each `motion_region_bbox` in `MotionResult.bboxes`, `PipelineOrchestrator` calls:
   ```
   FrameTransformationLayerInterface.get_frame(
       camera_id,
       CURRENT,
       motion_region_bbox,
       object_detection_contract.output_image_type,
       object_detection_contract.geometry_spec
   ) → ProcessedFrame
   ```
   `output_image_type` and `geometry_spec` come from the OD stage's `PipelineStageInputContract`, queried via `ObjectDetectionInterface.get_input_contract()` at initialization.

3. `PipelineOrchestrator` constructs `ObjectDetectionInput` from `ProcessedFrame.image` and `frame_packet` metadata:
   ```
   ObjectDetectionInput {
       frame_id       = frame_packet.frame_id,
       camera_id      = frame_packet.camera_id,
       timestamp_ms   = frame_packet.timestamp_ms,
       roi_image      = ProcessedFrame.image,
       roi_bbox_frame = motion_region_bbox,
       width          = ProcessedFrame.image.width,
       height         = ProcessedFrame.image.height
   }
   ```

4. `PipelineOrchestrator` calls:
   ```
   ObjectDetectionInterface.detect(ObjectDetectionInput) → PersonDetectionResult
   ```

5. `PersonDetectionResult.persons` contains zero or more `BoundingBox` entries in **ROI-local coordinates** — relative to `ObjectDetectionInput.roi_image`, not the full camera frame.

6. For each ROI-local `person_bbox`, `PipelineOrchestrator` calls:
   ```
   SpatialCoordinator.project_bbox_to_full_frame(
       person_bbox,
       ProcessedFrame.source_bbox_full_frame
   ) → BoundingBox (full-frame)
   ```
   `source_bbox_full_frame` comes from the FTL's `ProcessedFrame`, not from OD's output.

### Internal Components Responsible

| Component | Responsibility |
|---|---|
| `PipelineOrchestrator` | Constructs `ObjectDetectionInput`; calls `ObjectDetectionInterface.detect()`; passes ROI-local person bboxes to `SpatialCoordinator` |
| `ObjectDetectionInterface` | Abstract boundary RPM depends on; exposes `detect()` and `get_input_contract()` |
| `ObjectDetectionModule` | Concrete implementation satisfying `ObjectDetectionInterface` |
| `SpatialCoordinator` | Projects ROI-local person bboxes to full-frame coordinates using FTL spatial metadata |
| `FrameTransformationLayerInterface` | Supplies model-ready ROI image + `source_bbox_full_frame` to orchestrator |

### Coordinate Space
- OD input: ROI-local (origin = top-left of `roi_image`)
- OD output `persons`: ROI-local — relative to the same `roi_image`
- Full-frame projection: performed by `SpatialCoordinator` after OD returns, using `ProcessedFrame.source_bbox_full_frame`

### `get_input_contract()` Return Value (Current)
```python
{
    "output_image_type": OutputImageType.RGB_UINT8_HWC,
    "geometry_spec": {
        "width": 640,
        "height": 640,
        "resize_policy": ResizePolicy.LETTERBOX,
    }
}
```
This is returned by `ObjectDetectionModule.get_input_contract()` using types from `src/image_processing/shared/contracts.py`. The field name is `output_image_type` (matching `shared_contracts.md §5`).

---

## 4. Compatibility Verification Checklist

| Category | RPM / Shared Contract Definition | OD Implementation | Status |
|---|---|---|---|
| **API Methods** | | | |
| Primary method | `detect(ObjectDetectionInput) → PersonDetectionResult` | `detect(ObjectDetectionInput) → PersonDetectionResult` | ✅ |
| Init contract method | `get_input_contract() → PipelineStageInputContract` | Implemented; returns `RGB_UINT8_HWC`, `640×640 LETTERBOX` | ✅ |
| Internal method exposure | External callers must not use `process()` | `process()` retained internally; `detect()` is the primary entry point; `process` re-exported in `__init__.py` as legacy | ✅ |
| Sync/async | Synchronous | Synchronous | ✅ |
| **Input Fields** | | | |
| `frame_id` | `str`; preserved unchanged | `str` in `ObjectDetectionInput`; copied to `FrameMetadata`; returned in output | ✅ |
| `camera_id` | `str`; non-empty | `str`; validated non-empty in `InputValidator` | ✅ |
| `timestamp_ms` | `int`; present | `int` in `ObjectDetectionInput` and `FrameMetadata`; validated present and `isinstance(int)` | ✅ |
| `roi_image` | `Image` struct per `shared_contracts.md §6` | `Any` in TypedDict; runtime `np.ndarray`; `Image` struct not implemented in `shared/contracts.py` | ⚠️ Issue 1 / Issue 2 |
| `roi_bbox_frame` | `BoundingBox`; present; `width > 0, height > 0` | `BoundingBox` in `ObjectDetectionInput` and `FrameMetadata`; validated presence and positive dimensions | ✅ |
| `width` / `height` | Standalone fields in RPM §8.5.2 | Standalone in `ObjectDetectionInput` and `FrameMetadata`; validated positive; used in shape check | ✅ (OD spec §5.2 inconsistency — see Issue 3) |
| **`PipelineStageInputContract` field name** | `output_image_type` (`shared_contracts.md §5`; `shared/contracts.py`) | `output_image_type` in returned dict | ✅ (RPM spec §8.3 prose incorrect — see Issue 4) |
| **Image Validation** | | | |
| `color_format`, `layout`, `dtype`, `value_range` checks | Required by OD spec §5.3 | Not validated; impossible without `Image` struct | ⚠️ Issue 5 |
| ndarray shape vs width/height | Required by OD spec §5.3 | `shape[0] == height` and `shape[1] == width` validated | ✅ |
| **Coordinate System** | | | |
| OD output bbox space | ROI-local per RPM §8.5.2 | ROI-local — Ultralytics xyxy in input-image space; clipped to `[0, width] × [0, height]` from `FrameMetadata` | ✅ |
| Full-frame projection | Owned by `SpatialCoordinator` in RPM | OD has no projection logic | ✅ |
| **Output** | | | |
| Result type | `PersonDetectionResult` | `PersonDetectionResult` TypedDict | ✅ |
| `person_detected` | `bool` | `bool`; derived as `len(persons) > 0` | ✅ |
| `persons` | `list[BoundingBox]`; ROI-local | `list[BoundingBox]`; ROI-local | ✅ |
| `frame_id` in output | `str`; present | `str`; copied from metadata unchanged | ✅ |
| No confidence in output | Required | Absent from `PersonDetectionResult` | ✅ |
| **Configuration** | | | |
| Confidence threshold | Init-time config; not a per-call parameter | `person_confidence_threshold = 0.35` in `PersonDetectionConfig` | ✅ (OD spec §17 discrepancy — see Issue 6) |
| NMS threshold | Init-time config | `nms_iou_threshold = 0.35` | ✅ |
| Model path | Init-time config | `model_path = "yolo11m.pt"` | ✅ |
| **Error Handling** | | | |
| Validation failure | RPM catches stage exception per-region; continues with remaining regions | `InputValidator` raises `ValueError` / `TypeError` | ✅ |
| Empty detection | `persons = []`, `person_detected = false` | `persons = []`, `person_detected = false` | ✅ |
| **Stateless / Deterministic** | Required | No cross-frame state; same input → same output | ✅ |
| **Preprocessing ownership** | FTL owns all preprocessing | OD does no preprocessing | ✅ |

---

## 5. Current Mismatches

### Issue 1 — `roi_image` typed as `Any` / raw `np.ndarray`; OD spec declares it as `Image` struct (Medium)

**Files and sections:**
- `shared_contracts.md §6` — defines `Image` as the single canonical public image type; mandates "Public APIs must NOT expose raw `np.ndarray` directly."
- OD spec §5.2 — `roi_image (Image — see shared_contracts.md §6)` — explicitly types `roi_image` as `Image`
- RPM spec §8.5.2 — assigns `roi_image = ProcessedFrame.image`; `ProcessedFrame.image` is an `Image` struct per the FTL contract
- `src/image_processing/object_detection/module.py` — `ObjectDetectionInput` declares `roi_image: Any`
- `InputValidator.validate_input()` — checks `isinstance(model_ready_input, np.ndarray)` — validates and consumes a raw ndarray, not an `Image` struct

**Root cause:**
`Image` is not implemented in `shared/contracts.py` (see Issue 2), so the field cannot be typed as `Image` in the module.

**Effect:**
The validator and inference engine accept a raw ndarray. If the FTL passes a proper `Image` struct (as required by `shared_contracts.md §6` and the RPM spec), `isinstance(model_ready_input, np.ndarray)` will fail. The public API rule from `shared_contracts.md §6` is not enforced.

**Required change:** Object Detection implementation, contingent on Issue 2 being resolved first.

---

### Issue 2 — `Image` struct not implemented in `shared/contracts.py` (Medium)

**Files and sections:**
- `shared_contracts.md §6` — defines `Image` with fields: `data`, `width`, `height`, `color_format`, `layout`, `dtype`, `value_range`
- `src/image_processing/shared/contracts.py` — implements only: `ResizePolicy`, `GeometrySpec`, `OutputImageType`, `PipelineStageInputContract`
- `src/image_processing/object_detection/__init__.py` — does not export `Image`

**Effect:**
No module can comply with `shared_contracts.md §6` without `Image` defined in code. This is the root cause of Issue 1. Any future pipeline stage or test that tries to construct an `Image` struct will have no type to import from the shared contracts module.

**Required change:** Add `Image` TypedDict to `src/image_processing/shared/contracts.py`.

---

### Issue 3 — OD spec §5.2 claims standalone `width`/`height` were consolidated; rest of spec, RPM spec, and implementation all retain them (Medium)

**Files and sections:**
- OD spec §5.2 — states: "The standalone `width` and `height` fields previously present on `ObjectDetectionInput` have been consolidated into `roi_image.width` and `roi_image.height`."
- OD spec §7.2 — Input Validator responsibilities still lists: "Validate presence of required metadata fields (`camera_id`, `frame_id`, `timestamp_ms`, `width`, `height`, `roi_bbox_frame`)" — standalone `width` and `height` listed
- OD spec §10 — Processing Pipeline step 1 still lists: "Metadata validation (`camera_id`, `frame_id`, `width`, `height`)"
- RPM spec §8.5.2 — constructs `ObjectDetectionInput` with `width = ProcessedFrame.image.width`, `height = ProcessedFrame.image.height` as standalone fields
- RPM spec §8.8 step 8 — same standalone construction described
- `module.py` — `ObjectDetectionInput` has `width: int` and `height: int`; `FrameMetadata` has `width: int` and `height: int`; `InputValidator` validates `metadata["width"] > 0` and `metadata["height"] > 0`; shape check uses `metadata["width"]` and `metadata["height"]`

**Effect:**
OD spec §5.2 is internally inconsistent with OD spec §7.2 and §10, and out of sync with the implementation and RPM spec. The "consolidated" claim is incorrect — standalone `width` and `height` remain in `ObjectDetectionInput`, `FrameMetadata`, and the validator.

**Required change:** OD spec §5.2 documentation correction — remove the "consolidated" claim and accurately describe that `width` and `height` remain as standalone metadata fields.

---

### Issue 4 — RPM spec §8.3 uses field name `output_type`; correct name is `output_image_type` (Minor)

**Files and sections:**
- `shared_contracts.md §5` — `PipelineStageInputContract { output_image_type: OutputImageType; geometry_spec: GeometrySpec }`
- `src/image_processing/shared/contracts.py` — `class PipelineStageInputContract(TypedDict): output_image_type: OutputImageType`
- OD spec §4 — uses `output_image_type: RGB_UINT8_HWC` (correct)
- RPM spec §8.3 prose — "...obtained from the corresponding pipeline stage input contract (`stage_input_contract.output_type` and `stage_input_contract.geometry_spec`)" — `output_type` is wrong
- RPM spec §8.3 `get_frame` call — `object_detection_contract.output_type` — wrong field name

**Effect:**
Any code referencing `stage_input_contract["output_type"]` on a `PipelineStageInputContract` TypedDict would produce a `KeyError` at runtime. The correct key is `output_image_type`. OD spec and `shared/contracts.py` already use the correct name; RPM spec §8.3 is the outlier.

**Required change:** RPM spec §8.3 documentation correction — replace all `output_type` references with `output_image_type`.

---

### Issue 5 — `InputValidator` does not check `color_format`, `layout`, `dtype`, or `value_range` as required by OD spec §5.3 (Minor)

**Files and sections:**
- OD spec §5.3 — "`roi_image.color_format`, `roi_image.layout`, `roi_image.dtype`, and `roi_image.value_range` must match the expected `OutputImageType` for the configured model"; "`roi_image.data.shape` must be consistent with `roi_image.width`, `roi_image.height`, `roi_image.layout`, and `roi_image.color_format`"
- `module.py` `InputValidator.validate_input()` — validates ndarray type, 2D/3D shape, `shape[0] == metadata["height"]`, `shape[1] == metadata["width"]`; no `color_format`, `layout`, `dtype`, or `value_range` check

**Root cause:** These validations require `roi_image` to be an `Image` struct (Issues 1 and 2). With a raw ndarray, the Image metadata fields do not exist to validate.

**Effect:** An image with the wrong color format, layout, or dtype will pass validation silently. The YOLO model may receive an incompatible input without any diagnostic error.

**Required change:** Implement after Issues 1 and 2 are resolved — update `InputValidator` to validate `Image` metadata fields against the stage's declared `OutputImageType`.

---

### Issue 6 — Default confidence threshold: OD spec §17 says 0.5; implementation default is 0.35 (Minor)

**Files and sections:**
- OD spec §17 — "The default person confidence threshold is typically set to 0.5 (configurable)."
- `module.py` — `PersonDetectionConfig.person_confidence_threshold: float = 0.35`

**Effect:** Documentation inconsistency. The spec says "typically 0.5" but the implemented default is 0.35. Not a hard requirement per the word "typically," but a developer reading the spec will have incorrect expectations about default behavior.

**Required change:** OD spec §17 documentation correction — update the example value to reflect the implemented default of 0.35, or explicitly document the implemented default alongside the configurable range.

---

## 6. Recommended Resolution Plan

Steps are ordered by dependency. Steps 1 and 2 are prerequisites for Steps 3 and 4.

### Step 1 — Implement `Image` TypedDict in `shared/contracts.py`
**Resolves:** Issue 2 (foundational); unblocks Issues 1 and 5
**File:** `src/image_processing/shared/contracts.py`

Add:
```python
class Image(TypedDict):
    data:         np.ndarray
    width:        int
    height:       int
    color_format: str   # "RGB", "BGR", "GRAY"
    layout:       str   # "HWC", "CHW"
    dtype:        str   # "uint8", "float32"
    value_range:  str   # "[0,255]", "[0,1]", "[-1,1]"
```
Export from `src/image_processing/shared/__init__.py` and from `src/image_processing/object_detection/__init__.py`.

### Step 2 — Migrate `ObjectDetectionInput.roi_image` from `Any` to `Image`; update `detect()` and `InputValidator`
**Resolves:** Issues 1, 5
**File:** `src/image_processing/object_detection/module.py`

- Change `roi_image: Any` to `roi_image: Image` in `ObjectDetectionInput`.
- In `detect()`: pass `input["roi_image"]["data"]` as `model_ready_input` to `process()`.
- Update `InputValidator.validate_input()` to accept `roi_image: Image`, validate `isinstance(roi_image["data"], np.ndarray)`, validate `roi_image["width"] > 0` and `roi_image["height"] > 0`, and validate `color_format`, `layout`, `dtype`, and `value_range` against the `RGB_UINT8_HWC` contract declared by `get_input_contract()`.

### Step 3 — Correct OD spec §5.2 "consolidated" claim for standalone `width`/`height`
**Resolves:** Issue 3
**File:** `doc/image_processing_service/object_detection.md` §5.2

Remove the sentence: "The standalone `width` and `height` fields previously present on `ObjectDetectionInput` have been consolidated into `roi_image.width` and `roi_image.height`."
Replace with a description that accurately reflects the current state: `width` and `height` remain as standalone metadata fields in `ObjectDetectionInput`, set from `ProcessedFrame.image.width` and `ProcessedFrame.image.height` by the caller, and used by the validator for shape consistency checking.

### Step 4 — Fix `output_type` → `output_image_type` in RPM spec §8.3
**Resolves:** Issue 4
**File:** `doc/image_processing_service/RecognitionPipelineManager.md` §8.3

Replace all occurrences of `stage_input_contract.output_type`, `object_detection_contract.output_type`, `motion_contract.output_type`, `face_detection_contract.output_type`, and `face_recognition_contract.output_type` with the correct field name `output_image_type`.

### Step 5 — Update OD spec §17 default confidence threshold
**Resolves:** Issue 6
**File:** `doc/image_processing_service/object_detection.md` §17

Update the example default from 0.5 to 0.35 to match the implemented `PersonDetectionConfig.person_confidence_threshold`.

---

## 7. Final Compatibility Status

| Dimension | Status |
|---|---|
| **Overall Compatibility** | ⚠️ Substantially Compatible — all previously blocking issues resolved; six non-blocking spec-implementation gaps remain |
| **Critical blocking issues** | None |
| **Safe to integrate?** | Yes, at the current API surface level. `detect()` and `get_input_contract()` are both present and functional. Coordinate projection, output contract, and error handling are all aligned. Integration will work as long as the FTL passes a raw `np.ndarray` as `roi_image` for now; full `Image` struct compliance requires Step 1 and Step 2 above. |
| **Open medium issues** | Issue 1 (`roi_image` type mismatch — `Any` vs `Image`), Issue 2 (`Image` not in `shared/contracts.py`), Issue 3 (OD spec §5.2 inconsistency on `width`/`height`) |
| **Open minor issues** | Issue 4 (RPM spec `output_type` field name), Issue 5 (missing color/format/dtype validator checks), Issue 6 (default confidence threshold documentation) |
| **Coordinate system alignment** | ✅ — OD produces ROI-local bboxes; projection correctly owned by `SpatialCoordinator` in RPM |
| **Output contract alignment** | ✅ — `PersonDetectionResult` fields, types, and semantics match RPM expectations |
| **Error handling alignment** | ✅ — OD raises `ValueError`/`TypeError`; RPM catches per-region and continues with remaining regions |
| **Statelessness** | ✅ — No cross-frame state; deterministic output |
| **Preprocessing ownership** | ✅ — FTL owns all preprocessing; OD performs none |
| **Implementation readiness** | Ready for integration at current interface level. `Image` struct migration (Steps 1–2) is the next required step for full `shared_contracts.md` compliance. |
