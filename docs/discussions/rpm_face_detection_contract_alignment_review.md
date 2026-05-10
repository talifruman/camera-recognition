# RecognitionPipelineManager ↔ Face Detection Compatibility Review

**Review Date:** 2026-05-10
**Status:** NOT COMPATIBLE — 1 blocking critical mismatch; 3 medium mismatches
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

> **Note:** RecognitionPipelineManager has **no source code implementation** yet. The RPM side of this review is spec-only. The Face Detection module is fully implemented and is the authoritative truth for that side. Where spec and implementation diverge for Face Detection, the implementation is treated as ground truth.

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
| `timestamp_ms` | `frame_packet.timestamp_ms` | `int` — capture timestamp |
| `roi_image` | `ProcessedFrame.image` | **`Image` struct** (shared_contracts.md §6) — person ROI prepared by FTL |

`FaceDetectionInput` does not carry `roi_bbox_frame` or any spatial metadata. This is consistent across both specs and the implementation. The person-ROI position in full-frame coordinates is maintained by `PipelineOrchestrator` and used only for coordinate projection after face detection returns.

### 2.3 Person Bbox Source

`PersonResult.person_bbox` is a full-frame projected bounding box produced by `SpatialCoordinator` after Object Detection. It is used as the crop region passed to the Frame Transformation Layer to generate the person ROI image.

### 2.4 ROI Image Source

`ProcessedFrame.image` — the FTL returns a `ProcessedFrame` containing an `Image` struct (carrying `data`, `width`, `height`, `color_format`, `layout`, `dtype`, `value_range` as defined in shared_contracts.md §6). The FTL applies all color conversion, layout conversion, dtype conversion, normalization, and resizing before returning the image. RPM sets `FaceDetectionInput.roi_image = ProcessedFrame.image`, which is therefore an `Image` struct.

### 2.5 Whether Face Detection Receives Person ROI or Full Frame

Face Detection receives only the **person ROI image** — not the full frame. This is consistent across both specs and the implementation.

### 2.6 `roi_bbox_frame` Handling

Neither spec includes `roi_bbox_frame` in `FaceDetectionInput`. The Face Detection spec §2.2 explicitly states: *"FaceDetectionInput does not carry `roi_bbox_frame` or any spatial metadata."* The RPM spec §8.5.3 confirms: *"FaceDetectionInput does not carry `roi_bbox_frame`, spatial metadata, or ROI coordinates."* The implementation has no `roi_bbox_frame` field and the validator does not check for it. This was a previously reported critical mismatch — it is now fully resolved.

### 2.7 Coordinate Space of Input Image

The input `roi_image` is in **person-ROI-local coordinate space** (the person ROI is the origin). All pixel coordinates within the image are relative to the top-left corner of the cropped person region.

### 2.8 Coordinate Space of Returned Face Bbox

- **Face Detection spec §7.2:** `face_bbox` is expressed in **ROI-local coordinates relative to `roi_image`** (person-ROI-local). Full-frame projection is the responsibility of `RPM` / `PipelineOrchestrator`.
- **Implementation (`FaceDetectionOutputBuilder`):** Builds `DetectedFace(face_bbox=det.bbox, landmarks=...)` where `det.bbox` is the ROI-local bounding box as returned by the engine and postprocessor. No coordinate projection step exists.
- **RPM (§8.5.3, §8.8 step 11–12, §10):** Correctly expects ROI-local face bboxes. After receiving `FaceDetectionOutput`, calls `SpatialCoordinator.project_bbox_to_full_frame(DetectedFace.face_bbox, ProcessedFrame.source_bbox_full_frame)` to project each face bbox to full-frame coordinates.
- This is now **consistent** across both specs and the implementation.

### 2.9 Landmark Handling

- Face Detection `FaceDetectionOutputBuilder` outputs landmarks in **person-ROI-local coordinates relative to `roi_image`** (no projection step exists in the implementation).
- RPM §8.5.4 passes `DetectedFace.landmarks` directly to `FaceRecognitionInput.landmarks` without any coordinate transformation.
- Face Recognition spec §2.2 requires landmarks that are *"ROI-local relative to `face_roi_image`"* — meaning face-crop-local, not person-ROI-local.
- There is a coordinate-space mismatch between what Face Detection outputs (person-ROI-local) and what Face Recognition requires (face-crop-local). RPM currently passes them through unchanged.

### 2.10 `frame_id` and `camera_id` Handling

Both sides treat `frame_id` and `camera_id` as `str` values used for traceability, passed unchanged from input to output.

### 2.11 Image Contract

| Property | RPM Expectation | Face Detection Spec | Implementation |
|----------|----------------|---------------------|----------------|
| Image type | `Image` struct (shared_contracts.md §6) via `ProcessedFrame.image` | `Image` struct in `FaceDetectionInput.roi_image` | `Any` (TypedDict); validator checks `isinstance(roi_image, np.ndarray)` — **rejects `Image` struct** |
| Color format | Delegated to `face_detection_contract.output_type` | RGB (default) | `DetectorInputContract.color_format = "RGB"` |
| Layout | Delegated to stage contract | HWC | `DetectorInputContract.layout = "HWC"` |
| dtype | Delegated to stage contract | uint8 | `DetectorInputContract.dtype = "uint8"` |
| Value range | Delegated to stage contract | [0, 255] | `DetectorInputContract.value_range = "[0, 255]"` |
| Preprocessing ownership | FTL preprocesses; Face Detection validates | Upstream preprocesses; module validates | ✅ Consistent |

### 2.12 `get_input_contract()` Method

`FaceDetectionModule.get_input_contract()` is **implemented** and returns `PipelineStageInputContract(output_image_type=OutputImageType.RGB_UINT8_HWC, geometry_spec=self._config.geometry_spec)`. RPM §6.1 and §18 correctly require it on every pipeline stage interface. This was a previously reported critical mismatch — resolved at the implementation level. The Face Detection spec §9.1 and class diagram §13 do not yet document the method.

---

## 3. Compatibility Verification Checklist

| Category | RPM Definition | Face Detection Definition | Compatible? | Required Change |
|----------|---------------|--------------------------|-------------|-----------------|
| **API — method name** | `FaceDetectionInterface.detect_faces(input)` | `FaceDetectionModule.detect_faces(face_input)` | ✅ Yes | None |
| **API — return type** | `FaceDetectionOutput` | `FaceDetectionOutput` | ✅ Yes | None |
| **API — sync/async** | Synchronous | Synchronous | ✅ Yes | None |
| **API — `get_input_contract()`** | Required (§6.1); queried at initialization | Not in spec §9.1; **implemented** in `FaceDetectionModule` | ⚠️ Partial | Add to Face Detection spec §9.1 and class diagram §13 |
| **Input — `frame_id`** | `str`, present | `str`, required | ✅ Yes | None |
| **Input — `camera_id`** | `str`, present | `str`, required, non-empty | ✅ Yes | None |
| **Input — `timestamp_ms`** | `int` (uint64 in pseudo-struct notation) | `int`, required | ✅ Yes (minor type notation diff) | None functionally |
| **Input — `roi_image` type** | `Image` struct from `ProcessedFrame.image` (shared_contracts.md §6) | Spec: `Image` struct; impl: `np.ndarray` required by validator | ❌ No | Align implementation to accept and validate `Image` struct |
| **Input — `roi_bbox_frame`** | Not provided; confirmed excluded per §8.5.3 | Not present in spec or implementation | ✅ Yes | None |
| **Input — extra fields** | None added | None extra expected | ✅ Yes | None |
| **ROI — input is cropped person ROI** | Yes (from FTL) | Yes | ✅ Yes | None |
| **ROI — resizing before calling Face Detection** | FTL performs resize per stage contract | Expects pre-resized image matching contract | ✅ Yes | None |
| **Image — color format** | RGB_UINT8_HWC | RGB, HWC, uint8, [0,255] | ✅ Yes | None |
| **Image — preprocessing ownership** | FTL preprocesses; Face Detection validates | Upstream preprocesses; module validates | ✅ Yes | None |
| **Coordinate — face bbox output space** | RPM expects ROI-local; projects via `SpatialCoordinator` | ROI-local (spec §7.2 and implementation) | ✅ Yes | None |
| **Coordinate — no double projection** | Single `SpatialCoordinator` call after Face Detection | No internal projection in Face Detection | ✅ Yes | None |
| **Coordinate — landmark space (into FaceRecognition)** | Passed through without transformation | Person-ROI-local | ❌ No | RPM must transform to face-crop-local before `FaceRecognitionInput` |
| **`SpatialCoordinator` role** | Projects person and face bboxes to full-frame | Not applicable to Face Detection internals | ✅ Yes | None |
| **Landmark — required or optional** | Always present in `DetectedFace` | Always present per spec §7.2 | ✅ Yes | None |
| **Landmark — count** | 5-point canonical | 5-point canonical | ✅ Yes | None |
| **Landmark — structure** | `FaceLandmarks` (shared_contracts.md §8) | `FaceLandmarks` TypedDict | ✅ Yes | None |
| **Output — detection container field name** | `detections` (§8.8, §10, §18) | `detections: list[DetectedFace]` | ✅ Yes | §8.5.3 has one stale `face_bboxes` prose reference — fix there |
| **Output — face bbox field name** | `DetectedFace.face_bbox` (§8.8 step 12) | `DetectedFace.face_bbox` | ✅ Yes | Fix stale `face_bboxes` in §8.5.3 prose |
| **Output — landmarks field** | `DetectedFace.landmarks` | `DetectedFace.landmarks: FaceLandmarks` | ✅ Yes | None |
| **Output — confidence exposure** | Not exposed | Confidence is strictly internal | ✅ Yes | None |
| **Output — empty output handling** | `detections = []` | Returns `detections = []` on any error | ✅ Yes | None |
| **Output — multiple faces handling** | Iterates `FaceDetectionOutput.detections` | Returns all accepted faces | ✅ Yes | None |
| **Config — confidence threshold ownership** | Not referenced by RPM; internal | `FaceDetectionPostprocessor` owns it; init-time | ✅ Yes | None |
| **Config — `geometry_spec` in `FaceDetectionConfig`** | Not referenced directly | Present in impl; absent from spec §3.1 | ⚠️ Minor | Document in spec §3.1 |
| **`FaceDetectorEngine.detect()` signature** | N/A (internal interface) | Spec §9.2: `detect(prepared_roi: Image)`; impl: `detect(roi_image: np.ndarray)` | ❌ No | Align spec §9.2 and §13 with implementation |
| **BoundingBox naming** | `BoundingBox` (shared_contracts.md §1) | `BoundingBox` | ✅ Yes | None — `CanonicalBoundingBox` fully replaced |
| **Error handling** | Returns valid output with empty `persons` | Returns valid output with empty `detections` | ✅ Yes | None |
| **Diagrams — Face Detection class diagram §13** | N/A | Missing `get_input_contract()` method | ⚠️ Minor | Add method to diagram |
| **Diagrams — Face Detection sequence diagram §14** | N/A | Generic `Caller` participant | ⚠️ Minor | No action required — architecturally correct |

---

## 4. Detected Mismatches

---

### Mismatch 1 — `roi_image` Type: Implementation Requires `np.ndarray`; Spec and RPM Supply `Image` Struct

**Impact Level: Critical**

**Files and Sections:**
- `doc/image_processing_service/shared_contracts.md` §6
- `doc/image_processing_service/face_detection.md` §2.2, §5.2
- `doc/image_processing_service/RecognitionPipelineManager.md` §8.5.3, §8.8 step 11
- `src/image_processing/face_detection/module.py` — `FaceDetectionInput`, `FaceDetectionInputValidator`

**Structs/Classes/Functions Involved:**
- `FaceDetectionInput.roi_image`
- `Image` (shared_contracts.md §6)
- `ProcessedFrame.image` (FTL output — `Image` struct)
- `FaceDetectionInputValidator.validate()`
- `FaceDetectorEngine.detect()` (receives raw `np.ndarray` in implementation)

**Description:**

`shared_contracts.md §6` defines `Image` as the single canonical public image type and explicitly states:

> *"Public APIs must NOT expose raw `np.ndarray` directly. All image payloads crossing a public module boundary must be carried in an `Image` struct."*

The Face Detection spec §2.2 correctly adopts this rule — `FaceDetectionInput.roi_image` is declared as `Image`. Spec §5.2 describes the validator as checking `roi_image.data`, `roi_image.width`, `roi_image.height`, `roi_image.color_format`, `roi_image.layout`, `roi_image.dtype`, and `roi_image.value_range` — all fields of the shared `Image` struct.

The RPM spec §8.5.3 constructs `FaceDetectionInput` with `roi_image = ProcessedFrame.image` — which is an `Image` struct returned by the FTL.

The implementation diverges from the spec in three places:

1. `FaceDetectionInput` TypedDict declares `roi_image: Any` (documented in a comment as `numpy.ndarray at runtime`), not `Image`.
2. `FaceDetectionInputValidator.validate()` explicitly requires a raw `np.ndarray`:
   ```python
   if not isinstance(roi_image, np.ndarray):
       raise TypeError("roi_image must be a numpy.ndarray")
   ```
3. `FaceDetectionInputValidator._validate_roi_contract()` validates raw ndarray properties (`.ndim`, `.dtype`) rather than `Image` struct metadata fields (`color_format`, `layout`, `dtype`, `value_range`).

**Why This Is Problematic:**

When RPM supplies `FaceDetectionInput.roi_image = ProcessedFrame.image` (an `Image` struct), the validator's `isinstance(roi_image, np.ndarray)` check fails, raising a `TypeError`. This is caught by the outer `try/except Exception` block in `FaceDetectionModule.detect_faces()` and silently converted into an empty `FaceDetectionOutput` with `detections = []`. Every `detect_faces` call from RPM produces zero detections for every person on every frame, with no visible error to the caller.

Additionally, the validator does not currently check the image metadata fields (`color_format`, `layout`, `dtype`, `value_range`) that are defined in the spec's validation rules and serve as the mechanism for verifying the image matches the configured `DetectorInputContract`.

**Which Module Should Change:**

The Face Detection implementation must be updated. The spec is correct; the implementation has not yet been migrated to use the shared `Image` contract.

**Recommended Fix:**

1. Change `FaceDetectionInput.roi_image` TypedDict annotation from `Any` to the shared `Image` type.
2. Update `FaceDetectionInputValidator.validate()` to accept and validate an `Image` struct: check `roi_image.data is not None`, `roi_image.width > 0`, `roi_image.height > 0`, and that `roi_image.color_format`, `roi_image.layout`, `roi_image.dtype`, `roi_image.value_range` match the configured `DetectorInputContract`.
3. In `FaceDetectionModule._detect_faces_internal()`, pass `face_input["roi_image"].data` (the raw ndarray buffer) to `self._detector_engine.detect()`, keeping the engine interface simple.
4. The spec text in §5.2 already describes the correct validation behavior — only the implementation needs to change.

---

### Mismatch 2 — `FaceDetectorEngine.detect()` Signature: Spec Uses `Image`; Implementation Uses `np.ndarray`

**Impact Level: Medium**

**Files and Sections:**
- `doc/image_processing_service/face_detection.md` §9.2, §13
- `src/image_processing/face_detection/module.py` — `FaceDetectorEngine` Protocol

**Structs/Classes/Functions Involved:**
- `FaceDetectorEngine.detect()` (internal interface)
- `SCRFDFaceDetector.detect()` (implementation)

**Description:**

The Face Detection spec §9.2 defines the internal detector abstraction:

```
interface FaceDetectorEngine {
    detect(prepared_roi: Image) -> RawFaceDetections
}
```

The class diagram in §13 shows the same signature. The spec note in §9.2 states: *"The `prepared_roi` parameter passed to `FaceDetectorEngine.detect()` is the validated shared `Image` struct from `FaceDetectionInput.roi_image`."*

The implementation in `module.py` declares:

```python
class FaceDetectorEngine(Protocol):
    def detect(self, roi_image: np.ndarray) -> list[RawFaceDetection]: ...
```

Two discrepancies: parameter name (`prepared_roi` vs `roi_image`) and parameter type (`Image` struct vs `np.ndarray`).

This mismatch is downstream of Mismatch 1: the engine interface reflects the pre-migration state where the module passed a raw ndarray directly to the engine.

**Why This Is Problematic:**

A developer implementing a future `FaceDetectorEngine` by following the spec would write `detect(prepared_roi: Image)`, but the `FaceDetectionModule` currently passes a raw ndarray. The spec cannot be safely used as the integration guide for new engine implementations until it is aligned.

**Recommended Fix:**

**Option A (recommended):** Update spec §9.2 and class diagram §13 to document `detect(roi_image: np.ndarray)`, and add a note that `FaceDetectionModule` extracts `Image.data` from the validated `Image` struct before passing it to the engine. This keeps the engine interface free of `Image` struct dependency.

**Option B:** Migrate `FaceDetectorEngine.detect()` to accept `Image` in the implementation and update spec accordingly.

Option A is preferred because it avoids leaking the `Image` struct into the internal engine abstraction.

---

### Mismatch 3 — Landmark Coordinate Space: Face Detection Outputs Person-ROI-Local; Face Recognition Requires Face-Crop-Local

**Impact Level: Medium**

**Files and Sections:**
- `doc/image_processing_service/face_detection.md` §7.2
- `doc/image_processing_service/face_recognition.md` §2.2, §2.3
- `doc/image_processing_service/RecognitionPipelineManager.md` §8.5.4, §8.8 step 12

**Structs/Classes/Functions Involved:**
- `DetectedFace.landmarks` (Face Detection output)
- `FaceRecognitionInput.landmarks` (Face Recognition input)
- `PipelineOrchestrator` (RPM — constructs `FaceRecognitionInput`)

**Description:**

The Face Detection spec §7.2 states:

> *"landmarks are always present in every `DetectedFace` and are expressed in **ROI-local coordinates relative to `roi_image`**."*

`roi_image` is the **person ROI image** — the prepared crop of the person region. Landmark coordinates are therefore in person-ROI-local space.

The Face Recognition spec §2.2 defines:

```
FaceRecognitionInput {
    ...
    FaceLandmarks landmarks;   // all coordinates ROI-local relative to face_roi_image
}
```

*"ROI-local relative to `face_roi_image`"* means landmarks must be in the coordinate space of the **face crop image** (the small face region passed to Face Recognition), not the person ROI image.

RPM §8.5.4 and §8.8 step 12 construct `FaceRecognitionInput` with `landmarks = DetectedFace.landmarks` — passed through without any coordinate transformation.

**Why This Is Problematic:**

`FaceAligner` uses landmarks to geometrically align the face within the face crop image. ArcFace alignment requires landmarks in the coordinate space of the image being aligned. If landmarks reference positions in the person ROI image rather than the face crop, alignment targets are systematically wrong. The extracted embedding does not correctly represent the face. This is a silent correctness failure — no exception is raised.

**Which Module Should Change:**

RPM `PipelineOrchestrator` must add a landmark coordinate transformation step. The Face Detection spec §7.2 and Face Recognition spec §2.2 are both internally consistent and require no change.

**Recommended Fix:**

After `SpatialCoordinator` produces the projected full-frame face bbox (from `DetectedFace.face_bbox` in person-ROI-local space), and before constructing `FaceRecognitionInput`, `PipelineOrchestrator` must re-express landmarks in face-crop-local space:

```
for each landmark lm in DetectedFace.landmarks:
    face_local_lm.x = lm.x - DetectedFace.face_bbox.x
    face_local_lm.y = lm.y - DetectedFace.face_bbox.y
```

Document this transformation step explicitly in RPM spec §8.5.4 and §8.8 step 12.

---

### Mismatch 4 — Stale `face_bboxes` Reference in RPM §8.5.3 Prose

**Impact Level: Medium**

**Files and Sections:**
- `doc/image_processing_service/RecognitionPipelineManager.md` §8.5.3

**Structs/Classes/Functions Involved:**
- `FaceDetectionOutput.detections` (correct field name)
- `DetectedFace.face_bbox` (correct field name)

**Description:**

RPM §8.5.3 contains the sentence:

> *"`FaceDetectionOutput.face_bboxes` are ROI-local coordinates relative to the person ROI image. They must never be used as full-frame coordinates or exposed publicly."*

The actual `FaceDetectionOutput` struct has `detections: list[DetectedFace]`, not `face_bboxes`. Individual face bounding boxes are `DetectedFace.face_bbox`. The stale name `face_bboxes` appears only in this one prose sentence in §8.5.3. All other sections use the correct names:

- §8.8 step 11: *"If `FaceDetectionOutput.detections` is empty"* ✅
- §8.8 step 12: *"`SpatialCoordinator.project_bbox_to_full_frame(DetectedFace.face_bbox, ...)`"* ✅
- §10: correct field names ✅
- §18 compliance checklist: correct ✅

**Why This Is Problematic:**

A developer reading §8.5.3 in isolation would look for `output["face_bboxes"]` and encounter a `KeyError` at runtime.

**Recommended Fix:**

Replace the stale sentence in RPM §8.5.3 with:

> *"`FaceDetectionOutput.detections` carries zero or more `DetectedFace` entries; each `DetectedFace.face_bbox` is in ROI-local coordinates relative to the person ROI image. These coordinates must never be used as full-frame coordinates or exposed publicly."*

---

### Mismatch 5 — `FaceDetectionConfig.geometry_spec` Field Present in Implementation, Absent from Spec §3.1

**Impact Level: Minor**

**Files and Sections:**
- `doc/image_processing_service/face_detection.md` §3.1
- `src/image_processing/face_detection/module.py` — `FaceDetectionConfig`

**Structs/Classes/Functions Involved:**
- `FaceDetectionConfig` dataclass

**Description:**

Spec §3.1 defines `FaceDetectionConfig` with two fields: `confidence_threshold` and `detector_input_contract`. The implementation adds:

```python
geometry_spec: GeometrySpec = field(
    default_factory=lambda: GeometrySpec(width=0, height=0, resize_policy=ResizePolicy.NONE)
)
```

This field is consumed by `FaceDetectionModule.get_input_contract()` to return the stage-defined geometry specification to RPM. The spec does not document it, leaving the configuration contract incomplete.

**Recommended Fix:** Add `geometry_spec: GeometrySpec` to `FaceDetectionConfig` in spec §3.1, noting it is passed through `get_input_contract()` to inform the FTL about the required frame geometry for this stage.

---

### Mismatch 6 — `get_input_contract()` Not Documented in Face Detection Spec §9.1 or Class Diagram §13

**Impact Level: Minor**

**Files and Sections:**
- `doc/image_processing_service/face_detection.md` §9.1, §13

**Structs/Classes/Functions Involved:**
- `FaceDetectionModule.get_input_contract()` (implemented)

**Description:**

Spec §9.1 documents only `detect_faces(input: FaceDetectionInput) -> FaceDetectionOutput`. Class diagram §13 shows `FaceDetectionModule` with only that method. `FaceDetectionModule.get_input_contract() -> PipelineStageInputContract` is implemented but not documented.

RPM §6.1, §9.2, §13.1, and §18 all require `get_input_contract()` on every pipeline stage interface. The method is implemented and functional — this is a documentation-only mismatch.

**Recommended Fix:** Add `get_input_contract() -> PipelineStageInputContract` to Face Detection spec §9.1 and the `FaceDetectionModule` entry in class diagram §13.

---

### Mismatch 7 — `timestamp_ms` Declared as `uint64` in Spec Pseudo-Structs vs Python `int` in Implementation

**Impact Level: Minor**

**Files and Sections:**
- `doc/image_processing_service/face_detection.md` §2.2, §7.1
- `doc/image_processing_service/RecognitionPipelineManager.md` §3.1
- `src/image_processing/face_detection/module.py` — `FaceDetectionInput`, `FaceDetectionOutput`

**Description:**

Both module specs use `uint64 timestamp_ms` in pseudo-struct notation. The implementation declares `timestamp_ms: int`. Python `int` covers the value range of `uint64` but does not enforce unsigned semantics. No runtime impact.

**Recommended Fix:** Optionally add a note in Face Detection spec §2.2 that `timestamp_ms` must be non-negative and represents milliseconds since epoch. No code change required.

---

## 5. Previously Reported Mismatches — Now Resolved

| # | Old Mismatch | Resolution |
|---|-------------|------------|
| 1 | `roi_bbox_frame` missing from RPM-constructed `FaceDetectionInput` | **Resolved.** `roi_bbox_frame` has been removed from `FaceDetectionInput` in both specs and the implementation. Neither spec references it. |
| 2 | Coordinate system ownership conflict — double projection | **Resolved.** Face Detection outputs ROI-local face bboxes and landmarks. RPM correctly applies `SpatialCoordinator.project_bbox_to_full_frame` after receiving output. No double projection occurs. |
| 3 | `get_input_contract()` not implemented in `FaceDetectionModule` | **Resolved.** `FaceDetectionModule.get_input_contract()` is now implemented and returns the correct `PipelineStageInputContract`. RPM initialization can query it. Spec documentation still lags (see Mismatch 6). |
| 4 | Stale `face_bboxes` field name across RPM §8.5.3, §10, §18 | **Partially resolved.** §10 and §18 now correctly use `detections` and `face_bbox`. A single stale sentence remains in §8.5.3 (see Mismatch 4 above). |
| 5 | Landmark coordinate space — ambiguity in RPM §10 | **Partially resolved.** Coordinate spaces are now explicitly defined in both specs. The transformation gap in RPM is now an active mismatch (see Mismatch 3 above). |
| 6 | RPM §18 compliance checklist contained incorrect rule for `roi_bbox_frame` | **Resolved.** The incorrect rule has been removed from §18. The current §18 correctly reflects the updated contract. |
| 7 | Stale `PreparedROI` type in Face Detection spec §9.2 and class diagram §13 | **Resolved.** Spec §9.2 explicitly documents: *"PreparedROI is no longer a separate public or internal type — the shared Image struct serves this role directly."* |
| 8 | `CanonicalBoundingBox` (RPM) vs `BoundingBox` (Face Detection) naming | **Resolved.** `shared_contracts.md §1` establishes `BoundingBox` as the single canonical type. All specs use `BoundingBox`. |
| 9 | `timestamp_ms` `uint64` vs Python `int` notation | Still present as a minor notation difference — see Mismatch 7. |
| 10 | Face Detection sequence diagram §14 shows generic `Caller` | No action required. Generic `Caller` is architecturally correct for a module-agnostic spec. |

---

## 6. Recommended Resolution Plan

Steps are ordered by impact — blocking issues first, then correctness, then documentation cleanup.

1. **Fix `roi_image` type contract (Mismatch 1)** — Update `FaceDetectionInput.roi_image` to carry the shared `Image` struct; rewrite `FaceDetectionInputValidator.validate()` to validate `Image` struct fields against `DetectorInputContract`; pass `face_input["roi_image"].data` to the detector engine in `_detect_faces_internal()`. This is the sole blocking critical fix.

2. **Align `FaceDetectorEngine.detect()` spec with implementation (Mismatch 2)** — Update spec §9.2 and class diagram §13 to document `detect(roi_image: np.ndarray)` and note that the module extracts `Image.data` before calling the engine.

3. **Resolve landmark coordinate space in RPM (Mismatch 3)** — Add a landmark re-expression step in `PipelineOrchestrator` §8.5.4 and §8.8 step 12: subtract `DetectedFace.face_bbox.x` and `.y` from each landmark coordinate before constructing `FaceRecognitionInput`.

4. **Fix stale `face_bboxes` in RPM §8.5.3 prose (Mismatch 4)** — Replace with correct `FaceDetectionOutput.detections` / `DetectedFace.face_bbox` references.

5. **Document `geometry_spec` in Face Detection spec §3.1 (Mismatch 5)** — Add `geometry_spec: GeometrySpec` to the `FaceDetectionConfig` struct.

6. **Document `get_input_contract()` in Face Detection spec §9.1 and class diagram §13 (Mismatch 6)** — Add method signature and return type.

7. **Add optional `timestamp_ms` semantic note (Mismatch 7)** — No code change required.

---

## 7. Final Compatibility Status

### Overall Status: Not Compatible

### Summary Table

| Dimension | Status |
|-----------|--------|
| API method name | ✅ Compatible |
| Input — `roi_bbox_frame` absence | ✅ Compatible (both specs confirm exclusion) |
| Input — `roi_image` type | ❌ Not Compatible (`Image` struct vs `np.ndarray`) |
| Coordinate system — face bbox ROI-local output | ✅ Compatible |
| No double projection | ✅ Compatible |
| Stage interface — `get_input_contract()` | ✅ Compatible (implemented) / ⚠️ Spec doc incomplete |
| Landmark coordinate space (Face Detection → Face Recognition) | ❌ Not Compatible (person-ROI-local passed as face-crop-local) |
| Error handling | ✅ Compatible |
| BoundingBox type | ✅ Compatible (`BoundingBox` unified in shared_contracts.md) |
| Preprocessing ownership | ✅ Compatible |
| Confidence exposure | ✅ Compatible |
| Traceability metadata | ✅ Compatible |
| Field names — output structure | ✅ Compatible (§8.8, §10, §18) / ⚠️ One stale sentence in §8.5.3 |
| `FaceDetectorEngine.detect()` signature | ❌ Spec/impl diverge (internal only; no runtime gap until Mismatch 1 is fixed) |

### Implementation Readiness

**Not ready for integration.** The Face Detection module is internally complete, but `FaceDetectionInput.roi_image` and `FaceDetectionInputValidator` have not been migrated to the shared `Image` contract. RecognitionPipelineManager has no source code yet — it is spec-only.

### Blocking Issues

| # | Issue | Mismatch | Blocks |
|---|-------|----------|--------|
| 1 | `roi_image` validated as `np.ndarray`; RPM supplies `Image` struct; validator raises `TypeError` silently converted to empty output | Mismatch 1 | Every `detect_faces` call from RPM returns empty detections |

### Must Fix Before Integration

| # | Issue | Mismatch | Risk |
|---|-------|----------|------|
| 2 | `FaceDetectorEngine.detect()` spec/impl inconsistency | Mismatch 2 | Wrong reference contract for future engine implementations |
| 3 | Landmark coordinate transformation missing in RPM | Mismatch 3 | Silent face alignment failure; wrong embedding extracted per frame |
| 4 | Stale `face_bboxes` in RPM §8.5.3 prose | Mismatch 4 | `KeyError` risk for RPM developers reading §8.5.3 in isolation |

### Whether Integration Is Currently Safe

**Integration is not safe.** Connecting RPM (as currently specified) to the existing Face Detection implementation would produce:

- **Silent empty face detections on every frame** (Mismatch 1) — every call fails at the `isinstance(roi_image, np.ndarray)` validator check; the exception is silently converted to `detections = []`
- **Silent wrong face alignment output in Face Recognition** (Mismatch 3) — even if Mismatch 1 were patched, person-ROI-local landmarks would be passed to `FaceRecognitionInput.landmarks` which expects face-crop-local values; face alignment fails silently

Both failures are silent correctness errors with no visible exception reaching the caller. **The integration must not proceed until Mismatch 1 and Mismatch 3 are resolved.**
