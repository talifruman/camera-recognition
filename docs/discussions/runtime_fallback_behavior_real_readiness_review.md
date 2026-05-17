# Runtime Fallback Behavior — REAL Readiness Review

**Date:** 2026-05-13
**Scope:** Fallback and runtime-safety contracts across the five pipeline stages.
**Sources:**
- `src/image_processing/recognition_pipeline_manager/pipeline_orchestrator.py`
- `src/image_processing/motion_detection/module.py`
- `src/image_processing/object_detection/module.py`
- `src/image_processing/face_detection/module.py`
- `src/image_processing/face_recognition/module.py`
- `src/image_processing/person_directory/module.py`

---

## 1. Expected Fallback Contract

### Case A — No Motion Detected
| Field | Expected |
|---|---|
| `motion_detected` | `False` |
| Object Detection | Skipped |
| Face Detection | Skipped |
| Face Recognition | Skipped |
| Final output | `persons: []` |

### Case B — No Person Detected
| Field | Expected |
|---|---|
| `person_bboxes` | `[]` |
| Face Detection | Skipped |
| Face Recognition | Skipped |
| Final output | `persons: []` |

### Case C — No Face Detected
| Field | Expected |
|---|---|
| `detections` | `[]` |
| Face Recognition | Skipped |
| Final output | Person present in `persons[]`, `recognized_faces: []` |

### Case D — Recognition Below Threshold
| Field | Expected |
|---|---|
| `person_found` | `False` |
| `person_id` | `"UNKNOWN"` |
| Exception raised | No |
| Output object | Valid `FaceRecognitionOutput` returned |
| Effect on `recognized_faces` | Face omitted entirely |

### Case E — `person_id` Missing from PersonDirectory
| Field | Expected |
|---|---|
| `person_id` | `"UNKNOWN"` |
| `person_name` | `"UNKNOWN"` |
| `found` | `False` |
| `KeyError` raised | No |
| Crash | No |

---

## 2. Current Implementation Behavior

### Motion Detection (`MotionDetectionManager`)

- **No-motion path:** `FrameDifferencingMotionDetector.measure()` returns
  `MotionMeasurementResult(motion_fraction=<value>, bboxes=[])` when there are no
  significant contours. The decision policy then returns
  `MotionDetectionResultInternal(detected=False, bboxes=[])`, which the output
  builder converts to `MotionResult(detected=False, bboxes=[])`.
- **Exception guard:** The public `detect()` method wraps all execution in a
  `try/except Exception` that returns `MotionResult(detected=False, bboxes=[])` on
  any unhandled error (spec §11).
- **Validation failure:** Input validation failures return `detected=False` via the
  same path.
- **STUB behaviour:** `StubMotionDetectionAlgorithm` always returns
  `motion_fraction=0.2` with one synthetic bbox — **it never exercises the
  no-motion path.**

### Object Detection (`ObjectDetectionModule`)

- **No-person path:** `ResultBuilder.build()` returns
  `PersonDetectionResult(person_detected=False, persons=[])` when the filtered
  detections list is empty.
- **Motion gate:** Object Detection is never called when `motion_result["detected"]
  == False`. The gate lives in `PipelineOrchestrator.execute()` at line 167, not
  inside the module itself.
- **STUB behaviour:** The stub OD always returns 2 hardcoded person bboxes — **it
  never exercises the empty-persons path.**

### Face Detection (`FaceDetectionModule`)

- **No-face path:** `SCRFDFaceDetector.detect()` returns `[]` when no detections
  pass the confidence threshold and NMS. The module builds
  `FaceDetectionOutput(detections=[])`.
- **Exception guard:** `detect_faces()` wraps all execution in `try/except
  Exception`, returning `_empty_output(face_input)` on any error
  (`detections=[]`).
- **Person gate:** No explicit guard. The gate is implicit: Face Detection is only
  invoked inside `_process_person_faces()`, which is called once per person bbox.
  If `od_result["persons"]` is empty that method is never called at all.
- **STUB behaviour:** `StubFaceDetectorEngine` always returns exactly one detection
  — **it never exercises the empty-detections path.**

### Face Recognition (`FaceRecognitionModule`)

- **Below-threshold path:** `FaceRecognitionDecisionPolicy.decide()` returns
  `RecognitionDecision(person_found=False, person_id="UNKNOWN")` when
  `candidate.similarity < threshold` or `candidate is None` (empty gallery).
- **Exception guard:** `recognize()` wraps all execution in `try/except Exception`,
  returning `_no_match_output(face_input)` which always sets `person_found=False,
  person_id="UNKNOWN"`.
- **`ArcFaceEmbeddingEngine` exceptions:** `RuntimeError` and `ValueError` from the
  ONNX engine are caught by the module-level guard above — they do not propagate.
- **Face gate:** No explicit guard inside the module. The gate is implicit: Face
  Recognition is only invoked inside the `for detected_face in
  fd_output["detections"]` loop in `_process_person_faces()`. An empty list means
  the loop body never runs.
- **RPM discard rule:** After `recognize()` returns, `_process_face_recognition()`
  immediately returns without appending anything when `fr_output["person_found"] ==
  False`. Unrecognized faces are omitted from `recognized_faces` entirely (spec §11).

### PersonDirectory (`PersonDirectoryStore`)

- **Missing `person_id`:** `PersonDirectoryStore.get()` returns
  `_UNKNOWN_OUTPUT.copy()` — `PersonDirectoryOutput(person_id="UNKNOWN",
  person_name="UNKNOWN", found=False)` — when the key is absent, empty, or
  `"UNKNOWN"`.
- **No exception path:** The implementation never raises `KeyError` or any other
  exception for a missing key.
- **RPM enrichment:** PersonDirectory is only consulted when `fr_output
  ["person_found"] == True`. If the directory lookup returns `found=False` (e.g.,
  stale `person_id`), the face is still appended to `recognized_faces` with
  `found=False`, preserving the detected identity without crashing.

---

## 3. Gaps and Inconsistencies

### G1 — STUB implementations never cover fallback paths (Medium)

All three early-stage stubs are hardcoded to the happy path:

| Stub | Always returns |
|---|---|
| `StubMotionDetectionAlgorithm` | `motion_fraction=0.2`, one bbox (motion always detected) |
| Stub Object Detection | 2 person bboxes (persons always present) |
| `StubFaceDetectorEngine` | 1 detection with `confidence=0.95` (face always found) |

Cases A, B, and C are therefore **never exercised when the pipeline runs with stub
engines**. The fallback paths are only reachable in REAL mode.

### G2 — `PersonDirectory.get_person()` has no exception wrapper in RPM (Low)

In `_process_face_recognition()`, the call to `self._person_dir.get_person()` is
made with no surrounding `try/except`:

```python
pd_output = self._person_dir.get_person(recognized_person_id)
```

`PersonDirectoryStore` does not currently raise, but there is no defensive net if
`load()` was never called, if the store implementation changes, or if the backing
data is malformed at runtime.

### G3 — `FaceRecognitionOutput` comment contradicts implementation (Low, no runtime risk)

`module.py` line 85 carries the comment:

```python
person_id: str  # populated only when person_found = True (spec §3.2)
```

However, `_no_match_output()` always writes `person_id="UNKNOWN"` regardless of
`person_found`. The actual guard is in the RPM (`if not fr_output["person_found"]:
return`), so `person_id` on failure outputs is never consumed. There is no runtime
risk, but the comment is misleading to future readers.

### G4 — Implicit-only guards for Cases B and C (Info)

The guards that prevent Face Detection (Case B) and Face Recognition (Case C) from
running on empty inputs are **implicit loop guards** — an empty list means the loop
body never executes. This is correct Python, but there is no log trace or explicit
early-return to confirm the skip at runtime, which can complicate debugging.

---

## 4. Recommended Fixes

### R1 — Add STUB configurations that return empty outputs (addresses G1)

For each stub, add a configurable mode or a separate test-only variant that returns
no-motion / no-persons / no-faces. This allows integration tests to cover Cases A,
B, and C without switching to REAL engines.

No production code changes required.

### R2 — Wrap `PersonDirectory.get_person()` in a `try/except` in RPM (addresses G2)

```python
try:
    pd_output = self._person_dir.get_person(recognized_person_id)
except Exception:
    return  # skip this face — directory unavailable
```

This matches the defensive pattern already used for every other external call in
`_process_face_recognition()` and costs nothing in the happy path.

### R3 — Correct the `FaceRecognitionOutput.person_id` comment (addresses G3)

Change:

```python
person_id: str  # populated only when person_found = True (spec §3.2)
```

To:

```python
person_id: str  # "UNKNOWN" when person_found = False; caller must check person_found first
```

No behavior change; documentation only.

### R4 — Add trace-level log lines for implicit skips (addresses G4, optional)

In `_process_person_faces()` and `_process_face_recognition()`, a single
`logger.debug()` call after the relevant loop or guard would make no-face and
no-recognition paths observable in REAL runtime logs without changing any behavior.

---

## 5. Safety Verdict

**The pipeline is safe for REAL runtime fallback scenarios** for all five tested
cases:

| Case | Status | Notes |
|---|---|---|
| A — No motion | ✅ Fully implemented | Explicit guard in RPM; all stages skipped |
| B — No person | ✅ Implemented | Implicit loop guard; Face Det and Face Rec skipped |
| C — No face | ✅ Implemented | Implicit loop guard; Face Rec skipped |
| D — Below threshold | ✅ Implemented | `person_found=False`, face omitted, no exception |
| E — Missing in PersonDirectory | ✅ Fully implemented | `found=False`, `"UNKNOWN"` fallback, no crash |

**Two low-severity items** require attention before extended REAL operation:

1. **G2** — `PersonDirectory.get_person()` lacks a defensive `try/except` in RPM.
   Recommended fix R2 is a one-liner.
2. **G1** — Fallback paths A, B, C are only reachable with REAL engines. Stub-based
   integration tests give false confidence that these paths are covered.

No unsafe `None` leaks, no unhandled `KeyError` paths, and no missing output
standardization were found in the REAL code paths.
