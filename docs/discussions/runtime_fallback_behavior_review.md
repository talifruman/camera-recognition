# Runtime Fallback Behavior Review

**Date:** 2026-05-13  
**Scope:** REAL recognition pipeline fallback/runtime behavior contracts only.  
**No architecture, RPM design, or module switching changes.**

---

## 1. Expected Fallback Contracts

The following five scenarios define the expected runtime fallback behavior of the pipeline.

### A. No motion detected
- `motion_detected == False`
- Object Detection skipped
- Face Detection skipped
- Face Recognition skipped
- Final output: `RecognitionPipelineOutput(persons=[])`

### B. No person detected
- `person_bboxes == []`
- Face Detection skipped
- Face Recognition skipped
- Final output: `RecognitionPipelineOutput(persons=[])`

### C. No face detected
- `faces == []`
- Face Recognition skipped
- Final output: `PersonResult(recognized_faces=[])`  
  (Person entry still present; faces list is empty)

### D. Recognition below threshold
- `person_id == "UNKNOWN"`
- `found == False`
- No exception raised
- Valid `FaceRecognitionOutput` returned by the FR module
- In pipeline output: face is **omitted** from `recognized_faces` (not labeled UNKNOWN)

> **Design note:** `RecognizedFaceResult` contains only confirmed identities. Unrecognized faces
> are never appended to `recognized_faces` — they are silently dropped per the type docstring.
> This is intentional. `PersonResult.recognized_faces` will be empty if all faces failed recognition.

### E. `person_id` missing from PersonDirectory
- Fallback: `person_id = "UNKNOWN"`, `person_name = "UNKNOWN"`, `found = False`
- No `KeyError`
- No crash
- Valid `PersonDirectoryOutput` returned

---

## 2. Current Implementation Behavior

### 2.1 Motion Detection (`MotionDetectionManager.detect`)

| Behavior | Implementation |
|---|---|
| Detection returns `detected=False` when below threshold | ✅ `MotionDecisionPolicy.decide()` — dual condition: both `motion_fraction >= threshold` AND `len(bboxes) > 0` must hold |
| Returns `bboxes=[]` when `detected=False` | ✅ `MotionOutputBuilder.build()` always sets `bboxes=[]` on no-motion path |
| Any exception → `detected=False` | ✅ Top-level `try/except Exception` in `detect()` — spec §11 |
| Input validation failure → `detected=False` | ✅ Inner `try/except ValueError, TypeError` in `_process_internal()` — returns early, no exception propagated |

**Verdict:** Contract fully implemented. Fully safe.

---

### 2.2 Object Detection (`ObjectDetectionModule.detect`)

| Behavior | Implementation |
|---|---|
| Returns `persons=[]` when no persons above threshold | ✅ `PersonFilteringLayer.filter_persons()` returns empty list |
| Returns `persons=[]` when inference has no results | ✅ `Postprocessor.decode_and_nms()` returns empty list |
| `person_detected` flag is consistent with `len(persons)` | ✅ `ResultBuilder.build()` sets `person_detected = len(persons) > 0` |
| Empty list returned (never None) | ✅ All internal layers return `list[...]`, never None |
| Top-level exception swallowed → empty output | ❌ **OD does NOT have a top-level `try/except`** — validation errors and inference errors propagate to caller |

**Note:** The orchestrator wraps each OD call in `try/except Exception: return`, so pipeline integrity is preserved. However, OD is the **only stage module** without a self-contained exception boundary — unlike Motion Detection, Face Detection, and Face Recognition which all catch exceptions internally.

---

### 2.3 Face Detection (`FaceDetectionModule.detect_faces`)

| Behavior | Implementation |
|---|---|
| Returns `detections=[]` when no faces above confidence threshold | ✅ `FaceDetectionPostprocessor.accept()` filters by confidence |
| Returns `detections=[]` on any exception | ✅ Top-level `try/except Exception` in `detect_faces()` — spec §10 |
| `_empty_output()` uses `.get()` with defaults on malformed input | ✅ Safe — returns `detections=[]` even if input dict is incomplete |
| Empty list returned (never None) | ✅ Guaranteed |
| Landmark padding: fewer than 5 landmarks padded with `(0, 0)` | ⚠️ See Gap §3.2 |

**Verdict:** Contract fully implemented. Edge case in landmark padding (see Gap §3.2).

---

### 2.4 Face Recognition (`FaceRecognitionModule.recognize`)

| Behavior | Implementation |
|---|---|
| Below threshold → `person_found=False`, `person_id="UNKNOWN"` | ✅ `FaceRecognitionDecisionPolicy.decide()` — explicit `similarity < threshold` path |
| Empty gallery → `person_found=False`, `person_id="UNKNOWN"` | ✅ `FaceMatcher.find_best_match()` returns `None` on empty gallery; decision policy maps `None → UNKNOWN` |
| Any exception → `person_found=False`, `person_id="UNKNOWN"` | ✅ Top-level `try/except Exception` in `recognize()` — spec §11 |
| `_no_match_output()` safe on malformed input | ✅ Uses `.get()` with defaults |
| No exception propagated to caller | ✅ Guaranteed |

**Verdict:** Contract fully implemented. Fully safe.

---

### 2.5 PersonDirectory Enrichment (`PersonDirectory.get_person`)

| Behavior | Implementation |
|---|---|
| Missing `person_id` → `found=False`, UNKNOWN record | ✅ `PersonDirectoryStore.get()` uses `dict.get()` with explicit `None` check |
| Empty or "UNKNOWN" `person_id` input → UNKNOWN record | ✅ Explicit early-exit: `if not person_id or person_id == "UNKNOWN": return _UNKNOWN_OUTPUT.copy()` |
| No `KeyError` possible | ✅ Never uses `dict[key]` directly on records |
| Returns copy of canonical UNKNOWN, not shared reference | ✅ `.copy()` called on `_UNKNOWN_OUTPUT` |
| `get_person()` never raises | ✅ Documented and implemented — no exception paths |
| `load()` failure with `is_optional=True` → silent empty store | ✅ Load error silently initializes empty store; all lookups return UNKNOWN |

**Note:** If `load()` is never called before `get_person()`, the store is empty and all lookups silently return UNKNOWN. There is no `_loaded` guard to catch this misuse at runtime. See Gap §3.3.

**Verdict:** Contract fully implemented for all expected lookup scenarios.

---

### 2.6 Pipeline Orchestration (`PipelineOrchestrator.execute`)

The orchestrator is the chain that connects all fallback behaviors.

#### Stage skip chain

| Stage | Early-exit condition | What is skipped |
|---|---|---|
| All stages | `except Exception` during `ingest_frame()` | Entire pipeline → `persons=[]` |
| All stages | `except Exception` during CURRENT/PREVIOUS frame retrieval | Entire pipeline → `persons=[]` |
| All stages | `except PreviousFrameNotAvailableError` (cold start) | Entire pipeline → `persons=[]` |
| OD + FD + FR | `if not motion_result["detected"]` | Entire pipeline → `persons=[]` |
| FD + FR (per region) | `except Exception` in OD ROI retrieval or OD detect | Region skipped, other regions continue |
| FR (per person) | `except Exception` in FD ROI retrieval or FD detect | Person skipped, other persons continue |
| PersonDirectory (per face) | `except Exception` in FR ROI retrieval or FR recognize | Face skipped, other faces continue |
| PersonDirectory (per face) | `if not fr_output["person_found"]` | Face omitted, PersonDirectory not queried |
| PersonDirectory enrichment (per face) | `except Exception` in `PersonDirectory.get_person()` | UNKNOWN fallback applied; face still appended to `recognized_faces` |

#### None propagation

| Value | Can be None? | Guard |
|---|---|---|
| `od_result["persons"]` | No | List always returned |
| `full_person_bbox` (projected) | Yes | `if full_person_bbox is None: continue` |
| `fd_output["detections"]` | No | List always returned |
| `full_face_bbox` (projected) | Yes | `if full_face_bbox is None: continue` |
| `pd_output` (PersonDirectory result) | No | Dict always returned; all keys present |
| `PipelineResult.persons` | No | Always a list |

---

## 3. Gaps and Inconsistencies

### 3.1 Object Detection missing top-level exception boundary (Inconsistent — pipeline-safe)

**Status:** Inconsistent between OD and all other stage modules.

Motion Detection, Face Detection, and Face Recognition each have a `try/except Exception` at their public API boundary, guaranteeing they never propagate exceptions to callers. Object Detection does not.

The orchestrator compensates by wrapping each OD call in `try/except Exception: return`, so the pipeline is safe. However:
- OD cannot be used safely as a standalone module in REAL runtime without the orchestrator's wrapping.
- An OD inference crash silently drops the entire motion region — no partial person output.

**Risk level:** Low (pipeline-safe), but creates inconsistency in module-level safety guarantees.

**Recommended fix:** Add a top-level `try/except Exception` to `ObjectDetectionModule.detect()` that returns `PersonDetectionResult(person_detected=False, persons=[])` on any unhandled exception — matching the pattern in FD and FR.

---

### 3.2 Landmark zero-padding silently passes Face Recognition validation (Silent degradation)

**Status:** No crash, but potential accuracy issue in REAL runtime.

`FaceDetectionOutputBuilder._build_landmarks()` pads detections with fewer than 5 landmarks using `Point(x=0, y=0)`. The `FaceRecognitionInputValidator` accepts coordinates of `(0, 0)` because they are finite and technically within bounds when the image is larger than 0×0. This means:
- A bad detection with 2 real landmarks and 3 synthetic `(0, 0)` points will pass validation.
- `FaceAligner` will use those landmarks for affine transform computation.
- The resulting alignment and embedding will silently be degraded or incorrect.
- No skip occurs — the face proceeds through FR as if it had valid landmarks.

**Risk level:** Medium — silent accuracy loss in REAL runtime with partial landmark detections.

**Recommended fix:** In the orchestrator's `_process_face_recognition()` path (or in the FR input validator), detect padded zero-landmarks and skip the face. Alternatively, `FaceDetectionOutputBuilder` should return `None` or omit the detection when it cannot produce 5 real landmarks, rather than padding.

---

### 3.3 PersonDirectory has no load-state guard (Silent misuse risk)

**Status:** Not a crash, but a silent contract violation if `load()` is omitted.

`PersonDirectory.get_person()` is documented as "never raises; always returns a PersonDirectoryOutput." However, if `load()` is never called, the internal store is empty and every lookup silently returns UNKNOWN (`found=False`). There is no `_loaded` flag, no `NotYetLoadedError`, and no warning. This makes misconfigured deployments (where `load()` is missed in startup code) indistinguishable at runtime from a correctly loaded but empty directory.

**Risk level:** Low for crash safety; Medium for operational correctness in REAL deployment.

**Recommended fix:** Add a `_loaded: bool` flag to `PersonDirectory`. Set it to `True` in `load()`. In `get_person()`, raise `RuntimeError("PersonDirectory.load() must be called before get_person()")` if `_loaded` is `False`. This surfaces misconfigured deployments immediately.

---

### 3.4 Unrecognized faces are silently omitted, not labeled UNKNOWN (Design gap in contract clarity)

**Status:** Implemented intentionally, but inconsistent with naive reading of scenario D.

The expected fallback contract in scenario D states: "valid output object returned." This is true at the `FaceRecognitionModule.recognize()` level — the module returns `{person_found: False, person_id: "UNKNOWN"}` cleanly. However, at the pipeline level, the orchestrator checks `if not fr_output["person_found"]: return` and **omits the face entirely** from `recognized_faces`. The face does not appear in the final output as UNKNOWN.

The `RecognizedFaceResult` type docstring explicitly states: "Unrecognized faces are never present — they are omitted entirely." This is by design, but:
- Callers cannot distinguish "face detected but not recognized" from "no face detected."
- `PersonResult.recognized_faces` being empty is ambiguous.

**Risk level:** Not a safety issue. Purely a contract clarity issue.

**Recommended fix (optional):** Document this behavior explicitly in `RecognitionPipelineOutput` or `PersonResult` docstrings. No code change needed unless callers require visibility into unrecognized faces.

---

### 3.5 Silent region drop when FTL `get_frame()` fails in motion region (Spec-compliant, undocumented)

**Status:** Intentional per spec §11, but not documented in output contracts.

If `ftl.get_frame()` raises during OD ROI retrieval for a motion bbox, the orchestrator silently returns from `_process_motion_region()`, discarding all persons in that region. No indication appears in the final output. This is compliant with spec §11 ("spec-defined graceful degradation"), but callers have no way to detect partial frame processing.

**Risk level:** Low — spec-compliant. Warrants operational awareness.

**Recommended fix (optional):** No code change required. Consider adding a diagnostic log or a `processing_notes` field in future output schema versions if partial-frame observability becomes important.

---

### 3.6 Dead code: pre-allocated `empty` in `process_frame()` (Non-functional)

**Status:** Code smell, no runtime risk.

In `RecognitionPipelineManager.process_frame()`, a variable `empty` is pre-allocated at the top of the method. The validation-failure path at the `try/except ValueError` block does not use this variable — it creates a new `RecognitionPipelineOutput` inline. The pre-allocated `empty` is never returned.

**Risk level:** None. Dead code only.

**Recommended fix:** Remove the pre-allocated `empty` variable and use the inline construction in the `except` block as-is, or centralize into a helper.

---

## 4. Summary Table

| Scenario | Status | Notes |
|---|---|---|
| **A. No motion** | ✅ Fully implemented | Orchestrator exits with `persons=[]`; all downstream stages skipped |
| **B. No person** | ✅ Fully implemented | `persons=[]` list → loop body never executes → FD/FR never invoked |
| **C. No face** | ✅ Fully implemented | `detections=[]` → FR loop never executes → `PersonResult(recognized_faces=[])` returned |
| **D. Below threshold** | ✅ Implemented (design note) | FR module returns clean UNKNOWN output; orchestrator omits face from pipeline output (see §3.4) |
| **E. Missing person_id** | ✅ Fully implemented | No `KeyError`; returns `{found: False, person_id: "UNKNOWN", person_name: "UNKNOWN"}`. Also: `get_person()` exception → UNKNOWN fallback applied; face still included in `recognized_faces` |
| **OD exception boundary** | ⚠️ Inconsistent | OD alone lacks module-level exception guard; pipeline-safe via orchestrator (see §3.1) |
| **Landmark zero-padding** | ⚠️ Silent degradation | Padded `(0,0)` landmarks pass FR validation; no skip occurs (see §3.2) |
| **PersonDirectory load guard** | ⚠️ Missing guard | No `_loaded` flag; unloaded directory silently behaves as empty (see §3.3) |
| **Unrecognized face labeling** | ℹ️ Design intent | Unrecognized faces omitted from output (not labeled UNKNOWN) — intentional per type docstring (see §3.4) |
| **None propagation** | ✅ Guarded | All possible `None` return values from projection have explicit `if ... is None: continue` guards |

---

## 5. Recommended Fixes (Priority Order)

| Priority | Gap | Fix |
|---|---|---|
| **P1** | §3.2 — Landmark zero-padding passes FR | Skip face in orchestrator if any landmark is `(0, 0)`, OR have FD omit the detection when fewer than 5 real landmarks are found |
| **P2** | §3.1 — OD missing top-level exception guard | Add `try/except Exception` to `ObjectDetectionModule.detect()` returning empty `PersonDetectionResult` |
| **P3** | §3.3 — PersonDirectory no load-state guard | Add `_loaded: bool` flag; raise `RuntimeError` in `get_person()` if `load()` was not called |
| **P4** | §3.4 — Unrecognized face visibility | Add docstring clarification to `PersonResult` and `RecognitionPipelineOutput` |
| **P5** | §3.6 — Dead code | Remove unused `empty` pre-allocation in `process_frame()` |

---

## 6. Pipeline Safety Verdict for REAL Runtime

**The pipeline is safe for REAL runtime fallback scenarios.**

All five expected fallback contracts (A–E) are implemented and will not crash or produce `None` output on the documented failure paths. The end-to-end exception coverage across the orchestrator and most module boundaries ensures that individual stage failures degrade gracefully rather than propagate.

Three issues require attention before a production REAL deployment:

1. **Landmark zero-padding (P1)** — a silent accuracy risk that could produce incorrect identity matches in REAL runtime without any error signal.
2. **OD exception boundary inconsistency (P2)** — does not affect pipeline safety today but creates a maintenance trap if OD is ever used outside the orchestrator.
3. **PersonDirectory load-state (P3)** — a deployment misconfiguration risk: a forgotten `load()` call will not fail fast and will silently treat all persons as UNKNOWN.

None of these will cause a crash or an unhandled exception in the current integration. They represent degraded-output risks and operational misuse risks in REAL environments.
