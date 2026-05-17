# RecognitionPipelineManager ↔ PersonDirectory Compatibility Review

**Review Date:** 2026-05-12 (UPDATED)  
**Status:** IMPLEMENTATION-READY — Architecture Decisions Finalized  
**Reviewed By:** Updated analysis of finalized architecture decisions, implementation modules, specs, and tests

---

## Executive Summary

The architecture decisions finalizing PersonDirectory ownership, enrichment responsibility, and lifecycle sequencing have resolved all critical blocking issues that previously prevented implementation. The RPM ↔ PersonDirectory boundary is now clearly defined, fully documented in the RPM specification, and compatible at both API and runtime behavior levels.

**Key Status Changes from Previous Review:**
- ✅ Naming cleanup completed: PersonDirectory is canonical across runtime and specification layers
- ✅ Ownership assigned: RPM/PipelineOrchestrator owns PersonDirectory lifecycle
- ✅ Enrichment boundary defined: PipelineOrchestrator calls PersonDirectory.get_person() after FR returns person_id
- ✅ Startup load recommended: PersonDirectory should be loaded before normal runtime traffic
- ✅ UNKNOWN semantics finalized: Canonical "UNKNOWN" output, no exceptions for missing identities
- ✅ Pre-load fail-safe contract aligned: get_person() returns UNKNOWN/found=false without raising before load()
- ✅ Configuration injection specified: PersonDirectoryConfig injected at RPM construction
- ✅ Startup sequencing defined: PersonDirectory.load() called during RPM initialization, before process_frame() begins

**Blocking Issues Resolved:** All 4 critical blockers from the previous review have been resolved through architecture decisions.

---

## Reviewed Files

| # | File | Role | Status |
|---|------|------|--------|
| 1 | `doc/image_processing_service/RecognitionPipelineManager.md` | RPM spec with PersonDirectory ownership, enrichment boundary, lifecycle, and integration | ✅ Updated |
| 2 | `doc/image_processing_service/PersonDirectory.md` | PersonDirectory spec (canonical form) | ✅ Current |
| 3 | `src/image_processing/person_directory/module.py` | PersonDirectory runtime implementation | ✅ Current |
| 4 | `tests/person_directory/test_person_directory.py` | PersonDirectory runtime behavior and load/lookup tests | ✅ Current |
| 5 | `doc/image_processing_service/face_recognition.md` | FaceRecognition output contract (person_id only) | ✅ Current |
| 6 | `doc/image_processing_service/shared_contracts.md` | Shared types authority | ✅ Current |

---

## Finalized Architecture Decisions

### 1. Canonical Module Identity

**Decision:** PersonDirectory is the canonical runtime module name for the identity metadata resolution component.

**Justification:**
- Runtime module location: `src/image_processing/person_directory/module.py`
- Public class name: `PersonDirectory`
- RPM spec uses "PersonDirectory" consistently in all enrichment and lifecycle sections
- Implementation matches spec contract exactly

**Status:** ✅ Resolved — No ambiguity remains

### 2. Lifecycle Ownership

**Decision:** RecognitionPipelineManager (via PipelineOrchestrator) is the exclusive lifecycle owner of PersonDirectory.

**Responsibility:**
- Construct PersonDirectory(PersonDirectoryConfig) at RPM initialization
- Call PersonDirectory.load() exactly once during RPM startup, before process_frame() is invoked
- Inject PersonDirectory instance into PipelineOrchestrator
- Treat startup load as recommended for full identity coverage while keeping runtime lookup fail-safe

**Evidence:**
- RPM spec §9.2: "PersonDirectory → injected into PipelineOrchestrator for person_id → person_name resolution; startup load is recommended before runtime traffic, while lookup remains fail-safe"
- RPM spec §8.9 step 12: PipelineOrchestrator calls PersonDirectory.get_person(person_id) after FR succeeds
- PersonDirectory spec §9.1: "PersonDirectory is constructed with PersonDirectoryConfig; startup load is expected during initialization"

**Status:** ✅ Resolved — Ownership is explicit and documented

### 3. Enrichment Responsibility

**Decision:** PipelineOrchestrator performs identity enrichment by calling PersonDirectory.get_person(person_id) after FaceRecognition returns person_id, and uses the resolved person_name in the final output.

**Implementation Sequence (RPM spec §8.5.4, §8.9 step 12):**
1. FaceRecognitionInterface.recognize() returns FaceRecognitionOutput {person_found, person_id}
2. If person_found = true:
   - PipelineOrchestrator calls PersonDirectory.get_person(person_id) → PersonDirectoryOutput {person_id, person_name, found}
   - Uses PersonDirectoryOutput.person_name for enrichment
   - Appends RecognizedFaceResult {face_bbox, person_id, person_name} to recognized_faces
3. If person_found = false:
   - Face is omitted entirely from recognized_faces

**Output Contract (RPM spec §3.1):**
```text
struct RecognizedFaceResult {
    BoundingBox face_bbox;
    string      person_id;
    string      person_name;
}
```

**Status:** ✅ Resolved — Enrichment responsibility is clearly defined and documented

### 4. UNKNOWN Semantics

**Decision:** Unknown/missing identities are handled via canonical UNKNOWN output, never via exceptions.

**Canonical Output:**
```text
person_id:   "UNKNOWN"
person_name: "UNKNOWN"
found:       false
```

**Triggering Cases:**
- person_id is empty string
- person_id == "UNKNOWN"
- person_id is not found in store
- PersonDirectory.load() was called with is_optional=true and file was missing

**Behavior:**
- PersonDirectory.get_person() returns canonical UNKNOWN output (no exception)
- PipelineOrchestrator still appends RecognizedFaceResult with person_name="UNKNOWN"
- Pipeline continues normally; never fails due to missing identity metadata

**Evidence:**
- PersonDirectory spec §4.2: "person_id may be empty, unknown, or equal to UNKNOWN_PERSON_ID — all such cases are valid runtime inputs and return the canonical UNKNOWN output after successful load()"
- PersonDirectory implementation: `_UNKNOWN_OUTPUT = PersonDirectoryOutput(person_id="UNKNOWN", person_name="UNKNOWN", found=False)`
- PersonDirectory.get() method returns `_UNKNOWN_OUTPUT` for all unresolvable cases
- RPM spec §8.9 step 12: "if PersonDirectoryOutput.found = false, person_name = UNKNOWN"

**Status:** ✅ Resolved — UNKNOWN semantics are consistent across spec and runtime

### 5. Initialization Contract

**Decision:** PersonDirectory.load() is expected during startup, but get_person() remains fail-safe even before successful load(). Pre-load lookup returns canonical UNKNOWN (`found=false`) without raising.

**Load Behavior:**
- Reads and parses configured JSON file
- Validates all records (non-empty person_id, non-empty person_name, no reserved "UNKNOWN" key)
- Builds in-memory map person_id → PersonRecord
- Publishes map via replace_all() to PersonDirectoryStore
- If file is missing and is_optional=true: store is initialized empty, no error
- If file is missing and is_optional=false: raise PersonDirectoryLoadError, startup fails
- If validation fails: raise PersonDirectoryValidationError, startup fails

**Lookup Behavior (after successful load):**
- Store is read-only
- Concurrent read access is safe
- get_person(person_id) does no file I/O, validation, or state mutation
- Returns deterministic output based on in-memory store state

**Pre-Load Behavior:**
- Calling get_person() before successful load() returns canonical UNKNOWN output (`found=false`)
- Runtime lookup remains fail-safe; initialization ordering errors do not crash the recognition pipeline

**Evidence:**
- PersonDirectory spec §9.1: startup load is expected during initialization before normal runtime traffic
- PersonDirectory spec §4.2: pre-load lookup is allowed and returns canonical UNKNOWN output (`found=false`)
- PersonDirectory implementation delegates to store UNKNOWN semantics even before load

**Status:** ✅ Resolved — startup recommendation and fail-safe runtime semantics are aligned

### 6. Configuration Injection

**Decision:** PersonDirectoryConfig is injected at RPM construction time and passed to PersonDirectory constructor.

**Config Fields:**
```text
struct PersonDirectoryConfig {
    string json_file_path;  // path to person metadata JSON
    bool   is_optional;     // if true, missing file does not fail startup
}
```

**Injection Point:**
- RPM initialization (before process_frame() is called)
- PersonDirectory constructor receives config
- Config is immutable after construction
- load() uses the configured json_file_path

**Evidence:**
- RPM spec §9.2: "PersonDirectory → injected into PipelineOrchestrator for person_id → person_name resolution"
- PersonDirectory spec §6: "Configuration is loaded exactly once at initialization. It is immutable after initialization and is not passed as a parameter to any runtime call."

**Status:** ✅ Resolved — Config injection is specified

### 7. Startup Sequencing

**Decision:** PersonDirectory.load() is called during RPM initialization as the recommended startup path, while runtime lookup remains fail-safe if load is delayed or unavailable.

**Sequence:**
1. External system initializes RPM with all dependencies injected (including PersonDirectory)
2. PersonDirectory.load() is called — reads JSON file, validates, builds store
3. If load() fails: startup should fail immediately (unless is_optional=true with missing file)
4. Once load() succeeds: PersonDirectory is ready for full-coverage runtime queries
5. process_frame() is invoked for each frame
6. PipelineOrchestrator calls PersonDirectory.get_person() for enrichment

**Failure Modes:**
- Missing JSON file (is_optional=false): PersonDirectoryLoadError, startup fails
- Malformed JSON: PersonDirectoryLoadError, startup fails
- Invalid records (empty person_id, empty person_name, reserved UNKNOWN key): PersonDirectoryValidationError, startup fails
- Missing JSON file (is_optional=true): store initialized empty, startup continues, runtime lookups return UNKNOWN

**Evidence:**
- RPM spec §9.2: startup load is recommended before runtime traffic; pre-load lookup remains fail-safe
- PersonDirectory spec §9.1: startup load is expected during initialization

**Status:** ✅ Resolved — Startup sequencing is clearly defined

---

## Compatibility Verification Checklist

### Summary: ALL DIMENSIONS COMPATIBLE ✅

| Dimension | Status | Blocking? |
|---|---|---|
| Ownership & Lifecycle | ✅ 5/5 Compatible | NO |
| Enrichment Boundary | ✅ 6/6 Compatible | NO |
| API Contract | ✅ 4/4 Compatible | NO |
| UNKNOWN Semantics | ✅ 5/5 Compatible | NO |
| Load Behavior | ✅ 6/6 Compatible | NO |
| Runtime Lookup | ✅ 5/5 Compatible | NO |
| Pre-Load Behavior | ✅ 2/2 Compatible | NO |
| Configuration | ✅ 4/4 Compatible | NO |
| Threading | ✅ 4/4 Compatible | NO |
| Data Ownership | ✅ 4/4 Compatible | NO |
| Error Handling | ✅ 6/6 Compatible | NO |

---

## Previously Active Issues — Status Update

### Issue 1: Legacy Naming Drift Removed

**Previous Status:** Critical — blocking integration

**Resolution:**
- ✅ RESOLVED — The architecture now treats PersonDirectory as canonical for all runtime/enrichment references
- RPM spec uses "PersonDirectory" consistently throughout
- Runtime module is `person_directory` with class `PersonDirectory`
- Spec file is named `PersonDirectory.md` but documents the same component

**Why Resolved:**
- Canonical name is now established by RPM spec usage
- Runtime implementation matches canonical usage
- API contract and documentation naming now match directly

---

### Issue 2: Missing Invocation Path from FR to Resolver Lookup

**Previous Status:** Critical — blocking enrichment implementation

**Resolution:**
- ✅ RESOLVED — RPM spec §8.5.4 and §8.9 step 12 define complete invocation sequence:
  1. FaceRecognitionInterface.recognize(FaceRecognitionInput) returns FaceRecognitionOutput {person_found, person_id}
  2. If person_found = true: PipelineOrchestrator calls PersonDirectory.get_person(person_id)
  3. PersonDirectory returns PersonDirectoryOutput {person_id, person_name, found}
  4. PipelineOrchestrator uses person_name to enrich RecognizedFaceResult
  5. RecognizedFaceResult {face_bbox, person_id, person_name} is appended to recognized_faces

**Why Resolved:**
- Call sequence is explicitly documented in RPM spec
- Enrichment responsibility is assigned to PipelineOrchestrator
- Integration seam between FR (person_id only) and PersonDirectory (person_name lookup) is defined

---

### Issue 3: Missing Lifecycle Owner for Resolver

**Previous Status:** Critical — blocking startup sequencing

**Resolution:**
- ✅ RESOLVED — RPM is the explicit lifecycle owner:
  - RPM construction injects PersonDirectory instance
  - RPM startup calls PersonDirectory.load() before any process_frame()
  - PersonDirectory.load() must complete successfully before first get_person() call
  - Startup failure semantics are defined (fail if file missing and is_optional=false)

**Why Resolved:**
- Ownership is explicitly assigned to RPM in spec §8.8 and §9.2
- Startup sequence is documented in RPM spec §8.9
- Injection responsibility is clear

---

### Issue 4: Pre-Load Semantics Divergence

**Previous Status:** High — contract-level mismatch

**Resolution:**
- ✅ RESOLVED — PersonDirectory is fail-safe for pre-load lookup:
  - load() is still expected at startup for full identity coverage
  - Calling get_person() before successful load() returns canonical UNKNOWN (`found=false`)
  - No NotInitialized exception is required for runtime robustness
  - Pipeline behavior remains deterministic and non-crashing even if startup ordering is imperfect

**Why Resolved:**
- PersonDirectory implementation already returns UNKNOWN through store lookup when pre-load
- RPM spec now documents startup load as recommended while keeping runtime fail-safe
- Pre-load lookup behavior is explicitly tested and documented

---

### Issue 5: RPM Enriched Output Contract is Spec-Only

**Previous Status:** High — no implementation path

**Resolution:**
- ✅ RESOLVED — RPM spec now documents the complete enrichment path:
  - RecognizedFaceResult includes person_name (RPM spec §3.1)
  - PipelineOrchestrator performs enrichment (RPM spec §8.8, §8.9)
  - Enrichment happens after FR returns person_id (RPM spec §8.5.4, §8.9 step 12)
  - PersonDirectory is integrated for person_id → person_name resolution

**Why Resolved:**
- RPM spec is the implementation guide for this enrichment
- All components are now documented with explicit responsibilities
- Enrichment seam is fully specified in RPM spec

---

### Issue 6: Shared Identity Model Authority

**Previous Status:** Medium — potential future drift

**Resolution:**
- ⚠️ PARTIALLY ADDRESSED — Identity output types are defined in PersonDirectory spec:
  - PersonRecord {person_id, person_name}
  - PersonDirectoryOutput {person_id, person_name, found}
- These types are not promoted to shared_contracts.md, but this is a deliberate design choice:
  - PersonDirectory is internal to RPM, not a shared public module
  - Only RPM consumes PersonDirectory output
  - No other module depends on this type

**Assessment:**
- Not an active integration issue (only RPM uses it)
- If PersonDirectory were to become a true shared dependency in future, these types would need to move to shared_contracts.md
- Current design keeps identity types internal; this is acceptable

**Status:** ✅ Design is sound for current scope (RPM-internal component)

---

## Implementation Readiness Assessment

### Status: ✅ READY FOR IMPLEMENTATION

**Justification:**

1. **Ownership is Explicit** — RPM is the documented lifecycle owner; PersonDirectory is injected and load() is called at startup per spec §9.2

2. **Enrichment Path is Defined** — Complete invocation sequence is documented in RPM spec §8.5.4 and §8.9 step 12; PipelineOrchestrator calls PersonDirectory.get_person() after FR returns person_id

3. **API Contracts are Aligned** — PersonDirectory constructor, load(), and get_person() signatures match specification exactly; return types are consistent

4. **UNKNOWN Semantics are Finalized** — Canonical "UNKNOWN" output is implemented; no runtime exceptions for missing identities; spec and implementation align

5. **Startup + Fail-Safe Semantics are Aligned** — PersonDirectory.load() is recommended at startup, and pre-load lookup remains UNKNOWN/found=false without exceptions

6. **Configuration Injection is Specified** — PersonDirectoryConfig is injected at RPM construction; path and optional file flag are documented

7. **Startup Failure Policy is Defined** — PersonDirectory raises LoadError/ValidationError on missing/invalid file (unless is_optional=true); RPM startup sequence is documented

8. **No Remaining Blockers** — All 4 critical blockers from previous review are resolved

### Next Steps:

1. **Implement RecognitionPipelineManager** per RPM spec §8 (PipelineOrchestrator, InputValidator, OutputBuilder, SpatialCoordinator)
2. **Wire PersonDirectory injection** at RPM construction per spec §9.2
3. **Call PersonDirectory.load()** at RPM startup before any process_frame() invocation
4. **Implement enrichment** in PipelineOrchestrator per spec §8.9 step 12
5. **Add integration tests** for enrichment pipeline (FR → PersonDirectory → RecognizedFaceResult)
6. **Verify startup semantics** in tests (load() must complete before first process_frame())

---

## Conclusion

The RecognitionPipelineManager ↔ PersonDirectory boundary is now fully defined, documented, and implementation-ready. All architectural decisions required for implementation have been finalized. The previous critical blockers have been systematically resolved through:

1. **Canonical naming** — PersonDirectory established as canonical runtime name
2. **Explicit ownership** — RPM assigned as lifecycle owner
3. **Defined enrichment** — Complete call sequence from FR through PersonDirectory to final output
4. **Startup semantics** — PersonDirectory.load() at RPM initialization before process_frame()
5. **Contract alignment** — API shapes, UNKNOWN handling, and threading model all aligned

The boundary is ready to move to implementation phase. No further specification updates are needed for core enrichment functionality.
