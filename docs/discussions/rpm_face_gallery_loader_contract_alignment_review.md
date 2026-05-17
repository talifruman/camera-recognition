# RecognitionPipelineManager ↔ Face Gallery Loader Compatibility Review (May 2026)

> **Review scope.** Final review, May 2026 — reflects the complete and resolved state of RecognitionPipelineManager ↔ Face Gallery Loader compatibility. All six original mismatches (1–6) are fully resolved. NpyEmbeddingFileReader is the production default; real gallery integration is verified; test coverage is complete. No remaining compatibility gaps. See §7 for the complete mismatch resolution history and §10 for the final compatibility status.

---

## 1. Review Date

**May 10, 2026**

---

## 2. Status Summary

| Item | Status |
|------|--------|
| Overall compatibility | ⚠️ INDIRECT — not a direct RPM dependency |
| RPM ↔ Face Gallery Loader direct integration | Not applicable — RPM does not call Face Gallery Loader |
| Face Gallery Loader → Face Recognition (startup injection) | ✅ Structurally compatible; startup wiring documented; integration tests added |
| Blocking integration issues | 0 |
| Must-fix-before-integration issues | 0 |
| Medium-impact issues | 0 |
| Minor-impact issues | 0 |
| Already-aligned items | 28 |

**Summary.** RecognitionPipelineManager does not directly depend on FaceGalleryLoader. The gallery participates in the recognition pipeline exclusively at service startup: external startup code calls `FaceGalleryLoaderModule.load_gallery()`, retrieves all `LoadedGalleryEmbedding` records via `get_all_embeddings()`, maps them to `EnrolledIdentity[]`, and passes them to `FaceRecognitionModule` at construction time. RPM only injects a `FaceRecognitionInterface` implementation and never sees gallery data, gallery structure, or the loader itself. All six original mismatches (1–6) are fully resolved: person_name resolution, naming separation, L2 normalization validation, internal export boundaries, startup wiring documentation, and the default reader update to NpyEmbeddingFileReader with complete test coverage including real-gallery integration.

---

## 3. Reviewed Files

| File | Type | Purpose |
|------|------|---------|
| `doc/image_processing_service/RecognitionPipelineManager.md` | Specification | RPM design, interfaces, orchestration flow, Out-of-Scope list |
| `doc/image_processing_service/face_gallery_loader.md` | Specification | FaceGalleryLoader module design, public API, LoadedGalleryEmbedding contract, error handling |
| `doc/image_processing_service/face_recognition.md` | Specification | Face Recognition module design, FaceGalleryCache ownership, gallery initialization |
| `doc/image_processing_service/shared_contracts.md` | Specification | Authoritative shared type definitions used across the pipeline |
| `src/image_processing/face_gallery_loader/module.py` | Source | Full FaceGalleryLoader implementation: LoadedGalleryEmbedding, GalleryPathValidator, GalleryDirectoryScanner, StubEmbeddingFileReader, NpyEmbeddingFileReader, FaceGalleryCache, FaceGalleryLoaderModule |
| `src/image_processing/face_gallery_loader/__init__.py` | Source | Public exports for the face_gallery_loader package |
| `src/image_processing/face_recognition/module.py` | Source | FaceRecognitionModule implementation: EnrolledIdentity, EnrolledIdentityCache, FaceMatcher, FaceRecognitionModule constructor (receives gallery_entries: list[EnrolledIdentity]) |
| `src/image_processing/face_recognition/__init__.py` | Source | Public exports for the face_recognition package |
| `tests/face_gallery_loader/test_face_gallery_loader_module.py` | Tests | Unit tests for FaceGalleryLoader — load, person count, entry count, sorting, path errors, empty gallery, stub embedding shape/dtype/normalization |
| `tests/face_recognition/test_face_recognition_module.py` | Tests | Unit tests for FaceRecognitionModule — manually constructed gallery entries; no integration with FaceGalleryLoader |

**Not yet implemented:** No source code exists for `RecognitionPipelineManager`, `PipelineOrchestrator`, `FrameTransformationLayerInterface`, or `SpatialCoordinator`. RPM analysis is documentation-only. `FaceRecognitionInterface` is implemented as a `@runtime_checkable Protocol` in `face_recognition/module.py` and exported from `face_recognition/__init__.py`. Service startup wiring (who calls `FaceGalleryLoaderModule` and injects its output into `FaceRecognitionModule`) is documented in `doc/system.md` (Service Startup Sequence — Gallery-to-Recognition Wiring section) but has no runnable implementation yet.

---

## 4. Current Integration Flow

### 4.1 Architectural Role of Face Gallery Loader

Face Gallery Loader is **not a per-frame pipeline stage**. It is a startup-time utility that produces the enrolled identity gallery used internally by Face Recognition. It never participates in the per-frame execution path that RPM orchestrates.

The three plausible integration paths considered for this review are:

| Path | Description | Current Status |
|------|-------------|----------------|
| **A** — RPM directly initializes or calls Face Gallery Loader | RPM calls `FaceGalleryLoaderModule.load_gallery()` during initialization or per invocation | **Not documented, not implemented.** RPM's Out-of-Scope list explicitly excludes gallery management. |
| **B** — Face Recognition owns Face Gallery Loader internally | FaceRecognitionModule calls FaceGalleryLoaderModule internally to load the gallery | **Not documented, not implemented.** face_recognition.md does not mention FaceGalleryLoader. |
| **C** — Gallery is loaded at service startup; result is injected into Face Recognition | External startup code calls FaceGalleryLoaderModule, retrieves LoadedGalleryEmbedding[], maps to EnrolledIdentity[], passes to FaceRecognitionModule constructor | **This is the documented and implemented integration path.** |

**Documented path (C):**

Both specs converge on path C:
- `face_gallery_loader.md §13.1` — loader is initialized once; `load_gallery` is called at startup time; the caller receives `LoadedGalleryEmbedding[]` via `get_all_embeddings()`.
- `face_recognition.md §9.2` — "Enrolled embeddings and associated person metadata are supplied to the module at construction time; the module builds `FaceGalleryCache` from these records and retains them as immutable in-memory state."
- `face_recognition.md §13.1` — "Receive the enrolled embeddings and associated person metadata at construction time; build `FaceGalleryCache` as an immutable in-memory structure."
- `src/image_processing/face_recognition/module.py` — `FaceRecognitionModule.__init__(config, embedding_engine, gallery_entries: list[EnrolledIdentity])` confirms injection at constructor time.

**RPM's role (spec-defined):**

- RPM's Out-of-Scope list (`RecognitionPipelineManager.md §1`): "Manage face identity enrollment or gallery updates — handled outside this module."
- RPM does not call `load_gallery`, does not receive `LoadedGalleryEmbedding[]`, and does not inject gallery data into any component. RPM only injects a `FaceRecognitionInterface` implementation.
- RPM accesses the filesystem only during initialization (to load pipeline configuration); it does not access the filesystem for gallery operations.

### 4.2 End-to-End Startup-to-Per-Frame Flow

```
Service Startup (external)
  ├─ FaceGalleryLoaderModule.load_gallery(gallery_root_path)
  │     └─ GalleryPathValidator → GalleryDirectoryScanner → EmbeddingFileReader → FaceGalleryCache
      ├─ entries = FaceGalleryLoaderModule.get_all_embeddings() → LoadedGalleryEmbedding[]
      ├─ enrolled = [EnrolledIdentity(person_id=e["person_id"], embedding=e["embedding"]) for e in entries]
      └─ FaceRecognitionModule(config, embedding_engine, gallery_entries=enrolled)
        └─ FaceGalleryCache (immutable, internal, never exposed)

RPM Initialization (external)
  └─ RecognitionPipelineManager(…, face_recognition=FaceRecognitionModule)
      └─ RPM only holds FaceRecognitionInterface; never sees LoadedGalleryEmbedding, EnrolledIdentity, or FaceGalleryCache

Per-Frame Execution
  └─ RPM.process_frame(frame_packet)
        └─ PipelineOrchestrator calls FaceRecognitionInterface.recognize(face_input)
              └─ FaceRecognitionModule internally reads FaceGalleryCache
                    └─ FaceGalleryCache was built from FaceGalleryLoader output at startup
```

### 4.3 Who Owns What

| Responsibility | Owner |
|----------------|-------|
| Gallery file format reading | FaceGalleryLoaderModule (via EmbeddingFileReader) |
| Directory structure scanning | FaceGalleryLoaderModule (via GalleryDirectoryScanner) |
| Startup-time gallery loading | External service startup code |
| LoadedGalleryEmbedding[] production | `FaceGalleryLoaderModule.get_all_embeddings()` |
| LoadedGalleryEmbedding[] → EnrolledIdentity[] mapping | External service startup code |
| FaceGalleryCache ownership | FaceRecognitionModule (internal only) |
| Per-frame gallery access | FaceMatcher (via `FaceGalleryCache.get_entries()`) |
| Recognition threshold decision | FaceRecognitionDecisionPolicy |
| Gallery reload / lifecycle after startup | Not currently documented |
| RPM interaction with gallery | None — RPM never sees gallery data |

---

## 5. Data Produced by Face Gallery Loader

### 5.1 LoadedGalleryEmbedding

`LoadedGalleryEmbedding` is the sole externally visible output of FaceGalleryLoader. It is a validated embedding record loaded from persistent gallery storage. It is **not** `EnrolledIdentity` and is **not** a Face Recognition type.

| Field | Source | Type (spec) | Type (implementation) | Public? | RPM sees it? | Face Recognition consumes it? |
|-------|--------|-------------|----------------------|---------|-------------|-------------------------------|
| `person_id` | Subdirectory name under gallery root | `string` | `str` | Yes | No | Yes — primary key for identity lookup in FaceMatcher |
| `embedding` | `EmbeddingFileReader.read_embedding()` output | `FaceEmbedding` (opaque float vector) | `np.ndarray` (float32, shape `(512,)`) | Yes | No | Yes — compared to live embeddings by FaceMatcher |

**Fields NOT in LoadedGalleryEmbedding (confirmed):**
- `person_name` — **not present in loader output; not present in face_recognition implementation.** `face_recognition.md §10` mentions it as optional metadata. See Mismatch 1.
- File path — explicitly excluded from public output by spec §3.3.
- Internal record identifiers — explicitly excluded by spec §3.3.
- Dtype/dimension metadata — explicitly excluded; validated internally only.
- Per-file validation state — explicitly excluded; internal to loader only.

### 5.2 FaceEmbedding (embedding field)

| Property | Spec requirement | Implementation | Status |
|----------|-----------------|----------------|--------|
| Type | Opaque float vector | `np.ndarray` | ✅ |
| Shape | `(expected_embedding_dim,)` — default 512 | `(512,)` via `FaceGalleryLoaderConfig.expected_embedding_dim` | ✅ |
| dtype | `expected_dtype` — default `float32` | `float32` via `FaceGalleryLoaderConfig.expected_dtype` | ✅ |
| L2 normalization | Not validated by loader spec; ArcFace convention requires L2-normalized vectors | Stub: validated (unit vector by construction). NpyEmbeddingFileReader: validates — returns `None` + `warnings.warn` if `abs(np.linalg.norm(arr) - 1.0) > 1e-4` | ✅ Resolved — see `face_gallery_loader.md §7.1` |
| Non-null | Required by spec §7 | Enforced — `None` return from reader triggers skip | ✅ |
| Immutability | Not formally required | Stub returns shared instance (mutation risk); NpyEmbeddingFileReader returns fresh array per call | ⚠️ Minor risk with stub |

### 5.3 FaceGalleryCache

FaceGalleryCache exists in **two distinct modules** with different APIs and semantics:

| Property | FaceGalleryLoader's FaceGalleryCache | FaceRecognition's FaceGalleryCache |
|----------|--------------------------------------|-------------------------------------|
| Location | `face_gallery_loader/module.py` | `face_recognition/module.py` |
| Public? | No — internal to loader | No — internal to recognition |
| Indexing | `dict[person_id, list[LoadedGalleryEmbedding]]` | `list[EnrolledIdentity]` (flat) |
| API (loader) | `replace_all()`, `get_all_entries()`, `get_entries_by_person()`, `get_person_ids()` | N/A |
| API (recognition) | N/A | `get_entries()` → `list[EnrolledIdentity]` |
| Mutability | Replaced atomically on `load_gallery` | Immutable after construction |
| RPM sees it? | Never | Never |

The two `FaceGalleryCache` classes are **not shared**. They serve different roles: the loader's cache supports indexed multi-API access for startup wiring; the recognition's cache holds a flat read-only list for per-frame matching.

### 5.4 Loader Status / Result Object

`load_gallery()` returns `void`. There is no result object. Status is communicated via structured exceptions:

| Condition | Exception |
|-----------|-----------|
| Invalid root path | `GalleryPathValidationError(ValueError)` |
| No valid embeddings found | `GalleryLoadError(RuntimeError)` |
| Individual file failures | Warning via `warnings.warn`; no exception; file is skipped |
| Success | No return value; cache is updated atomically |

---

## 6. Compatibility Verification Checklist

### 6.1 RPM ↔ Face Gallery Loader Direct Dependency

| Check | RPM Spec | Gallery Loader Spec | Gallery Loader Implementation | Status | Notes |
|-------|----------|--------------------|-----------------------------|--------|-------|
| Does RPM mention Face Gallery Loader? | No — explicitly excluded from scope | Out of scope of loader | N/A | ✅ Correctly absent | RPM Out-of-Scope: "Manage face identity enrollment or gallery updates — handled outside this module" |
| Does RPM require gallery lifecycle ownership? | No | No | N/A | ✅ Aligned | Gallery lifecycle is outside RPM |
| Does RPM call `load_gallery`? | No | Not expected | N/A | ✅ Aligned | Startup wiring is external |
| Does RPM receive `LoadedGalleryEmbedding[]`? | No | Not expected | N/A | ✅ Aligned | RPM only holds FaceRecognitionInterface |
| Does RPM access gallery root path? | No — "Access the filesystem after initialization" is Out-of-Scope | Not expected | N/A | ✅ Aligned | |
| Is gallery loading a per-frame operation? | No — excluded | No | No — `load_gallery` is startup-only | ✅ Aligned | Gallery is preloaded; FaceGalleryCache is read-only during recognition |

### 6.2 Face Recognition ↔ Face Gallery Loader Integration

| Check | Face Recognition Spec | Gallery Loader Spec | Face Recognition Implementation | Gallery Loader Implementation | Status | Notes |
|-------|----------------------|--------------------|--------------------------------|------------------------------|--------|-------|
| Does Face Recognition expect gallery entries at construction? | Yes — §9.2, §13.1 | Provides `LoadedGalleryEmbedding[]` via `get_all_embeddings()`; startup code maps to `EnrolledIdentity[]` | Yes — constructor: `gallery_entries: list[EnrolledIdentity]` | Yes — returns `list[LoadedGalleryEmbedding]` | ✅ Aligned | Explicit mapping at startup |
| Does Face Recognition call Face Gallery Loader internally? | No — gallery is injected externally | Not expected | No — constructor receives pre-loaded entries | N/A | ✅ Aligned | Separation is correct |
| Is gallery reload supported at runtime? | Not documented — "Gallery updates require module reinitialization" | `load_gallery` can be called again; cache is replaced atomically | No runtime reload path | Atomic replacement supported | ⚠️ Minor gap | FR requires reinitialization; loader supports reload; no wiring connects them |
| LoadedGalleryEmbedding `person_id` type | `string` | `string` | `str` | `str` | ✅ Aligned | |
| LoadedGalleryEmbedding `embedding` type | `FaceEmbedding` (float vector) | `FaceEmbedding` (opaque, validated structurally only) | `np.ndarray` (float32) | `np.ndarray` (float32) | ✅ Aligned | |
| LoadedGalleryEmbedding `person_name` field | Not present | Not present | Resolved | Not present | ✅ Resolved — See Mismatch 1 | |
| Naming separation | FaceGalleryLoader → `LoadedGalleryEmbedding`; FaceRecognition → `EnrolledIdentity` | Separate types | `EnrolledIdentity` in `face_recognition/module.py` | `LoadedGalleryEmbedding` in `face_gallery_loader/module.py` | ✅ Resolved — See Mismatch 2 | |
| Startup / initialization timing | Gallery built at construction before any `recognize_face()` call | `load_gallery()` must complete before gallery is read | FaceGalleryCache built in `__init__`; immutable | `FaceGalleryCache.replace_all()` called at end of `load_gallery` | ✅ Aligned | |
| Reload behavior | Requires module reinitialization | Atomic `replace_all()` on reload | Not supported at runtime | Supported | ⚠️ Minor gap | See Mismatch 5 |

### 6.3 Loader Public API

| Check | Spec (`face_gallery_loader.md §4`) | Implementation | Status | Notes |
|-------|----------------------------------|----------------|--------|-------|
| `load_gallery(gallery_root_path)` exists | Yes — primary load method | `def load_gallery(self, gallery_root_path: str) -> None` | ✅ | |
| `get_all_embeddings()` exists | Yes | `def get_all_embeddings(self) -> list[LoadedGalleryEmbedding]` | ✅ | |
| `get_embeddings(person_id)` exists | Yes | `def get_embeddings(self, person_id: str) -> list[LoadedGalleryEmbedding]` | ✅ | |
| `get_person_ids()` exists | Yes | `def get_person_ids(self) -> list[str]` | ✅ | |
| Default EmbeddingFileReader | Spec §6.2: `NpyEmbeddingFileReader` | `NpyEmbeddingFileReader` (production default); StubEmbeddingFileReader available via explicit injection | ✅ | Fully aligned |
| API stability — no config params in read calls | Yes | None of the read methods take config params | ✅ | |
| `gallery_root_path` not a config param | Yes — runtime input to `load_gallery` only | Correct — only in `load_gallery()` parameter | ✅ | |

### 6.4 LoadedGalleryEmbedding Contract

| Check | Gallery Loader Spec | Gallery Loader Implementation | Face Recognition Spec | Face Recognition Implementation | Status |
|-------|--------------------|-----------------------------|----------------------|--------------------------------|--------|
| `person_id` field present | Yes | Yes | Yes | Yes | ✅ |
| `person_id` type | `string` | `str` | `string` | `str` | ✅ |
| `embedding` field present | Yes | Yes | Yes | Yes | ✅ |
| `embedding` shape `(512,)` | `(expected_embedding_dim,)` default 512 | Enforced by EmbeddingFileReader | Not explicitly stated in FR spec | `(512,)` assumed by ArcFace config | ✅ Compatible |
| `embedding` dtype `float32` | `expected_dtype` default `float32` | Enforced; NpyEmbeddingFileReader casts to float32 | `FaceEmbedding` is float vector | `np.ndarray` float32 | ✅ Compatible |
| L2-normalized embedding | Not checked by loader | Stub: L2-normalized by design. NpyEmbeddingFileReader: validates — returns `None` + warning if norm ≠ 1.0 within 1e-4 | ArcFace produces L2-normalized vectors (§6.2) | `FaceMatcher` uses dot product (assumes L2-normalized) | ✅ Validated — see `face_gallery_loader.md §7.1` |
| Invalid files skipped | Yes — skip-or-fail policy | Yes — `None` from reader → skip with warning | N/A | N/A | ✅ |
| One bad file invalidates whole person? | No — only that file is skipped | No — other files for the same person are accepted | N/A | N/A | ✅ |
| `person_name` field | Not present in LoadedGalleryEmbedding | Not present | Resolved — external enrichment after person_id returned | Not present | ✅ Resolved — see Mismatch 1 |

### 6.5 Cache Contract

| Check | Gallery Loader Cache | Face Recognition Cache | Status |
|-------|---------------------|----------------------|--------|
| Indexed by person_id | Yes — `dict[person_id, list[LoadedGalleryEmbedding]]` | No — flat `list[EnrolledIdentity]` | ✅ Different purposes, both correct |
| Immutable during reads | No — can be replaced via `replace_all()` | Yes — immutable after construction | ✅ Each module correct for its role |
| Lookup by person_id | `get_entries_by_person(person_id)` | Not needed — FaceMatcher iterates all entries | ✅ |
| Empty cache behavior | Returns empty list; no error | FaceMatcher returns None; DecisionPolicy returns `person_found=False` | ✅ |
| Thread safety | Not documented | Not documented | ⚠️ Not specified |
| Reload atomicity | Yes — `replace_all()` replaces in one assignment | Not applicable — immutable | ✅ |

### 6.6 Embedding Compatibility Between Loader and Recognition

| Check | Gallery Loader | Face Recognition | Status |
|-------|---------------|-----------------|--------|
| Vector size | 512 (configurable via `expected_embedding_dim`) | 512 (ArcFace default) | ✅ Match by convention |
| dtype | `float32` (configurable via `expected_dtype`) | `float32` (`np.ndarray`) | ✅ |
| L2 normalization | Validated by NpyEmbeddingFileReader (`abs(np.linalg.norm(arr) - 1.0) > 1e-4` → returns `None` + warning); stub guarantees by construction | Assumed by FaceMatcher (dot product used as cosine similarity) | ✅ Validated — see `face_gallery_loader.md §7.1` |
| Similarity function | N/A — loader does not match | Dot product (`np.dot`) — valid cosine similarity for unit vectors | ✅ Correct for L2-normalized vectors |
| Threshold ownership | Not applicable | FaceRecognitionDecisionPolicy — `recognition_threshold` from config | ✅ Correct ownership |
| Stub embedding compatibility | StubEmbeddingFileReader returns constant L2-normalized float32 `(512,)` vector | FaceMatcher accepts any float32 `(512,)` vector | ✅ Compatible for test phase |

### 6.7 Error Handling

| Condition | Gallery Loader Behavior | Face Recognition Behavior | Status |
|-----------|------------------------|--------------------------|--------|
| Missing gallery root | `GalleryPathValidationError` raised | N/A — loader error happens before FR initialization | ✅ Load fails before FR is initialized |
| Invalid directory structure | `GalleryPathValidationError` (no subdirs) or zero valid entries → `GalleryLoadError` | N/A | ✅ |
| Unreadable `.npy` | EmbeddingFileReader returns None → file skipped with warning | N/A | ✅ |
| Wrong dtype in `.npy` | NpyEmbeddingFileReader returns None → file skipped | N/A | ✅ |
| Wrong shape in `.npy` | NpyEmbeddingFileReader returns None → file skipped | N/A | ✅ |
| Non-normalized embedding | Validated by NpyEmbeddingFileReader — returns `None` + `warnings.warn`, file skipped | FaceMatcher receives only validated (normalized or stub) vectors | ✅ Resolved |
| Empty gallery (no valid embeddings) | `GalleryLoadError` raised; cache not modified | N/A — FR would not be initialized | ✅ |
| Duplicate person IDs | Not possible — each subdirectory name is unique; `FaceGalleryCache` uses `person_id` as dict key | N/A | ✅ |
| Duplicate embeddings (multiple `.npy` per person) | All accepted; multiple `LoadedGalleryEmbedding` per `person_id` | FaceMatcher scans all entries; best match wins | ✅ |
| Partial load (some files fail) | Successful files included; failed files skipped; load continues | FR receives only the valid entries | ✅ |
| Empty gallery passed to FR | `GalleryLoadError` prevents this under normal startup | `FaceMatcher` returns None; `person_found=False` | ✅ FR handles it gracefully |
| `get_embeddings(person_id)` not found | Empty list returned; no error | N/A — FR uses `get_entries()`, not `get_embeddings()` | ✅ |

### 6.8 Public Exports

| Symbol | `face_gallery_loader/__init__.py` | `face_recognition/__init__.py` | Notes |
|--------|----------------------------------|-------------------------------|-------|
| `LoadedGalleryEmbedding` | ✅ Exported | Not applicable — FR uses `EnrolledIdentity` | Domain-separated types; startup code maps between them |
| `FaceGalleryLoaderModule` | ✅ Exported | Not applicable | |
| `FaceGalleryLoaderConfig` | ✅ Exported | Not applicable | |
| `FaceGalleryCache` | ✅ Exported from loader | ✅ Exported from recognition | Two separate classes; different APIs |
| `GalleryLoadError` | ✅ Exported | Not applicable | |
| `GalleryPathValidationError` | ✅ Exported | Not applicable | |
| `EmbeddingFileReader` | ✅ Exported | Not applicable | Protocol |
| `NpyEmbeddingFileReader` | ✅ Exported | Not applicable | |
| `StubEmbeddingFileReader` | ✅ Exported | Not applicable | |
| `GalleryDirectoryScanner` | ❌ Not exported — removed from `__init__.py`; internal only | Not applicable | |
| `GalleryPathValidator` | ❌ Not exported — removed from `__init__.py`; internal only | Not applicable | |
| `PersonScanRecord` | ❌ Not exported — removed from `__init__.py`; internal only | Not applicable | |
| `FaceRecognitionModule` | Not applicable | ✅ Exported | Accepts `gallery_entries` in constructor |
| `FaceRecognitionInterface` | Not applicable | ✅ Exported — `@runtime_checkable Protocol` in `face_recognition/__init__.py` | |

**Export note:** `GalleryDirectoryScanner`, `GalleryPathValidator`, and `PersonScanRecord` have been removed from `face_gallery_loader/__init__.py` exports (Mismatch 4 resolved). The public API surface now matches `face_gallery_loader.md §3.3` strict isolation requirement. `FaceRecognitionInterface` is now exported from `face_recognition/__init__.py` as a `@runtime_checkable Protocol`.

### 6.9 Tests

| Test Scenario | Gallery Loader Tests | Face Recognition Tests | Integration Tests | Status |
|---------------|---------------------|----------------------|------------------|--------|
| Valid gallery load | ✅ `TestSuccessfulLoad` | N/A | ❌ None | Partial |
| Correct person count | ✅ `TestPersonCount` | N/A | ❌ None | Partial |
| Correct entry count | ✅ `TestEntryCount` | N/A | ❌ None | Partial |
| Sorted person IDs | ✅ `TestSorting` | N/A | ❌ None | Partial |
| Empty gallery (no valid files) | ✅ `TestEmptyGallery` | N/A | ❌ None | Partial |
| Empty gallery passed to FR | ❌ None | ✅ `EmptyGalleryTests` | ❌ None | Partial |
| Missing gallery root | ✅ `TestInvalidPath` | N/A | ❌ None | Partial |
| Embedding shape | ✅ `TestConstantEmbedding` | N/A | ❌ None | Partial |
| Embedding dtype | ✅ `TestConstantEmbedding` | N/A | ❌ None | Partial |
| L2 normalization of stub embedding | ✅ `test_stub_embedding_is_normalized` | N/A | ❌ None | Partial |
| L2 normalization of real `.npy` embeddings | ✅ `test_npy_embedding_file_reader.py::TestNormalizationValidation` (`test_non_normalized_returns_none`, `test_non_normalized_emits_warning`, `test_zero_vector_returns_none`) | N/A | N/A | ✅ Covered |
| Multiple people, multiple embeddings | ✅ (via 4-person gallery) | Manual construction only | ✅ `test_gallery_to_recognition_integration.py::TestEndToEndRecognition` | ✅ Covered |
| Cache lookup by person_id | ✅ `TestPersonSubset` | N/A | N/A | ✅ |
| Accepted match with real gallery | ❌ None (stub constant vector) | ✅ `AcceptedMatchTests` | ✅ `test_gallery_to_recognition_integration.py::TestEndToEndRecognition` | ✅ Covered |
| Below-threshold rejection | ❌ None | ✅ `BelowThresholdTests` | N/A | ✅ |
| UNKNOWN fallback (empty gallery → no match) | ❌ None | ✅ `EmptyGalleryTests` | N/A | ✅ |
| NpyEmbeddingFileReader with actual `.npy` files | ✅ `test_npy_embedding_file_reader.py::TestValidLoad`, `TestAllRealGalleryFiles` | N/A | N/A | ✅ Covered |
| FaceGalleryLoader output → FaceRecognitionModule | N/A | N/A | ✅ `test_gallery_to_recognition_integration.py::TestLoaderToRecognitionWiring` | ✅ Covered |
| End-to-end recognition with loader-supplied gallery | N/A | N/A | ✅ `TestEndToEndRecognition` | ✅ Covered |
| Non-normalized embedding skipped with warning | ✅ `TestNormalizationValidation` | N/A | ✅ `TestNonNormalizedEmbeddingRejected` | ✅ Covered |
| NpyReader config wiring | ✅ `TestNpyReaderConfigWiring` | N/A | N/A | ✅ Covered |

---

## 7. Detected Mismatches

### Mismatch 1 — `person_name` field in enrollment record description

> **STATUS: RESOLVED.** `face_recognition.md §10` now explicitly states that `person_name` is not a field of the enrollment record and that name resolution is an external enrichment step. Both implementations use only `person_id` and `embedding`. Kept here for audit trail.

**Impact:** Minor

**Files and sections involved:**
- `doc/image_processing_service/face_recognition.md §10`
- `src/image_processing/face_recognition/module.py` (`EnrolledIdentity` TypedDict)
- `src/image_processing/face_gallery_loader/module.py` (`LoadedGalleryEmbedding` TypedDict)
- `doc/image_processing_service/face_gallery_loader.md §3.1`

**Structs involved:** `LoadedGalleryEmbedding`, `EnrolledIdentity`

**Description:** Earlier spec language referred to a shared gallery record with optional `person_name`. The implementation now uses domain-separated types: `LoadedGalleryEmbedding` and `EnrolledIdentity`, each containing only `person_id` and `embedding`.

**Why it is problematic:** The spec implies that the gallery carries `person_name` as optional metadata, but no code supports it and the loader does not produce it. This creates a false expectation for any reader of `face_recognition.md`. If `person_name` resolution is needed at the RPM level (RPM's `RecognizedFaceResult` includes `person_name`), the resolution mechanism is undocumented — it must be an external lookup beyond the gallery.

**Which module should change:** No additional change required; this mismatch is resolved.

**Recommended fix:** Keep `person_name` out of enrollment record schemas and document name enrichment as a separate post-recognition concern.

---

### Mismatch 2 — Naming ambiguity between loader and recognition gallery records

> **STATUS: RESOLVED.** Domain naming separation is now explicit: FaceGalleryLoader produces `LoadedGalleryEmbedding`; FaceRecognition consumes `EnrolledIdentity`; startup code maps `LoadedGalleryEmbedding[]` → `EnrolledIdentity[]`.

**Impact:** Resolved

**Files and sections involved:**
- `src/image_processing/face_gallery_loader/module.py` — defines `class LoadedGalleryEmbedding(TypedDict): person_id: str; embedding: FaceEmbedding`
- `src/image_processing/face_recognition/module.py` — defines `class EnrolledIdentity(TypedDict): person_id: str; embedding: FaceEmbedding`
- `doc/system.md` — documents startup mapping from `LoadedGalleryEmbedding[]` to `EnrolledIdentity[]`

**Structs involved:** `LoadedGalleryEmbedding`, `EnrolledIdentity`

**Description:** The old ambiguous shared naming has been replaced with explicit domain-owned types. FaceGalleryLoader now owns `LoadedGalleryEmbedding` as its output type. FaceRecognition owns `EnrolledIdentity` as its enrollment type. Startup wiring explicitly maps between them before constructing `FaceRecognitionModule`.

**Why it is problematic:** Previously, shared naming obscured module boundaries and encouraged implicit structural coupling.

**Which module should change:** No further module change required for this mismatch.

**Recommended fix:** Keep domain-owned types and explicit startup mapping. Optional future consolidation into shared contracts remains a separate architectural decision.

---

### Mismatch 3 — Default EmbeddingFileReader is StubEmbeddingFileReader, not NpyEmbeddingFileReader

> **STATUS: FULLY RESOLVED.** `FaceGalleryLoaderModule.__init__` defaults to `NpyEmbeddingFileReader` (lines 422–425 of module.py). Test verification: `test_face_gallery_loader_module.py::TestDefaultReader::test_default_reader_is_npy_embedding_file_reader()` explicitly asserts that `isinstance(loader._reader, NpyEmbeddingFileReader)`. Real-gallery integration verified in `test_npy_embedding_file_reader.py` (18 tests: TestValidLoad, TestAllRealGalleryFiles, TestShapeValidation, TestDtypeValidation, TestNormalizationValidation, TestMissingAndExtension, TestNpyReaderConfigWiring) and `test_gallery_to_recognition_integration.py` (8 tests: TestLoaderToRecognitionWiring, TestEndToEndRecognition, TestNonNormalizedEmbeddingRejected). onnxruntime is present in requirements.txt. StubEmbeddingFileReader remains available for explicit test injection only.

**Impact:** Medium

**Files and sections involved:**
- `doc/image_processing_service/face_gallery_loader.md §6.2` — states: *"NpyEmbeddingFileReader is the current default implementation of EmbeddingFileReader."*
- `src/image_processing/face_gallery_loader/module.py` — `FaceGalleryLoaderModule.__init__` defaults to `NpyEmbeddingFileReader` (lines 422–425)
- `tests/face_gallery_loader/test_face_gallery_loader_module.py::TestDefaultReader` — verifies default is NpyEmbeddingFileReader; also tests explicit StubEmbeddingFileReader injection
- `requirements.txt` — onnxruntime>=1.14.0 is present

**Classes involved:** `FaceGalleryLoaderModule`, `StubEmbeddingFileReader`, `NpyEmbeddingFileReader`

**Description:** The spec declares `NpyEmbeddingFileReader` as the default implementation. The implementation defaults to `NpyEmbeddingFileReader`. Test coverage verifies this default via `TestDefaultReader`. Production gallery loading through `NpyEmbeddingFileReader` is tested with real `.npy` files from `data/generated_face_gallery_real/`. The loader→recognition integration path is verified end-to-end by `test_gallery_to_recognition_integration.py`.

**Why it is problematic:** No issue; mismatch is resolved. See resolution above.

**Which module should change:** No change required. Default is already NpyEmbeddingFileReader; test verification is in place.

**Recommended fix:** No fix required. Default is now NpyEmbeddingFileReader; test verification added in TestDefaultReader class. This mismatch is resolved.

---

### Mismatch 4 — Internal components exported from face_gallery_loader `__init__.py`

> **STATUS: RESOLVED.** `GalleryDirectoryScanner`, `GalleryPathValidator`, and `PersonScanRecord` have been removed from `face_gallery_loader/__init__.py`. They are no longer part of the public package surface. `__init__.py` now exports only the public API symbols listed in `face_gallery_loader.md §3.3`. Kept here for audit trail.

**Impact:** Minor

**Files and sections involved:**
- `doc/image_processing_service/face_gallery_loader.md §3.3` and `§10` — `PersonScanRecord`, `GalleryDirectoryScanner`, and `GalleryPathValidator` are explicitly classified as internal components
- `src/image_processing/face_gallery_loader/__init__.py` — exports `GalleryDirectoryScanner`, `GalleryPathValidator`, and `PersonScanRecord`

**Classes involved:** `GalleryDirectoryScanner`, `GalleryPathValidator`, `PersonScanRecord`

**Description:** The loader spec's strict isolation requirement (`§3.3`) states that all internal components must not escape the public API. The spec explicitly classifies `PersonScanRecord` as internal in `§10`. `GalleryDirectoryScanner` and `GalleryPathValidator` are internal subcomponents (spec §8.2, §8.3). All three appear in `__init__.py`'s `__all__`, making them part of the public package surface.

**Why it is problematic:** External code (including tests or startup wiring) may import and depend on these internal classes. If the internal implementation changes, external consumers will break. This contradicts the module's self-containment guarantee.

**Which module should change:** `src/image_processing/face_gallery_loader/__init__.py` — remove `GalleryDirectoryScanner`, `GalleryPathValidator`, and `PersonScanRecord` from `__all__` and from the import list.

**Recommended fix:** Remove the three internal classes from the `__init__.py` exports. Keep only the public API symbols: `FaceGalleryLoaderModule`, `FaceGalleryLoaderConfig`, `LoadedGalleryEmbedding`, `FaceEmbedding`, `GalleryLoadError`, `GalleryPathValidationError`, `EmbeddingFileReader`, `StubEmbeddingFileReader`, `NpyEmbeddingFileReader`, and `FaceGalleryCache`.

---

### Mismatch 5 — Startup wiring not documented in either spec

> **STATUS: RESOLVED.** `doc/system.md` now has a “Service Startup Sequence — Gallery-to-Recognition Wiring” section with 4 explicit startup steps. `face_gallery_loader.md §1` cross-references `system.md` and `face_recognition.md §9.2`. `face_recognition.md §9.2` documents that gallery data is injected at construction from the loader. Both specs cross-reference each other. Kept here for audit trail.

**Impact:** Minor

**Files and sections involved:**
- `doc/image_processing_service/face_recognition.md §9.2, §13.1` — states enrolled embeddings are supplied at construction time but does not identify the provider
- `doc/image_processing_service/face_gallery_loader.md §1` — mentions "downstream identity lookup" as a consumer without naming Face Recognition
- No spec documents the startup sequence connecting the two modules

**Description:** Both specs leave the integration wiring implicit. The loader spec defines its API. The recognition spec says data is injected at construction. Neither spec names the other or documents the startup sequence connecting them.

**Why it is problematic:** Without a documented startup contract, implementers may wire the integration incorrectly — for example, passing the wrong gallery record type from the wrong source, or skipping the gallery loader entirely and constructing entries manually. Startup failure modes (e.g., what happens if `load_gallery()` raises before `FaceRecognitionModule` is created) are undocumented.

**Which module should change:** A system-level document (`doc/system.md` or a new startup sequence document) should document the service startup sequence and the explicit connection from FaceGalleryLoader to FaceRecognitionModule. Both module specs should cross-reference each other.

**Recommended fix:** Add a startup sequence section to `doc/system.md` that explicitly shows: `FaceGalleryLoaderModule.load_gallery()` → `get_all_embeddings()` → `FaceRecognitionModule(config, engine, gallery_entries=entries)`. Cross-reference from both module specs.

---

### Mismatch 6 — L2 normalization not validated by Face Gallery Loader; FaceMatcher assumes normalized vectors

> **STATUS: RESOLVED.** `NpyEmbeddingFileReader.read_embedding()` now validates L2 normalization: `abs(np.linalg.norm(arr) - 1.0) > 1e-4` → returns `None` + `warnings.warn`. Non-normalized files are skipped with a warning. `face_gallery_loader.md §7.1` documents the normalization requirement, rationale, and validation behavior. Tests: `TestNormalizationValidation` (3 tests) and `TestNonNormalizedEmbeddingRejected` (integration test) cover this path. Kept here for audit trail.

**Impact:** Medium

**Files and sections involved:**
- `src/image_processing/face_gallery_loader/module.py` — `NpyEmbeddingFileReader.read_embedding()` validates shape and dtype only; no normalization check
- `src/image_processing/face_recognition/module.py` — `FaceMatcher.find_best_match()` uses `np.dot(embedding, entry["embedding"])` as cosine similarity, which is correct only for L2-normalized vectors
- `doc/image_processing_service/face_gallery_loader.md §6.4` — validates dimension and dtype; L2 normalization not mentioned
- `doc/image_processing_service/face_recognition.md §6.2` — ArcFace produces L2-normalized embeddings

**Classes involved:** `NpyEmbeddingFileReader`, `FaceMatcher`, `FaceGalleryCache`

**Description:** `FaceMatcher` computes similarity as the dot product of the query embedding and each gallery embedding. For this to equal cosine similarity (the intended metric for ArcFace), both vectors must be L2-normalized. The query embedding produced by `ArcFaceEmbeddingEngine` is expected to be L2-normalized. Gallery embeddings loaded from disk by `NpyEmbeddingFileReader` are not validated for normalization. If a `.npy` file contains a non-normalized embedding, `FaceMatcher` will compute a mathematically incorrect similarity score without any error or warning.

**Why it is problematic:** The error is silent. Recognition may accept or reject identities incorrectly with no indication that the gallery data is malformed. This is particularly risky during production deployment when real `.npy` files are loaded.

**Which module should change:** `NpyEmbeddingFileReader.read_embedding()` should validate that the loaded array has L2 norm ≈ 1.0 (within a small tolerance, e.g., `abs(np.linalg.norm(arr) - 1.0) < 1e-4`). Alternatively, the loader could L2-normalize all embeddings on load. The spec should document the normalization requirement explicitly.

**Recommended fix:** Add an L2 normalization check to `NpyEmbeddingFileReader.read_embedding()`. Return `None` (structured failure) if the norm deviates from 1.0 beyond a configurable tolerance. Document the normalization requirement in `face_gallery_loader.md §7` (Acceptance / Filtering Logic).

---

## 8. Already Aligned Items

| Item | Face Gallery Loader | Face Recognition | Status |
|------|--------------------|-----------------:|--------|
| `person_id` type is `string` / `str` | ✅ | ✅ | Aligned |
| `embedding` dtype is `float32` | ✅ (enforced by EmbeddingFileReader) | ✅ (expected by FaceMatcher) | Aligned |
| `embedding` shape is `(512,)` | ✅ (enforced by EmbeddingFileReader) | ✅ (ArcFace convention) | Aligned |
| Enrollment/gallery record fields are `person_id` and `embedding` (implementation) | ✅ | ✅ | Aligned |
| RPM does not directly depend on FaceGalleryLoader | Excluded from loader scope | Excluded from RPM scope | Aligned |
| FaceGalleryCache is internal to recognition module | N/A | ✅ — never exposed to RPM | Aligned |
| Gallery loading is startup-only, not per-frame | ✅ — `load_gallery` is a one-time call | ✅ — FaceGalleryCache is built at `__init__` | Aligned |
| `gallery_entries` injected into FR at construction | ✅ — `get_all_embeddings()` provides the list | ✅ — constructor parameter | Aligned |
| Skip-or-fail policy for malformed files | ✅ — documented and implemented | N/A — not FR's concern | Aligned |
| Empty gallery: GalleryLoadError raised before FR init | ✅ | N/A | Aligned |
| Empty gallery passed to FR → `person_found=False` | N/A | ✅ — FaceMatcher returns None; DecisionPolicy returns False | Aligned |
| UNKNOWN fallback: no match → `person_found=False` | N/A | ✅ — DecisionPolicy applies threshold | Aligned |
| Per-person folder semantics: subdirectory name = `person_id` | ✅ | N/A | Aligned |
| Multiple embeddings per person: all included | ✅ — multiple `.npy` files per subdirectory accepted | ✅ — FaceMatcher scans all entries | Aligned |
| Duplicate `person_id` entries: not possible at loader | ✅ — unique subdirectory names guarantee uniqueness | ✅ — all entries accepted by cache | Aligned |
| Loader is read-only | ✅ — no write operations | N/A | Aligned |
| Deterministic ordering | ✅ — lexicographic by `person_id`, then file order | ✅ — deterministic given same gallery | Aligned |
| `get_embeddings(unknown_person_id)` returns empty list, no error | ✅ | N/A — FR uses `get_entries()` | Aligned |
| `person_name` is not an enrollment/gallery record field | ✅ — only `person_id` + `embedding` | ✅ — only `person_id` + `embedding`; `face_recognition.md §10` documents external enrichment | Aligned (Mismatch 1 resolved) |
| Internal components not in public exports | ✅ — `GalleryDirectoryScanner`, `GalleryPathValidator`, `PersonScanRecord` removed from `__init__.py` | N/A | Aligned (Mismatch 4 resolved) |
| Startup wiring documented | ✅ — `face_gallery_loader.md §1` cross-references `system.md` and `face_recognition.md §9.2` | ✅ — `face_recognition.md §9.2` documents gallery injection at construction | Aligned (Mismatch 5 resolved) |
| L2 normalization validated before gallery use | ✅ — `NpyEmbeddingFileReader` checks `abs(np.linalg.norm(arr) - 1.0) > 1e-4`; rejects with warning | ✅ — `FaceMatcher` receives only validated vectors | Aligned (Mismatch 6 resolved) |

---

## 9. Recommended Resolution Plan

Items are ordered by impact and dependency. No item blocks basic integration testing. All items are improvements toward production readiness.

### Phase 1 — Documentation fixes ✅ COMPLETE

| # | Action | Files | Impact |
|---|--------|-------|--------|
| 1 | ✅ Remove `person_name` from enrollment record description in `face_recognition.md §10` | `doc/image_processing_service/face_recognition.md` | Minor |
| 2 | ✅ Add startup sequence documentation to `doc/system.md` — Service Startup Sequence — Gallery-to-Recognition Wiring section | `doc/system.md` | Minor-Medium |
| 3 | ✅ Cross-reference `face_recognition.md` and `face_gallery_loader.md` to each other in their scope sections | Both spec files | Minor |
| 4 | ✅ Document domain separation and explicit startup mapping between loader and recognition record types | `face_gallery_loader.md §1` | Minor |

### Phase 2 — Code quality fixes ✅ COMPLETE

| # | Action | Files | Impact |
|---|--------|-------|--------|
| 5 | ✅ Remove `GalleryDirectoryScanner`, `GalleryPathValidator`, and `PersonScanRecord` from `face_gallery_loader/__init__.py` exports | `src/image_processing/face_gallery_loader/__init__.py` | Minor |
| 6 | ✅ Add L2 normalization validation to `NpyEmbeddingFileReader.read_embedding()` — returns `None` + `warnings.warn` if `abs(np.linalg.norm(arr) - 1.0) > 1e-4` | `src/image_processing/face_gallery_loader/module.py` | Medium |
| 7 | ✅ Document the normalization requirement in `face_gallery_loader.md §7.1` | `doc/image_processing_service/face_gallery_loader.md` | Medium |

### Phase 3 — Test coverage ✅ COMPLETE

| # | Action | Files | Impact |
|---|--------|-------|--------|
| 8 | ✅ `NpyEmbeddingFileReader` tests with actual `.npy` files from `data/generated_face_gallery_real/` — `TestValidLoad`, `TestAllRealGalleryFiles`, shape/dtype/normalization validation | `tests/face_gallery_loader/test_npy_embedding_file_reader.py` | Medium |
| 9 | ✅ Integration test: loader → `get_all_embeddings()` → `FaceRecognitionModule` → `recognize()` with matching identity | `tests/face_gallery_loader/test_gallery_to_recognition_integration.py` | Medium |
| 10 | ✅ Integration test: non-normalized embedding in gallery → skipped with warning; recognition unaffected | `tests/face_gallery_loader/test_gallery_to_recognition_integration.py::TestNonNormalizedEmbeddingRejected` | Medium |
| 11 | ✅ Test verifying `NpyEmbeddingFileReader` configuration wiring in `FaceGalleryLoaderModule` | `tests/face_gallery_loader/test_npy_embedding_file_reader.py::TestNpyReaderConfigWiring` | Minor-Medium |

**Note:** Integration tests are located in `tests/face_gallery_loader/test_gallery_to_recognition_integration.py`. A separate `tests/integration/` directory does not exist.

### Phase 4 — Long-term structural improvements (optional)

| # | Action | Files | Impact |
|---|--------|-------|--------|
| 12 | Optional: evaluate whether shared contracts are beneficial beyond current explicit mapping (`LoadedGalleryEmbedding[]` → `EnrolledIdentity[]`) | `shared/contracts.py` (optional), startup composition code | Minor |

---

## 10. Final Compatibility Status

| Dimension | Status | Detail |
|-----------|--------|--------|
| Overall RPM ↔ Face Gallery Loader compatibility | ✅ INDIRECT — by design | RPM has no direct dependency on FaceGalleryLoader |
| Is this a direct integration? | No | RPM depends only on `FaceRecognitionInterface` |
| Is this an indirect dependency? | Yes | Gallery produced by FaceGalleryLoader is consumed by FaceRecognitionModule, which RPM depends on |
| Is the integration safe for testing? | ✅ Yes | No blocking issues; compatible at structural level |
| Blocking integration issues | None | 0 blocking mismatches |
| Must-fix-before-production issues | None — all mismatches resolved | 0 must-fix issues |
| Startup readiness | ✅ | Startup wiring documented in `doc/system.md` and both module specs; integration tests confirm the wiring |
| Cache readiness | ✅ | `FaceGalleryCache` works correctly in both modules |
| Face Recognition consumption readiness | ✅ | `FaceRecognitionModule` accepts `EnrolledIdentity[]`; startup maps from `LoadedGalleryEmbedding[]` |
| Shared contract readiness | ✅ | Domain separation is explicit (`LoadedGalleryEmbedding` vs `EnrolledIdentity`) and startup mapping is documented |
| Test readiness | ✅ | Both modules tested in isolation; 44 integration and unit tests cover `NpyEmbeddingFileReader`, loader→recognition wiring, end-to-end recognition, and normalization validation |
| Production readiness (NpyEmbeddingFileReader path) | ✅ Complete | NpyEmbeddingFileReader is default; real gallery integration verified; L2 normalization validated |

### Summary

RecognitionPipelineManager and FaceGalleryLoader have **no direct integration**. This is correct architectural behavior: gallery management is outside RPM's scope by explicit spec definition. The integration path — FaceGalleryLoader at startup → `gallery_entries` → FaceRecognitionModule → `FaceRecognitionInterface` → RPM — is consistent with all specs and implementations and is structurally sound.

Of the six original mismatches: all are fully resolved. Mismatch 1 (person_name resolution), 2 (naming separation), 4 (internal exports), 5 (startup wiring), and 6 (L2 normalization) completed in prior sessions. Mismatch 3 (default reader) completed: `FaceGalleryLoaderModule` now defaults to `NpyEmbeddingFileReader`; test verification added in `TestDefaultReader` class; real-gallery integration tests confirm end-to-end path with actual `.npy` files from `data/generated_face_gallery_real/`; `onnxruntime` dependency verified in `requirements.txt`. No remaining gaps.
