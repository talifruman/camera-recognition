# RecognitionPipelineManager ? Frame Transformation Layer Compatibility Review

## Review Date
- 2026-05-12

## Current Status
- Pre-verification status: PARTIAL, with known API ambiguity around GeometrySpec/BoundingBox public contract boundaries and missing policy coverage evidence.
- Post-verification status: READY FOR RPM IMPLEMENTATION (with non-blocking hardening follow-ups).

## Reviewed Artifacts
- doc/image_processing_service/RecognitionPipelineManager.md
- doc/image_processing_service/frame_transformation_layer.md
- doc/image_processing_service/shared_contracts.md
- src/image_processing/shared/contracts.py
- src/image_processing/frame_transformation_layer/module.py
- src/image_processing/frame_transformation_layer/__init__.py
- src/image_processing/object_detection/module.py
- tests/frame_transformation_layer/test_frame_transformation_layer_module.py

## Final API Contract Summary
- FTL public boundary uses shared contracts for caller-facing types:
  - get_frame(camera_id, temporal_selector, region_bbox, output_type, geometry_spec) where region_bbox uses shared BoundingBox shape and geometry_spec accepts shared GeometrySpec dict shape.
- ProcessedFrame contains:
  - image: shared Image
  - source_bbox_full_frame: shared BoundingBox-shaped dict in full-frame coordinates
  - spatial_transform: mandatory and always populated
- Supported ResizePolicy behaviors in FTL converter:
  - NONE
  - LETTERBOX
- Temporal semantics are defined and test-backed:
  - CURRENT exists after first ingest
  - PREVIOUS unavailable until second ingest
  - per-camera CURRENT/PREVIOUS isolation
- Concurrency evidence exists for multi-camera state isolation and no cross-camera contamination.

## Boundary Contract Table
| Boundary | Expected Contract | Verified In Code | Test Evidence | Status |
|---|---|---|---|---|
| ingest_frame(frame_packet) | Canonical FramePacket ingest and store as full-frame shared Image | Yes | test_ingest_stores_full_frame_base_image | Resolved |
| get_frame(..., region_bbox, ..., geometry_spec) | Accept shared BoundingBox and shared GeometrySpec at boundary | Yes (duck-typed field validation/usage) | test_get_frame_accepts_shared_bounding_box_typeddict, test_get_frame_accepts_shared_geometry_spec_typeddict | Resolved |
| ProcessedFrame.image | shared Image with np.ndarray data | Yes | test_get_frame_returns_processed_frame | Resolved |
| ProcessedFrame.source_bbox_full_frame | Full-frame shared BoundingBox coordinates | Yes | multiple get_frame/crop tests | Resolved |
| ProcessedFrame.spatial_transform | Mandatory metadata for projection | Yes | NONE/LETTERBOX tests | Resolved |
| ResizePolicy.NONE | Identity transform | Yes | test_none_geometry_keeps_crop_dimensions, test_spatial_transform_none_identity | Resolved |
| ResizePolicy.LETTERBOX | Uniform scale + padding to exact target | Yes | test_letterbox_geometry_target_dims_and_transform, test_spatial_transform_letterbox_includes_padding | Resolved |
| CURRENT/PREVIOUS | Cold start + rotation semantics | Yes | temporal tests in FTL suite | Resolved |
| Concurrency expectations | Multi-camera isolation under parallel usage assumptions | Partial but present | test_concurrent_ingest_multiple_cameras, test_concurrent_current_previous_isolation, test_no_cross_camera_contamination | Hardening gap (expanded stress tests) |

## Shared Contract Authority Check
Duplicate-type search classification:
- BoundingBox
  - src/image_processing/shared/contracts.py: authoritative shared public type
  - src/image_processing/object_detection/module.py: now imports shared BoundingBox (resolved public duplicate)
  - src/image_processing/frame_transformation_layer/module.py: internal helper dataclass remains, but no longer exported from FTL package API
- GeometrySpec
  - src/image_processing/shared/contracts.py: authoritative shared public type
  - src/image_processing/frame_transformation_layer/module.py: internal helper dataclass remains for internal convenience; public boundary accepts shared dict-shape
- Image
  - authoritative shared type only (public)

Public export cleanup performed:
- Removed BoundingBox and GeometrySpec from src/image_processing/frame_transformation_layer/__init__.py exports.

## Resolved Mismatches
- Resolved API mismatch: FTL now supports the shared ResizePolicy values end-to-end (NONE/LETTERBOX).
- Resolved API mismatch: FTL public package no longer exports duplicate BoundingBox/GeometrySpec public boundary types.
- Resolved API mismatch: object_detection public BoundingBox now uses shared authoritative contract import.
- Resolved missing evidence: shared TypedDict compatibility tests for get_frame inputs are present and passing.
- Resolved missing evidence: spatial_transform behavior across all policies is test-backed.
- Resolved missing evidence: multi-camera CURRENT/PREVIOUS isolation and no cross-camera contamination tests are present and passing.

## Active Issues (By Category)
### Runtime Hardening Gap
- Concurrency coverage is functional but minimal. Current tests verify logical isolation, not prolonged contention/stress patterns.

### Missing Concurrency Evidence
- No dedicated high-volume stress test for concurrent get_frame while concurrent ingest on same camera with randomized region requests.

### Future Optimization Decision
- Image ownership policy is currently contract-based (read-only consumer behavior). 
- FTL currently uses copy operations in critical paths (`np.copy()` on base build and crop), but this remains an implementation detail, not a guaranteed optimization policy.
- Decision pending: enforce defensive-copy semantics everywhere vs. preserve current performance profile with read-only contract.

### Actual RPM Blockers
- None remaining.

## Required Fixes Before RPM Implementation
- None.

## Required Tests Before RPM Implementation
- None mandatory blockers remain.
- Recommended hardening (non-blocking):
  - Add contention-heavy concurrency stress test matrix (same-camera ingest/get_frame interleaving at high iteration counts).
  - Add explicit inverse-projection helper tests at RPM SpatialCoordinator level once RPM code exists.

## Open Questions
- Should long-running concurrency stress tests be promoted to required CI gates pre-production, or tracked as post-RPM hardening?
- Should FTL eventually expose explicit immutable image wrappers, or continue with read-only-by-contract semantics for shared Image.data?

## Final Compatibility Status
- Final status: COMPATIBLE and READY.
- Spatial_transform definition sufficiency: Yes, fully defined for NONE/LETTERBOX and documented for inverse projection.
- LETTERBOX projection specification: Yes.
- CURRENT/PREVIOUS semantics: Yes, fully specified and test-backed.
- Shared contracts authoritative with no duplicate public boundary types: Yes.

## Final Decision
- Can RPM implementation begin now? Yes.
- Which issues must be fixed before coding RPM? None.
- Which issues can be tracked as hardening after initial RPM implementation?
  - Concurrency stress expansion
  - Performance/ownership optimization policy decision for image data handling
- Is spatial_transform fully defined enough for RPM to use safely? Yes.
- Is LETTERBOX projection fully specified? Yes.
- Are CURRENT/PREVIOUS semantics fully specified? Yes.
- Are shared contracts authoritative with no duplicate public boundary types? Yes.
