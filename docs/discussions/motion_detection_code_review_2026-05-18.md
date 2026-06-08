# Motion Detection Code Review - Engineering Hot-Path Review (2026-05-18)

## Scope

- Module under review: `src/image_processing/motion_detection/module.py`
- Contract and operational requirements: `doc/image_processing_service/motion_detection.md`
- Conventions baseline: `docs/PythonCode/Living_Code_Review_Conventions.md`
- Test baseline: `tests/motion_detection/test_motion_detection_module.py`

This review is implementation-focused and evidence-based. Findings are prioritized by:

1. correctness
2. production safety
3. concurrency safety
4. real-time performance
5. maintainability
6. readability

## Findings Summary

| Issue | Severity | Real-Time / Production Impact | Location |
| --- | --- | --- | --- |
| Silent exception swallowing in public API | CRITICAL | False negative motion, undiagnosable production failures, broken SLO visibility | `module.py::MotionDetectionManager.detect` |
| Unbounded debug-frame retention + full-frame copying | CRITICAL | Memory churn, allocator pressure, long-tail latency spikes | `module.py::FrameDifferencingMotionDetector.measure`, `module.py::MotionDetectionManager._process_internal` |
| Unbounded per-camera state lifecycle (no idle eviction) | MAJOR | Memory growth over long uptimes, stale camera state retained forever | `module.py::_motion_history_by_camera_id`, `_apply_temporal_persistence` |
| O(n^2) bbox merge loop in frame hot path | MAJOR | Throughput collapse with contour bursts, frame-time variance under stress | `module.py::_merge_bboxes_for_frame`, `_should_merge` |
| Missing config fail-fast validation at manager boundary | MAJOR | Silent misconfiguration, unstable behavior across deployments | `module.py::MotionDetectionConfig`, `FrameDifferencingMotionDetector.__init__` |
| Repeated full-frame operations and per-frame kernel allocation | MINOR | Extra CPU and allocation overhead on every frame | `module.py::measure`, `_compute_changed_ratio` |

## Hot-Path Cost Table

| Operation | Current Behavior | Approximate Cost Driver | Risk Pattern |
| --- | --- | --- | --- |
| Debug image capture | 5-8 frame-sized arrays copied per call (when debug images are populated) | At 1920x1080 uint8: ~2.1 MB per copy; 6 copies ~12.4 MB/frame | Allocation churn + cache pressure |
| Thresholding | Threshold done in `_compute_changed_ratio` and again in main pipeline | Extra full-frame pass (`O(H*W)`) | Redundant pixel scans |
| Morphology kernel | `np.ones((3,3), dtype=np.uint8)` allocated every call | Small but per-frame repeated allocation | Hot-path allocation noise |
| Bbox merge | Pairwise merging loop with repeated distance/IoU checks | `O(n^2)` per pass, multi-pass until fixed point | Stress-time frame spikes |
| Global alignment (enabled) | GFTT + LK optical flow + affine + warp each frame | High per-frame compute at larger resolutions | Real-time budget overrun risk |

## Production-Safety Table

| Failure Mode | Current Behavior | User-Visible Effect | Required Direction |
| --- | --- | --- | --- |
| Runtime exception in algorithm path | Caught by broad `except Exception`, returns `detected=False` | Internal failure is indistinguishable from real no-motion | Emit structured error telemetry and explicit failure reason |
| Invalid config values | Mostly normalized/coerced, not validated centrally | Misbehavior without startup failure | Validate config once at initialization and fail loudly |
| Camera state lifecycle | State keyed by `camera_id`, no inactivity expiry | Long-running process accumulates stale states | Add bounded lifecycle (TTL/LRU/manual prune policy) |

---

# Finding 1: Silent Exception Swallowing Masks Production Failures

Severity:

- CRITICAL

Location:

- file: `src/image_processing/motion_detection/module.py`
- class: `MotionDetectionManager`
- function: `detect`
- approximate lines: 721-733

Current Code:

```python
def detect(self, input: MotionDetectionInput) -> MotionResult:
    """Single RPM-facing public stage API.

    Accepts a MotionDetectionInput whose frames carry shared Image structs.
    Returns MotionResult(detected=False, bboxes=[]) for any invalid input
    or algorithm failure (spec §11).
    """
    try:
        return self._process_internal(input)
    except Exception:
        self._last_debug_info = {}
        self._last_debug_images = {}
        # Spec §11: all unhandled exceptions → detected=False
        return self._output_builder.build(
            MotionDetectionResultInternal(detected=False, bboxes=[])
        )
```

Problem:

The public API swallows all runtime exceptions and maps them to normal "no motion" output. This erases error semantics at the module boundary.

Why It Matters:

- correctness risk: algorithm failures become false negatives instead of explicit failures.
- hidden behavior: caller cannot distinguish "scene has no motion" from "motion detector crashed".
- reviewability problem: incident postmortems cannot reconstruct root cause from API behavior.
- conventions conflict: `docs/PythonCode/Living_Code_Review_Conventions.md` explicitly flags silent fallback as an anti-pattern.

Real-Time / Production Impact:

- monitoring blind spot: availability/error SLOs cannot detect detector failures.
- hidden failure mode: pipeline appears healthy while output quality degrades.
- operational impact: can suppress alerts in motion-driven systems (security/event triggers).

Suggested Direction:

Keep API compatibility if needed (`detected=False` fallback), but publish failure metadata and emit structured logging/metrics on exception paths.

Suggested Pattern:

```python
def detect(self, input: MotionDetectionInput) -> MotionResult:
    try:
        return self._process_internal(input)
    except (ValueError, TypeError):
        # Contract violations are expected user-input failures.
        self._record_validation_failure(input)
        return MotionResult(detected=False, bboxes=[])
    except Exception as exc:
        # Runtime failures must be observable.
        self._record_runtime_failure(exc)
        self._last_debug_info = {"runtime_error": type(exc).__name__}
        self._last_debug_images = {}
        return MotionResult(detected=False, bboxes=[])
```

---

# Finding 2: Full-Frame Debug Copies and Manager-Level Recopy Inflate Memory Churn

Severity:

- CRITICAL

Location:

- file: `src/image_processing/motion_detection/module.py`
- class: `FrameDifferencingMotionDetector`
- function: `measure`
- approximate lines: 263-336
- related location: `MotionDetectionManager._process_internal` lines 819-821

Current Code:

```python
debug_images: dict[str, np.ndarray] = {
    "raw_previous_frame": prev2d.copy(),
    "current_frame": curr2d.copy(),
}

diff_before = cv2.absdiff(prev_work, curr_work)
debug_images["diff_before_alignment"] = diff_before.copy()

...

debug_images.setdefault("aligned_previous_frame", aligned_previous.copy())
debug_images["diff_after_alignment"] = diff.copy()

...

debug_images["final_cleaned_motion_mask"] = mask.copy()
```

```python
self._last_debug_images = {
    key: value.copy() for key, value in measurement.debug_images.items()
}
```

Problem:

The hot path copies multiple full-size arrays into `debug_images`, then the manager copies all of them again. This duplicates frame memory traffic inside the per-frame execution path.

Why It Matters:

- ownership issue: debug-state ownership is unclear (algorithm owns one copy, manager owns another).
- lifecycle issue: `_last_debug_images` persists until next call/reset; no explicit cap for retained bytes.
- architectural violation: debug capture work is on the same critical path as detection logic.
- conventions mismatch: hot-path rules discourage unnecessary `.copy()` and debug artifact accumulation.

Real-Time / Production Impact:

- measurable memory churn example: 1080p gray frame is about 2.1 MB; 6 copies in algorithm + manager recopy can exceed 12-20 MB/frame depending on GMC branch.
- latency effect: allocator pressure and cache eviction increase p95/p99 frame latency under sustained FPS.
- scalability risk: multi-camera deployments multiply this cost linearly by active stream count.

Suggested Direction:

- Move debug image capture behind an explicit debug mode flag.
- Avoid manager recopy when possible; keep immutable references or ring-buffer snapshots with strict capacity.
- Define ownership: detector produces optional debug handles, manager stores bounded debug artifacts only when enabled.

Suggested Pattern:

```python
@dataclass(slots=True)
class DebugCapturePolicy:
    enabled: bool = False
    max_images: int = 2

def _capture_debug(self, key: str, img: np.ndarray) -> None:
    if not self._debug_policy.enabled:
        return
    if len(self._debug_images) >= self._debug_policy.max_images:
        return
    self._debug_images[key] = img.copy()
```

---

# Finding 3: Per-Camera Temporal State Has No Idle Eviction Policy

Severity:

- MAJOR

Location:

- file: `src/image_processing/motion_detection/module.py`
- class: `MotionDetectionManager`
- function: `_apply_temporal_persistence`
- approximate lines: 873-924

Current Code:

```python
self._motion_history_by_camera_id: dict[str, _CameraMotionState] = {}
...
state = self._motion_history_by_camera_id.setdefault(camera_id, _CameraMotionState())
state.history.append(list(current_bboxes))
max_history = max(1, int(self._config.max_history_frames))
if len(state.history) > max_history:
    state.history = state.history[-max_history:]
```

Problem:

History is bounded per camera, but the camera map itself is unbounded over process lifetime. Cameras that stop sending frames are never expired unless external code calls reset methods.

Why It Matters:

- ownership/lifecycle clarity: module owns camera-scoped state without ownership expiry contract.
- concurrency implication: if this manager is shared across many camera feeds, stale state persists indefinitely.
- hidden behavior: memory growth is workload-duration dependent, not immediately visible in tests.

Real-Time / Production Impact:

- long-running services accumulate dormant camera entries.
- memory footprint drifts upward with camera churn (dynamic fleets, reconnect cycles).
- increased GC overhead and state traversal overhead over time.

Suggested Direction:

Track last-seen timestamp per camera state and evict inactive entries by TTL or bounded LRU policy in hot-safe amortized cleanup steps.

Suggested Pattern:

```python
@dataclass(slots=True)
class _CameraMotionState:
    tracks: list[_PersistentTrack] = field(default_factory=list)
    history: list[list[BoundingBox]] = field(default_factory=list)
    last_seen_ms: int = 0

def _prune_inactive_cameras(self, now_ms: int) -> None:
    expired = [
        cam_id for cam_id, state in self._motion_history_by_camera_id.items()
        if now_ms - state.last_seen_ms > self._config.camera_state_ttl_ms
    ]
    for cam_id in expired:
        self._motion_history_by_camera_id.pop(cam_id, None)
```

---

# Finding 4: BBox Merge Uses Repeated Pairwise Scans in Hot Path

Severity:

- MAJOR

Location:

- file: `src/image_processing/motion_detection/module.py`
- class: `MotionDetectionManager`
- function: `_merge_bboxes_for_frame`, `_should_merge`
- approximate lines: 826-872, 926-931

Current Code:

```python
merged = [bbox for bbox in normalized if bbox is not None]
changed = True
while changed:
    changed = False
    next_boxes: list[BoundingBox] = []
    consumed = [False] * len(merged)

    for i, base_bbox in enumerate(merged):
        if consumed[i]:
            continue

        candidate = base_bbox
        consumed[i] = True

        for j in range(i + 1, len(merged)):
            if consumed[j]:
                continue

            other = merged[j]
            if self._should_merge(candidate, other):
                candidate = self._union_bboxes(candidate, other)
                consumed[j] = True
                changed = True
```

Problem:

Merging performs repeated pairwise scans and can execute multiple passes until convergence. Complexity grows quickly with contour bursts and fragmented masks.

Why It Matters:

- correctness under load: frame deadlines can be missed even when correctness logic is otherwise right.
- real-time stability: runtime variance grows with scene noise and contour count.
- production safety: frame drops or backlog growth can appear only in high-motion episodes.

Real-Time / Production Impact:

- per-frame cost can degrade toward repeated `O(n^2)` passes.
- in noisy scenes (many small contours), merge stage can dominate frame budget.
- multi-camera concurrency multiplies worst-case CPU demand.

Suggested Direction:

Use spatial partitioning (grid bucketing) to reduce candidate comparisons, then union-find to merge connected groups in one pass.

Suggested Pattern:

```python
def merge_bboxes_grid(bboxes: list[BoundingBox], cell: int) -> list[BoundingBox]:
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, b in enumerate(bboxes):
        for key in covered_cells(b, cell):
            buckets[key].append(idx)
    dsu = DisjointSet(len(bboxes))
    for idxs in buckets.values():
        for i, j in pairwise_candidates(idxs):
            if should_merge(bboxes[i], bboxes[j]):
                dsu.union(i, j)
    return union_groups(dsu, bboxes)
```

---

# Finding 5: Config Validation Is Not Centralized or Fail-Fast

Severity:

- MAJOR

Location:

- file: `src/image_processing/motion_detection/module.py`
- class: `MotionDetectionConfig`, `FrameDifferencingMotionDetector`, `MotionDetectionManager`
- approximate lines: 54-77, 211-245, 682-719

Current Code:

```python
@dataclass(slots=True)
class MotionDetectionConfig:
    motion_threshold: int = 25
    motion_fraction_threshold: float = 0.05
    min_bbox_area: int = 100
    ...
    max_history_frames: int = 8
```

```python
self._motion_threshold = motion_threshold
self._min_bbox_area = min_bbox_area
self._max_features = max(1, int(max_features))
self._min_feature_matches = max(1, int(min_feature_matches))
self._global_motion_changed_ratio_threshold = max(0.0, float(global_motion_changed_ratio_threshold))
```

Problem:

Config values are mostly coerced in-place instead of validated once with explicit invariant checks and clear failure messages.

Why It Matters:

- correctness risk: invalid ranges can silently alter behavior.
- production safety: config mistakes pass startup and fail only under load.
- conventions conflict: review conventions require startup validation.

Real-Time / Production Impact:

- deployment drift: different configs produce hard-to-compare behavior.
- rollback risk: subtle config typo can alter detector sensitivity without visibility.

Suggested Direction:

Introduce a dedicated config validator at manager construction and reject invalid configs with actionable messages.

Suggested Pattern:

```python
def validate_config(cfg: MotionDetectionConfig) -> None:
    if not (0 <= cfg.motion_threshold <= 255):
        raise ValueError(f"motion_threshold must be in [0,255], got {cfg.motion_threshold}")
    if not (0.0 <= cfg.motion_fraction_threshold <= 1.0):
        raise ValueError("motion_fraction_threshold must be in [0.0,1.0]")
    if cfg.max_features < cfg.min_feature_matches:
        raise ValueError("max_features must be >= min_feature_matches")
```

---

# Finding 6: Redundant Full-Frame Thresholding and Per-Frame Kernel Allocation

Severity:

- MINOR

Location:

- file: `src/image_processing/motion_detection/module.py`
- class: `FrameDifferencingMotionDetector`
- functions: `measure`, `_compute_changed_ratio`
- approximate lines: 323-335, 463-468

Current Code:

```python
changed_ratio_after, _ = self._compute_changed_ratio(diff)
...
_, mask = cv2.threshold(diff, self._motion_threshold - 1, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), dtype=np.uint8)
if self._morph_open_iterations > 0:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self._morph_open_iterations)
```

```python
def _compute_changed_ratio(self, diff: np.ndarray) -> tuple[float, np.ndarray]:
    total_pixels = diff.shape[0] * diff.shape[1]
    _, mask = cv2.threshold(diff, self._motion_threshold - 1, 255, cv2.THRESH_BINARY)
    changed_pixel_count = int(np.count_nonzero(mask))
    return changed_pixel_count / max(1, total_pixels), mask
```

Problem:

The code thresholds `diff` in `_compute_changed_ratio`, discards the produced mask, then thresholds again for contour pipeline. Kernel allocation is also repeated per call.

Why It Matters:

- avoidable CPU work in per-frame loop.
- repeated allocation in hot path (small but constant overhead).
- makes profiling noisier and optimization harder.

Real-Time / Production Impact:

- extra `O(H*W)` pass per frame.
- small per-frame overhead becomes measurable at high FPS and multi-stream loads.

Suggested Direction:

Return and reuse one threshold mask for both ratio and contour extraction, and pre-create morphology kernel in constructor.

Suggested Pattern:

```python
def _threshold_and_ratio(self, diff: np.ndarray) -> tuple[np.ndarray, float]:
    _, mask = cv2.threshold(diff, self._motion_threshold - 1, 255, cv2.THRESH_BINARY)
    ratio = float(np.count_nonzero(mask)) / float(mask.size)
    return mask, ratio
```

---

## Test Coverage Review (Evidence-Based Gaps)

Existing strengths are substantial:

- Error fallback behavior is covered in `ManagerBehaviorTests.test_algorithm_exception_returns_no_motion`.
- GMC behavior and fallback switches are covered in `GlobalMotionCompensationTests`.
- Merge/persistence basics and per-camera isolation are covered in `BboxMergingAndTemporalPersistenceTests`.

Concrete missing scenarios and escaped regressions:

| Missing Scenario | Why Existing Tests Miss It | Regression That Can Escape |
| --- | --- | --- |
| Concurrent calls to one `MotionDetectionManager` instance across threads | Current tests are single-threaded and sequential | Data races on shared mutable fields (`_last_debug_info`, `_last_debug_images`, `_motion_history_by_camera_id`) can corrupt output/debug state |
| Idle camera state eviction over long uptime | Tests verify history behavior, not state lifecycle expiry | Stale camera state accumulation causes memory drift in production |
| Config boundary rejection (`motion_threshold<0`, `>255`, `fraction>1`) | Tests validate algorithm outcomes, not fail-fast config invariants | Invalid config silently coerces behavior and causes hard-to-diagnose sensitivity shifts |
| Performance guardrails under contour bursts | No latency/allocation assertions in tests | Merge and copy overhead regressions can pass CI and fail real-time SLA |
| Debug-mode memory budget behavior | Tests do not assert retained debug-image bytes/limits | Debug artifact accumulation can trigger memory pressure in sustained operation |

Recommended test additions:

1. Threaded stress test for concurrent `detect` calls on same manager instance with multiple camera IDs.
2. Long-run lifecycle test validating TTL/LRU camera-state eviction semantics.
3. Config validation tests that assert startup failure on invalid ranges.
4. Synthetic contour-burst benchmark test with a max frame-time budget assertion.
5. Debug capture budget test asserting bounded retained image count/bytes.

---

## Anti-Pattern Consolidation

Observed repeated anti-patterns mapped to conventions:

| Anti-Pattern | Evidence in Motion Module | Conventions Link |
| --- | --- | --- |
| Silent fallback | Broad exception handling in `detect` returns normal no-motion output | `Living_Code_Review_Conventions.md` -> `Anti-Pattern: Silent Fallback` |
| Hot-path unnecessary copies | Multiple `.copy()` calls for debug arrays in frame path | `unnecessary .copy() calls are avoided` |
| Missing config fail-loud validation | Coercion/normalization instead of strict invariant checks | `configuration should be validated at startup` |

Suggested new anti-pattern entry for conventions:

- **Anti-Pattern: Unbounded Debug Retention in Hot Path**
- Bad: collecting and retaining full-frame debug artifacts without explicit budget/lifecycle.
- Good: opt-in debug capture with bounded artifact count/bytes and explicit ownership.
- Rule: debug observability must not change hot-path memory complexity or latency class.

---

## Contract Alignment Notes

The module spec states:

- `No frame-pixel storage` (module spec section 5)
- `Real-time capable` (module spec section 5)

Current implementation tension:

- debug image retention and recopy behavior effectively retain full-frame pixel artifacts beyond immediate per-call use.
- repeated hot-path allocations and pairwise merge scans create avoidable pressure against real-time guarantees.

This is not a functional correctness failure by itself, but it is a production-safety and latency-risk mismatch against stated non-functional requirements.

---

## Execution Priority

| Priority | Action |
| --- | --- |
| Immediate | Make runtime failures observable in `detect` and add failure telemetry |
| Immediate | Gate and bound debug image capture to remove full-frame copy churn |
| Next | Add centralized config validation with fail-fast behavior |
| Next | Add lifecycle eviction for camera state map |
| Scheduled | Replace pairwise merge with spatially partitioned merging and add performance regression tests |
