# Motion Detection Config Gap Analysis — REAL Mode Readiness

**Date:** 2026-05-13  
**Scope:** Configuration sufficiency for switching `MotionDetectionManager` from stub to REAL.  
**Sources:**
- Spec: `doc/image_processing_service/motion_detection.md` §6.2, §9.1
- Implementation: `src/image_processing/motion_detection/module.py`

---

## 1. Existing Config Fields (Spec §9.1 + Implementation)

```python
@dataclass(slots=True)
class MotionDetectionConfig:
    motion_threshold: int = 25             # per-pixel abs-diff threshold
    motion_fraction_threshold: float = 0.05  # fraction of changed pixels required
    min_bbox_area: int = 100               # minimum bbox area; smaller boxes discarded
```

### What each field controls

| Field | Consumed By | Effect |
|---|---|---|
| `motion_threshold` | `FrameDifferencingMotionDetector` | Pixel classified as "changed" if `absdiff >= motion_threshold` |
| `motion_fraction_threshold` | `MotionDecisionPolicy` | `detected = True` only if `motion_fraction >= threshold AND bboxes non-empty` |
| `min_bbox_area` | `FrameDifferencingMotionDetector` | Contour bbox discarded if `width * height < min_bbox_area` |

All three fields are injected at construction. No per-call runtime overrides exist and none are needed.

---

## 2. Missing Config Fields Required for REAL

The current spec does not define fields for the following REAL-mode concerns:

### 2.1 `implementation_type` / stub vs. real mode switch

**Status: Missing from spec and config.**

Stub vs. real selection is currently done purely via constructor injection (`algorithm` parameter). There is no config-level `implementation_type` field. For REAL production use, this is acceptable — callers inject `FrameDifferencingMotionDetector` (or omit it, since it is the default). However, there is no config-driven way to switch implementations without code changes. If runtime or environment-based selection is required, this field is a gap.

**Verdict:** Gap for config-driven switching. Acceptable for manual injection model.

---

### 2.2 Gaussian blur pre-processing (`blur_kernel_size`)

**Status: Missing — implicit zero (no blur).**

`FrameDifferencingMotionDetector.measure()` applies `cv2.absdiff` directly on raw grayscale arrays with no pre-processing. Real camera footage contains sensor noise, compression artifacts, and fine-grained texture that produces spurious pixel differences below any reasonable `motion_threshold`. Without a blur step, these produce fragmented bboxes and elevated `motion_fraction` on static scenes.

A `blur_kernel_size: int = 0` field (0 = disabled, odd values ≥ 3 apply `cv2.GaussianBlur`) is not in the spec or config. The threshold default of 25 is the only current noise mitigation, but it does not address spatial fragmentation.

**Verdict:** Missing. Relevant for noise-heavy real footage. Requires both spec update and implementation.

---

### 2.3 Morphological post-processing parameters

**Status: Missing — implicit none.**

After `cv2.threshold`, the binary mask on real footage is typically fragmented — individual motion regions appear as many small disconnected components. A dilation pass consolidates them into fewer, larger contours that better represent the actual moving object.

No morphological parameters exist in spec or config:
- `dilation_kernel_size: int` — size of the structuring element
- `dilation_iterations: int` — number of dilation passes

**Verdict:** Missing. Relevant for real footage where fragmented masks reduce bbox quality.

---

### 2.4 ROI exclusion zones

**Status: Not in scope per spec §1 ("Out of Scope"); missing from config.**

The module currently processes the full frame. Static camera scenes with dynamic backgrounds (waving foliage, road traffic outside a monitored area) can trigger false positives. ROI exclusion would require the caller to crop the input before passing it in — the module's `get_input_contract()` already states `ResizePolicy.NONE` and returns native dimensions, meaning ROI handling belongs upstream in the FTL.

**Verdict:** Architecturally out of scope for this module. No config field needed here. The caller (RPM/FTL) is responsible for providing a cropped region if ROI exclusion is needed.

---

### 2.5 Downscale target resolution

**Status: Not in spec or config.**

The module always operates at native input resolution. For high-resolution cameras this can be costly. A configurable `processing_width` / `processing_height` would allow downscaling before differencing with results mapped back to original coordinates.

**Verdict:** Missing, but not required for a basic REAL switch. Relevant as a future performance optimization.

---

## 3. Hardcoded Runtime Values That Should Move to Config

These values are embedded directly in `FrameDifferencingMotionDetector.measure()` and are not exposed in `MotionDetectionConfig`:

| Hardcoded Value | Location | Notes |
|---|---|---|
| `cv2.RETR_EXTERNAL` | `cv2.findContours(mask, cv2.RETR_EXTERNAL, ...)` | External contours only; correct for motion use-case but untunable |
| `cv2.CHAIN_APPROX_SIMPLE` | `cv2.findContours(..., cv2.CHAIN_APPROX_SIMPLE)` | Memory-efficient approximation; no practical reason to change |
| Gaussian blur kernel = 0 (none) | No blur call exists | Real footage noise mitigation is absent |
| Morphological ops = none | No morphology call exists | No dilation/erosion to consolidate mask fragments |

The contour retrieval and approximation flags are low-priority; they are sensible defaults. The absence of blur and morphological ops is the material gap.

---

## 4. Is the Current Config Sufficient for REAL Mode?

**Verdict: Minimally sufficient to run; not robust for production footage.**

| Criterion | Status |
|---|---|
| Can `MotionDetectionManager` run in REAL mode with current config? | **Yes** — `FrameDifferencingMotionDetector` is the default; no code changes needed |
| Will default values (25 / 0.05 / 100) work on real camera footage? | **Likely needs tuning** — sensor noise and scene specifics affect all three |
| Is noise/fragmentation controllable via config? | **No** — no blur or morphology config; only `motion_threshold` provides indirect noise control |
| Is cold-start handled? | **Yes** — module is stateless; cold-start is an FTL/RPM concern entirely outside this module |
| Is background subtraction applicable? | **No** — `FrameDifferencingMotionDetector` is a frame-differencing algorithm; background subtraction (MOG2/KNN) is a different algorithm family not in scope |
| Are input image expectations met? | **Yes** — `GRAYSCALE_UINT8_HWC` is enforced by `InputValidator`; the contract is sufficient |

**Blocking gaps for REAL switch:** None — the module can run REAL today.  
**Quality gaps:** `blur_kernel_size` and morphological parameters are absent; their omission means real footage quality depends entirely on tuning `motion_threshold`.

---

## 5. Recommended Final `MotionDetectionConfig`

To fully cover REAL-mode needs without breaking the current API:

```python
@dataclass(slots=True)
class MotionDetectionConfig:
    # --- existing fields (no change) ---
    motion_threshold: int = 25
    motion_fraction_threshold: float = 0.05
    min_bbox_area: int = 100

    # --- recommended additions for REAL ---
    blur_kernel_size: int = 0
    # 0 = disabled; odd values >= 3 apply GaussianBlur before absdiff.
    # Reduces sensor noise and fine texture false-positives.

    dilation_kernel_size: int = 0
    # 0 = disabled; odd values >= 3 apply binary mask dilation after threshold.
    # Consolidates fragmented motion mask regions into contiguous blobs.

    dilation_iterations: int = 1
    # Number of dilation passes when dilation_kernel_size > 0.
```

The `0 = disabled` convention preserves backward compatibility: existing code using the current 3-field config continues to work without change, and defaults produce identical output to the current implementation.

An `implementation_type` field is **not recommended** — algorithm selection via constructor injection is cleaner and already works. A string flag in config adds no value over injecting the right class.

---

## 6. Open Questions / Unresolved Gaps

1. **Default value tuning**: The current defaults (25 / 0.05 / 100) were designed for grayscale normalized frames. Actual camera resolution and lighting conditions will affect whether these values suppress false positives without suppressing true motion. Calibration against real footage is required before production use.

2. **Blur field belongs in spec §9.1**: If `blur_kernel_size` is added to the implementation, the spec must be updated in §9.1 (Configuration Parameters) and §8.3 (MotionDetectionAlgorithm) to describe the new pre-processing step and its integration into `FrameDifferencingMotionDetector`.

3. **Morphology field belongs in spec §6.2**: The spec's pipeline description in §6.2 lists 5 steps (absdiff → threshold → contours → bbox → area filter). Dilation would insert between threshold and contours. The spec must document this step before the implementation adds it.

4. **Downscale for high-res cameras**: If cameras deliver 4K or higher resolution, the per-frame absdiff computation will be expensive. A configurable processing resolution (downscale before differencing, no coordinate remapping needed since output coordinates are ROI-local) should be considered as a follow-up.

5. **Stub selection in REAL environment**: There is currently no guard preventing `StubMotionDetectionAlgorithm` from being injected in a production environment. Whether this needs an assertion or documentation convention is unresolved.
