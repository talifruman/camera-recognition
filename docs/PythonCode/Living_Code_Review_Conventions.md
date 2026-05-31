# Living Code Review Conventions

Use this document during code review and when updating production code.
Keep it practical, extend it when repeated issues appear, and use the
detailed conventions below as the reference standard.

## 1. Code Review Checklist

### Readability

- names clearly explain intent
- complex conditions are extracted into named variables when that improves
    readability
- deep nesting is avoided
- unclear abbreviations are not introduced
- code is understandable without excessive comments
- comments explain why, not what

### Function Quality

- each function has a single responsibility
- return types stay consistent across paths
- functions are easy to test in isolation
- hidden side effects are avoided
- duplicated logic is extracted or justified
- abstraction levels are not mixed in the same function

### Ownership & Lifecycle

- ownership and lifecycle are explicit
- responsibility boundaries are clear
- shared mutable state ownership is obvious
- cleanup responsibility is defined
- resource ownership is not ambiguous

### Error Handling

- bare except blocks are not used
- failures are not silent
- error messages are meaningful and actionable
- fallback behavior is explicit
- exceptions are handled intentionally rather than broadly suppressed

### Performance / Real-Time Efficiency

- model loading does not happen in per-frame hot paths
- file I/O is not placed inside tight loops
- unnecessary `.copy()` calls are avoided
- NumPy and OpenCV views are preferred when safe
- repeated expensive operations are cached when appropriate
- the same frame is not repeatedly resized or converted without need
- validation is not duplicated across layers
- logging is minimized inside frame loops
- debug rendering is disabled by default
- locks are held for the shortest safe duration
- buffers and caches are bounded or cleaned up
- vectorized operations are preferred over Python loops when practical
- hot-path code is measurable and profileable
- unnecessary temporary allocations are avoided

#### Real-Time Rules

- any per-frame code is hot-path code
- hot-path code must avoid blocking operations
- hot-path code must avoid unnecessary allocations
- expensive resources should load once at initialization
- performance-sensitive code should be measurable

### Testing

- tests are deterministic
- sleep-based tests are avoided
- edge cases are covered
- fallback behavior is tested
- assertions are meaningful and specific
- tests do not depend on execution order

## 2. Severity Levels

### CRITICAL

Violates correctness, data integrity, concurrency safety, lifecycle safety,
real-time guarantees, or production stability.

Requires immediate fix before merge.

### MAJOR

Significant maintainability, readability, abstraction, coupling,
reviewability, or testability issue.

Should normally be fixed before merge.

### MINOR

Style, naming, formatting, or consistency issue.

Should be fixed unless intentionally deferred.

Examples:

```text
[CRITICAL] Hidden frame copy inside hot path
[MAJOR] Mixed abstraction levels in orchestration function
[MINOR] Unclear variable naming
```

## 3. Review Priorities

1. correctness
2. production safety
3. concurrency safety
4. real-time performance
5. maintainability
6. readability
7. style consistency

Style issues should not distract from correctness or production risks.

## 4. Review Decision

- APPROVED
- APPROVED WITH MINOR ISSUES
- CHANGES REQUIRED
- BLOCKED (critical issue)

Critical production-safety, correctness, concurrency, or hot-path violations
block approval.

## 5. Hot Path Rules

Code is hot-path code if it runs per frame, per detection, per face, per
camera loop, inside synchronization pipelines, or inside queue-processing
paths.

- hot-path code receives stricter review standards
- allocations should be minimized
- blocking operations should be avoided
- repeated transformations should be avoided
- expensive resources must load once
- hot-path code should be measurable and profileable

## 6. Reviewability

- understand it quickly
- debug it safely
- extend it safely
- review it without tracing hidden behavior
- identify ownership boundaries easily

## 7. Observability

- important failures are observable
- logs include enough context for debugging
- production issues are diagnosable
- silent operational failure is avoided
- hot paths expose timing/metrics when needed
- logs should not flood hot paths

## 8. Dependency Rules

- avoid circular dependencies
- avoid hidden cross-layer dependencies
- dependencies should flow predictably
- modules should not know unnecessary implementation details
- avoid dependency sprawl
- module boundaries should remain explicit

## 9. Configuration Rules

- avoid hardcoded paths
- avoid unexplained hardcoded thresholds
- environment-specific values belong in config
- configuration should be validated at startup
- defaults should be explicit
- invalid configuration should fail loudly

## 10. Debug Code Rules

- temporary debug code should be removable
- debug rendering should be configurable
- debug logging should not change behavior
- debug-only code should not leak into production paths
- debug artifacts should not accumulate silently

## 11. Anti-Patterns

Add new repeated review failures here using this format:
- `### Anti-Pattern: <name>`
- `Bad:`
- `Good:`
- `Rule:`

### Anti-Pattern: Silent Fallback

Bad:

```python
def load_threshold(path: str) -> float:
    try:
        return float(Path(path).read_text().strip())
    except Exception:
        return 0.5
```

Good:

```python
def load_threshold(path: str) -> float:
    try:
        return float(Path(path).read_text().strip())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Threshold file not found: {path}"
        ) from exc
```

Rule: Do not hide failures behind silent defaults; make fallbacks explicit,
documented, and tested because silent fallback shifts defects downstream.

### Anti-Pattern: Hidden Mutation

Bad:

```python
def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame[:] = frame / 255.0
    return frame
```

Good:

```python
def normalize_frame(frame: np.ndarray) -> np.ndarray:
    normalized_frame = frame.astype(np.float32) / 255.0
    return normalized_frame
```

Rule: Make mutation explicit in the API contract because hidden mutation
creates action at a distance and breaks caller assumptions.

### Anti-Pattern: Repeated Logic

Bad:

```python
if packet.pixel_format != "RGB":
    raise ValueError("Unsupported pixel format")
```

Good:

```python
def validate_pixel_format(pixel_format: str) -> None:
    if pixel_format != "RGB":
        raise ValueError("Unsupported pixel format")
```

Rule: Centralize shared logic once and reuse it across layers.

### Anti-Pattern: Boolean Flag Explosion

Bad:

```python
def process_frame(frame, save_debug=False, crop=False, normalize=False):
    pass
```

Good:

```python
class FrameProcessingConfig:
    def __init__(self, save_debug: bool, crop_enabled: bool,
                 normalize_enabled: bool):
        self.save_debug = save_debug
        self.crop_enabled = crop_enabled
        self.normalize_enabled = normalize_enabled
```

Rule: Replace clusters of boolean flags with explicit configuration objects
or separate functions.

### Anti-Pattern: Mixed Abstraction Levels

Bad:

```python
def run_pipeline(frame):
    validate_frame(frame)
    resized = cv2.resize(frame, (112, 112))
    log_result("resized frame")
    return {"image": resized, "status": "ok"}
```

Good:

```python
def run_pipeline(frame):
    validate_frame(frame)
    resized_frame = resize_frame(frame)
    return build_pipeline_result(resized_frame)
```

Rule: Keep orchestration code at orchestration level and low-level
processing in dedicated helpers.

### Anti-Pattern: Implicit Assumptions

Bad:

```python
def project_box(box, roi_offset):
    return {
        "x": box[0] + roi_offset[0],
        "y": box[1] + roi_offset[1],
    }
```

Good:

```python
def project_box(box: BoundingBox, roi_offset: Point) -> BoundingBox:
    validate_bounding_box(box)
    return {
        "x": box["x"] + roi_offset["x"],
        "y": box["y"] + roi_offset["y"],
        "width": box["width"],
        "height": box["height"],
    }
```

Rule: Make assumptions about inputs, coordinate systems, and state explicit
in types, validation, or documentation.

### Anti-Pattern: Giant Utility Files

Bad:

```python
# utils.py
def parse_config(...):
    pass

def resize_frame(...):
    pass

def compare_embeddings(...):
    pass
```

Good:

```python
# config_loader.py
def parse_config(...):
    pass

# frame_transformer.py
def resize_frame(...):
    pass
```

Rule: Keep helpers with the domain or layer that owns their behavior.

### Anti-Pattern: Deep Nested Conditions

Bad:

```python
if frame is not None:
    if frame.size > 0:
        if is_enabled:
            if has_target:
                process(frame)
```

Good:

```python
if frame is None:
    return
if frame.size == 0:
    return
if not is_enabled or not has_target:
    return
process(frame)
```

Rule: Prefer guard clauses and extracted conditions over deep nesting.

### Anti-Pattern: Repeated Coordinate Logic

Bad:

```python
face_x = roi_x + local_face_x
face_y = roi_y + local_face_y
```

Good:

```python
projected_face_box = coordinate_projector.project(
    local_face_box,
    roi_box,
)
```

Rule: Centralize coordinate projection in one well-tested owner.

### Anti-Pattern: Unnecessary Frame or Image Copies

Bad:

```python
frame = np.frombuffer(raw_bytes, dtype=np.uint8).copy()
cropped = image[y:y+h, x:x+w].copy()
```

Good:

```python
frame = np.frombuffer(raw_bytes, dtype=np.uint8)
cropped = image[y:y+h, x:x+w]
```

Rule: Do not copy frame or image data unless mutation, ownership, or
lifecycle requirements make the copy necessary.

### Anti-Pattern: Large try/except Blocks

Bad:

```python
try:
    frame = load_frame(path)
    metadata = load_metadata(path)
    detections = detector.detect(frame)
    persist_result(detections)
except Exception:
    return None
```

Good:

```python
frame = load_frame(path)
metadata = load_metadata(path)

try:
    detections = detector.detect(frame)
except DetectorError as exc:
    raise RuntimeError("Detection failed") from exc

persist_result(detections)
```

Rule: Keep try/except scopes narrow and aligned with one intentional
recovery boundary.

## 12. Code Smells

- large functions
- excessive booleans
- inconsistent return types
- repeated `.copy()`
- giant classes
- TODO accumulation
- duplicated validation
- excessive comments
- huge orchestrator methods
- repeated conversions
- circular imports
- over-coupled modules

## 13. Lessons Learned

- avoid mixing orchestration and low-level processing in the same function
- avoid bypassing shared transformation layers without a clear design reason
- centralize coordinate projection logic instead of re-deriving it per caller
- validate once and reuse results across layers
- avoid duplicate transformations of the same frame or image
- keep module responsibilities explicit and reviewable

## 14. How To Use During Reviews

- use this document during every code review
- when a repeated issue appears: fix the code, add a rule or anti-pattern,
  and add a lesson learned when it reveals broader knowledge

Reusable review prompts:

```text
Review this change against Living Code Review Conventions.
Focus on readability, correctness, testability, and production safety.
Call out hot-path allocation, duplicate validation, hidden mutation,
unclear ownership, and review-process gaps.
```

```text
Review this module as if it runs in a large real-time production system.
Identify anti-patterns, code smells, missing tests, and violations of the
real-time rules in this document.
```

## 15. Review Output Format

Every review should separate findings into:

- Critical Issues
- Major Issues
- Minor Issues
- Suggested Improvements
- Suggested Convention Additions

Each finding should explain why it matters, its impact on
maintainability/performance/safety, and the suggested fix direction.

## 16. When To Add New Rules

Add new rules when:

- the same issue appears repeatedly
- a bug escaped review
- a production issue was discovered
- a performance issue was discovered
- a confusing implementation slowed development
- the same review comment repeats frequently

## 17. Simplicity Rule

Prefer simple explicit code over premature abstraction.

Avoid:

- unnecessary wrappers
- speculative abstractions
- generic frameworks for one use case
- configuration systems without clear need
- abstraction layers without ownership clarity

## 18. Production Safety

- avoid silent state corruption
- avoid implicit shared mutable state
- fail loudly when correctness is uncertain
- validate destructive operations
- prefer explicit ownership and lifecycle rules
- avoid hidden cross-module side effects

## 19. Review Questions

- Is this code easy to modify safely?
- Can this fail silently?
- Is this doing unnecessary work?
- Is ownership clear?
- Is this safe under concurrency?
- Is this measurable/debuggable?
- Would another engineer understand this quickly?
- Does this introduce unnecessary coupling?
- Is this safe for real-time execution?

## 20. Convention Addition Style

New convention additions should:

- stay concise
- stay practical
- stay review-oriented
- avoid long theoretical explanations
- prioritize examples and explicit rules

## 21. Detailed Coding Conventions

The sections below preserve the full detailed coding guidance and examples
used as the reference standard during implementation and review.

### 1. Function Design

#### Function Length
- **Maximum 50 lines per function** (excluding docstrings)
- If a function exceeds 50 lines, break it into smaller, focused functions
- Aim for functions that do one thing and do it well (Single Responsibility Principle)

#### Function Naming
- Use **lowercase with underscores** (snake_case) for function names
- Start with a verb describing the action: `get_`, `set_`, `calculate_`, `validate_`, `process_`, `build_`, etc.
- Be descriptive and explicit about what the function does

**Good Examples:**
```python
def validate_email_format(email):
    pass

def calculate_total_cost(items):
    pass

def extract_frame_metadata(video_path):
    pass
```

**Bad Examples:**
```python
def process(x):  # Too vague
    pass

def do_stuff(data):  # Unclear intent
    pass

def f(a, b):  # Not descriptive
    pass
```

### 2. Variable Naming

#### General Rules
- Use **lowercase with underscores** (snake_case) for variable names
- Use **UPPERCASE with underscores** for constants
- Use **PascalCase** for class names only

#### Naming Conventions
- Use meaningful, descriptive names that explain the variable's purpose
- Avoid single-letter variables except for loop indices (`i`, `j`, `k`) or very temporary values
- Boolean variables should start with `is_`, `has_`, `should_`, `can_`, or `was_`

**Good Examples:**
```python
person_count = 10
is_valid_input = True
max_retries = 3
FRAME_WIDTH = 1920
user_email_address = "user@example.com"
detected_faces = []
```

**Bad Examples:**
```python
pc = 10  # Abbreviation unclear
valid = True  # Doesn't indicate it's a boolean
m = 3  # No meaning
frame_w = 1920  # Abbreviated constant name
email = "user@example.com"  # When it's actually an address
faces = []  # Unclear if this is raw input or processed result
```

### 3. Class Naming

- Use **PascalCase** for class names
- Use nouns or noun phrases
- Make the name self-explanatory

**Good Examples:**
```python
class PersonDirectory:
    pass

class FrameTransformationLayer:
    pass

class FaceDetectionEngine:
    pass
```

### 4. Code Structure

#### Imports
- Group imports in this order:
  1. Standard library imports
  2. Third-party imports
  3. Local application imports
- Separate groups with blank lines

**Example:**
```python
import os
import json
from pathlib import Path

import cv2
import numpy as np

from src.image_processing.face_detection import FaceDetector
from src.utils.validators import validate_path
```

#### Comments and Docstrings

##### Docstrings (Required for all functions and classes)
- Use triple quotes (`"""`)
- Include description, parameters, return value, and raises

**Example:**
```python
def validate_frame_quality(frame, min_brightness=50):
    """
    Validate if a frame meets quality requirements.
    
    Args:
        frame (numpy.ndarray): Input image frame to validate
        min_brightness (int): Minimum average brightness threshold (0-255)
    
    Returns:
        bool: True if frame meets quality standards, False otherwise
    
    Raises:
        ValueError: If frame is not a valid numpy array
        TypeError: If min_brightness is not an integer
    """
    if not isinstance(frame, np.ndarray):
        raise ValueError("Frame must be a numpy array")
    if not isinstance(min_brightness, int):
        raise TypeError("min_brightness must be an integer")
    
    return np.mean(frame) > min_brightness
```

##### Inline Comments
- Use sparingly - code should be self-explanatory
- Only explain *why*, not *what*
- Use `#` for single-line comments

**Good Comment:**
```python
# Adjust brightness compensation for low-light conditions
brightness_factor = 1.2 if avg_brightness < 100 else 1.0
```

**Bad Comment:**
```python
# Multiply brightness_factor by 1.2
brightness_factor = 1.2  # Not helpful
```

### 5. Code Formatting

#### Line Length
- Maximum **79 characters** per line (PEP 8 recommendation)
- Break long lines with backslash or parentheses

#### Spacing
- Use **4 spaces** for indentation (never tabs)
- 2 blank lines between top-level functions and classes
- 1 blank line between methods in a class

#### Operator Spacing
- One space around operators: `x = y + z`
- No space around `=` in keyword arguments: `func(param=value)`

**Good Example:**
```python
def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    delta_x = x2 - x1
    delta_y = y2 - y1
    distance = (delta_x ** 2 + delta_y ** 2) ** 0.5
    return distance
```

### 6. Error Handling

#### Exception Handling
- Catch specific exceptions, not generic `Exception`
- Use `try-except-finally` appropriately
- Provide meaningful error messages

**Good Example:**
```python
def load_configuration(config_path):
    """Load configuration from file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in config file: {config_path}")
```

**Bad Example:**
```python
def load_configuration(config_path):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except:  # Too generic
        print("Error")  # Unhelpful message
```

### 7. Function Parameters

#### Parameter Design
- Maximum **3-4 parameters** per function
- If more parameters needed, use a configuration object or dictionary
- Use type hints

**Good Example:**
```python
def process_frame(frame: np.ndarray, config: dict) -> dict:
    """Process frame with specified configuration."""
    pass
```

**Parameter Limits Exceeded - Refactor:**
```python
# ❌ Too many parameters
def process_frame(frame, width, height, blur_kernel, 
                  brightness, contrast, saturation, rotation):
    pass

# ✅ Use configuration object
def process_frame(frame: np.ndarray, config: ProcessingConfig):
    pass
```

### 8. Type Hints

- Use type hints for function parameters and return values
- Benefits: Better IDE support, easier debugging, self-documenting code

**Example:**
```python
from typing import List, Dict, Optional, Tuple

def find_faces_in_frame(
    frame: np.ndarray,
    confidence_threshold: float = 0.5
) -> List[Dict[str, any]]:
    """
    Detect faces in a frame.
    
    Args:
        frame: Input image frame
        confidence_threshold: Minimum confidence for detection (0-1)
    
    Returns:
        List of face detections with bounding boxes and confidence scores
    """
    pass

def get_config(config_path: str) -> Optional[dict]:
    """Load configuration, returning None if file doesn't exist."""
    pass
```

### 9. Conditional Logic

#### If Statements
- Keep conditions readable and simple
- Avoid deeply nested conditions (max 3 levels)
- Use early return to reduce nesting

**Good Example:**
```python
def validate_frame(frame):
    """Validate frame with early returns."""
    if frame is None:
        return False
    if frame.size == 0:
        return False
    if np.mean(frame) < 10:
        return False
    return True
```

**Bad Example:**
```python
def validate_frame(frame):
    if frame is not None:
        if frame.size > 0:
            if np.mean(frame) >= 10:
                return True
    return False
```

### 10. Loop Usage

#### For Loops
- Use descriptive variable names in loops
- Use enumerate when you need the index
- Prefer list comprehensions for simple transformations

**Good Example:**
```python
# Using enumerate
for idx, frame in enumerate(frames):
    process_frame(frame, idx)

# List comprehension
squared_values = [x ** 2 for x in values]

# Dictionary comprehension
name_to_id = {person.name: person.id for person in people}
```

**Bad Example:**
```python
# Unclear loop variable
for x in frames:
    process_frame(x, len(x))

# Inefficient - recalculating length
squared_values = []
for x in values:
    squared_values.append(x ** 2)
```

### 11. Constants

- Define constants at the module level
- Use ALL_CAPS with underscores
- Group related constants together with comments

**Example:**
```python
# Frame processing constants
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_RATE = 30

# Face detection constants
MIN_FACE_CONFIDENCE = 0.5
MIN_FACE_SIZE = 20

# File paths
DEFAULT_CONFIG_PATH = "config/default.yaml"
OUTPUT_DIRECTORY = "output/"
```

### 12. Avoiding Magic Numbers

#### What Are Magic Numbers?

**Magic numbers** are literal numeric values hardcoded directly in code without explanation or named context. They make code harder to understand, maintain, and debug.

**Bad Examples:**
```python
# ❌ What do 0, 1, 2, 3, 4, 5 represent?
def _build_landmarks(raw_landmarks: list[Point]) -> FaceLandmarks:
    if len(raw_landmarks) < 5:
        zero: Point = {"x": 0, "y": 0}
        padded = list(raw_landmarks) + [zero] * (5 - len(raw_landmarks))
    else:
        padded = raw_landmarks[:5]
    return FaceLandmarks(
        left_eye=padded[0],      # What is index 0?
        right_eye=padded[1],     # What is index 1?
        nose=padded[2],          # What is index 2?
        mouth_left=padded[3],    # What is index 3?
        mouth_right=padded[4],   # What is index 4?
    )

# ❌ What do these numbers mean?
if brightness < 100:
    factor = 1.2
elif brightness < 50:
    factor = 1.5
```

#### Solution: Define Named Constants

Define constants at the module level with clear names that explain the value's purpose.

**Good Example:**
```python
# Landmark configuration constants
EXPECTED_LANDMARK_COUNT = 5
DEFAULT_LANDMARK_COORDINATE = 0

# Landmark indices (matches SCRFD keypoint order)
LANDMARK_LEFT_EYE_INDEX = 0
LANDMARK_RIGHT_EYE_INDEX = 1
LANDMARK_NOSE_INDEX = 2
LANDMARK_MOUTH_LEFT_INDEX = 3
LANDMARK_MOUTH_RIGHT_INDEX = 4

# Brightness thresholds
BRIGHTNESS_DARK_THRESHOLD = 50
BRIGHTNESS_LOW_THRESHOLD = 100
BRIGHTNESS_DARK_FACTOR = 1.5
BRIGHTNESS_LOW_FACTOR = 1.2
BRIGHTNESS_NORMAL_FACTOR = 1.0


def _build_landmarks(raw_landmarks: list[Point]) -> FaceLandmarks:
    """Build canonical 5-point landmarks structure from raw detection."""
    if len(raw_landmarks) < EXPECTED_LANDMARK_COUNT:
        # Pad with zero landmarks
        zero: Point = {"x": DEFAULT_LANDMARK_COORDINATE, "y": DEFAULT_LANDMARK_COORDINATE}
        padded = list(raw_landmarks) + [zero] * (
            EXPECTED_LANDMARK_COUNT - len(raw_landmarks)
        )
    else:
        padded = raw_landmarks[:EXPECTED_LANDMARK_COUNT]
    
    return FaceLandmarks(
        left_eye=padded[LANDMARK_LEFT_EYE_INDEX],
        right_eye=padded[LANDMARK_RIGHT_EYE_INDEX],
        nose=padded[LANDMARK_NOSE_INDEX],
        mouth_left=padded[LANDMARK_MOUTH_LEFT_INDEX],
        mouth_right=padded[LANDMARK_MOUTH_RIGHT_INDEX],
    )


def calculate_brightness_factor(brightness: int) -> float:
    """Calculate compensation factor based on brightness level."""
    if brightness < BRIGHTNESS_DARK_THRESHOLD:
        return BRIGHTNESS_DARK_FACTOR
    elif brightness < BRIGHTNESS_LOW_THRESHOLD:
        return BRIGHTNESS_LOW_FACTOR
    return BRIGHTNESS_NORMAL_FACTOR
```

#### Rules for Constants

1. **All hardcoded numbers must have named constants** (except loop counters: `i`, `j`, `k`)
2. **Group related constants together** with descriptive comments
3. **Use descriptive constant names** that explain what the number represents
4. **Place constants at module level**, before class definitions
5. **For indices or counts**, use constants ending in `_INDEX`, `_COUNT`, or `_SIZE`
6. **For thresholds or limits**, use constants ending in `_THRESHOLD`, `_LIMIT`, or `_MAX`/`_MIN`

#### Exception: Loop Indices

Loop counters using single letters are acceptable:
```python
# ✅ GOOD - loop indices are exceptions
for i in range(len(items)):
    for j in range(len(items[i])):
        process(items[i][j])

# ✅ GOOD - enumerate with descriptive variable
for idx, item in enumerate(detections):
    process_detection(item, idx)
```

#### Exception: Temporary/Self-Explanatory Values

Very simple, self-explanatory contexts may skip constants (use judgment):
```python
# ✅ GOOD - intent is clear, no constant needed
if score > 0.5:
    mark_as_valid()

# ❌ BAD - purpose is unclear
if detection_score > 0.4733:  # Why 0.4733?
    accept_detection()
```

---

### 13. Function Decomposition Example

#### Problem: Function Over 50 Lines

```python
# ❌ BAD - Single function doing too much
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    detections = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detection
        faces = cascade.detectMultiScale(blurred)
        
        # Extraction
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            # ... more processing
        
        # Storage
        frames.append(frame)
        detections.append(faces)
    
    cap.release()
    return frames, detections
```

#### Solution: Decomposed Functions

```python
# ✅ GOOD - Each function has a single responsibility

def load_video_frames(video_path: str) -> List[np.ndarray]:
    """Load all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert frame to grayscale and apply blur."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def detect_faces(frame: np.ndarray) -> List[Tuple]:
    """Detect faces in a frame."""
    preprocessed = preprocess_frame(frame)
    return cascade.detectMultiScale(preprocessed)


def extract_face_regions(
    frame: np.ndarray,
    faces: List[Tuple]
) -> List[np.ndarray]:
    """Extract face regions from frame."""
    face_regions = []
    for (x, y, w, h) in faces:
        face_regions.append(frame[y:y+h, x:x+w])
    return face_regions


def process_video(video_path: str) -> Tuple[List, List]:
    """Orchestrate video processing pipeline."""
    frames = load_video_frames(video_path)
    detections = []
    
    for frame in frames:
        faces = detect_faces(frame)
        detections.append(faces)
    
    return frames, detections
```

### 14. Documentation Standards

#### Module-Level Docstring
Every Python file should start with a module docstring:

```python
"""
Module for face detection and recognition operations.

This module provides classes and functions for detecting faces in images
and video frames, extracting face features, and matching them against
known face databases.

Classes:
    FaceDetector: Main class for face detection
    FaceRecognizer: Face recognition and matching

Functions:
    validate_frame: Check if frame meets quality requirements
    extract_face_features: Extract embedding vectors from faces
"""

import os
import cv2
```

#### Class Docstring

```python
class FaceDetector:
    """Detect faces in images using cascade classifier.
    
    Attributes:
        model_path (str): Path to cascade classifier model
        scale_factor (float): Scale factor for multiscale detection
        min_neighbors (int): Minimum neighbors for detection validation
    """
    
    def __init__(self, model_path: str):
        """Initialize detector with cascade classifier model."""
        self.model_path = model_path
        self.cascade = cv2.CascadeClassifier(model_path)
```

### 15. Common Pitfalls to Avoid

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Functions > 50 lines | Hard to test, debug, reuse | Break into smaller functions |
| Vague variable names | Code is hard to understand | Use descriptive names |
| No type hints | IDE can't help, harder to debug | Add type hints |
| Bare `except:` | Catches all exceptions, hides bugs | Catch specific exceptions |
| Magic numbers | Unclear purpose and maintenance | Define named constants |
| Global state | Functions not independent, testing hard | Pass parameters instead |
| No docstrings | Documentation unclear, bad for teams | Document all functions/classes |
| Deeply nested code | Hard to read and test | Use early returns, extract functions |

### 16. Tools and Linters

#### Recommended Tools
- **Black**: Code formatter (enforces consistent style)
- **Pylint**: Code quality analyzer
- **Flake8**: PEP 8 compliance checker
- **mypy**: Static type checker

#### Setup Example
```bash
pip install black pylint flake8 mypy
```

#### Usage
```bash
black src/  # Format all Python files
flake8 src/  # Check PEP 8 compliance
mypy src/  # Check type hints
pylint src/  # Quality analysis
```

### 17. Testing Best Practices

#### Test Function Naming
- Start with `test_` prefix
- Be descriptive about what is being tested

```python
def test_validate_email_with_valid_address():
    """Test email validation with correct format."""
    assert validate_email("user@example.com") is True

def test_validate_email_with_invalid_format():
    """Test email validation rejects malformed address."""
    assert validate_email("invalid.email") is False
```

### 18. Memory Efficiency and Allocation

#### Avoid Unnecessary Copies

**Problem**: Copying data unnecessarily wastes memory and CPU time.

**Bad Examples:**
```python
# ❌ Unnecessary copy of numpy array
arr = np.frombuffer(bytes_data, dtype=np.uint8).reshape(height, width, 3).copy()

# ❌ Copying array slice (slice already creates view)
cropped = image_data[y:y+h, x:x+w, :].copy()

# ❌ List concatenation creates copy
new_list = list(old_list) + [element]
```

**Impact**: 
- 1080p RGB frame copy: ~6MB per operation
- At 30 FPS: 180 MB/s wasted allocation
- Garbage collection pressure and latency spikes

**Good Practices:**
```python
# ✅ Use read-only view (no copy)
arr = np.frombuffer(bytes_data, dtype=np.uint8).reshape(height, width, 3)

# ✅ Use array view for crops (no copy)
cropped = image_data[y:y+h, x:x+w, :]

# ✅ Use view (if contract is read-only)
view = original_array  # Share reference, not copy

# ✅ Only copy when modifying
if needs_modification:
    arr = original_array.copy()
```

#### Define Resource Cleanup

**Problem**: Resources left open or never freed cause memory leaks.

**Bad Examples:**
```python
# ❌ No way to cleanup frames from stopped cameras
class FrameStore:
    def __init__(self):
        self._frames_by_camera = {}  # Never cleaned
    # No cleanup method

# ❌ Files/connections never explicitly closed
def process_video(path):
    cap = cv2.VideoCapture(path)
    # ... process
    # Missing: cap.release()
```

**Good Practice:**
```python
# ✅ Explicit cleanup method
class FrameStore:
    def cleanup_camera(self, camera_id: str) -> None:
        """Free frame storage for specified camera."""
        with self._lock:
            if camera_id in self._frames_by_camera:
                del self._frames_by_camera[camera_id]

# ✅ Use context managers for resources
def process_video(path: str) -> None:
    with cv2.VideoCapture(path) as cap:
        # ... process
        pass  # Automatically released
```

---

### 19. Real-Time and Performance Considerations

#### Avoid Blocking Operations in Hot Paths

**Problem**: CPU-bound operations block real-time processing, causing latency spikes.

**Bad Examples:**
```python
# ❌ BLOCKING - PIL resize blocks thread (30-50ms per frame)
def get_frame(self, frame_data):
    pil_img = PILImage.fromarray(frame_data)
    resized = pil_img.resize((target_w, target_h))  # BLOCKS
    return np.asarray(resized)

# ❌ BLOCKING - Synchronous I/O blocks processing
def load_model(path):
    return pickle.load(open(path))  # BLOCKS until loaded
```

**Good Practice:**
```python
# ✅ Use faster libraries (OpenCV instead of PIL)
import cv2
def get_frame(self, frame_data):
    resized = cv2.resize(frame_data, (target_w, target_h))  # 5-10x faster
    return resized

# ✅ Load expensive resources at initialization
def __init__(self, model_path: str):
    self.model = pickle.load(open(model_path))  # Load once, not per-frame
```

#### Latency Budget

**Principle**: Know your latency budget and stay within it.

**Example:**
```python
# REAL-TIME REQUIREMENT: 30 FPS → 33ms per frame
# REAL-TIME REQUIREMENT: 60 FPS → 16.7ms per frame

# ❌ BAD - exceeds budget
def process_frame(frame):
    # Takes 50ms total:
    # PIL resize: 30-50ms
    # PIL convert: 10-20ms
    # → EXCEEDS 33ms budget, violates real-time

# ✅ GOOD - within budget
def process_frame(frame):
    # Takes 5-10ms total:
    # OpenCV resize: 2-3ms
    # NumPy operations: 2-5ms
    # → WITHIN 33ms budget ✓
```

---

### 20. Resource Management and Thread Safety

#### Use Locks Appropriately

**Problem**: Improper locking causes contention or race conditions.

**Bad Examples:**
```python
# ❌ Lock held too long
def get(self, key):
    with self._lock:
        value = self._dict.get(key)
        processed = expensive_process(value)  # Blocks others
        return processed

# ❌ Using RLock (recursive) when Lock (non-recursive) would do
self._lock = threading.RLock()  # Slower than Lock
```

**Good Practice:**
```python
# ✅ Hold lock only when necessary
def get(self, key):
    with self._lock:
        value = self._dict.get(key)  # Short operation
    # Lock released before expensive operation
    processed = expensive_process(value)
    return processed

# ✅ Use non-recursive Lock when possible
self._lock = threading.Lock()  # Faster, simpler
```

#### Avoid Duplicate Operations

**Rule**: Don't repeat validation or computation if already done.

**Bad Examples:**
```python
# ❌ FramePacketValidator checks format
class FramePacketValidator:
    def validate(self, packet):
        if packet.pixel_format != "RGB":
            raise InvalidFramePacketFormatError()

# ❌ Then BaseImageBuilder checks SAME things again
class BaseImageBuilder:
    def build(self, packet):
        if packet.pixel_format != "RGB":  # DUPLICATE
            raise InvalidFramePacketFormatError()
```

**Good Practice:**
```python
# ✅ Validate once, reuse result
class Pipeline:
    def ingest(self, packet):
        self._validator.validate(packet)  # Validate once
        image = self._builder.build(packet)  # Don't re-validate
```

---

### 21. Caching and Object Pooling

#### Cache Expensive Computations

**Rule**: Avoid recomputing the same result multiple times.

**Bad Examples:**
```python
# ❌ PIL resize done repeatedly for same input
def get_frame(self, frame, output_size):
    pil_img = PILImage.fromarray(frame)
    resized = pil_img.resize(output_size)  # Recomputed every call
    return np.asarray(resized)

# ❌ Constants recalculated in loops
for frame in frames:
    resized = frame * (1.0 / 255.0)  # Division computed every iteration
```

**Good Practice:**
```python
# ✅ Cache expensive operations
class FrameConverter:
    def __init__(self):
        self._resize_cache = {}
    
    def get_frame(self, frame, output_size):
        cache_key = (id(frame), output_size)
        if cache_key in self._resize_cache:
            return self._resize_cache[cache_key]
        
        pil_img = PILImage.fromarray(frame)
        resized = np.asarray(pil_img.resize(output_size))
        self._resize_cache[cache_key] = resized
        return resized

# ✅ Define constants, don't recalculate
NORMALIZATION_FACTOR = 1.0 / 255.0
for frame in frames:
    normalized = frame * NORMALIZATION_FACTOR
```

---

### 22. Common Memory and Performance Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Unnecessary `.copy()` | Wastes 40-50% memory | Use views, only copy when modifying |
| No resource cleanup | Memory leaks in long-running services | Add explicit cleanup methods |
| Blocking operations in hot path | Violates real-time requirements | Use faster libs (OpenCV vs PIL) |
| Lock held too long | Thread contention, blocking | Minimize lock scope |
| Duplicate validation | CPU waste | Validate once, reuse result |
| No caching | Repeated expensive computation | Cache expensive operations |
| Creating objects in loops | GC pressure, latency spikes | Use object pooling or reuse |
| Unbounded data structures | Memory leak | Add cleanup and limits |
| Using RLock when Lock suffices | Slower performance | Use Lock unless recursion needed |
| No performance profiling | Doesn't know actual bottlenecks | Profile and measure real impact |

---

### 23. Summary

| Aspect | Rule |
|--------|------|
| Function Length | ≤ 50 lines (excl. docstrings) |
| Max Parameters | 3-4, use config objects for more |
| Naming | snake_case for functions/vars, PascalCase for classes |
| Line Length | ≤ 79 characters |
| Indentation | 4 spaces (never tabs) |
| Type Hints | Required for all function signatures |
| Docstrings | Required for all functions and classes |
| Comments | Explain *why*, not *what* |
| Error Handling | Specific exceptions with meaningful messages |
| Code Nesting | Maximum 3 levels, prefer early returns |
| Memory Copies | Use views not copies; only copy when modifying |
| Resource Cleanup | Define cleanup methods; use context managers |
| Real-Time Latency | Know latency budget; avoid blocking operations |
| Lock Management | Hold locks minimally; use Lock not RLock when possible |
| Duplicate Operations | Validate/compute once; reuse results |
| Caching | Cache expensive computations; define constants |
| Performance Profiling | Profile hot functions; measure real impact |


