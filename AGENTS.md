# AGENTS.md — Milon Exercise Engine

## Package overview

`milon_engine` is a Python package that wraps MediaPipe pose estimation into a clean three-component pipeline for real-time exercise rep counting. The package is front-end agnostic: the same `FrameProcessor` works whether the caller is a Streamlit app, a Raspberry Pi capture loop, or a plain `cv2.imshow` window.

---

## Main components

### 1. `PoseEstimator` (`core/pose_estimator.py`)
MediaPipe wrapper. Takes a raw BGR frame, converts it to RGB internally, runs inference, and returns two things:

- `landmark_list` — flat list of `NormalizedLandmark` objects used by the exercise logic
- `raw_results` — the full MediaPipe results object passed to the Visualizer for skeleton drawing

**Public API:** `detect(frame) -> (landmark_list | None, raw_results)`

---

### 2. `Exercise` (`exercises/base.py` + subclasses)
Stateful rep counter for a specific exercise. Receives landmarks per frame and runs a state machine (`waiting → aligning → ready → counting`) to detect up/down transitions and increment the rep count.

- **Base class** provides shared geometry helpers (`choose_side`, `calculate_angle`, `get_reference_y`) and the `@abstractmethod evaluate()` contract.
- **Subclasses:** `Squat`, `PushUp`, `LegRaise`
- Configuration (landmark names, reference point, angle tolerance) is loaded from a YAML file in `milon_engine/configs/`.
- Calibration (down/up thresholds) is loaded from a separate YAML in `outputs/calibration/`.

**Public API:** `evaluate(landmarks) -> ExerciseResult`

---

### 3. `Visualizer` (`core/visualizer.py`)
Stateless drawing layer. Takes a frame, the raw MediaPipe results, and an `ExerciseResult`, composes all visual layers, and returns an annotated copy of the frame. The original frame is never mutated.

Layers drawn (all under the single public method):
- Pose skeleton (via `mp_drawing`)
- Rep count / stage / angle panel (top-left)
- System status banner (bottom)
- "No person detected" overlay when landmarks are absent

**Public API:** `render(frame, raw_results, result) -> annotated_frame`

---

### 4. `FrameProcessor` (`core/frame_processor.py`)
Orchestrator. Holds references to all three components and exposes a single method that runs the full pipeline in order.

**Public API:** `process_frame(frame) -> annotated_frame`

---

### 5. `ExerciseResult` (`core/models.py`)
Typed dataclass that is the contract between `Exercise` and `Visualizer`.

```
angle        float | None   — current joint angle in degrees
rep_count    int            — cumulative rep count
stage        str | None     — "up" | "down" | None
system_stage str            — "waiting" | "aligning" | "ready" | "counting"
side         str | None     — "left" | "right" | None
```

---

## Frame flow

```
raw BGR frame
      │
      ▼
PoseEstimator.detect(frame)
      │
      ├── landmark_list ──► Exercise.evaluate(landmarks)
      │                             │
      │                             ▼
      │                       ExerciseResult
      │                             │
      └── raw_results ─────────────►┤
                                    ▼
                          Visualizer.render(frame, raw_results, result)
                                    │
                                    ▼
                             annotated_frame
                          (returned to caller)
```

The caller (Streamlit / cv2 / Raspberry Pi script) is solely responsible for **capturing** frames and **displaying** the returned annotated frame. No display logic lives inside the package.

---

## Adding a new exercise

1. Create `milon_engine/configs/<name>.yaml` with `landmarks`, `reference_point`, and `angle_tolerance`.
2. Subclass `Exercise` in `milon_engine/exercises/<name>.py`, implement `evaluate(landmarks) -> ExerciseResult`.
3. Export it from `milon_engine/exercises/__init__.py`.
4. Provide a calibration YAML at `outputs/calibration/<name>.yaml`.
