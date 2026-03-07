import os
import yaml
from typing import Optional

from milon_engine.exercises.base import Exercise
from milon_engine.core.models import ExerciseResult


class PushUp(Exercise):
    """Rep counter for push-ups.

    Uses shoulder-elbow-wrist angle combined with vertical shoulder
    displacement to detect down/up transitions.
    """

    # Time the athlete must hold the aligned position before counting starts
    MIN_ALIGN_SEC: float = 0.7
    MAX_ALIGN_SEC: float = 3.0

    def __init__(
        self, config: dict, fps: float = 30.0, calibration_path: Optional[str] = None
    ):
        super().__init__(config, fps)

        self._down_shift_min: float = 0.05
        self._up_shift_max: float = 0.02
        self.align_counter: int = 0
        self.has_been_up: bool = False

        calib_path = (
            calibration_path
            or f"outputs/calibration/{config['exercise_name'].lower()}.yaml"
        )

        if os.path.exists(calib_path):
            with open(calib_path, "r") as f:
                calib = yaml.safe_load(f) or {}

            self.down_threshold = float(calib["down_threshold"])
            self.up_threshold = float(calib["up_threshold"])
            self._down_shift_min = float(calib["down_shift_delta"])
            self._up_shift_max = float(calib["up_shift_delta"])
            print(f"Loaded push-up calibration from: {calib_path}")

    def evaluate(self, landmarks: list) -> ExerciseResult:
        """Process landmarks and return updated exercise state."""
        min_align_frames = max(1, int(self.fps * self.MIN_ALIGN_SEC))
        max_align_frames = max(min_align_frames, int(self.fps * self.MAX_ALIGN_SEC))

        angle, pts, side = self.choose_side(landmarks)

        if angle is None or pts is None or side is None:
            self.system_stage = "waiting"
            self.stage = None
            return ExerciseResult(
                angle=None,
                rep_count=self.rep_count,
                stage=self.stage,
                system_stage=self.system_stage,
            )

        ref_y = self.get_reference_y(landmarks, side)
        if ref_y is None:
            return ExerciseResult(
                angle=float(angle),
                rep_count=self.rep_count,
                stage=self.stage,
                system_stage=self.system_stage,
                side=self.side_selected,
            )

        # Calibration not loaded yet — skip state machine
        if self.up_threshold is None or self.down_threshold is None:
            return ExerciseResult(
                angle=float(angle),
                rep_count=self.rep_count,
                stage=self.stage,
                system_stage=self.system_stage,
                side=self.side_selected,
            )

        down_thr, up_thr = self._compute_angle_thresholds()

        # --- State machine ---
        if self.system_stage == "waiting":
            if angle > max(120.0, up_thr - 5.0):
                self.system_stage = "aligning"
                self.align_counter = 0

        elif self.system_stage == "aligning":
            if angle > max(120.0, up_thr - 5.0):
                self.align_counter += 1
            else:
                self.align_counter = 0

            if self.baseline_axis_val is None:
                self.baseline_axis_val = ref_y
            else:
                self.baseline_axis_val = 0.9 * self.baseline_axis_val + 0.1 * ref_y

            if (
                self.align_counter > min_align_frames
                or self.align_counter > max_align_frames
            ):
                self.baseline_axis_val = ref_y
                self.system_stage = "ready"
                self.stage = "up"

        elif self.system_stage == "ready":
            if angle < min(120.0, down_thr + 10.0):
                self.system_stage = "counting"

        elif self.system_stage == "counting":
            shift = ref_y - (
                self.baseline_axis_val if self.baseline_axis_val is not None else ref_y
            )

            if angle < down_thr and shift > self._down_shift_min:
                self.stage = "down"

            if self.stage == "down" and angle > up_thr and shift < self._up_shift_max:
                self.rep_count += 1
                self.stage = "up"
                print(f"Push-up rep {self.rep_count}")

            if self.baseline_axis_val is not None:
                self.baseline_axis_val = 0.995 * self.baseline_axis_val + 0.005 * ref_y

        return ExerciseResult(
            angle=float(angle),
            rep_count=self.rep_count,
            stage=self.stage,
            system_stage=self.system_stage,
            side=self.side_selected,
        )

    def reset(self) -> None:
        super().reset()
        self.align_counter = 0
        self.has_been_up = False
