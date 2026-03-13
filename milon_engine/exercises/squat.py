import os
import yaml
from typing import Optional

from milon_engine.exercises.base import Exercise
from milon_engine.core.models import ExerciseResult


class Squat(Exercise):
    """Rep counter for squats.

    Uses hip-knee-ankle angle combined with vertical hip displacement to
    detect down/up transitions.
    """

    def __init__(
        self, config: dict, fps: float = 30.0, calibration_path: Optional[str] = None
    ):
        super().__init__(config, fps)

        self.down_shift_delta: float = 0.0
        self.up_shift_delta: float = 0.0
        self.align_counter: int = 0

        _default_calib = os.path.join(
            os.path.dirname(__file__),
            "..",
            "calibration",
            f"{config['exercise_name'].lower()}.yaml",
        )
        calib_path = calibration_path or os.path.normpath(_default_calib)

        if os.path.exists(calib_path):
            with open(calib_path, "r") as f:
                calib = yaml.safe_load(f)

            self.down_threshold = float(calib["down_threshold"])
            self.up_threshold = float(calib["up_threshold"])
            self.down_shift_delta = float(calib["down_shift_delta"])
            self.up_shift_delta = float(calib["up_shift_delta"])
            print(f"Loaded squat calibration from: {calib_path}")

            # Apply angle tolerance from config YAML
            tol = self.config.get("angle_tolerance", 0)
            self.down_threshold *= 1 + tol / 100
            self.up_threshold *= 1 - tol / 100

    def evaluate(self, landmarks: list) -> ExerciseResult:
        """Process landmarks and return updated exercise state."""
        angle, pts, side = self.choose_side(landmarks)

        if angle is None or pts is None or side is None:
            self.system_stage = "waiting"
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

        up_thr = self.up_threshold
        down_thr = self.down_threshold

        # --- State machine ---
        if self.system_stage == "waiting":
            if angle > up_thr - 10:
                self.system_stage = "aligning"
                self.align_counter = 0

        elif self.system_stage == "aligning":
            if angle > up_thr - 10:
                self.align_counter += 1
            else:
                self.align_counter = 0

            if self.align_counter > int(self.fps * 0.7):
                self.baseline_axis_val = ref_y
                self.system_stage = "ready"
                print("Squat: ready to start")

        elif self.system_stage == "ready":
            if angle < down_thr:
                self.system_stage = "counting"

        elif self.system_stage == "counting":
            shift = (
                ref_y - self.baseline_axis_val
                if self.baseline_axis_val is not None
                else 0.0
            )
            is_down = (angle < down_thr) and (shift > self.down_shift_delta)
            is_up = (angle > up_thr) and (shift < self.up_shift_delta)

            if is_down:
                self.stage = "down"
            if is_up and self.stage == "down":
                self.rep_count += 1
                self.stage = "up"
                print(f"Squat rep {self.rep_count}")

            # Smooth baseline drift correction
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
