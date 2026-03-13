import os
import yaml
from typing import Optional

from milon_engine.exercises.base import Exercise
from milon_engine.core.models import ExerciseResult


class LegRaise(Exercise):
    """Rep counter for leg raises.

    Uses hip-knee-ankle angle to detect up/down transitions.
    The first descent into the down position counts as rep 1.
    """

    MIN_ALIGN_SEC: float = 0.5
    MAX_ALIGN_SEC: float = 2.0

    def __init__(
        self, config: dict, fps: float = 30.0, calibration_path: Optional[str] = None
    ):
        super().__init__(config, fps)

        self.align_counter: int = 0
        self.has_started: bool = False
        self.has_been_up: bool = False
        self.baseline_angle: Optional[float] = None

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
            self.down_threshold = calib.get("down_threshold")
            self.up_threshold = calib.get("up_threshold")
            if self.down_threshold is not None:
                self.down_threshold = float(self.down_threshold)
            if self.up_threshold is not None:
                self.up_threshold = float(self.up_threshold)
            print(f"Loaded leg-raise calibration from: {calib_path}")

    def evaluate(self, landmarks: list) -> ExerciseResult:
        """Process landmarks and return updated exercise state."""
        min_align_frames = max(1, int(self.fps * self.MIN_ALIGN_SEC))

        angle, pts, side = self.choose_side(landmarks)

        if angle is None or side is None:
            self.system_stage = "waiting"
            return ExerciseResult(
                angle=None,
                rep_count=self.rep_count,
                stage=self.stage,
                system_stage=self.system_stage,
            )

        # Calibration not loaded yet — skip state machine
        if self.up_threshold is None or self.down_threshold is None:
            return ExerciseResult(
                angle=float(angle),
                rep_count=self.rep_count,
                stage=self.stage or "waiting",
                system_stage=self.system_stage,
                side=self.side_selected,
            )

        down_thr, up_thr = self._compute_angle_thresholds()

        # --- Alignment state machine ---
        if self.system_stage == "waiting":
            if angle > up_thr - 5:
                self.system_stage = "aligning"
                self.align_counter = 0

        elif self.system_stage == "aligning":
            if angle > up_thr - 5:
                self.align_counter += 1
            else:
                self.align_counter = 0

            if self.align_counter > min_align_frames:
                self.system_stage = "ready"
                self.baseline_angle = angle

        elif self.system_stage == "ready":
            if angle < down_thr:
                self.system_stage = "counting"

        # Not counting yet — return early
        if self.system_stage != "counting":
            return ExerciseResult(
                angle=float(angle),
                rep_count=self.rep_count,
                stage=self.stage or "waiting",
                system_stage=self.system_stage,
                side=self.side_selected,
            )

        # --- Rep counting ---
        is_down = angle < down_thr
        is_up = angle > up_thr

        if is_up:
            self.stage = "up"
            self.has_been_up = True

        elif is_down:
            if self.stage != "down":
                self.stage = "down"

                if not self.has_started:
                    # First descent counts immediately
                    self.rep_count += 1
                    self.has_started = True
                    self.has_been_up = True
                elif self.has_been_up:
                    # Full up→down cycle completed
                    self.rep_count += 1
                    self.has_been_up = False
                    print(f"Leg-raise rep {self.rep_count}")

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
        self.baseline_angle = None
        self.has_started = False
        self.has_been_up = False
