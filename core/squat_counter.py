from .exercise_base import ExerciseBase
import os, yaml

class SquatCounter(ExerciseBase):


    def __init__(self, config: dict, calibration_path=None):
        super().__init__(config)

        calib_path = calibration_path or f"outputs/calibration/{config['exercise_name'].lower()}.yaml"

        # load calibration thresholds
        if os.path.exists(calib_path):
            with open(calib_path, "r") as f:
                calib = yaml.safe_load(f)

            self.down_threshold   = float(calib["down_threshold"])
            self.up_threshold     = float(calib["up_threshold"])
            self.down_shift_delta = float(calib["down_shift_delta"])
            self.up_shift_delta   = float(calib["up_shift_delta"])
            print(f"oaded squat calibration from: {calib_path}")

        # apply angle tolerance from config yaml
        tol = self.config.get("angle_tolerance", 0)
        self.down_threshold *= (1 + tol / 100)
        self.up_threshold   *= (1 - tol / 100)

        self.align_counter = 0

    # main
    def update(self, landmarks, fps):
        # angle and side detection
        angle, pts, side = self.choose_side(landmarks)
        if angle is None or pts is None:
            self.system_stage = "waiting"
            return {"angle": None}

        # reference Y from yaml
        ref_y = self.get_reference_y(landmarks, side)
        if ref_y is None:
            return {"angle": float(angle)}

        # set baseline position
        if self.system_stage == "waiting":
            if angle > self.up_threshold - 10:
                self.system_stage = "aligning"
                self.align_counter = 0

        elif self.system_stage == "aligning":
            if angle > self.up_threshold - 10:
                self.align_counter += 1
            else:
                self.align_counter = 0

            if self.align_counter > int(fps * 0.7):
                self.baseline_axis_val = ref_y
                self.system_stage = "ready"
                print("Ready to start")

        # counting on descent
        elif self.system_stage == "ready":
            if angle < self.down_threshold:
                self.system_stage = "counting"

        # check vertical shift
        elif self.system_stage == "counting":
            shift = ref_y - self.baseline_axis_val if self.baseline_axis_val else 0
            axis_shift_ok_down = shift > self.down_shift_delta
            axis_shift_ok_up   = shift < self.up_shift_delta

            # repetitions
            is_down = (angle < self.down_threshold) and axis_shift_ok_down
            is_up   = (angle > self.up_threshold)   and axis_shift_ok_up

            if is_down:
                self.stage = "down"

            if is_up and self.stage == "down":
                self.rep_count += 1
                self.stage = "up"
                print(f"✅ Rep {self.rep_count} detected")

            # smooth baseline correction
            if self.baseline_axis_val:
                self.baseline_axis_val = 0.995 * self.baseline_axis_val + 0.005 * ref_y

        # Debug
        shift_val = (ref_y - self.baseline_axis_val) if self.baseline_axis_val else 0
        print(
            f"[DEBUG][SQUAT] Stage:{self.stage}, System:{self.system_stage}, "
            f"Angle:{angle:.2f}°, ↓Thr:{self.down_threshold:.1f}, ↑Thr:{self.up_threshold:.1f}, "
            f"Shift:{shift_val:+.3f}, Reps:{self.rep_count}"
        )

        return {
            "angle": float(angle),
            "ref_y": float(ref_y),
            "rep_count": int(self.rep_count),
            "stage": self.stage,
            "system_stage": self.system_stage,
            "side": self.side_selected or side
        }
