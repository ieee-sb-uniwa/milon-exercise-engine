from .exercise_base import ExerciseBase
import os, yaml

class PushUpCounter(ExerciseBase):


    def __init__(self, config: dict, calibration_path=None):
        super().__init__(config)
        calib_path = calibration_path or f"outputs/calibration/{config['exercise_name'].lower()}.yaml"

        #  defaults fallbacks
        self._base_down_thr = 90.0
        self._base_up_thr   = 140.0
        self._down_shift_min_default = 0.05
        self._up_shift_max_default   = 0.02

        # load calibration YAML
        with open(calib_path, "r") as f:
            calib = yaml.safe_load(f) or {}

        self._base_down_thr = float(calib["down_threshold"])
        self._base_up_thr   = float(calib["up_threshold"])
        self._down_shift_min = float(calib["down_shift_delta"])
        self._up_shift_max   = float(calib["up_shift_delta"])
        print(f" Loaded calibration from: {calib_path}")

        # angle tolerance (%) 
        self.angle_tolerance_pct = float(self.config.get("angle_tolerance", 0))

        # states
        self.system_stage = "waiting"   # waiting -> aligning -> ready -> counting
        self.stage = None               # down/ up
        self.align_counter = 0
        self.baseline_axis_val = None   # baseline_shoulder_y

        self.rep_count = 0
        self.has_been_up = False  

        # aligning time before starting
        self.MIN_ALIGN_SEC = 0.7
        self.MAX_ALIGN_SEC = 3.0
        self.min_align_frames = 0
        self.max_align_frames = 0
        self._fps_last = 30.0  

    # thresholds with tolerance
    def _compute_angle_thresholds(self):
        tol = self.angle_tolerance_pct / 100.0
        down_thr = self._base_down_thr * (1.0 + tol)
        up_thr   = self._base_up_thr   * (1.0 - tol)
        return down_thr, up_thr

    # main function
    def update(self, landmarks, fps):
        # calculate how many frames the user must hold a straight pose to be ready
        if fps and fps > 0:
            self._fps_last = fps
            self.min_align_frames = max(1, int(self._fps_last * self.MIN_ALIGN_SEC))
            self.max_align_frames = max(self.min_align_frames, int(self._fps_last * self.MAX_ALIGN_SEC))

        # angle and side detection
        angle, pts, side = self.choose_side(landmarks)
        if angle is None or pts is None:
            self.system_stage = "waiting"
            self.stage = None
            return {"angle": None, "system_stage": self.system_stage, "rep_count": int(self.rep_count)}

        # shoulder y-axis reference
        ref_y = self.get_reference_y(landmarks, side)
        if ref_y is None:
            return {"angle": float(angle), "system_stage": self.system_stage, "rep_count": int(self.rep_count)}

        # angle & shift thresholds
        down_thr, up_thr = self._compute_angle_thresholds()
        down_shift_min = self._down_shift_min
        up_shift_max   = self._up_shift_max

        # detect upright position
        if self.system_stage == "waiting":
            if angle > max(120.0, up_thr - 5.0):
                self.system_stage = "aligning"
                self.align_counter = 0

        # set baseline position
        elif self.system_stage == "aligning":
            if angle > max(120.0, up_thr - 5.0):
                self.align_counter += 1
            else:
                self.align_counter = 0

            if self.baseline_axis_val is None:
                self.baseline_axis_val = ref_y
            else:
                self.baseline_axis_val = 0.9 * self.baseline_axis_val + 0.1 * ref_y

            if self.align_counter > self.min_align_frames or self.align_counter > self.max_align_frames:
                self.baseline_axis_val = ref_y
                self.system_stage = "ready"
                self.stage = "up"   

        # detect initial descent
        elif self.system_stage == "ready":
            if angle < min(120.0, down_thr + 10.0):
                self.system_stage = "counting"

        # count reps based on angle & shift
        elif self.system_stage == "counting":
            shift = ref_y - (self.baseline_axis_val if self.baseline_axis_val is not None else ref_y)

            if angle < down_thr and shift > down_shift_min:
                self.stage = "down"

            if self.stage == "down" and angle > up_thr and shift < up_shift_max:
                self.rep_count += 1
                self.stage = "up"

            self.baseline_axis_val = 0.995 * self.baseline_axis_val + 0.005 * ref_y

        # debug log 
        shift_val = ref_y - (self.baseline_axis_val if self.baseline_axis_val is not None else ref_y)
        print(
            f"[DEBUG][PUSHUP] Stage:{self.stage}, System:{self.system_stage}, "
            f"Angle:{angle:.2f}°, ↓Thr:{down_thr:.2f}, ↑Thr:{up_thr:.2f}, "
            f"Shift:{shift_val:+.3f}, down_shift_min:{down_shift_min:.3f}, up_shift_max:{up_shift_max:.3f}, "
            f"reps:{self.rep_count}"
        )

        # return payload
        return {
            "angle": float(angle),
            "ref_y": float(ref_y),
            "rep_count": int(self.rep_count),
            "stage": self.stage,
            "system_stage": self.system_stage,
            "side": self.side_selected or side
        }

    def reset(self):
        super().reset()
        self.system_stage = "waiting"
        self.stage = None
        self.align_counter = 0
        self.baseline_axis_val = None
        self.rep_count = 0
        self.has_been_up = False
