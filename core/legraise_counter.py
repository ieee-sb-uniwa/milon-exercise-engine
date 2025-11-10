from .exercise_base import ExerciseBase
import os, yaml

class LegRaiseCounter(ExerciseBase):


    def __init__(self, config: dict, calibration_path=None):
        super().__init__(config)
        calib_path = calibration_path or f"outputs/calibration/{config['exercise_name'].lower()}.yaml"

        # loads calibration yaml
        if os.path.exists(calib_path):
            with open(calib_path, "r") as f:
                calib = yaml.safe_load(f)
                self.down_threshold = calib.get("down_threshold")
                self.up_threshold = calib.get("up_threshold")
        else:
            self.down_threshold = None
            self.up_threshold = None
                
        # internal states 
        self.has_started = False
        self.has_been_up = False  

        # system stages for baseline logic
        self.system_stage = "waiting"  # waiting → aligning → ready → counting
        self.align_counter = 0
        self.baseline_angle = None

        # aligning time and frames
        self.MIN_ALIGN_SEC = 0.5
        self.MAX_ALIGN_SEC = 2.0
        self.min_align_frames = 0
        self.max_align_frames = 0
        self.angle_tolerance_pct = float(config.get("angle_tolerance", 0))
        self._fps_last = 30.0 

    # thresholds with tolerance
    def _compute_angle_thresholds(self):
        tol = self.angle_tolerance_pct / 100.0
        down_thr = self.down_threshold * (1.0 + tol)
        up_thr   = self.up_threshold   * (1.0 - tol)
        return down_thr, up_thr

    # main
    def update(self, landmarks, fps):
        # timing through fps
        if fps and fps > 0:
            self._fps_last = fps
            self.min_align_frames = max(1, int(self._fps_last * self.MIN_ALIGN_SEC))
            self.max_align_frames = max(self.min_align_frames, int(self._fps_last * self.MAX_ALIGN_SEC))

        # calculate angle and side
        angle, pts, side = self.choose_side(landmarks)
        if angle is None:
            self.system_stage = "waiting"
            return {"angle": None, "system_stage": self.system_stage, "rep_count": int(self.rep_count)}

        down_thr, up_thr = self._compute_angle_thresholds()

        # baseline logic
        # when the leg is extended above the threshold, the system enters the "aligning" stage.
        if self.system_stage == "waiting":
            if angle > up_thr - 5:
                self.system_stage = "aligning"
                self.align_counter = 0

        # if the extended posture is held steady for a few frames, it transitions to "ready".
        elif self.system_stage == "aligning":
            if angle > up_thr - 5:
                self.align_counter += 1
            else:
                self.align_counter = 0

            if self.align_counter > self.min_align_frames:
                self.system_stage = "ready"
                self.baseline_angle = angle

        # once the leg lowers below the down threshold, the system enters "counting" mode.
        elif self.system_stage == "ready":
            if angle < down_thr:
                self.system_stage = "counting"

        # until the system is in "counting", no repetitions are counted
        if self.system_stage != "counting":
            return {
                "angle": float(angle),
                "rep_count": int(self.rep_count),
                "stage": self.stage or "waiting",
                "system_stage": self.system_stage,
            }

        is_down = angle < down_thr
        is_up   = angle > up_thr

        # if the angle goes above the up threshold, mark stage as "up" and note that the leg has been raised
        if is_up:
            self.stage = "up"
            self.has_been_up = True

        # If the angle drops below the down threshold and it wasn't already in "down" stage
        elif is_down:
            if self.stage != "down":
                self.stage = "down"

                # if this is the first rep after entering "counting", it counts immediately
                if not self.has_started:
                    self.rep_count += 1
                    self.has_started = True
                    self.has_been_up = True  

                # otherwise, only count a rep if the leg was previously up (ensures up→down cycle)
                # after counting, reset has_been_up is set to False to wait for the next full cycle
                elif self.has_been_up:
                    self.rep_count += 1
                    self.has_been_up = False

        # debug log
        print(
            f"[DEBUG][LEG_RAISE] Stage:{self.stage}, "
            f"System:{self.system_stage}, "
            f"Angle:{angle:.2f}, "
            f"↓{down_thr:.1f}, ↑{up_thr:.1f}, "
            f"Reps:{self.rep_count}"
        )

        # visualizer
        return {
            "angle": float(angle),
            "rep_count": int(self.rep_count),
            "stage": self.stage,
            "system_stage": self.system_stage,
            "side": self.side_selected or side,
        }


    def reset(self):
        super().reset()
        self.system_stage = "waiting"
        self.align_counter = 0
        self.baseline_angle = None
        self.has_started = False
        self.has_been_up = False
