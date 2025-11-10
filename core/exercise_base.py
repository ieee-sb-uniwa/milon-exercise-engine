import numpy as np
from collections import deque
import mediapipe as mp
import cv2, yaml
from core.pose_estimator import PoseEstimator  

# landmark names to enums for mediapipe
NAME2ENUM = {
    name: getattr(mp.solutions.pose.PoseLandmark, name)
    for name in dir(mp.solutions.pose.PoseLandmark)
    if name.isupper()
}

# base class for all exercise counters (push-up, squat, leg raise)
# uses YAML config and calculates side selection, angles, calibration, and auto-start logic.
class ExerciseBase:

    def __init__(self, config: dict, pose_estimator=None):
        self.config = config
        self.pose_estimator = pose_estimator or PoseEstimator()

        # loads general options from YAML config
        self.preferred_side = config.get("preferred_side", "auto").lower()
        self.angle_tolerance = config.get("angle_tolerance", 20)
        self.reference_point = config.get("reference_point", "hip").lower()

        lm_cfg = config.get("landmarks", {})
        self.lm_left_names = lm_cfg.get("left", [])
        self.lm_right_names = lm_cfg.get("right", [])
        self.lm_left = [NAME2ENUM[n] for n in self.lm_left_names if n in NAME2ENUM]
        self.lm_right = [NAME2ENUM[n] for n in self.lm_right_names if n in NAME2ENUM]

        # system state variables
        self.rep_count = 0
        self.stage = None
        self.system_stage = "waiting"  # waiting → counting
        self.baseline_axis_val = None
        self.align_counter = 0
        self.side_selected = None
        self._side_votes = deque(maxlen=15)

        # calibration thresholds
        self.down_threshold = None
        self.up_threshold = None

        # auto-start detection settings
        self.auto_start_enabled = True               
        self._angle_history = deque(maxlen=10)       
        self._auto_started = False                   

    # calculates the angle between 3 points
    @staticmethod 
    def calculate_angle(a, b, c): 
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    # returns the x,y of the landmarks we need
    def _coords_from_ids(self, landmarks, ids): 
        return [[landmarks[i.value].x, landmarks[i.value].y] for i in ids]
    
    # calculates the angle between the specific 3 points from the yaml
    def _angle_for_side(self, landmarks, side: str):
        ids = self.lm_left if side == "left" else self.lm_right
        if len(ids) < 3:
            return None, None
        pts = self._coords_from_ids(landmarks, ids)
        return self.calculate_angle(pts[0], pts[1], pts[2]), pts
    
    # returns y of the reference_point
    def _landmark_y_if_available(self, target_name: str, pts_coords, pts_names):
        if target_name in pts_names:
            idx = pts_names.index(target_name)
            return pts_coords[idx][1]
        return None
    
    # gets left or right reference_point
    def get_reference_y(self, landmarks, side):
        ref_map = {
            "hip": ("LEFT_HIP", "RIGHT_HIP"),
            "knee": ("LEFT_KNEE", "RIGHT_KNEE"),
            "ankle": ("LEFT_ANKLE", "RIGHT_ANKLE"),
            "shoulder": ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
            "foot": ("LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"),
        }

        ref_pair = ref_map.get(self.reference_point, ref_map["hip"])
        ref_name = ref_pair[0] if side == "left" else ref_pair[1]

        pts_names = self.lm_left_names if side == "left" else self.lm_right_names
        ids = self.lm_left if side == "left" else self.lm_right
        pts = self._coords_from_ids(landmarks, ids)
        return self._landmark_y_if_available(ref_name, pts, pts_names)

    # chooses the visible side of the athlete (left or right)
    def choose_side(self, landmarks):
        left_angle, left_pts = self._angle_for_side(landmarks, "left")
        if left_angle is not None:
            self.side_selected = "left"
            return left_angle, left_pts, "left"

        right_angle, right_pts = self._angle_for_side(landmarks, "right")
        if right_angle is not None:
            self.side_selected = "right"
            return right_angle, right_pts, "right"

        return None, None, None

    # starts counting once the athlete moves or if he is already in position.
    def detect_auto_start(self, angle):
        if not self.auto_start_enabled or self.system_stage != "waiting":
            return
        self._angle_history.append(angle)
        if len(self._angle_history) == self._angle_history.maxlen:
            delta = abs(self._angle_history[-1] - self._angle_history[0])
            if delta > 8 or (self.up_threshold and angle > self.up_threshold - 5):
                self.system_stage = "counting"
                self.stage = "up"
                self._auto_started = True
                print("Auto-start — ready for counting")

    # saves calibration thresholds to a YAML for use in real-time repetition counting
    def train_from_video(self, video_path, output_yaml):
        cap = cv2.VideoCapture(video_path)
        angles, ref_ys = [], []
        frame_count = 0

        print(f"Training from video: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            landmarks = self.pose_estimator.detect(frame)
            if landmarks is None:
                continue

            # calculate angle and side
            angle, _, side = self.choose_side(landmarks)
            if angle is None:
                continue

            # calculate y movement
            ref_y = self.get_reference_y(landmarks, side)
            if ref_y is None:
                continue

            angles.append(angle)
            ref_ys.append(ref_y)

        cap.release()

        if not angles:
            print("No valid angles found")
            return

        # threshold angles
        min_angle = float(np.min(angles))
        max_angle = float(np.max(angles))
        down_th = round(min_angle + (max_angle - min_angle) * 0.25, 2)
        up_th = round(max_angle - (max_angle - min_angle) * 0.2, 2)

        ref_y_min = float(np.min(ref_ys))
        ref_y_max = float(np.max(ref_ys))
        shift_range = abs(ref_y_max - ref_y_min)
        down_shift_delta = round(shift_range * 0.25, 5)
        up_shift_delta = round(shift_range * 0.1, 5)

        # determines if vertical movement is enough to use in repetition counting 
        use_vertical_shift = shift_range > 0.03  # if the shift changes over 3%
        calibration_data = {
            "min_angle": min_angle,
            "max_angle": max_angle,
            "down_threshold": down_th,
            "up_threshold": up_th,
            "ref_y_min": ref_y_min,
            "ref_y_max": ref_y_max,
            "down_shift_delta": down_shift_delta,
            "up_shift_delta": up_shift_delta,
            "ref_y_range": round(shift_range, 5),
            "use_vertical_shift": bool(use_vertical_shift),
            "frames_analyzed": frame_count
        }

        with open(output_yaml, "w") as f:
            yaml.dump(calibration_data, f, sort_keys=False)

        print(f"Calibration saved to {output_yaml}")
        print(calibration_data)

    def reset(self):
        self.rep_count = 0
        self.stage = None
        self.system_stage = "waiting"
        self.baseline_axis_val = None
        self.align_counter = 0
        self.side_selected = None
        self._side_votes.clear()
        self._angle_history.clear()
        self.down_threshold = self.up_threshold = None
        self._auto_started = False
