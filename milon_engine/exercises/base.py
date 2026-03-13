from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

import numpy as np
from mediapipe.tasks.python.vision import PoseLandmark
import yaml

from milon_engine.core.models import ExerciseResult


# Map landmark name strings → MediaPipe PoseLandmark enum values
NAME2ENUM = {
    name: getattr(PoseLandmark, name) for name in dir(PoseLandmark) if name.isupper()
}


class Exercise(ABC):
    """Base class for all exercise counters (push-up, squat, leg raise).

    Subclasses must implement evaluate() and may override reset().

    Args:
        config: Dict loaded from a YAML exercise config file.
        fps:    Frames-per-second of the video source. Used to convert
                time-based alignment thresholds into frame counts.
    """

    def __init__(self, config: dict, fps: float = 30.0):
        self.config = config
        self.fps = fps

        # Rep counting state
        self.rep_count: int = 0
        self.stage: Optional[str] = None  # "up" | "down" | None
        self.system_stage: str = "waiting"  # waiting → aligning → ready → counting
        self.baseline_axis_val: Optional[float] = None
        self.side_selected: Optional[str] = None

        # Angle thresholds (populated by subclass via calibration load)
        self.down_threshold: Optional[float] = None
        self.up_threshold: Optional[float] = None
        self.angle_tolerance: float = float(config.get("angle_tolerance", 20))

        # Landmark config from YAML
        lm_cfg = config.get("landmarks", {})
        self._lm_left_names: List[str] = lm_cfg.get("left", [])
        self._lm_right_names: List[str] = lm_cfg.get("right", [])
        self._lm_left = [NAME2ENUM[n] for n in self._lm_left_names if n in NAME2ENUM]
        self._lm_right = [NAME2ENUM[n] for n in self._lm_right_names if n in NAME2ENUM]

        self._preferred_side: str = config.get("preferred_side", "auto").lower()
        self._reference_point: str = config.get("reference_point", "hip").lower()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate(self, landmarks: list) -> ExerciseResult:
        """Process a landmark list and return the current exercise state.

        Args:
            landmarks: List of NormalizedLandmark from PoseEstimator.detect().

        Returns:
            ExerciseResult with current angle, rep count, stage, etc.
        """

    # ------------------------------------------------------------------
    # Shared geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_angle(a, b, c) -> float:
        """Return the angle (degrees) at vertex b formed by points a-b-c."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
            a[1] - b[1], a[0] - b[0]
        )
        angle = np.abs(radians * 180.0 / np.pi)
        return 360.0 - angle if angle > 180.0 else angle

    def _coords_from_ids(self, landmarks: list, ids: list) -> List[List[float]]:
        """Extract [x, y] coordinates for each landmark id."""
        return [[landmarks[i.value].x, landmarks[i.value].y] for i in ids]

    def _angle_for_side(
        self, landmarks: list, side: str
    ) -> Tuple[Optional[float], Optional[list]]:
        """Compute joint angle and key-point coords for the given side."""
        ids = self._lm_left if side == "left" else self._lm_right
        if len(ids) < 3:
            return None, None
        pts = self._coords_from_ids(landmarks, ids)
        return self.calculate_angle(pts[0], pts[1], pts[2]), pts

    def choose_side(
        self, landmarks: list
    ) -> Tuple[Optional[float], Optional[list], Optional[str]]:
        """Select the visible side of the athlete and return (angle, pts, side).

        Tries left first, then right.  Returns (None, None, None) when
        neither side has enough visible landmarks.
        """
        for side in ("left", "right"):
            angle, pts = self._angle_for_side(landmarks, side)
            if angle is not None:
                self.side_selected = side
                return angle, pts, side
        return None, None, None

    def get_reference_y(self, landmarks: list, side: str) -> Optional[float]:
        """Return the y-coordinate of the reference landmark for the given side."""
        ref_map = {
            "hip": ("LEFT_HIP", "RIGHT_HIP"),
            "knee": ("LEFT_KNEE", "RIGHT_KNEE"),
            "ankle": ("LEFT_ANKLE", "RIGHT_ANKLE"),
            "shoulder": ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
            "foot": ("LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"),
        }
        ref_pair = ref_map.get(self._reference_point, ref_map["hip"])
        ref_name = ref_pair[0] if side == "left" else ref_pair[1]

        names = self._lm_left_names if side == "left" else self._lm_right_names
        ids = self._lm_left if side == "left" else self._lm_right
        pts = self._coords_from_ids(landmarks, ids)

        if ref_name in names:
            idx = names.index(ref_name)
            return pts[idx][1]
        return None

    # ------------------------------------------------------------------
    # Threshold helpers
    # ------------------------------------------------------------------

    def _compute_angle_thresholds(self) -> Tuple[float, float]:
        """Return (down_thr, up_thr) adjusted by the configured tolerance."""
        tol = self.angle_tolerance / 100.0
        down_thr = (self.down_threshold or 0.0) * (1.0 + tol)
        up_thr = (self.up_threshold or 0.0) * (1.0 - tol)
        return down_thr, up_thr

    def load_calibration(self, yaml_path: str) -> None:
        """Load down/up thresholds from a calibration YAML file."""
        with open(yaml_path) as f:
            calib = yaml.safe_load(f)
        self.down_threshold = float(calib["down_threshold"])
        self.up_threshold = float(calib["up_threshold"])

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all rep-counting and system state."""
        self.rep_count = 0
        self.stage = None
        self.system_stage = "waiting"
        self.baseline_axis_val = None
        self.side_selected = None
