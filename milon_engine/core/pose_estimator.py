import pathlib
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as mp_base
from typing import Optional, Tuple, Any

# Bundled .task model file — ships with the package so no network download
# is needed at runtime (important on Streamlit Cloud).
_BUNDLED_MODEL = str(
    pathlib.Path(__file__).parent.parent / "models" / "pose_landmarker_lite.task"
)


class PoseEstimator:
    """MediaPipe pose wrapper (Tasks API — mediapipe >= 0.10.30).

    Takes a BGR frame and returns pose landmarks.
    All MediaPipe-specific details are encapsulated here; callers only
    interact with detect().
    """

    def __init__(
        self,
        min_detection_conf: float = 0.5,
        min_tracking_conf: float = 0.5,
        model_complexity: int = 0,  # kept for API compatibility, ignored (lite only)
    ):
        options = vision.PoseLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path=_BUNDLED_MODEL),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_conf,
            min_pose_presence_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame) -> Tuple[Optional[list], Any]:
        """Process a BGR frame and return (landmark_list, raw_results).

        - landmark_list: flat list of NormalizedLandmark for exercise logic,
          or None if no pose was detected.
        - raw_results:   PoseLandmarkerResult, passed to Visualizer.render()
          for skeleton drawing.
        """
        # Tasks API requires RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)

        if not results.pose_landmarks:
            return None, results

        # pose_landmarks[0] is the landmark list for the first (only) detected pose
        return results.pose_landmarks[0], results
