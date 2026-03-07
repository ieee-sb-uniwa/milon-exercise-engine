import cv2
import mediapipe as mp
from typing import Optional, Tuple, Any


class PoseEstimator:
    """MediaPipe pose wrapper.

    Takes a BGR frame and returns pose landmarks.
    All MediaPipe-specific details are encapsulated here; callers only
    interact with detect().
    """

    def __init__(
        self,
        min_detection_conf: float = 0.5,
        min_tracking_conf: float = 0.5,
        model_complexity: int = 0,  # 0=lite, 1=full, 2=heavy
    ):
        self._pose = mp.solutions.pose.Pose(
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
            model_complexity=model_complexity,
        )

    def detect(self, frame) -> Tuple[Optional[list], Any]:
        """Process a BGR frame and return (landmark_list, raw_results).

        - landmark_list: list of NormalizedLandmark for exercise logic,
          or None if no pose was detected.
        - raw_results:   the raw MediaPipe results object, passed to
          Visualizer.render() for skeleton drawing.
        """
        # MediaPipe requires RGB; this is an internal implementation detail
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self._pose.process(image)
        image.flags.writeable = True

        landmarks = (
            list(results.pose_landmarks.landmark) if results.pose_landmarks else None
        )
        return landmarks, results
