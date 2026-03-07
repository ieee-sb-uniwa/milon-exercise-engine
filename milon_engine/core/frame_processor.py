import numpy as np

from milon_engine.core.pose_estimator import PoseEstimator
from milon_engine.core.visualizer import Visualizer
from milon_engine.exercises.base import Exercise


class FrameProcessor:
    """Orchestrates the full per-frame pipeline.

    Flow:
        frame → PoseEstimator.detect()
              → Exercise.evaluate()
              → Visualizer.render()
              → annotated_frame

    The caller (Streamlit, Raspberry Pi script, etc.) is responsible for
    capturing frames and displaying the returned annotated frame.
    """

    def __init__(
        self,
        exercise: Exercise,
        estimator: PoseEstimator,
        visualizer: Visualizer,
    ):
        self.exercise = exercise
        self.estimator = estimator
        self.visualizer = visualizer

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run the full pipeline on a single BGR frame.

        Args:
            frame: Raw BGR frame (e.g. from cv2.VideoCapture or Streamlit).

        Returns:
            Annotated BGR frame ready for display.
        """
        # 1. Pose estimation
        landmarks, raw_results = self.estimator.detect(frame)

        # 2. No detection — render fallback overlay
        if landmarks is None:
            return self.visualizer.render(frame, raw_results, None)

        # 3. Exercise logic
        result = self.exercise.evaluate(landmarks)

        # 4. Visualize
        return self.visualizer.render(frame, raw_results, result)
