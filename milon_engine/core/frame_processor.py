from milon_engine.core import Visualizer
from milon_engine.core import PoseEstimator
from milon_engine.exercises.base import Exercise


class FrameProcessor:
    """Coordinates pose detection → exercise logic → visualization"""

    def __init__(
        self, exercise: Exercise, pose_estimator: PoseEstimator, visualizer: Visualizer
    ):
        self.exercise = exercise
        self.pose_estimator = pose_estimator
        self.visualizer = visualizer

    def process_frame(self, frame):
        """Main pipeline"""
        # 1. Extract landmarks
        landmarks = self.pose_estimator.detect(frame)
        if landmarks is None:
            return self.visualizer.render_no_detection(frame)

        # 2. Exercise logic
        result = self.exercise.evaluate(landmarks)

        # 3. Visualization
        annotated_frame = self.visualizer.render(
            frame,
            landmarks=landmarks,
            angle=result["angle"],
            reps=result["reps"],
            stage=result["stage"],
            feedback=result["feedback"],
        )

        return annotated_frame


if __name__ == "__main__":
    # Setup
    import cv2
    from milon_engine.exercises import PushUp
    import yaml

    # with open("config/pushup.yaml", "r") as f:
    #     config = yaml.safe_load(f)

    config = "config/pushup.yaml"
    pose_estimator = PoseEstimator()
    exercise = PushUp(config)
    exercise.load_calibration("pushup_calib.yaml")
    visualizer = Visualizer()

    processor = FrameProcessor(exercise, pose_estimator, visualizer)

    # Run
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Single responsibility: process frame
        annotated = processor.process_frame(frame)

        cv2.imshow("Counter", annotated)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
