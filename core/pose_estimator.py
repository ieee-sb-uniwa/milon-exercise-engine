import cv2
import mediapipe as mp

# Mediapipe pose wrapper
class PoseEstimator:

    def __init__(self, source=0, min_detection_conf=0.5, min_tracking_conf=0.5, model_complexity=0): # 0=light
        self.source = source
        self.cap = cv2.VideoCapture(source)

        # detects FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0 or self.fps is None:
            self.fps = 30.0

        # Mediapipe Pose initialization
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
            model_complexity=model_complexity
        )

        # drawing skeletons for visualization
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    # video or camera
    def read_frame(self):
        if not self.cap.isOpened():
            print("VideoCapture is not open.")
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    # processes frame for pose detection (RGB conversion for mediapipe)
    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return results, image

    def detect(self, frame):
        results, _ = self.process_frame(frame)
        return results.pose_landmarks.landmark if results.pose_landmarks else None

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
