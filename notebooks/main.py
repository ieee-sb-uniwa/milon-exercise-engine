import cv2
from milon_engine import FrameProcessor, PoseEstimator, Visualizer, Squat, load_config

config = load_config("squat")
exercise = Squat(config, fps=30.0, calibration_path="outputs/calibration/squat.yaml")
processor = FrameProcessor(exercise, PoseEstimator(), Visualizer())

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    annotated = processor.process_frame(frame)
    cv2.imshow("Milon", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
