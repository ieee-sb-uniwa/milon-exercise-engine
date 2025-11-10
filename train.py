import argparse
import os, cv2, yaml
import importlib
from core.exercise_base import ExerciseBase
from core.pose_estimator import PoseEstimator
from core.visualizer import Visualizer

# loads the exercise counter class
def load_model_class(cfg: dict):

    module_path = cfg.get("module")
    class_name = cfg.get("counter_class")

    if not module_path or not class_name:
        raise ValueError("Missing 'module' or 'counter_class' in YAML config.")

    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load class '{class_name}' from '{module_path}': {e}")

def main():

    # arguments
    parser = argparse.ArgumentParser(description="Calibration from reference video")
    parser.add_argument("--exercise", type=str, required=True, help="Exercise name (e.g. pushup, squat, legraise)")
    parser.add_argument("--video", type=str, required=True, help="Path to reference video file")
    args = parser.parse_args()

    exercise = args.exercise.lower()
    video_path = args.video
    config_path = f"configs/{exercise}.yaml"

    # load YAML config 
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"Loaded config for: {exercise}")
    print(f"Reference video: {video_path}")

    # creates calibration yaml
    os.makedirs("outputs/calibration", exist_ok=True)
    calib_output = f"outputs/calibration/{exercise}.yaml"

    # "trains" from video, runs calibration and save thresholds
    model = ExerciseBase(config=cfg)
    model.train_from_video(video_path, calib_output)
    print(f"Calibration completed and saved to {calib_output}")

    pose_engine = PoseEstimator(source=video_path)
    ModelClass = load_model_class(cfg)
    counter = ModelClass(config=cfg, calibration_path=calib_output)
    viz = Visualizer(pose_engine.mp_drawing, pose_engine.mp_pose)

    while True:
        frame = pose_engine.read_frame()
        if frame is None:
            break

        results, image = pose_engine.process_frame(frame)
        if results.pose_landmarks:
            out = counter.update(results.pose_landmarks.landmark, fps=pose_engine.fps)

            # visualization
            if out.get("angle") is not None:
                viz.draw_feedback(image, out["rep_count"], out["stage"], out["angle"])
                viz.draw_system_status(image, out["system_stage"])
                viz.draw_skeleton(image, results)

        # display
        cv2.imshow(f"{exercise.capitalize()} Preview (After Calibration)", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    pose_engine.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
