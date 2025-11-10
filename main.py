import cv2, yaml, os
import argparse
import importlib
from core.pose_estimator import PoseEstimator
from core.visualizer import Visualizer

# in the same way as train.py , loads the yaml and expects the arguments
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


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Exercise Counter")
    parser.add_argument("--exercise", type=str, required=True,
                        help="Exercise name (e.g. pushup, squat, legraise)")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source (0=webcam or path to video file)")
    return parser.parse_args()


def main():
    args = parse_args()
    exercise = args.exercise.lower()

    # load config yaml
    config_path = f"configs/{exercise}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"⚠️ Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # load Calibration yaml
    calib_path = f"outputs/calibration/{exercise}.yaml"
    if not os.path.exists(calib_path):
        print(f"Calibration file not found for {exercise}. Run train.py first!")
    else:
        print(f"Calibration found: {calib_path}")

    # video source
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    print(f"Loaded config for: {exercise}")
    print(f"Source: {'Webcam' if isinstance(source, int) else source}")

    # initialize parameters
    pose_engine = PoseEstimator(source=source)
    ModelClass = load_model_class(cfg) # loads the rep counter
    counter = ModelClass(config=cfg, calibration_path=calib_path)
    viz = Visualizer(pose_engine.mp_drawing, pose_engine.mp_pose)

    # main loop
    while True:
        frame = pose_engine.read_frame()
        if frame is None:
            break

        results, image = pose_engine.process_frame(frame)
        if results.pose_landmarks:
            out = counter.update(results.pose_landmarks.landmark, fps=pose_engine.fps)

            if out.get("angle") is not None:
                viz.draw_feedback(image, out["rep_count"], out["stage"], out["angle"])
                viz.draw_system_status(image, out["system_stage"])
                viz.draw_skeleton(image, results)

        cv2.imshow(f"{exercise.capitalize()} Counter", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    pose_engine.release()
    cv2.destroyAllWindows()
    print("Session ended.")


if __name__ == "__main__":
    main()
