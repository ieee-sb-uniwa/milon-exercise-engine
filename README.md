# MILON Pose Analysis Engine

Exercise form analysis and biomechanical assessment using computer vision and pose estimation.

![Milon Exercise Engine](pose_estimator.png)

## Project Structure

```
milon-exercise-engine/
├── data/
│   ├── raw/
│   │   ├── videos/          # Raw exercise videos
│   │   └── poses/           # Pre-extracted pose data
│   ├── processed/           # Processed sequences
│   └── annotations/         # Form quality labels, rep counts
├── notebooks/
│   ├── 01_pose_estimation_baseline.ipynb
│   ├── 02_biomechanics_features.ipynb
│   ├── 03_rep_counting.ipynb
│   └── 04_form_scoring.ipynb
├── src/
│   ├── pose/
│   │   ├── estimator.py     # MediaPipe/OpenPose wrapper
│   │   └── tracker.py       # Multi-person tracking
│   ├── biomechanics/
│   │   ├── angles.py        # Joint angle calculations
│   │   └── form_analyzer.py # Form quality assessment
│   ├── exercise/
│   │   ├── squat.py         # Squat-specific analysis
│   │   ├── deadlift.py      # Deadlift analysis
│   │   ├── pushup.py
│   │   └── ...
│   ├── temporal/
│   │   ├── rep_counter.py   # Repetition counting
│   │   ├── pattern_recognition.py  # LSTM classifier
│   │   └── quality_scorer.py       # Rep quality scoring
│   └── utils/
│       ├── video_processor.py
│       ├── visualization.py
│       └── metrics.py
├── configs/
│   └── mediapipe_config.yaml
├── models/
│   ├── pretrained/          # MediaPipe, OpenPose weights
│   └── checkpoints/         # Trained exercise classifiers
├── experiments/
├── tests/
└── requirements.txt
```

## Framework Design

### Modular Pipeline Architecture

```python
# Example usage
from src.pose import PoseEstimator
from src.exercise import SquatAnalyzer
from src.temporal import RepCounter, QualityScorer

# 1. Pose Estimation
estimator = PoseEstimator(backend='mediapipe')
poses = [estimator.estimate(frame) for frame in video_frames]

# 2. Exercise-Specific Analysis
analyzer = SquatAnalyzer()
form_feedback = analyzer.analyze(poses)

# 3. Temporal Analysis
counter = RepCounter(exercise_type='squat')
reps = counter.count_reps(angle_sequence)

# 4. Quality Scoring
scorer = QualityScorer()
quality = scorer.score_rep(rep_data)
```

## Pre-trained Models & Datasets

### Pose Estimation Models (Pre-trained, Free)

- **MediaPipe Pose**: Lightweight, real-time, mobile-friendly
  - Download: Automatic via `pip install mediapipe`
  - Speed: 30-60 FPS on CPU
  - Accuracy: Good for exercise tracking
- **OpenPose** (Alternative): More accurate, slower
  - Requires: CUDA GPU
  - Download: CMU OpenPose repository

### Datasets

**Free/Academic:**

- **NTURGB+D**: 56K action sequences (includes exercises)
- **Penn Action**: 2,326 videos with pose annotations
- **LSP (Leeds Sports Pose)**: 2K sport images

**Custom Collection Needed:**

- Exercise form videos with expert annotations
- Multi-angle recordings for 3D analysis
- Good/bad form examples for each exercise

### Temporal Models (Train from scratch)

- **Exercise Classifier**: LSTM on pose sequences
- **Rep Counter**: Peak detection + LSTM validation
- **Form Scorer**: Rule-based + ML hybrid

## Setup

```bash
pip install -r requirements.txt

# Download MediaPipe models (automatic)
python -c "import mediapipe as mp; mp.solutions.pose"
```

## Quick Start

```bash
# Test pose estimation on video
python src/pose/estimator.py --video data/raw/videos/squat_demo.mp4

# Run squat analysis
python src/exercise/squat.py --video data/raw/videos/squat_demo.mp4
```

## Deployment Considerations

**NOT recommended for smartphone app due to:**

- High computational requirements (real-time video processing)
- Battery drain from continuous inference
- Camera angle limitations (single viewpoint)

**Recommended deployment:**

- Desktop/tablet application for home use
- Gym kiosk system with fixed cameras
- Cloud processing with video upload (non-real-time)

## Related Repositories

- **Nutrition Module**: [milon-nutrition-training](https://github.com/your-org/milon-nutrition-training)
- **Nutrition App**: [milon-nutrition-app](https://github.com/your-org/milon-nutrition-app)

## License

[Your License]
