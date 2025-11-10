# Rep Counting Methodology and System Overview

## 1. Objective
The goal of this system is to **automatically detect and count exercise repetitions** (reps) from video footage, using geometric analysis of body posture.  
The approach relies entirely on **pose estimation and kinematic features**.

---

## 2. Pose Estimation Setup
We use **MediaPipe Pose** to extract human body landmarks from each video frame.  
Each landmark corresponds to a key joint (such as the shoulder, elbow, hip, knee, or ankle), defined in normalized 2D coordinates.

The pose estimator runs with:
- **model complexity:** `0` (lightweight mode for real-time performance)
- **min_detection_confidence:** `0.5`
- **min_tracking_confidence:** `0.5`

This configuration provides sufficient accuracy for exercise tracking while maintaining high FPS for smooth visualization. Model complexity can optionally be set to 1 during calibration for improved landmark accuracy, and 0 for real-time inference.

---

## 3. Angle Calculation
Once the body joints are detected, we compute **angles between relevant triplets of landmarks** to describe the movement of each limb.  
The angle between three points (A–B–C) represents how much the middle joint (B) is bent or extended.

For example:
- Shoulder–Elbow–Wrist → elbow flexion
- Shoulder–Hip–Ankle → hip flexion
- Hip–Knee–Ankle → knee flexion

These angular values change as the user moves, providing a simple and robust signal to detect motion phases such as “up” and “down”.

---

## 4. Counting Logic: General Structure
The repetition counting mechanism follows a **four-stage state machine**:

1. **Waiting:** The system waits for the person to appear and maintain a stable position.
2. **Aligning:** Once the person reaches an upright or fully extended posture, the system locks a reference baseline (either an angle or a body coordinate).
3. **Ready:** The system confirms that the user is stable and ready to begin counting.
4. **Counting:** The system continuously monitors the angles and position changes to identify full movement cycles and increment the repetition count.

This structure prevents false counts and ensures stability before starting. The system transitions automatically between these states without manual input, using dynamic auto start logic that detects whether the user begins already in the “up” posture or moves into it.

---

## 5. Exercise-Specific Methodologies

### 🟢 Push-Up Counter
**Measured Joint:** Shoulder–Elbow–Wrist  
**Additional Reference:** Vertical position of the shoulder

**Concept:**
- The elbow angle indicates the arm’s flexion.  
  A larger angle means the arms are extended (“up”), while a smaller angle means the elbows are bent (“down”).
- To improve robustness, the system also measures the **vertical shift of the shoulders** relative to a baseline position.
  This helps confirm that the entire body is moving downwards, not just the elbows.

**How it works:**
1. The system first detects a fully extended position and sets that as the starting baseline.
2. When the elbow angle decreases and the shoulder moves downward, the system recognizes the “down” phase.
3. When the angle increases again and the shoulders return near the baseline, the system registers one complete repetition.

This combination of **angle + vertical motion** ensures that only full push-ups are counted, avoiding partial movements.

---

### 🟣 Leg Raise Counter
**Measured Joint:** Shoulder–Hip–Ankle  
**Additional Reference:** None

**Concept:**
- The hip angle increases when the leg is raised and decreases when it is lowered.
- This single angle is sufficient to describe the entire leg raise motion, so no additional body shift is required.

**How it works:**
1. The system starts when the user holds a flat, extended posture (leg straight).
2. Once the leg lowers below a certain angular threshold, the system begins active counting.
3. Every full cycle from a raised (up) position to a lowered (down) position and back up again is considered one repetition.

The simplicity of this method performs detection without vertical reference points.

---

### 🟠 Squat Counter
**Measured Joint:** Hip–Knee–Ankle  
**Additional Reference:** Vertical position of the hip

**Concept:**
- The knee angle decreases as the user lowers into a squat and increases as they stand up.
- The vertical position of the hips is also tracked to ensure that the motion corresponds to an actual squat (not just slight knee bending).

**How it works:**
1. The system identifies the upright standing posture and uses the hip position as the baseline.
2. When both the knee angle decreases and the hips move downward, the system marks the “down” phase.
3. When the angle increases again and the hips rise close to the baseline, one repetition is counted.

---

## 6. Calibration
Calibration is not performed per user but is instead derived from a **reference training video** of each exercise.  
During calibration:
- The system analyzes a clean performance of the movement.
- It identifies the maximum and minimum joint angles.
- It extracts and stores threshold values for “up” and “down” phases.
- It optionally records vertical position deltas when needed (for push-ups or squats).

The calibration also determines whether vertical displacement is significant (if the range exceeds ~3%), enabling or disabling the use of the reference Y-axis shift in future sessions. These calibration values are saved as a configuration yaml file for each exercise and are later reused across all sessions.

---

## 7. Tolerance Handling
Human motion naturally varies from one repetition to another.  
To account for these small inconsistencies, a **tolerance margin** is applied to each threshold.  
This ensures that minor deviations in angle or position do not prevent a valid repetition from being counted.

---

## 8. Stability and Timing
Before starting to count, the system requires the user to maintain the initial posture for a short period (around one second).  
This prevents false starts and ensures that the baseline (either angle or position) is stable.

---

## 9. Summary
In summary, the repetition counting pipeline works as follows:
1. Capture video frames and detect body landmarks using MediaPipe Pose.  
2. Compute joint angles for the relevant body parts.  
3. Identify the baseline posture before starting.  
4. Monitor changes in joint angles and body position during motion.  
5. Recognize “up” and “down” transitions that define one complete repetition.  
6. Use calibration data from a training video and apply small tolerance margins to make counting smooth and realistic.  

---

## 10. Run Commands

To run the counting system via webcam (real-time), execute the following commands from the project root:

▶️ Push-Ups

python main.py --exercise pushup --source 0

▶️ Leg Raises

python main.py --exercise legraise --source 0

▶️ Squats

python main.py --exercise squat --source 0


## 11. Calibration (Train Commands)

To generate or re-train calibration data from reference videos and preview the calibration results visually, use:

▶️ Leg Raise Calibration

python train.py --exercise legraise --video data/reference_videos/legraise_example.mp4

▶️ Push-Up Calibration

python train.py --exercise pushup --video data/reference_videos/pushup_example.mp4

▶️ Squat Calibration

python train.py --exercise squat --video data/reference_videos/squat_example.mp4



These commands:

Analyze the reference video to extract movement patterns.

Compute automatic thresholds for up/down transitions.

Save calibration parameters into outputs/calibration/<exercise>.yaml.

Automatically replay the video with a live overlay (skeleton, angle, and rep detection) for visual verification.
