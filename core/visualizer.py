import cv2
import numpy as np

class Visualizer:

    def __init__(self, mp_drawing, mp_pose):
        self.mp_drawing = mp_drawing
        self.mp_pose = mp_pose

        self.colors = {
            "accent": (0, 255, 127),
            "warning": (0, 165, 255),
            "error": (50, 50, 255),
            "neutral": (220, 220, 220),
            "text": (20, 20, 20),
            "panel": (40, 40, 40),
            "white": (255, 255, 255),
            "border": (80, 80, 80),
        }

    #draw skeleton with mediapipe
    def draw_skeleton(self, image, results):
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 127), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(255, 100, 180), thickness=2, circle_radius=2),
        )

    # repetitions feedback
    def draw_feedback(self, image, rep_count, stage, angle=None):

        c = self.colors

        # position
        x1, y1 = 15, 15
        x2, y2 = 270, 135

        # panel background
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), c["panel"], -1)
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        cv2.rectangle(image, (x1, y1), (x2, y2), c["border"], 2)

        # text (REPS, STAGE, ANGLE)
        cv2.putText(image, 'REPS', (25, 45), cv2.FONT_HERSHEY_DUPLEX, 0.7, c["white"], 2, cv2.LINE_AA)
        cv2.putText(image, f'{rep_count}', (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, c["accent"], 3, cv2.LINE_AA)

        cv2.putText(image, 'STAGE', (25, 90), cv2.FONT_HERSHEY_DUPLEX, 0.7, c["white"], 2, cv2.LINE_AA)
        stage_color = (
            c["warning"] if stage == "down"
            else c["accent"] if stage == "up"
            else c["neutral"]
        )
        cv2.putText(image, f'{(stage or "-").upper()}', (150, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, stage_color, 3, cv2.LINE_AA)

        if angle is not None:
            cv2.putText(image, f'{int(angle)}°', (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c["accent"], 2, cv2.LINE_AA)


    # state status feedback
    def draw_system_status(self, image, system_stage):

        c = self.colors

        status_text, color = {
            "waiting": ("Searching for user...", c["error"]),
            "aligning": ("Aligning posture...", c["warning"]),
            "ready": ("Ready to start!", c["accent"]),
            "counting": ("Counting repetitions...", c["accent"]),
        }.get(system_stage, ("", c["white"]))

        if not status_text:
            return

        # overlay position
        overlay = image.copy()
        h, w = image.shape[:2]
        panel_y1, panel_y2 = h - 80, h - 30

        # panel
        cv2.rectangle(overlay, (40, panel_y1), (w - 40, panel_y2), c["panel"], -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        cv2.rectangle(image, (40, panel_y1), (w - 40, panel_y2), c["border"], 2)

        # text
        cv2.putText(
            image,
            status_text,
            (60, h - 45),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )
