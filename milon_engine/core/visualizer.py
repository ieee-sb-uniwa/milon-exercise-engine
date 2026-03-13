import cv2
import numpy as np
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections
from typing import Optional, Any

from milon_engine.core.models import ExerciseResult


class Visualizer:
    """Renders pose landmarks and exercise feedback onto a frame.

    All drawing logic is consolidated under render(), which is the single
    public method callers should use.
    """

    def __init__(self):
        self._colors = {
            "accent": (0, 255, 127),
            "warning": (0, 165, 255),
            "error": (50, 50, 255),
            "neutral": (220, 220, 220),
            "text": (20, 20, 20),
            "panel": (40, 40, 40),
            "white": (255, 255, 255),
            "border": (80, 80, 80),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        raw_results: Any,
        result: Optional[ExerciseResult],
    ) -> np.ndarray:
        """Compose all visual layers and return the annotated frame.

        Args:
            frame:       Original BGR frame from the camera/video.
            raw_results: Raw MediaPipe results object (for skeleton drawing).
                         Pass None when no detection occurred.
            result:      ExerciseResult from Exercise.evaluate(), or None
                         when no landmarks were detected.

        Returns:
            Annotated BGR frame (a copy — the original is not mutated).
        """
        annotated = frame.copy()

        if result is None or raw_results is None or not raw_results.pose_landmarks:
            return self._render_no_detection(annotated)

        self._draw_skeleton(annotated, raw_results)
        self._draw_feedback(annotated, result.rep_count, result.stage, result.angle)
        self._draw_system_status(annotated, result.system_stage)

        return annotated

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_skeleton(self, image: np.ndarray, raw_results: Any) -> None:
        """Draw the MediaPipe pose skeleton onto image in-place."""
        landmark_drawing_spec = mp_drawing.DrawingSpec(
            color=(0, 255, 127), thickness=2, circle_radius=2
        )
        connection_drawing_spec = mp_drawing.DrawingSpec(
            color=(255, 100, 180), thickness=2, circle_radius=2
        )
        mp_drawing.draw_landmarks(
            image,
            raw_results.pose_landmarks[0],
            PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec,
            connection_drawing_spec,
        )

    def _draw_feedback(
        self,
        image: np.ndarray,
        rep_count: int,
        stage: Optional[str],
        angle: Optional[float],
    ) -> None:
        """Draw the rep-count / stage / angle panel in the top-left corner."""
        c = self._colors
        x1, y1, x2, y2 = 15, 15, 270, 135

        # semi-transparent background panel
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), c["panel"], -1)
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        cv2.rectangle(image, (x1, y1), (x2, y2), c["border"], 2)

        # REPS label + value
        cv2.putText(
            image,
            "REPS",
            (25, 45),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            c["white"],
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{rep_count}",
            (150, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            c["accent"],
            3,
            cv2.LINE_AA,
        )

        # STAGE label + value
        cv2.putText(
            image,
            "STAGE",
            (25, 90),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            c["white"],
            2,
            cv2.LINE_AA,
        )
        stage_color = (
            c["warning"]
            if stage == "down"
            else c["accent"]
            if stage == "up"
            else c["neutral"]
        )
        cv2.putText(
            image,
            (stage or "-").upper(),
            (150, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            stage_color,
            3,
            cv2.LINE_AA,
        )

        # optional angle
        if angle is not None:
            cv2.putText(
                image,
                f"{int(angle)}\u00b0",
                (25, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                c["accent"],
                2,
                cv2.LINE_AA,
            )

    def _draw_system_status(self, image: np.ndarray, system_stage: str) -> None:
        """Draw a status banner at the bottom of the frame."""
        c = self._colors
        status_map = {
            "waiting": ("Searching for user...", c["error"]),
            "aligning": ("Aligning posture...", c["warning"]),
            "ready": ("Ready to start!", c["accent"]),
            "counting": ("Counting repetitions...", c["accent"]),
        }
        status_text, color = status_map.get(system_stage, ("", c["white"]))
        if not status_text:
            return

        h, w = image.shape[:2]
        panel_y1, panel_y2 = h - 80, h - 30

        overlay = image.copy()
        cv2.rectangle(overlay, (40, panel_y1), (w - 40, panel_y2), c["panel"], -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        cv2.rectangle(image, (40, panel_y1), (w - 40, panel_y2), c["border"], 2)

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

    def _render_no_detection(self, frame: np.ndarray) -> np.ndarray:
        """Overlay a dark tint and a warning message when no person is detected."""
        c = self._colors
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), c["panel"], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(
            frame,
            "No person detected",
            (int(w / 2) - 150, int(h / 2)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.5,
            c["error"],
            3,
            cv2.LINE_AA,
        )
        return frame
