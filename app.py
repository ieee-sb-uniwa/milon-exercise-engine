import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from milon_engine import FrameProcessor, PoseEstimator, Visualizer, load_config
from milon_engine.exercises import Squat, PushUp, LegRaise

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Milon Exercise Counter", layout="centered")
st.title("Milon Exercise Counter")
st.write("Επίλεξε άσκηση, δώσε άδεια στην κάμερα και ξεκίνα!")

# ── Exercise selection ────────────────────────────────────────────────────────

EXERCISES = {
    "Squat": ("squat", Squat),
    "Push-up": ("pushup", PushUp),
    "Leg Raise": ("legraise", LegRaise),
}

choice = st.selectbox("Άσκηση", list(EXERCISES.keys()))
config_name, ExerciseClass = EXERCISES[choice]

# Store the current selection in session_state so the processor can read it.
st.session_state["config_name"] = config_name
st.session_state["ExerciseClass"] = ExerciseClass

# ── Video processor ───────────────────────────────────────────────────────────
# We use a FIXED key ("exercise_stream") so the webrtc_streamer is never
# torn down when the user switches exercises — avoiding the Python 3.14
# shutdown bug ('NoneType' has no attribute 'is_alive').
# Instead, the processor checks session_state each frame and swaps the
# internal pipeline when the exercise changes.


class ExerciseProcessor(VideoProcessorBase):
    def __init__(self):
        self._current_choice = st.session_state.get("config_name")
        self._build_pipeline()

    def _build_pipeline(self):
        cfg_name = st.session_state.get("config_name", "squat")
        cls = st.session_state.get("ExerciseClass", Squat)
        config = load_config(cfg_name)
        exercise = cls(config)
        self.processor = FrameProcessor(exercise, PoseEstimator(), Visualizer())

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Rebuild pipeline if the user switched exercise
        new_choice = st.session_state.get("config_name")
        if new_choice != self._current_choice:
            self._current_choice = new_choice
            self._build_pipeline()

        img = frame.to_ndarray(format="bgr24")
        annotated = self.processor.process_frame(img)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ── WebRTC streamer ───────────────────────────────────────────────────────────
# ICE servers: Google STUN + open-relay.metered.ca TURN (free, no account needed).
# TURN is required on Streamlit Cloud because the server sits behind a firewall
# and peer-to-peer WebRTC connections cannot be established with STUN alone.

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {
            "urls": "turn:openrelay.metered.ca:80",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": "turn:openrelay.metered.ca:443",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": "turn:openrelay.metered.ca:443?transport=tcp",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
}

st.info(
    "Πάτα **START** για να ενεργοποιήσεις την κάμερα. "
    "Σταθεροποίησε τη στάση σου και το σύστημα θα ξεκινήσει αυτόματα."
)

webrtc_streamer(
    key="exercise_stream",  # σταθερό key — δεν γίνεται teardown κατά την αλλαγή άσκησης
    video_processor_factory=ExerciseProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
