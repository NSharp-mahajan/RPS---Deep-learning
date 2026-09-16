"""
ROCK • PAPER • SCISSORS — AI Gesture Classifier
------------------------------------------------
Live real-time webcam prediction application using a custom CNN.
Captures continuous video frames via WebRTC, runs model-aligned
preprocessing (128x128, RGB, normalized [0, 1]), applies prediction
smoothing, and renders live predictions with confidence and class probabilities.
"""

from collections import deque
from pathlib import Path
import threading
import time
from typing import Optional, Tuple

# Safe imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import numpy as np
import streamlit as st

# Attempt to import WebRTC components
try:
    import av
    from streamlit_webrtc import (
        RTCConfiguration,
        VideoProcessorBase,
        WebRtcMode,
        webrtc_streamer,
    )
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


# ==============================================================================
# Configuration & Constants
# ==============================================================================

# Target frame dimensions for the trained CNN model
IMG_SIZE: Tuple[int, int] = (128, 128)

# Model class mapping (0: Rock, 1: Paper, 2: Scissors)
CLASS_NAMES: list[str] = ["Rock", "Paper", "Scissors"]

CLASS_ICONS: dict[str, str] = {
    "Rock": "✊ ROCK",
    "Paper": "✋ PAPER",
    "Scissors": "✌️ SCISSORS",
}

# Dynamic, portable path resolution relative to this application file
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODEL_PATH: Path = PROJECT_ROOT / "models" / "rps_cnn.keras"

# Public STUN servers for WebRTC connectivity
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})


# ==============================================================================
# Model Loading & Caching
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    """
    Load the trained Keras model from disk once.
    Cached across sessions so it is never reloaded per frame or rerun.
    """
    if not path.exists():
        return None

    import tensorflow as tf
    try:
        loaded_model = tf.keras.models.load_model(str(path))
        return loaded_model
    except Exception as err:
        st.error(f"Error loading trained model from '{path}': {err}")
        return None


# ==============================================================================
# Preprocessing Pipeline (Exact Match to Training)
# ==============================================================================

def preprocess_frame(bgr_image: np.ndarray, target_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """
    Preprocess a raw BGR frame from OpenCV / WebRTC for the custom CNN:
      1. Convert BGR to RGB.
      2. Resize to 128 x 128 using bilinear interpolation.
      3. Cast pixel values to float32.
      4. Normalize to [0.0, 1.0] by dividing by 255.0.
      5. Add batch dimension -> (1, 128, 128, 3).
    """
    if CV2_AVAILABLE:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
    else:
        from PIL import Image
        # BGR to RGB via array slicing
        rgb_arr = bgr_image[:, :, ::-1] if len(bgr_image.shape) == 3 and bgr_image.shape[2] == 3 else bgr_image
        pil_img = Image.fromarray(rgb_arr)
        resized_pil = pil_img.resize(target_size, Image.Resampling.BILINEAR)
        normalized = np.asarray(resized_pil, dtype=np.float32) / 255.0

    batch_tensor = np.expand_dims(normalized, axis=0)
    return batch_tensor


def predict_gesture(model, preprocessed_tensor: np.ndarray) -> np.ndarray:
    """
    Run forward pass through the custom CNN and return raw softmax probabilities.
    Output shape: (3,) corresponding to [Rock, Paper, Scissors].
    """
    raw_probs = model.predict(preprocessed_tensor, verbose=0)
    return raw_probs[0]


# ==============================================================================
# Live Video Stream Processor with Prediction Smoothing
# ==============================================================================

if WEBRTC_AVAILABLE:
    class LiveVideoProcessor(VideoProcessorBase):
        """
        Processes continuous incoming video frames in a background thread:
          - Preprocesses each frame
          - Performs model inference
          - Smooths predictions over recent frames to prevent visual jitter
          - Draws an on-frame HUD overlay
          - Exposes thread-safe state for UI rendering
        """

        def __init__(self, model):
            self.model = model
            self.lock = threading.Lock()
            # History buffer of recent probabilities for smoothing (size 5)
            self.history = deque(maxlen=5)
            self.latest_label: str = "Awaiting hand gesture..."
            self.latest_confidence: float = 0.0
            self.latest_probabilities: dict[str, float] = {
                "Rock": 0.0,
                "Paper": 0.0,
                "Scissors": 0.0,
            }
            self.has_prediction: bool = False

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            if self.model is not None:
                try:
                    # Preprocess frame
                    tensor = preprocess_frame(img, target_size=IMG_SIZE)

                    # Model prediction
                    raw_probs = predict_gesture(self.model, tensor)

                    # Apply lightweight prediction smoothing
                    with self.lock:
                        self.history.append(raw_probs)
                        smoothed_probs = np.mean(self.history, axis=0)

                        pred_idx = int(np.argmax(smoothed_probs))
                        self.latest_label = CLASS_NAMES[pred_idx]
                        self.latest_confidence = float(smoothed_probs[pred_idx]) * 100.0
                        self.latest_probabilities = {
                            name: float(smoothed_probs[i]) * 100.0
                            for i, name in enumerate(CLASS_NAMES)
                        }
                        self.has_prediction = True

                        current_label = self.latest_label
                        current_conf = self.latest_confidence

                    if CV2_AVAILABLE:
                        # Draw subtle HUD banner on the video frame
                        h, w, _ = img.shape
                        overlay = img.copy()
                        cv2.rectangle(overlay, (0, 0), (w, 55), (15, 23, 42), -1)
                        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

                        # Text HUD
                        hud_text = f"{current_label.upper()}  {current_conf:.1f}%"
                        cv2.putText(
                            img,
                            hud_text,
                            (20, 38),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (56, 189, 248),
                            2,
                            cv2.LINE_AA,
                        )

                except Exception as e:
                    # Non-fatal error handling: keep frame flowing
                    if CV2_AVAILABLE:
                        cv2.putText(
                            img,
                            f"Inference error: {e}",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
            else:
                # Model not loaded overlay
                if CV2_AVAILABLE:
                    h, w, _ = img.shape
                    cv2.rectangle(img, (0, 0), (w, 50), (0, 0, 150), -1)
                    cv2.putText(
                        img,
                        "Model not loaded (models/rps_cnn.keras missing)",
                        (15, 33),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

            return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==============================================================================
# UI Component Renderers
# ==============================================================================

def render_prediction_panel(label: str, confidence: float, probabilities: dict[str, float], active: bool):
    """Render the right-side prediction card, confidence bar, and probabilities."""
    st.markdown("### PREDICTION")

    if active:
        display_label = CLASS_ICONS.get(label, label.upper())
        # Styled prediction card
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                border: 2px solid #38bdf8;
                border-radius: 12px;
                padding: 22px;
                text-align: center;
                box-shadow: 0 4px 14px rgba(56, 189, 248, 0.2);
                margin-bottom: 20px;
            ">
                <div style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; letter-spacing: 1px;">
                    {display_label}
                </div>
                <div style="font-size: 1.8rem; font-weight: 600; color: #38bdf8; margin-top: 6px;">
                    {confidence:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Confidence")
        st.progress(min(max(confidence / 100.0, 0.0), 1.0), text=f"{confidence:.1f}%")

        st.markdown("#### Class Probabilities")
        for name in CLASS_NAMES:
            prob = probabilities.get(name, 0.0)
            col_l, col_r = st.columns([1, 3])
            with col_l:
                st.write(f"**{name}**")
            with col_r:
                st.progress(min(max(prob / 100.0, 0.0), 1.0), text=f"{prob:.1f}%")

    else:
        # Inactive initial state
        st.markdown(
            """
            <div style="
                background-color: #1e293b;
                border: 2px dashed #475569;
                border-radius: 12px;
                padding: 24px;
                text-align: center;
                margin-bottom: 20px;
            ">
                <div style="font-size: 1.3rem; color: #94a3b8;">
                    Show Rock, Paper, or Scissors to the camera.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Confidence")
        st.progress(0.0, text="0.0%")

        st.markdown("#### Class Probabilities")
        for name in CLASS_NAMES:
            col_l, col_r = st.columns([1, 3])
            with col_l:
                st.write(f"**{name}**")
            with col_r:
                st.progress(0.0, text="0.0%")


# ==============================================================================
# Main Application
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Rock • Paper • Scissors — AI Gesture Classifier",
        page_icon="✊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Header section
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 10px;">
            <h1 style="margin: 0; font-size: 2.5rem; letter-spacing: 2px;">ROCK • PAPER • SCISSORS</h1>
            <h3 style="margin: 4px 0; color: #38bdf8; font-weight: 500;">AI Gesture Classifier</h3>
            <p style="color: #94a3b8; font-size: 1.05rem;">Real-time hand gesture recognition using a custom CNN</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Load Model
    model = load_model(MODEL_PATH)

    # Status Bar: Model and Camera
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        if model is not None:
            st.success("Model status: **Model loaded ✓**")
        else:
            st.error("Model status: **Model file not found**")

    with col_stat2:
        if not WEBRTC_AVAILABLE:
            st.error("Camera status: **Camera unavailable (missing streamlit-webrtc)**")
        else:
            st.info("Camera status: **Camera ready**")

    # Missing Model Error Banner (if applicable)
    if model is None:
        st.warning(
            f"⚠️ **Trained Model Required**\n\n"
            f"The application checked for the trained model at:\n"
            f"```text\n{MODEL_PATH}\n```\n\n"
            f"**Action Required**: Save your trained model to `models/rps_cnn.keras`. In your notebook, run:\n"
            f"```python\n"
            f"import os\n"
            f"os.makedirs('models', exist_ok=True)\n"
            f"regularized_model.save('models/rps_cnn.keras')\n"
            f"```\n"
            f"After saving the file, refresh this page to begin live predictions."
        )

    # Missing Dependency Banner (if applicable)
    if not WEBRTC_AVAILABLE:
        st.error(
            "⚠️ **Missing Dependency for Live Video**\n\n"
            "Please install the required streaming packages by running:\n"
            "```bash\npip install streamlit-webrtc av\n```"
        )
        return

    # Two-Column Layout: Left (Live Camera) | Right (Prediction Panel)
    col_camera, col_prediction = st.columns([1.2, 1.0], gap="large")

    with col_camera:
        st.markdown("### LIVE CAMERA")
        st.caption("Click **START** below to enable continuous camera prediction.")

        # WebRTC Live Video Streamer
        webrtc_ctx = webrtc_streamer(
            key="rps-live-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=lambda: LiveVideoProcessor(model),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if webrtc_ctx.state.playing:
            st.success("🟢 Streaming live video... Showing real-time CNN predictions.")
        else:
            st.info("Waiting for video stream. Click **START** above to begin.")

    with col_prediction:
        prediction_placeholder = st.empty()

        # Initial inactive render
        with prediction_placeholder.container():
            render_prediction_panel(
                label="Awaiting hand gesture...",
                confidence=0.0,
                probabilities={"Rock": 0.0, "Paper": 0.0, "Scissors": 0.0},
                active=False,
            )

    # Live UI update loop while camera is active
    if webrtc_ctx.state.playing:
        while webrtc_ctx.state.playing:
            if webrtc_ctx.video_processor:
                with webrtc_ctx.video_processor.lock:
                    has_pred = webrtc_ctx.video_processor.has_prediction
                    lbl = webrtc_ctx.video_processor.latest_label
                    conf = webrtc_ctx.video_processor.latest_confidence
                    probs = dict(webrtc_ctx.video_processor.latest_probabilities)

                if has_pred:
                    with prediction_placeholder.container():
                        render_prediction_panel(lbl, conf, probs, active=True)
            time.sleep(0.08)

    # Model Information Footer Panel
    st.divider()
    st.markdown("### MODEL INFORMATION")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.metric(label="Architecture", value="Custom CNN + Dropout")
    with info_col2:
        st.metric(label="Input Dimension", value="128 × 128 × 3")
    with info_col3:
        st.metric(label="Classes", value="Rock • Paper • Scissors")
    with info_col4:
        st.metric(label="Recorded Test Accuracy", value="96.51%")

    st.caption("Note: Test accuracy is 96.51%. Actual prediction confidence depends on lighting, framing, and hand orientation.")


if __name__ == "__main__":
    main()
