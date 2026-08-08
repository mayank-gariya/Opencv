import streamlit as st
import pathlib
import cv2 as cv
import numpy as np
from PIL import Image
import tempfile
import os
from modules.detector import CascadeDetector
from modules.processor import draw_rectangles, draw_circles, put_text
from css import load_css
import imageio_ffmpeg
import subprocess
import hashlib

# ---------- WebRTC imports ----------
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Detection Dashboard",
    page_icon="🎮",
    layout="wide"
)

# ---------- LOAD CSS ----------
load_css()

# ---------- PATHS ----------
BASE_DIR = pathlib.Path(__file__).parent
CASCADE_DIR = BASE_DIR / "cascade"

# ---------- SESSION STATE INIT ----------
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None
# For live webcam, we store the current detector and params
if "live_detector" not in st.session_state:
    st.session_state.live_detector = None
if "live_params" not in st.session_state:
    st.session_state.live_params = {}

# ---------- HELPER FUNCTIONS (unchanged) ----------
def hex_to_bgr(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)

def load_image_from_upload(uploaded_file) -> np.ndarray:
    if uploaded_file is None:
        return None
    img = Image.open(uploaded_file)
    img_rgb = np.array(img.convert("RGB"))
    return cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)

def get_image_from_camera(camera_input) -> np.ndarray:
    if camera_input is None:
        return None
    img = Image.open(camera_input)
    img_rgb = np.array(img.convert("RGB"))
    return cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)

def process_image(img: np.ndarray, detector: CascadeDetector, params: dict) -> np.ndarray:
    if img is None or detector is None:
        return img
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rects = detector.detect(
        gray,
        scaleFactor=params["scale"],
        minNeighbors=params["neighbors"],
        minSize=params["min_size"]
    )
    shape = params["shape"]
    color = params["color_bgr"]
    thickness = params["thickness"]
    if shape == "Rectangle":
        img = draw_rectangles(img, rects, color, thickness)
    elif shape == "Circle":
        img = draw_circles(img, rects, color, thickness)
    if params["text"]:
        img = put_text(img, params["text"], params["text_pos"],
                       params["font_scale"], color, thickness)
    return img

def process_video(input_path: str, detector: CascadeDetector, params: dict) -> str:
    # ... (your existing function, unchanged) ...
    # (I'm omitting the full function here for brevity, but keep it as in your original)
    # Make sure it's present in your actual code.
    pass  # Placeholder – you must copy your original process_video here

# ---------- LIVE WEBCAM TRANSFORMER (reads from session_state) ----------
class LiveVideoTransformer(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Get the most current detector and params from session_state
        detector = st.session_state.get("live_detector")
        params = st.session_state.get("live_params", {})
        try:
            if detector is not None:
                processed = process_image(img, detector, params)
            else:
                processed = img
        except Exception as e:
            # If anything fails, return original frame to avoid crashing
            st.warning(f"Live processing error: {e}")
            processed = img
        return processed

# ---------- SIDEBAR CONTROLS ----------
with st.sidebar:
    st.title("🎛️ Controls")
    st.subheader("Input Source")
    source = st.radio(
        "Choose source",
        ["Image", "Camera", "Video", "Live Webcam"],
        index=0
    )
    st.divider()

    cascade_files = list(CASCADE_DIR.glob("*.xml"))
    if not cascade_files:
        st.warning("No cascade XML files found in 'cascade/' folder.")
        cascade_names = []
    else:
        cascade_names = [f.name for f in cascade_files]

    selected_cascade = st.selectbox(
        "Cascade Classifier",
        cascade_names,
        index=0 if cascade_names else None
    )

    if selected_cascade:
        cascade_path = CASCADE_DIR / selected_cascade
        try:
            detector = CascadeDetector(cascade_path)
        except Exception as e:
            st.error(f"Failed to load cascade: {e}")
            detector = None
    else:
        detector = None

    # Update session_state for live webcam
    st.session_state.live_detector = detector

    st.divider()
    st.subheader("Detection Settings")
    scale_factor = st.slider("Scale Factor", 1.01, 2.0, 1.1, 0.01)
    min_neighbors = st.slider("Min Neighbors", 1, 20, 5)
    min_size = st.text_input("Min Size (w,h)", value="30,30")
    try:
        min_w, min_h = map(int, min_size.split(","))
        min_size_tuple = (min_w, min_h)
    except:
        min_size_tuple = (30, 30)

    st.divider()
    st.subheader("Drawing Options")
    shape = st.selectbox("Shape", ["None", "Rectangle", "Circle"], index=0)
    color_hex = st.color_picker("Color", value="#FF8C00")
    color_bgr = hex_to_bgr(color_hex)
    thickness = st.slider("Thickness", 1, 10, 2)
    text = st.text_input("Text Overlay", value="")
    font_scale = st.slider("Font Scale", 0.5, 3.0, 1.0, 0.1)
    text_pos_x = st.slider("Text X", 0, 500, 10)
    text_pos_y = st.slider("Text Y", 0, 500, 30)
    text_pos = (text_pos_x, text_pos_y)

    detection_params = {
        "scale": scale_factor,
        "neighbors": min_neighbors,
        "min_size": min_size_tuple,
        "shape": shape,
        "color_bgr": color_bgr,
        "thickness": thickness,
        "text": text,
        "font_scale": font_scale,
        "text_pos": text_pos,
    }
    # Update session_state for live webcam
    st.session_state.live_params = detection_params

    st.divider()
    st.caption("Built with ❤️ using OpenCV & Streamlit")

# ---------- MAIN AREA ----------
st.header("🎯 Detection Dashboard")
st.info("Upload or capture an image/video and apply cascade detection with custom overlays.")

img_original = None
img_processed = None

if source == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="image_upload")
    if uploaded_file:
        img_original = load_image_from_upload(uploaded_file)
    else:
        st.info("📤 Please upload an image to get started.")

elif source == "Camera":
    camera_input = st.camera_input("Take a picture")
    if camera_input:
        img_original = get_image_from_camera(camera_input)
    else:
        st.info("📸 Click the camera button to capture an image.")

elif source == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="video_upload")
    if uploaded_video:
        video_bytes = uploaded_video.getvalue()
        video_hash = hashlib.md5(video_bytes).hexdigest()
        if (st.session_state.get("video_hash") != video_hash or
            st.session_state.get("processed_video_path") is None):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_bytes)
                input_video_path = tmp.name
            st.session_state.video_path = input_video_path
            st.session_state.video_hash = video_hash
            st.session_state.processed_video_path = None
            if detector is not None:
                with st.spinner("🔍 Processing video frame-by-frame..."):
                    try:
                        output_video_path = process_video(input_video_path, detector, detection_params)
                        st.session_state.processed_video_path = output_video_path
                    except Exception as e:
                        st.error(f"Failed to process video: {e}")
            else:
                st.warning("Please select a valid cascade classifier.")
    if (st.session_state.get("processed_video_path") and
        os.path.exists(st.session_state.processed_video_path)):
        st.subheader("🎯 Processed Video")
        st.video(st.session_state.processed_video_path)
    elif (st.session_state.get("video_path") and
          os.path.exists(st.session_state.video_path)):
        st.subheader("🎥 Original Video")
        st.video(st.session_state.video_path)
    else:
        st.info("📤 Please upload a video.")

elif source == "Live Webcam":
    st.subheader("📹 Live Webcam Feed")
    # The transformer will use the latest detector and params from session_state
    webrtc_streamer(
        key="live-webcam",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=LiveVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,   # simpler threading
    )

# ---------- Display processed image for Image/Camera ----------
if source in ["Image", "Camera"] and img_original is not None:
    if detector is not None:
        img_processed = process_image(img_original, detector, detection_params)
        img_display = cv.cvtColor(img_processed, cv.COLOR_BGR2RGB)
        st.image(img_display, use_container_width=True, caption="Processed Output")
    else:
        img_display = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)
        st.image(img_display, use_container_width=True, caption="Original (no detector)")
