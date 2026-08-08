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

# ---------- SESSION STATE ----------
if "live_detector" not in st.session_state:
    st.session_state.live_detector = None
if "live_params" not in st.session_state:
    st.session_state.live_params = {}

# ---------- HELPER FUNCTIONS ----------
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

# ---------- LIVE WEBCAM TRANSFORMER ----------
class LiveVideoTransformer(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        detector = st.session_state.get("live_detector")
        params = st.session_state.get("live_params", {})
        try:
            if detector is not None:
                processed = process_image(img, detector, params)
            else:
                processed = img
        except Exception as e:
            st.warning(f"Live processing error: {e}")
            processed = img
        return processed

# ---------- SIDEBAR CONTROLS ----------
with st.sidebar:
    st.title("🎛️ Controls")
    st.subheader("Input Source")
    source = st.radio(
        "Choose source",
        ["Image", "Camera", "Live Webcam"],   # Video removed
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
    st.session_state.live_params = detection_params

    st.divider()
    st.caption("Built with ❤️ using OpenCV & Streamlit")

# ---------- MAIN AREA ----------
st.header("🎯 Detection Dashboard")
st.info("Upload or capture an image, or use live webcam with cascade detection.")

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

elif source == "Live Webcam":
    st.subheader("📹 Live Webcam Feed")
    webrtc_streamer(
        key="live-webcam",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=LiveVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
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
