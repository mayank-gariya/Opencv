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

# ---------- HELPER FUNCTIONS ----------
def hex_to_bgr(hex_color: str) -> tuple:
    """Convert hex (#RRGGBB) to BGR tuple."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV uses BGR


def load_image_from_upload(uploaded_file) -> np.ndarray:
    """Load an uploaded image file into a BGR numpy array."""
    if uploaded_file is None:
        return None
    img = Image.open(uploaded_file)
    img_rgb = np.array(img.convert("RGB"))
    return cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)


def get_image_from_camera(camera_input) -> np.ndarray:
    """Convert camera input to BGR numpy array."""
    if camera_input is None:
        return None
    img = Image.open(camera_input)
    img_rgb = np.array(img.convert("RGB"))
    return cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)


def process_image(
    img: np.ndarray,
    detector: CascadeDetector,
    params: dict
) -> np.ndarray:
    """Apply detection and drawing to the image."""
    if img is None:
        return None

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
        img = put_text(
            img,
            params["text"],
            params["text_pos"],
            params["font_scale"],
            color,
            thickness
        )

    return img
def process_video(
    input_path: str,
    detector: CascadeDetector,
    params: dict
) -> str:
    """
    Process an entire video using OpenCV and return
    the path of the processed video.
    """

    cap = cv.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Could not open video.")

    # Get video properties
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Fallback FPS if OpenCV cannot detect it
    if fps <= 0:
        fps = 30.0

    # Create temporary output video
    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )
    output_path = output_file.name
    output_file.close()

    # MP4 codec
    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    writer = cv.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create output video.")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        processed_frame = process_image(
            frame,
            detector,
            params
        )

        writer.write(processed_frame)
    cap.release()
    writer.release()

    return output_path

# ---------- SIDEBAR CONTROLS ----------
with st.sidebar:
    st.title("🎛️ Controls")

    st.subheader("Input Source")
    source = st.radio(
        "Choose source",
        ["Image", "Camera", "Video"],
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

    st.divider()

    st.subheader("Detection Settings")
    scale_factor = st.slider(
        "Scale Factor",
        min_value=1.01,
        max_value=2.0,
        value=1.1,
        step=0.01
    )
    min_neighbors = st.slider(
        "Min Neighbors",
        min_value=1,
        max_value=20,
        value=5
    )
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
    thickness = st.slider("Thickness", min_value=1, max_value=10, value=2)

    text = st.text_input("Text Overlay", value="")
    font_scale = st.slider("Font Scale", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    text_pos_x = st.slider("Text X", min_value=0, max_value=500, value=10)
    text_pos_y = st.slider("Text Y", min_value=0, max_value=500, value=30)
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

    st.divider()
    st.caption("Built with ❤️ using OpenCV & Streamlit")


# ---------- MAIN AREA ----------
st.header("🎯 Detection Dashboard")
st.info("Upload or capture an image/video and apply cascade detection with custom overlays.")

img_original = None
img_processed = None

if source == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
        key="image_upload"
    )
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

```python
else:  # Video

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video:

        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as tmp:

            tmp.write(uploaded_video.read())
            input_video_path = tmp.name

        # Store input path
        st.session_state.video_path = input_video_path

        if detector is not None:
            with st.spinner("🔍 Processing video..."):
                try:
                    output_video_path = process_video(
                        input_video_path,
                        detector,
                        detection_params
                    )
                    st.session_state.processed_video_path = (
                        output_video_path
                    )
                except Exception as e:
                    st.error(
                        f"Failed to process video: {e}"
                    )
        else:
            st.warning(
                "Please select a valid cascade classifier."
            )

    if (
        "processed_video_path" in st.session_state
        and st.session_state.processed_video_path
        and os.path.exists(
            st.session_state.processed_video_path
        )
    ):
        st.subheader("🎯 Processed Video")
        st.video(
            st.session_state.processed_video_path
        )

    elif (
        "video_path" in st.session_state
        and st.session_state.video_path
        and os.path.exists(
            st.session_state.video_path
        )
    ):

        st.subheader("🎥 Original Video")
        st.video(
            st.session_state.video_path
        )
    else:
        st.info("📤 Please upload a video.")
```



if img_original is not None:
    if detector is not None:
        img_processed = process_image(img_original, detector, detection_params)
        img_display = cv.cvtColor(img_processed, cv.COLOR_BGR2RGB)
        st.image(img_display, use_container_width=True, caption="Processed Output")
    else:
        img_display = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)
        st.image(img_display, use_container_width=True, caption="Original (no detector)")
else:
    st.warning("No image available. Please provide input from the selected source.")
