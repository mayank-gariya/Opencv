import cv2 as cv
from pathlib import Path

try:
    from cv2 import CascadeClassifier
except ImportError:
    try:
        # Some builds put it in cv2.cv2
        from cv2 import cv2 as cv2_native
        CascadeClassifier = cv2_native.CascadeClassifier
    except (ImportError, AttributeError):
        # Last resort: use cv2.CascadeClassifier
        if hasattr(cv, "CascadeClassifier"):
            CascadeClassifier = cv.CascadeClassifier
        else:
            raise ImportError(
                "Cannot find CascadeClassifier in cv2. "
                "Please ensure you have a stable OpenCV version "
                "(e.g., opencv-python-headless==4.9.0.80)."
            )

class CascadeDetector:
    def __init__(self, cascade_path: Path):
        self.cascade = CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise ValueError(f"Failed to load cascade from {cascade_path}")