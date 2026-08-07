import cv2 as cv
from pathlib import Path

# Try multiple ways to get CascadeClassifier
try:
    from cv2 import CascadeClassifier
except ImportError:
    try:
        from cv2 import cv2 as cv2_native
        CascadeClassifier = cv2_native.CascadeClassifier
    except (ImportError, AttributeError):
        if hasattr(cv, "CascadeClassifier"):
            CascadeClassifier = cv.CascadeClassifier
        else:
            raise ImportError(
                "Cannot find CascadeClassifier in cv2. "
                "Please ensure you have a stable OpenCV version "
                "(e.g., opencv-python-headless==4.9.0.80)."
            )

class CascadeDetector:
    """Wrapper for OpenCV cascade classifiers."""
    
    def __init__(self, cascade_path: Path):
        self.cascade = CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise ValueError(f"Failed to load cascade from {cascade_path}")
    
    def detect(self, gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        """Return list of rectangles (x, y, w, h)."""
        return self.cascade.detectMultiScale(
            gray_img,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize
        )
