import cv2 as cv
import numpy as np
from pathlib import Path

class CascadeDetector:
    """Wrapper for OpenCV cascade classifiers."""
    
    def __init__(self, cascade_path: Path):
        self.cascade = cv.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise ValueError(f"Failed to load cascade from {cascade_path}")
    
    def detect(self, gray_img: np.ndarray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        """Return list of rectangles (x, y, w, h)."""
        return self.cascade.detectMultiScale(
            gray_img,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize
        )