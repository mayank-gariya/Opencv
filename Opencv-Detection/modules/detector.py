import cv2 as cv
import numpy as np
from pathlib import Path

class CascadeDetector:
    def __init__(self, cascade_path: Path):
        # Check if CascadeClassifier exists
        if not hasattr(cv, "CascadeClassifier"):
            raise ImportError(
                "OpenCV's CascadeClassifier is missing. "
                "Try reinstalling opencv-python."
            )
        self.cascade = cv.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise ValueError(f"Failed to load cascade from {cascade_path}")
