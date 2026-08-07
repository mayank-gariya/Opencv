import cv2 as cv
from cv2 import CascadeClassifier 
from pathlib import Path

class CascadeDetector:
    def __init__(self, cascade_path: Path):
        # No attribute check needed – we use the imported class
        self.cascade = CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise ValueError(f"Failed to load cascade from {cascade_path}")
