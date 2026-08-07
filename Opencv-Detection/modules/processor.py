import cv2 as cv
import numpy as np

def draw_rectangles(img: np.ndarray, rects, color: tuple, thickness: int = 2):
    """Draw rectangles around each detection."""
    for (x, y, w, h) in rects:
        cv.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    return img

def draw_circles(img: np.ndarray, rects, color: tuple, thickness: int = 2):
    """Draw circles around each detection (center + radius)."""
    for (x, y, w, h) in rects:
        center = (x + w // 2, y + h // 2)
        radius = int((w + h) / 4)
        cv.circle(img, center, radius, color, thickness)
    return img

def put_text(
    img: np.ndarray,
    text: str,
    position: tuple,
    font_scale: float,
    color: tuple,
    thickness: int = 2,
    font=cv.FONT_HERSHEY_SIMPLEX
):
    """Overlay text on the image."""
    if text:
        cv.putText(img, text, position, font, font_scale, color, thickness)
    return img