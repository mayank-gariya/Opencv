import cv2
import mediapipe as mp
import numpy as np
import math
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hand skeleton connections layout
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index finger
    (9, 10), (10, 11), (11, 12),          # Middle finger
    (13, 14), (14, 15), (15, 16),         # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm Knuckles
]

model_path = r'D:\opencv-practice\hand_landmarker.task'
background_path = r'D:\opencv-practice\Gemini_Generated_Image_cxkqwocxkqwocxkq.png'

# Initialize MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
volume_pct = 0

MIN_DISTANCE = 20   
MAX_DISTANCE = 180  

# Load background image
background_template = cv2.imread(background_path)
if background_template is None:
    raise FileNotFoundError(f"Could not load background image from {background_path}")

# Resize background image to fit comfortably on screen (1100x662)
background_template = cv2.resize(background_template, (1100, 662))

BG_Y1, BG_Y2 = 145, 510  
BG_X1, BG_X2 = 95, 650   

# Automatically match webcam frame size to your background box slice shape
VIDEO_WIDTH = BG_X2 - BG_X1    
VIDEO_HEIGHT = BG_Y2 - BG_Y1  

# Track start time for video stream timestamp calculation
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break        
    
    frame = cv2.flip(frame, 1)
    
    # Resize camera feed to match the exact calculated slot size dynamically
    processed_frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
    fh, fw, _ = processed_frame.shape
    
    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    timestamp = int((time.time() - start_time) * 1000)
    results = detector.detect_for_video(mp_image, timestamp)
    
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            pixel_points = []
            for landmark in hand_landmarks:
                cx, cy = int(landmark.x * fw), int(landmark.y * fh)
                pixel_points.append((cx, cy))  
            
            thumb_tip = pixel_points[4]
            index_tip = pixel_points[8]
            
            # Draw standard skeletal connections onto processed_frame
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(pixel_points) and end_idx < len(pixel_points):
                    cv2.line(processed_frame, pixel_points[start_idx], pixel_points[end_idx], (255, 0, 0), 2)
            
            cv2.line(processed_frame, thumb_tip, index_tip, (0, 0, 255), 3)
                    
            for pt in pixel_points:
                cv2.circle(processed_frame, pt, 5, (0, 255, 0), -1)
                
            cv2.circle(processed_frame, thumb_tip, 8, (0, 255, 255), -1)
            cv2.circle(processed_frame, index_tip, 8, (0, 255, 255), -1)
            
            dx = index_tip[0] - thumb_tip[0]
            dy = index_tip[1] - thumb_tip[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance <= MIN_DISTANCE:
                volume_pct = 0
            elif distance >= MAX_DISTANCE:
                volume_pct = 100
            else:
                volume_pct = int(((distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)) * 100)
            
            center_x, center_y = (thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2
            if distance < MIN_DISTANCE + 5:
                cv2.circle(processed_frame, (center_x, center_y), 10, (0, 0, 255), -1)
            
            cv2.putText(processed_frame, f'{volume_pct}%', (15, 25), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,0, 255), 2)

    display_image = background_template.copy()

    display_image[BG_Y1:BG_Y2, BG_X1:BG_X2] = processed_frame

    # Update dynamic volume text readout on the dashboard
    cv2.putText(display_image, f"{volume_pct}%", (810, 315), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 255), 4)

    cv2.imshow("Volume controller", display_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()