import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle


with open(r'D:\opencv-practice\sign_model.p','rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']

labels_dict = {0: 'happy', 1: 'sad', 2: 'python'}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (9, 10), (10, 11), (11, 12),           # Middle finger
    (13, 14), (14, 15), (15, 16),          # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm Knuckles
]

model_path = r'D:\opencv-practice\hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp == 0:
        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
    results = detector.detect_for_video(mp_image, timestamp)
    
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            pixel_points = []
            data_aux = []
            
            for landmark in hand_landmarks:
                cx ,cy = int(landmark.x * w) ,int(landmark.y * h)
                pixel_points.append((cx,cy))
                
                data_aux.append(landmark.x)
                data_aux.append(landmark.y)
                
            if len(data_aux) == 42:
                predictions = model.predict([np.asarray(data_aux)])
                predictions_text = labels_dict[int(predictions[0])]
                
                x_min = min([pt[0] for pt in pixel_points])
                y_min = min([pt[1] for pt in pixel_points])
                cv2.putText(frame, predictions_text, (x_min, y_min - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            
            # Draw Connections using OpenCV
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(pixel_points) and end_idx < len(pixel_points):
                    cv2.line(frame, pixel_points[start_idx], pixel_points[end_idx], (255, 0, 0), 2)
            
            # Draw Joint Circles
            for pt in pixel_points:
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)
                
    cv2.imshow("Sign Language Detector (Tasks API)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()