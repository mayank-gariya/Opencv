import cv2 as cv
import numpy as np
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    with open('face_database.pkl', 'rb') as f:
        face_db = pickle.load(f)
except FileNotFoundError:
    print("Database file not found! Please run train_or_embed.py first.")
    exit()
    
base_options = python.BaseOptions(model_asset_path=r'D:\\opencv-practice\\face_landmarker_v2_with_blendshapes.task')
# --- Update your Options Setup ---
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    min_face_detection_confidence=0.4, 
    min_face_presence_confidence=0.4
)
detector = vision.FaceLandmarker.create_from_options(options)


cap = cv.VideoCapture(0)
THRESHOLD = 0.5

while True:
    ret, frame = cap.read()
    if not ret: break
    
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    detection_res = detector.detect(mp_image)
    identity = "Unknown Face"
    confidence_text = ""
    
    if detection_res.face_blendshapes:
        current_feat = np.array([b.score for b in detection_res.face_blendshapes[0]])
        
        best_match = None
        min_dis = float('inf')
        
        for user_name ,saved_feat in face_db.items():
            distance = np.linalg.norm(current_feat - saved_feat)
            
            if distance < min_dis:
                min_dis = distance
                best_match = user_name
                
        if min_dis < THRESHOLD:
            identity = f"Welcome, {best_match.capitalize()}!"
            color = (0, 255, 0) # Green border for authorized users
        else:
            identity = "Access Denied: Unknown"
            color = (0, 0, 255) # Red border for locked out attempts
            
        confidence_text = f"Dist: {min_dis:.4f}"
    else:
        color = (255, 0, 0)
    
    # UI Elements overlay
    cv.putText(frame, identity, (30, 50), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if confidence_text:
        cv.putText(frame, confidence_text, (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
    cv.imshow('Face Login System Portal', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()