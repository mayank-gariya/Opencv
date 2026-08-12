import os
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path=r'D:\\opencv-practice\\face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

direc = './know_face'
database = {}

print("Extracting features from face database...")

if os.path.exists(direc):
    for user_name in os.listdir(direc):
        user_path = os.path.join(direc, user_name)
        
        if os.path.isdir(user_path):
            user_features = []
            
            for img_name in os.listdir(user_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(user_path, img_name)
                    
                    image = mp.Image.create_from_file(img_path)
                    detection_res = detector.detect(image)
                    
                    # Extract blendshape scores if a face is found
                    if detection_res.face_blendshapes:
                        # Extract the 52 float scores as a flat feature vector
                        blendshape_scores = [b.score for b in detection_res.face_blendshapes[0]]
                        user_features.append(blendshape_scores)
            
            if user_features:
                # Average the features across all 100 images to get a clean baseline profile
                database[user_name] = np.mean(user_features, axis=0)
                print(f"Successfully profiles user: {user_name}")

with open('face_database.pkl', 'wb') as f:
    pickle.dump(database, f)

print("Face feature signatures saved to face_database.pkl!")
