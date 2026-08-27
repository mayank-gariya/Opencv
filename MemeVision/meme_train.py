import cv2 as cv
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

MemeDataPath = r'D:\opencv-practice\MemeData'
model_path = r'D:\opencv-practice\hand_landmarker.task'
face_model_path = r'D:\opencv-practice\face_landmarker_v2_with_blendshapes.task'

# Initialize Detectors
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_hands=1)
hand_detector = vision.HandLandmarker.create_from_options(options)

face_base_options = python.BaseOptions(model_asset_path=face_model_path)
face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options, 
    running_mode=vision.RunningMode.IMAGE, 
    output_face_blendshapes=True, 
    num_faces=1
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

data_dir = MemeDataPath
data = []
label = []

# Using 52 standard face blendshape categories as our maximum feature size
MAX_FEATURES = 52 

print("Starting dataset scanning...")
for dir_ in os.listdir(data_dir):
    dir_path = os.path.join(data_dir, dir_)
    if not os.path.isdir(dir_path):
        continue
        
    print("\n" + "="*40)
    print(f"Folder found: '{dir_}'")
    print("Choose extraction mode for this folder:")
    print("1. Hand Landmarks")
    print("2. Facial Blendshapes (Expressions)")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice not in ['1', '2']:
        print(f"Skipping folder '{dir_}' due to invalid choice.")
        continue

    print(f"Processing images in '{dir_}'...")
    for img_path in os.listdir(dir_path):
        img = cv.imread(os.path.join(dir_path, img_path))
        if img is None:
            continue
            
        rgb_frame = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        data_aux = []
        
        if choice == '1':
            results = hand_detector.detect(mp_image)
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    for landmark in hand_landmarks:
                        data_aux.append(landmark.x)
                        data_aux.append(landmark.y)
                if len(data_aux) == 42:
                    padded_aux = data_aux + [0.0] * (MAX_FEATURES - len(data_aux))
                    data.append(padded_aux)
                    label.append(int(dir_))
                    
        else:
            results = face_detector.detect(mp_image)
            if results.face_blendshapes:
                # Extract the direct 52 structural matrix scores
                for blendshape in results.face_blendshapes[0]:
                    data_aux.append(blendshape.score)
                if len(data_aux) == MAX_FEATURES:
                    data.append(data_aux)
                    label.append(int(dir_))

hand_detector.close()
face_detector.close()

if len(data) > 0:
    print("\nTraining classifier on mixed dataset...")
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True, stratify=label)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    print(f"Model Training Complete. Accuracy Score: {accuracy_score(y_test, model.predict(x_test)) * 100:.2f}%")
    
    with open(r'D:\opencv-practice\meme_mixed_model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    print("Model saved successfully with blendshapes integration.")
else:
    print("\nError: No features extracted.")
