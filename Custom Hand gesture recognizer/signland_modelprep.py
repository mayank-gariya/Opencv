import os
import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Setup MediaPipe Landmarker for Static Images
model_path = r'D:\opencv-practice\hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

Data_dir = './data'
data = []
labels = []

print("Extracting features from dataset images...")
for dir_ in os.listdir(Data_dir):
    dir_path = os.path.join(Data_dir, dir_)
    if not os.path.isdir(dir_path):
        continue
        
    for img_path in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            continue
            
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect landmarks on a static image
        results = detector.detect(mp_image)
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                data_aux = []
                for landmark in hand_landmarks:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)
                
                # Ensure we only process instances that output exactly 42 coordinates (21 joints * x,y)
                if len(data_aux) == 42:
                    data.append(data_aux)
                    labels.append(int(dir_))

detector.close()

# Train Machine Learning Model
if len(data) > 0:
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    print(f"Model Training Complete. Accuracy Score: {accuracy_score(y_test, model.predict(x_test)) * 100:.2f}%")

    # Export Model weights
    with open('sign_model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
else:
    print("Error: No hand landmarks detected. Ensure hand images are clear and model path is correct.")
