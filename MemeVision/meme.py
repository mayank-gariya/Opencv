import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle

try:
    with open(r'D:\opencv-practice\meme_mixed_model.p', 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
    print("Trained model loaded successfully.")
except FileNotFoundError:
    print("Error: Trained model file not found.")
    exit()

model_path = r'D:\opencv-practice\hand_landmarker.task'
face_model_path = r'D:\opencv-practice\face_landmarker_v2_with_blendshapes.task'

hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_path), num_hands=1)
)
face_detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=face_model_path), 
        num_faces=1, 
        output_face_blendshapes=True
    )
)

MODEL_EXPECTED_FEATURES = 52 
labels = ['angryCat','giveMeMoney','Iknow','middle','shock','cute','totalpeace','waitWaht','wannafight']

cap = cv.VideoCapture(0)
print("Starting webcam... Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    detected_label = "Scanning..."
    hand_data = []
    
    # 1. Check Hand Tracking First
    hand_res = hand_detector.detect(mp_image)
    if hand_res.hand_landmarks:
        for hand_landmarks in hand_res.hand_landmarks:
            for landmark in hand_landmarks:
                hand_data.append(landmark.x)
                hand_data.append(landmark.y)
        if len(hand_data) == 42:
            padded_hand = hand_data + [0.0] * (MODEL_EXPECTED_FEATURES - len(hand_data))
            prediction = model.predict([padded_hand])
            detected_label = f"Hand: {labels[int(prediction[0])]}"
            
    # 2. Check Expression Blendshapes if No Hand Is Up
    if len(hand_data) == 0:
        face_res = face_detector.detect(mp_image)
        # MediaPipe delivers blendshapes as a list of lists containing category items
        if face_res.face_blendshapes and len(face_res.face_blendshapes) > 0:
            face_data = []
            for blendshape_category in face_res.face_blendshapes[0]:
                face_data.append(blendshape_category.score)
                
            if len(face_data) == MODEL_EXPECTED_FEATURES:
                prediction = model.predict([face_data])
                detected_label = f"Face Expression: {labels[int(prediction[0])]}"

    cv.putText(frame, detected_label, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    cv.imshow('Meme Classification Test', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hand_detector.close()
face_detector.close()
cv.destroyAllWindows()
