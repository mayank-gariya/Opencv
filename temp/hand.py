import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

import numpy as np


model_path = r'D:\opencv-practice\hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv.VideoCapture(0)
canvas = None
xp,yp = 0,0

while cap.isOpened():
    ret , frame = cap.read()
    
    if not ret:break
    
    frame = cv.flip(frame,1)
    h,w,c = frame.shape
    if canvas is None: canvas = np.zeros_like(frame)
    
    # conveting the frame to rgb then converting frame to mediapipe object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=cv.cvtColor(frame,cv.COLOR_BGR2RGB))
    timestamp = int(cap.get(cv.CAP_PROP_POS_MSEC)) # creates the time stamp for the object
    
    # run hand and detection
    results = detector.detect_for_video(mp_image, timestamp)
    
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            idx_tip = hand_landmarks[8]
            mid_tip = hand_landmarks[12]
            
            cx, cy = int(idx_tip.x * w), int(idx_tip.y * h)
            mx, my = int(mid_tip.x * w), int(mid_tip.y * h)
            
            if cy < my: # Drawing mode
                cv.circle(frame, (cx, cy), 10, (0, 0, 255), cv.FILLED)
                if xp == 0 and yp == 0: xp, yp = cx, cy
                cv.line(canvas, (xp, yp), (cx, cy), (255, 0, 0), 5)
                xp, yp = cx, cy
            else:
                xp, yp = 0, 0
                cv.circle(frame, (cx, cy), 10, (0, 255, 0), cv.FILLED)

    # Combine canvas and camera stream
    gray_canvas = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray_canvas, 20, 255, cv.THRESH_BINARY_INV)
    thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    combined = cv.bitwise_and(frame, thresh)
    combined = cv.bitwise_or(combined, canvas)

    cv.imshow("Virtual Canvas (Tasks API)", combined)
    if cv.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv.destroyAllWindows()
    