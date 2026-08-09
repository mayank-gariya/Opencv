import cv2 as cv
import numpy as np

# OPTION A: Automatically use OpenCV's internal safe path for HUMAN faces (Recommended)
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    raise IOError("Failed to load cascade classifier XML file. Verify your file path or installation.")

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab webcam frame.")
        break 
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    cv.imshow('Camera is running...', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()
