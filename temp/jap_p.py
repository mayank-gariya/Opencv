import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret , frame = cap.read()
    
    grayF = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayF,1.1,5)
    
    for (x, y, w, h) in faces:
        # cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        face_img = frame[y:y+h,x:x+w]
        blur = cv.GaussianBlur(face_img,(35,35),0)
        frame[y:y+h,x:x+w] = blur
        
    cv.imshow('ved',frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()