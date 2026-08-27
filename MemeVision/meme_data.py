import cv2 as cv
import numpy
import os 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

Base_dir = 'MemeData'

if not os.path.exists(Base_dir):
    os.mkdir(Base_dir)

meme_imgs_path = r'C:\Users\admin\Desktop\MemeImgs'

meme_label = []

for meme in os.listdir(meme_imgs_path):
    meme_label.append(meme.split('.')[0])

# print(meme_label)
data_size = 100

cap = cv.VideoCapture(0)

for idx , reac in enumerate(meme_label):
    class_path = os.path.join(Base_dir,str(idx))
    
    if not os.path.exists(class_path):
        os.mkdir(class_path)
        
    print(f'reactions ready to capture for {reac}')
    
    while True:
        ret , frame = cap.read()
        frame = cv.flip(frame,1)
        if not ret:
            continue
        cv.putText(frame, f'Class: {reac}. Press Q to Start', (50, 50), 
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow('Data Collector', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < data_size:
        ret, frame = cap.read()
        frame = cv.flip(frame,1)
        
        if not ret:
            continue
        cv.imshow('Data Collector', frame)
        cv.waitKey(25)
        cv.imwrite(os.path.join(class_path, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv.destroyAllWindows()
print("Dataset collection complete!")
