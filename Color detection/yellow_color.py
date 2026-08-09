import cv2 as cv
import numpy as np
from PIL import Image

cap = cv.VideoCapture(0)

def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv.cvtColor(c,cv.COLOR_BGR2HSV)
    
    ll = hsvC[0][0][0] - 10 ,100,100
    ul = hsvC[0][0][0] + 10 ,255,255
    
    ll = np.array(ll,dtype=np.uint8)
    ul = np.array(ul,dtype=np.uint8)
    
    return ll , ul

color = [0,255,255]

while True:
    ret , frame = cap.read(0)
    
    # color detection 
    hsvImg = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    lower_limit , upper_limit = get_limits(color)
    
    mask = cv.inRange(hsvImg,lower_limit,upper_limit)
    
    mask_ = Image.fromarray(mask)
    
    bbox = mask_.getbbox()
    
    if bbox is not None:
        x1 , y1, x2 , y2 = bbox
        frame = cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),6)
        
    cv.imshow('frame',frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()