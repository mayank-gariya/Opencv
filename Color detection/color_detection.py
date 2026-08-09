import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

lower_blue = np.array([90,50, 20], dtype=np.uint8)
upper_blue = np.array([140,255, 255], dtype=np.uint8)

lower_yellow = np.array([20,113, 20], dtype=np.uint8)
upper_yellow = np.array([40,255, 255], dtype=np.uint8)

lower_green = np.array([45,85, 20], dtype=np.uint8)
upper_green= np.array([77,255, 255], dtype=np.uint8)

lower_red1 = np.array([1,100, 50], dtype=np.uint8)
upper_red1= np.array([5,100, 255], dtype=np.uint8)

lower_red2 = np.array([175,100, 50], dtype=np.uint8)
upper_red2= np.array([179,255, 255], dtype=np.uint8)

blue = {
    'color': (255,0,0),
    'pos': (0,50),
    'text': 'Blue'
 }
red = {
    'color': (0,0,255),
    'pos': (140,50),
    'text': 'Red'
 }
yellow = {
    'color': (0,255,255),
    'pos': (270,50),
    'text': 'Yellow'
 }
green = {
    'color': (0,255,0),
    'pos': (460,50),
    'text': 'Green'
}

def drawText():
    cv.rectangle(frame,(0,0),(640,55),0,-1)
    cv.putText(frame,blue['text'],blue['pos'],0,2,(255,255,255),2,cv.LINE_AA)
    cv.putText(frame,red['text'],red['pos'],0,2,(255,255,255),2,cv.LINE_AA)
    cv.putText(frame,yellow['text'],yellow['pos'],0,2,(255,255,255),2,cv.LINE_AA)
    cv.putText(frame,green['text'],green['pos'],0,2,(255,255,255),2,cv.LINE_AA)
    

def drawMatches(mask,color):
    cnts,_ = cv.findContours(mask,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    
    for c in cnts:
        a = cv.contourArea(c)
        if a> 1000:      
            cv.putText(frame,color['text'],color['pos'],0,2,color['color'],2, cv.LINE_AA)
    return

frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

frame_size = (frame_width,frame_height)
fourcc = cv.VideoWriter_fourcc(*'mp4v')

out = cv.VideoWriter(r'D:\opencv-practice\video.mp4',fourcc,20.0,frame_size)
    
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if not cap.isOpened():
        exit()
    
    out.write(frame)
    
    drawText()
    
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    blueMask = cv.inRange(frame_hsv, lower_blue, upper_blue)
    yellowMask = cv.inRange(frame_hsv, lower_yellow, upper_yellow)
    greenMask = cv.inRange(frame_hsv, lower_green, upper_green)
    redMask1 = cv.inRange(frame_hsv, lower_red1, upper_red1)
    redMask2 = cv.inRange(frame_hsv, lower_red2, upper_red2)

    redMask = cv.add(redMask1,redMask2)
    
    drawMatches(blueMask, blue)
    drawMatches(yellowMask,yellow)
    drawMatches(greenMask,green)
    drawMatches(redMask,red)
    
    cv.imshow("Video", frame)


    if (cv.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
out.release()
cv.destroyAllWindows()