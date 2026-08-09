import cv2 as cv
import matplotlib.pyplot as plt
import os 
import numpy as np

# print(cv.__version__) --> 5.5.0

# reading the image
def read_image(imgpath):
    img = cv.imread(imgpath)
    cv.imshow('img',img)
    cv.waitKey(0)
    
# read_image('D:\opencv-practice\dog.png')

# writing the image 
def write_image(inImg , imgpath):
    img = cv.imread(inImg)
    cv.imwrite(imgpath,img)
    
# reading the vedio 
def vediowebcam():
    # creates the capture like the camera that captures the vedio
    cap = cv.VideoCapture(0)
    
    # quick check if not open breakk 
    if not cap.isOpened():
        exit()
    
    # infinte loop to read each image 
    while True:
        # read the camera and return the frame (images) and ret (return value to check)
        ret , frame = cap.read()
        # qucick check and only allow to show the camera when return is true 
        if ret:
            cv.imshow('webcam',frame)
        # wait for 1 and if pressed q on keyboard break the loop and stop the camera
        if cv.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows() # break the camera

# reaing vedio from mp4 vedio from folder
def vedioFromFolder(filepath):
    cap = cv.VideoCapture(filepath)
    
    while cap.isOpened():
        ret , frame = cap.read()
        
        if not ret:
            break
        
        cv.imshow('video',frame)
        
        fps = cv.CAP_PROP_FPS
        if fps == 0: fps = 30 
        delay = int(1000/fps)
        
        if cv.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv.destroyAllWindows()
    
def capturingVideo(outpath):
    outPath = os.path.join(outpath,'demovedio.mp4')
    
    cap = cv.VideoCapture(0)
    
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width,frame_height)
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    
    out = cv.VideoWriter(outPath,fourcc,20.0,frame_size)
    
    while cap.isOpened():
        ret , frame = cap.read()
        if not ret:
            print('not working great with break ...')
            break
        
        out.write(frame)
        
        cv.imshow('Recording....',frame)
        
        if cv.waitKey(1) == ord('q'):
            break
        
    cap.release()
    out.release()
    cv.destroyAllWindows()
        
# capturingVideo(r'D:\opencv-practice')
# vedioFromFolder(r'D:\opencv-practice\demovedio.mp4')
# vediowebcam()

# now rgb 
def purecolors():
    zeros = np.zeros((100,100))
    ones = np.ones((100,100))
    rImg = cv.merge((255*ones,zeros,zeros))
    bImg = cv.merge((zeros,zeros,255*ones))
    gImg = cv.merge((zeros,255*ones,zeros))
    
    plt.figure()
    plt.subplot(231)
    plt.imshow(bImg)
    
    plt.subplot(232)
    plt.imshow(rImg)
    plt.subplot(233)
    plt.imshow(gImg)
    
    plt.show()
    
# purecolors()

# now gray scale
def gray(imgpath):
    img = cv.imread(imgpath)
    b,g,r = cv.split(img)
    
    plt.figure() 
    plt.subplot(131)
    plt.imshow(b,cmap='gray')
    plt.show()
    
    
gray(r'D:\opencv-practice\dog.png')   