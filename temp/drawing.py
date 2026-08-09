import cv2 as cv
import matplotlib.pyplot as plt
import os 
import numpy as np

image = cv.imread(r'D:\opencv-practice\dog.png')

if image is None:
    print('invalid image')

# image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# pt1 = (20,100)
# pt2 = (100,100)

# color = (0,0,255)
# thickness = 4

# line = cv.line(image,pt1,pt2,color,thickness)
# cv.imshow('drawing line on image',line)
# cv.waitKey(0)
# cv.destroyAllWindows()

# rectangle 
# image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# pt1 = (20,100)
# pt2 = (100,300)

# color = (0,0,255)
# thickness = 4

# line = cv.rectangle(image,pt1,pt2,color,thickness)
# cv.imshow('drawing line on image',line)
# cv.waitKey(0)
# cv.destroyAllWindows()

# circle 
# color = (0,0,255)
# thickness = 4

# line = cv.circle(image,center=(210,100),radius=20,color=color,thickness=2)
# cv.imshow('drawing line on image',line)
# cv.waitKey(0)
# cv.destroyAllWindows()

# adding text

text_img = cv.putText(image,'hey i am labra',(150,300),cv.FONT_HERSHEY_COMPLEX,1.2,(0,200,255),2)
cv.imshow('adding text',text_img)
cv.waitKey(0)
cv.destroyAllWindows()


