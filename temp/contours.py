import cv2 as cv
import numpy as np

img = cv.imread(r'D:\opencv-practice\car.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

solbeX = cv.Sobel(gray,cv.CV_64F,1,0,ksize=3)
solbelY = cv.Sobel(gray,cv.CV_64F,0,1,ksize=3)

mag = cv.magnitude(solbeX,solbelY)

mag = cv.convertScaleAbs(mag)


# lapasian 
lapacian = cv.Laplacian(gray,cv.CV_64F)
lapacian_abs = cv.convertScaleAbs(lapacian)

cv.imshow('contour image is ',mag)
cv.imshow('lapacian ',lapacian_abs)
cv.waitKey(0)
cv.destroyAllWindows()