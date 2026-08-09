import cv2 as cv
import numpy as np
# Read the image
img = cv.imread(r'D:\opencv-practice\dog.png')

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# # Apply Gaussian blur
# blurred = cv.GaussianBlur(gray, (5, 5), 0)
# median = cv.medianBlur(img,3)
# # Detect edges using Canny
# edges = cv.Canny(blurred, 50, 150)
# sharpen_kernel = np.array([
#     [0,-1,0],
#     [-1,5,-1],
#     [0,-1,0]
# ])

# sharpen = cv.filter2D(img,-1,sharpen_kernel)
# thresold binary method 
ret , thers_img = cv.threshold(gray,150,255,cv.THRESH_BINARY)

# Display all results
# cv.imshow('blurred',blurred)
# cv.imshow('Grayscale', gray)
# cv.imshow('Edges', edges)
# cv.imshow('median blur', median)
cv.imshow('Original', img)
cv.imshow('sharpen',thers_img)

cv.waitKey(0)
cv.destroyAllWindows()