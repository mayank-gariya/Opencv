# # image segmentation 

# import cv2 as cv
import numpy as np
import cv2

img = cv2.imread(r'D:\opencv-practice\tom.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, thres = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#noise removal 

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thres,cv2.MORPH_OPEN,kernel, iterations = 2)
 
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
 
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
 
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)
 
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
 
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

cv2.imshow('watershed image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Reshape the image to a 2D array of pixels (rows * cols, 3 channels)
# pixel_vals = img.reshape((-1, 3))
# pixel_vals = np.float32(pixel_vals)

# # Define criteria: stop if accuracy (epsilon) = 1.0 or iterations = 100
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)

# # Set number of clusters (K)
# k = 3
# retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# # Convert data back into 8-bit values and reconstruct the image
# centers = np.uint8(centers)
# segmented_data = centers[labels.flatten()]
# segmented_image = segmented_data.reshape((img.shape))

# cv2.imshow('K-Means Segmentation', segmented_image)
# cv2.waitKey(0)

# img = cv.imread(r'D:\opencv-practice\tom.png',cv.IMREAD_GRAYSCALE)
# _,thres = cv.threshold(img,0,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# # removing small noise with morophological ditration and erode
# kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
# cleaned = cv.morphologyEx(thres,cv.MORPH_OPEN,kernel)
