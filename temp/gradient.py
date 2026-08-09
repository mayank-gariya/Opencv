import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv.imread(r'D:\opencv-practice\dog.png')

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def gradient():
    plt.figure(figsize=(10, 5))
    
    # Original Image
    plt.subplot(121)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Gray')
    plt.axis('off')
    
    # Laplacian Gradient (Fixed the typo to cv.CV_64F)
    laplacian_raw = cv.Laplacian(gray, cv.CV_64F, ksize=5) 
    
    # Convert back to absolute 8-bit depth for proper visualization
    laplacian_abs = np.uint8(np.absolute(laplacian_raw))
    
    plt.subplot(122)
    plt.imshow(laplacian_abs, cmap='gray')
    plt.title('Laplacian Edge Detection')
    plt.axis('off')
    
    plt.show()

gradient()
