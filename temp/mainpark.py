import cv2 as cv
import pickle
import numpy as np

# 1. Load the reference image for exact dimensions
ref_img = cv.imread(r'D:\opencv-practice\park.png')
if ref_img is None:
    print("Error: Reference image 'park.png' not found.")
    exit()

target_height, target_width = ref_img.shape[:2]

# 2. Load the saved parking spot positions
try:
    with open('carparkpos', 'rb') as f:
        poslist = pickle.load(f)
except FileNotFoundError:
    print("Error: 'carparkpos' file not found.")
    poslist = []
    
width, height = 50, 100  

# 3. Define the parking space checker function
def checkcar(img_processed, img_display):
    space_counter = 0

    for pos in poslist:
        x, y = pos
        
        # Crop the spot from the processed (thresholded) image
        img_crop = img_processed[y:y+height, x:x+width]
        
        # Count white pixels (edges/features) inside the cropped area
        count = cv.countNonZero(img_crop)
        
        # Threshold: Adjust 900 if your video needs more/less sensitivity
        if count < 900:
            color = (0, 255, 0)  # Green for Empty
            thickness = 2
            space_counter += 1
        else:
            color = (0, 0, 255)  # Red for Occupied
            thickness = 2
            
        # Draw the bounding box and pixel count text on the display image
        cv.rectangle(img_display, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cv.putText(img_display, str(count), (x, y + height - 5), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
    # Display total free spaces on top of the frame
    cv.putText(img_display, f'Free Spaces: {space_counter}/{len(poslist)}', (60, 50), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

# 4. Video processing loop
cap = cv.VideoCapture(r'D:\opencv-practice\gfg1-ezgifcom-resize-video.mp4')

while True:
    if cap.get(cv.CAP_PROP_POS_FRAMES) == cap.get(cv.CAP_PROP_FRAME_COUNT):
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        
    ret, frame = cap.read()
    if not ret:
        break
        
    # Resize frame to match reference image
    frame = cv.resize(frame, (target_width, target_height))
    
    # 5. Image Pre-processing for detection
    img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (3, 3), 1)
    
    # Convert image to binary (edges look like white pixels)
    img_threshold = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv.THRESH_BINARY_INV, 25, 16)
    
    # Clean up random noise points
    img_median = cv.medianBlur(img_threshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    img_dilated = cv.dilate(img_median, kernel, iterations=1)
    
    # 6. Run the parking status check
    checkcar(img_dilated, frame)
         
    cv.imshow('Parking Space Detector', frame)

    
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
