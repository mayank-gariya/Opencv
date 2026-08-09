import cv2 as cv 
import pickle

try:
    with open('carparkpos','rb') as f:
        poslist= pickle.load(f)
except:
    poslist = []
    
img = cv.imread(r'D:\opencv-practice\park.png') 

if img is None:
    print("Error: Image not found or path is incorrect.")
    exit()

# Keep a clean copy of the original image to redraw fresh rectangles each frame
img_original = img.copy() 
width, height = 50, 100 

def mouseclick(event, x, y, flags, params): 
    # Use left button down (EVENT_LBUTTONDOWN) as it is standard for clicks
    if event == cv.EVENT_LBUTTONDOWN: 
        poslist.append((x, y))
        
    if event == cv.EVENT_RBUTTONDOWN:
        for posidx , pos in enumerate(poslist):
            x1,y1 = pos
            if x1 < x < x1+width and y1<y<y1 + height:
                poslist.pop(posidx)
                
                
    with open('carparkpos','wb') as f:
        pickle.dump(poslist,f)
# Name the window first
cv.namedWindow('park')
# Pass the window name to setMouseCallback
cv.setMouseCallback('park', mouseclick) 

while True: 
    # Reset the image to the clean copy each frame
    img = img_original.copy()
    
    for pos in poslist: 
        cv.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 2) 
        
    cv.imshow('park', img) 
    
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break 

cv.destroyAllWindows()
