import cv2
import numpy as np

PHONE_WIDTH_MM = 80.45
PHONE_HEIGHT_MM = 169.48

cap = cv2.VideoCapture(0)

pixels_per_mm = None

def midpoint(a, b):
    return ((a[0]+b[0])/2, (a[1]+b[1])/2)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)

    edges = cv2.Canny(blur,50,100)
    edges = cv2.dilate(edges,None,1)
    edges = cv2.erode(edges,None,1)

    contours,_ = cv2.findContours(edges,
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours,
                      key=cv2.contourArea,
                      reverse=True)

    pixels_per_mm = None

    for c in contours:

        if cv2.contourArea(c) < 5000:
            continue

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        (cx,cy),(w,h),angle = rect

        # Keep longer side as height
        if w > h:
            w,h = h,w

        # First large rectangle becomes reference phone
        if pixels_per_mm is None:
            pixels_per_mm = w / PHONE_WIDTH_MM

        width_mm = w / pixels_per_mm
        height_mm = h / pixels_per_mm

        cv2.drawContours(frame,[box],0,(0,255,0),2)

        cv2.putText(frame,
                    f"W: {width_mm:.1f} mm",
                    (int(cx)-60,int(cy)-20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2)

        cv2.putText(frame,
                    f"H: {height_mm:.1f} mm",
                    (int(cx)-60,int(cy)+10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2)

    cv2.imshow("Phone Measurement",frame)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()