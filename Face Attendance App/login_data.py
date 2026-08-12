import cv2 as cv
import os

direc = './know_face'
if not os.path.exists(direc):
    os.makedirs(direc)

user_name = input("Enter the username for registration: ").strip().lower()
user_dir = os.path.join(direc, user_name)

if not os.path.exists(user_dir):
    os.makedirs(user_dir)
else:
    print("User already exists! Overwriting existing data.")

cap = cv.VideoCapture(0)
data_size = 100
counter = 0

print("Position your face in front of the camera and press 'S' to start capturing...")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    cv.putText(frame, "Press 'S' to Start Data Collection", (30, 40), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.imshow('Registration Window', frame)
    
    if cv.waitKey(1) & 0xFF == ord('s'):
        break

# Capture loop
while counter < data_size:
    ret, frame = cap.read()
    if not ret: continue
    
    # Save directly to the specific user's folder
    img_path = os.path.join(user_dir, f'{counter}.jpg')
    cv.imwrite(img_path, frame)
    
    # Visual feedback
    cv.putText(frame, f"Capturing: {counter}/{data_size}", (30, 40), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.imshow('Registration Window', frame)
    cv.waitKey(20)  # Tiny delay to allow expressions to change slightly
    counter += 1

cap.release()
cv.destroyAllWindows()
print(f"Dataset collection complete for user: {user_name}!")
