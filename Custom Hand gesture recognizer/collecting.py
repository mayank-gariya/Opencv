import cv2
import os

Data_dir = './data'
if not os.path.exists(Data_dir):
    os.makedirs(Data_dir)
    
classes = ['happy', 'sad', 'python']
data_size = 100

cap = cv2.VideoCapture(0)

for idx, class_name in enumerate(classes):
    class_path = os.path.join(Data_dir, str(idx))
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    print(f'Ready to collect data for class: "{class_name}". Press "Q" to start.')
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, f'Class: {class_name}. Press Q to Start', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Data Collector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < data_size:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow('Data Collector', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_path, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
print("Dataset collection complete!")
