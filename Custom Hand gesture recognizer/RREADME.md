# Real-Time Sign Language Detection with MediaPipe Tasks API

An advanced real-time computer vision system that classifies complex hand signs (`happy`, `sad`, and `python`) from dynamic live video feeds. Built on the modern asynchronous **MediaPipe Tasks API**, this pipeline bypasses traditional illumination and background noise issues by focusing entirely on normalized 3D hand geometry.

---

## 📌 Features
* **Google MediaPipe Tasks Framework:** Utilizes cutting-edge landmarking configurations running on strict `RunningMode.VIDEO` temporal processing profiles.
* **Custom OpenCV Layout Engine:** Renders dynamic bounding text overlays and solid colored label block metrics based on absolute coordinate offsets.
* **Failsafe Core Design:** Native exception handlers intercept dropped webcam frame matrices and report missing local task file paths cleanly.
* **High-Density Feature Vectorization:** Flattens 21 unique joint coordinates into isolated numerical profiles for lightweight machine learning engines.

---

## 🛠️ GitHub Repository Quickstart

Initialize and build your local development environment using these commands:

```bash
# Clone the repository
git clone https://github.com
cd sign-language-detection

# Set up an isolated python execution profile
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install third-party framework dependencies
pip install opencv-python mediapipe scikit-learn numpy
```

> 🛑 **Task Asset Dependency:** Download the official pre-trained asset module file `hand_landmarker.task` from Google MediaPipe and place it in your target directory structure location (e.g., `D:\opencv-practice\hand_landmarker.task`).

---

## 📂 Repository File Blueprint
```text
├── data/                    # Dynamic automated image dataset subfolders
│   ├── 0/                   # Captured snapshots for sign: "happy"
│   ├── 1/                   # Captured snapshots for sign: "sad"
│   └── 2/                   # Captured snapshots for sign: "python"
├── images/                  # Document illustration asset folders
│   ├── demo_inference.gif   # Dynamic prediction animation asset
│   └── hand_landmarks.png   # Hand joint guide layout mapping chart
├── collect_data.py          # Script 1: Dataset webcam framework pipeline
├── train_model.py           # Script 2: ML weight compilation and evaluation
├── inference.py             # Script 3: Real-time production live tracking loop
└── sign_model.p             # Exported Random Forest model binary weights
```

---

## 💻 Source Code Components

### 1. `collect_data.py`
Generates clean dataset directory formats and captures `100` precise image buffers for each class.

```python
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
```

### 2. `train_model.py`
Analyzes stored source images via static `RunningMode.IMAGE` logic and exports a standalone serialized Random Forest model binary.

```python
import os
import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model_path = r'D:\opencv-practice\hand_landmarker.task'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"MediaPipe task asset bundle missing at location: {model_path}")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

Data_dir = './data'
data = []
labels = []

print("Extracting features from dataset images...")
for dir_ in os.listdir(Data_dir):
    dir_path = os.path.join(Data_dir, dir_)
    if not os.path.isdir(dir_path):
        continue
        
    for img_path in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            continue
            
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        try:
            results = detector.detect(mp_image)
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    data_aux = []
                    for landmark in hand_landmarks:
                        data_aux.append(landmark.x)
                        data_aux.append(landmark.y)
                    
                    if len(data_aux) == 42:
                        data.append(data_aux)
                        labels.append(int(dir_))
        except Exception as e:
            print(f"Skipping corrupt frame element {img_path}: {e}")

detector.close()

if len(data) > 0:
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    print(f"Model Training Complete. Accuracy Score: {accuracy_score(y_test, model.predict(x_test)) * 100:.2f}%")

    with open('sign_model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
else:
    print("Error: Extraction pipeline empty. Ensure images are populated correctly inside directories.")
```

### 3. `inference.py`
Executes real-time webcam frame processing streams, generating visual metrics and dynamic text boxes on live frame configurations.

```python
import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle

if not os.path.exists('sign_model.p'):
    raise FileNotFoundError("Trained classifier file 'sign_model.p' not found. Run train_model.py first.")

with open('sign_model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

labels_dict = {0: 'happy', 1: 'sad', 2: 'python'}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (9, 10), (10, 11), (11, 12),           # Middle finger
    (13, 14), (14, 15), (15, 16),          # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm Knuckles
]

model_path = r'D:\opencv-practice\hand_landmarker.task'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"MediaPipe task asset missing at path location: {model_path}")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp == 0:
        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
    try:
        results = detector.detect_for_video(mp_image, timestamp)
    except Exception as e:
        print(f"MediaPipe processing error bypassed: {e}")
        continue
    
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            pixel_points = []
            data_aux = []
            
            for landmark in hand_landmarks:
