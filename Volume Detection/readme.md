# 🎚️ AI Hand Gesture Volume Controller

<img width="1402" height="1122" alt="32cba086-2ea4-4afc-8c13-f527e44644a9" src="https://github.com/user-attachments/assets/56d1eaba-a96a-428e-b581-5dc7c52c77db" />


A real-time **computer vision project** that uses your webcam and **hand landmark detection** to control a virtual volume level using a simple hand gesture.

The project detects the **thumb and index finger**, calculates the distance between them, and maps that distance to a volume percentage from **0% to 100%**.

> 🤏 Move your thumb and index finger closer → Lower volume  
> 🫰 Move them farther apart → Higher volume

* * *

## ✨ Features

*   🎥 Real-time webcam hand tracking
    
*   🖐️ 21-point hand landmark detection
    
*   🤏 Thumb + index finger distance calculation
    
*   🎚️ Distance mapped to volume percentage
    
*   📊 Real-time volume display
    
*   🦴 Visual hand skeleton
    
*   🔴 Gesture feedback when fingers are close
    
*   🖥️ Custom dashboard-style background
    
*   ⚡ Real-time processing using MediaPipe and OpenCV
    

* * *

## 🧠 How It Works

The project follows a simple computer vision pipeline:

    Webcam
       ↓
    Capture Frame
       ↓
    OpenCV Preprocessing
       ↓
    MediaPipe Hand Detection
       ↓
    Detect 21 Hand Landmarks
       ↓
    Find Thumb + Index Finger
       ↓
    Calculate Distance
       ↓
    Map Distance → 0–100%
       ↓
    Display Volume Dashboard
    

### 1\. Capture Webcam Input

OpenCV accesses the computer's webcam:

    cap = cv2.VideoCapture(0)
    

Each frame is captured continuously and flipped horizontally to make the interaction feel natural.

* * *

### 2\. Detect the Hand

The project uses **MediaPipe Hand Landmarker** to detect the hand and its landmarks.

MediaPipe provides **21 landmarks** for a hand, including:

*   Wrist
    
*   Thumb joints
    
*   Index finger joints
    
*   Middle finger joints
    
*   Ring finger joints
    
*   Pinky joints
    

These landmarks are represented using normalized `x` and `y` coordinates.

* * *

### 3\. Find Thumb and Index Finger

The project uses two important landmarks:

    thumb_tip = pixel_points[4]
    index_tip = pixel_points[8]
    

These represent:

*   `4` → Thumb tip
    
*   `8` → Index finger tip
    

The distance between these two points is used as the gesture input.

* * *

### 4\. Calculate Finger Distance

The Euclidean distance formula is used:

    distance = √((x₂ - x₁)² + (y₂ - y₁)²)
    

In Python:

    dx = index_tip[0] - thumb_tip[0]
    dy = index_tip[1] - thumb_tip[1]
    
    distance = math.sqrt(dx**2 + dy**2)
    

This gives the distance between the thumb and index finger in pixels.

* * *

### 5\. Convert Distance into Volume

Two thresholds are defined:

    MIN_DISTANCE = 20
    MAX_DISTANCE = 180
    

The distance is then mapped to a percentage:

    20 pixels  →  0%
    180 pixels → 100%
    

For values between these limits:

    volume_pct = int(
        ((distance - MIN_DISTANCE) /
        (MAX_DISTANCE - MIN_DISTANCE)) * 100
    )
    

This creates a smooth volume-control gesture.

* * *

## 🎨 Visual Interface

The project uses a custom background image as a dashboard.

The webcam feed is inserted into a predefined area:

    BG_X1, BG_X2 = 95, 650
    BG_Y1, BG_Y2 = 145, 510
    

The final interface contains:

*   Webcam feed
    
*   Hand skeleton
    
*   Finger landmarks
    
*   Thumb-index distance indicator
    
*   Current volume percentage
    
*   Custom dashboard background
    

* * *

## 🛠️ Technologies Used

| Technology | Purpose |
| --- | --- |
| Python | Main programming language |
| OpenCV | Webcam and image processing |
| MediaPipe | Hand landmark detection |
| NumPy | Numerical operations |
| Math | Distance calculation |
| Time | Video timestamps |

* * *

## 📦 Installation

### 1\. Clone the repository

    git clone <your-repository-url>
    cd <your-project-folder>
    

### 2\. Create a virtual environment

    python -m venv venv
    

Activate it on Windows:

    venv\Scripts\activate
    

### 3\. Install dependencies

    pip install opencv-python mediapipe numpy
    

* * *

## 📁 Project Structure

A recommended project structure is:

    hand-gesture-volume-controller/
    │
    ├── main.py
    ├── hand_landmarker.task
    ├── background.png
    ├── requirements.txt
    └── README.md
    

> **Note:** Keep the MediaPipe `.task` model and background image inside the project directory instead of using an absolute path such as `D:\opencv-practice\...`.

* * *

## ⚙️ Configuration

Update these paths according to your project structure:

    model_path = r'D:\opencv-practice\hand_landmarker.task'
    
    background_path = r'D:\opencv-practice\Gemini_Generated_Image_cxkqwocxkqwocxkq.png'
    

For a GitHub-friendly project, you can use relative paths instead:

    model_path = "hand_landmarker.task"
    background_path = "background.png"
    

This makes the project easier for other developers to run.

* * *

## ▶️ Running the Project

Run:

    python main.py
    

Your webcam should open automatically.

### Gesture Controls

| Gesture | Result |
| --- | --- |
| 🤏 Fingers very close | 0% volume |
| 🫰 Fingers partially separated | Intermediate volume |
| ✋ Fingers far apart | 100% volume |
| Q key | Exit application |

* * *

## 📐 Mathematical Concept

The main mathematical concept used in this project is **Euclidean Distance**.

For two points:

    P₁ = (x₁, y₁)
    P₂ = (x₂, y₂)
    

The distance is:

    d = √((x₂ - x₁)² + (y₂ - y₁)²)
    

The calculated distance is then normalized between the minimum and maximum distance.

This is a simple example of converting a **physical gesture into a numerical control signal**.

* * *

## 🚀 Possible Improvements

This project can be extended further.

### 🔊 Control the Actual System Volume

Currently, the project calculates and displays the volume percentage.

It can be extended to control the **actual operating-system volume** using libraries such as:

*   Pycaw on Windows
    
*   PulseAudio/PipeWire tools on Linux
    
*   Platform-specific audio APIs
    

### ✋ Support Multiple Gestures

Additional gestures could be added:

    🤏 Pinch        → Volume control
    ✊ Fist         → Mute
    ☝️ One finger   → Play/Pause
    ✌️ Two fingers  → Next track
    🖐️ Open hand    → Stop
    

### 👥 Multiple Hand Detection

The current configuration uses:

    num_hands=1
    

It could be extended to detect both hands.

### 📊 Add a Real Volume Slider

Instead of only displaying:

    75%
    

the dashboard could include a dynamic visual volume bar.

### 🤖 Add Gesture Classification

A machine-learning classifier could be added to recognize more complex hand gestures.

* * *

## 🧩 What I Learned From This Project

This project helped me understand how **computer vision can connect physical human interaction with software controls**.

Key concepts explored:

*   Real-time computer vision
    
*   Webcam processing
    
*   MediaPipe hand tracking
    
*   Hand landmarks
    
*   Coordinate systems
    
*   Euclidean distance
    
*   Normalization
    
*   Gesture-based interaction
    
*   OpenCV image manipulation
    
*   Real-time video processing
    

* * *

## 🔮 Future Goal

The bigger idea behind this project is to explore **Human-Computer Interaction (HCI)** using AI and computer vision.

Instead of interacting with software only through a keyboard and mouse, computer vision can allow us to interact naturally through:

**hands → gestures → AI/computer vision → software actions**

* * *

## 👨‍💻 Author

**Mayank Gariya**

Aspiring Machine Learning Engineer  
Learning, building, and growing through practical AI/ML projects.

* * *

## ⭐ If You Like This Project

If you found this project interesting, consider giving the repository a ⭐ and following the project as I continue experimenting with **Python, Computer Vision, Machine Learning, and AI**.
