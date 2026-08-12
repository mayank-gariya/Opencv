# 🔐 Biometric Streak Portal & Face Attendance App
  ![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
  ![Vision Core](https://img.shields.io/badge/mediapipe%20Labs-%236933FF.svg?style=for-the-badge&logo=streamlit&logoColor=white)
  ![Airtabel](https://img.shields.io/badge/airtabel%20Labs-%236933FF.svg?style=for-the-badge&logo=airtabel%20&logoColor=white)

<p align="center">
  <img width="922" height="518" alt="image" src="https://github.com/user-attachments/assets/d7e7b357-7adc-483b-957d-2d2d71b10da0" />
</p>

A production-ready, biometric face authentication login portal and automated daily streak tracking application. The system leverages **Streamlit** for its responsive frontend user interface, **MediaPipe Face Landmarker (v2 with Blendshapes)** for high-precision 52-dimension facial structural tracking, and **Airtable API** as a persistent cloud database engine.

---

## 🏗️ Folder Structure

Based on your local project layout, your workspace workspace directory should look like this:

```text
Face Attendance App/
│
├── face_landmarker_v2_with_blendshapes.task  # MediaPipe facial geometry asset file
├── face_database.pkl                         # Local fallback backup database matrix
├── final_proj.py                             # Native OpenCV webcam fallback / local debugger script
├── login.py                                  # Core Streamlit Web UI application script
├── login_data.py                             # Configuration loader and security module
└── train_model.py                            # Script to register baseline local embeddings
```

---

## 🔄 System Architecture & Data Flow

The application isolates functional logic paths into two distinct procedural workflows based on the navigation selection:


<img width="1229" height="698" alt="image" src="https://github.com/user-attachments/assets/bf009854-536b-4ae4-829f-003d07ca82eb" />

---

## 🚀 Installation & Getting Started

### 1. Environment Setup & Core Dependencies
Ensure your environment runs Python 3.9 or higher. Initialize dependencies inside your terminal environment:
```bash
pip install streamlit opencv-python numpy mediapipe pyairtable
```

### 2. Model Asset Requirements
Download the official `face_landmarker_v2_with_blendshapes.task` bundle provided by Google MediaPipe Vision tasks. Update your absolute file system asset pointer inside your program configuration block:
```python
model_path = r'D:\opencv-practice\face_landmarker_v2_with_blendshapes.task'
```

### 3. Airtable Cloud Configuration
To tie your login workflows to the backend data layer, update your API connection string keys within `login.py` (or safely store them using Streamlit environment secrets variables):
```python
AIRTABLE_TOKEN = "your_personal_access_token_here"
BASE_ID = "your_base_id_here"
TABLE_NAME = "UserStreaks"
```

### 4. Database Schema Setup
Your targeted Airtable grid must map identically to the following field structures:
* **`username`** : `Single line text` (Unique key constraint)
* **`blendshapes`** : `Long text` (Stores serialized JSON feature matrix lists)
* **`streak`** : `Number` (Integer formatting settings)
* **`last_login`** : `Single line text` or `Date` (Tracks update times formatted as `YYYY-MM-DD`)

---

## 💻 Running the Programs

### Launching the Core Web Interface (Streamlit Application)
To kick off the primary interactive biometric registration and access portal dashboard, execute:
```bash
streamlit run login.py
```

### Running Local System Debugging Tools (OpenCV Script)
To test or debug camera capture buffers or check raw hardware feed bindings independently without the web layer, execute:
```bash
python final_proj.py
```

---

## 🧠 Algorithmic Insights

### 1. Expression-Agnostic Structural Matching
Unlike rigid pixel or direct coordinate tracking systems, this pipeline reads structural ratios. MediaPipe transforms raw pixels into **52 distinct face blendshape weights** mapping eyes, brows, cheek locations, and jaw structures. 

### 2. Distance Classification Threshold
Identity verification computes divergence through **Euclidean Vector Normalization (`np.linalg.norm`)**:
$$\text{Distance} = \sqrt{\sum_{i=1}^{52} (C_i - S_i)^2}$$
Where $C$ represents the live captured feature score and $S$ represents the stored cloud baseline registration. A fine-tuned distance restriction threshold of `< 0.16` provides strict security protection against spoofing attempts while maintaining smooth tracking under changing light environments.
