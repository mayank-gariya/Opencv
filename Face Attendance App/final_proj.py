import streamlit as st
import numpy as np
import cv2 as cv
import json
from datetime import datetime, date
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pyairtable import Table


AIRTABLE_TOKEN = "your_personal_access_token_here"
BASE_ID = "your_base_id_here"
TABLE_NAME = "UserStreaks"


try:
    table = Table(AIRTABLE_TOKEN, BASE_ID, TABLE_NAME)
except Exception as e:
    st.error(f"Airtable Connection Error: {e}")

@st.cache_resource
def load_face_landmarker():
    # Ensure this path points to your local absolute file path
    model_path = r'D:\opencv-practice\face_landmarker_v2_with_blendshapes.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        num_faces=1,
        min_face_detection_confidence=0.4
    )
    return vision.FaceLandmarker.create_from_options(options)

detector = load_face_landmarker()

def extract_blendshapes_from_bytes(image_bytes):
    """Converts Streamlit camera bytes to MediaPipe Image and extracts features."""
    file_bytes = np.frombuffer(image_bytes.read(), dtype=np.uint8)
    opencv_img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    rgb_frame = cv.cvtColor(opencv_img, cv.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_res = detector.detect(mp_image)
    
    # 1. Verify detection_res and that at least one face was detected
    if detection_res and detection_res.face_blendshapes:
        # 2. Access index 0 to get the list of blendshapes for the FIRST detected face
        first_face_blendshapes = detection_res.face_blendshapes[0]
        
        # 3. Safely extract scores from the categories within that single face list
        return [b.score for b in first_face_blendshapes]
        
    return None


st.set_page_config(page_title="Biometric Streak Portal", page_icon="🔐", layout="centered")
st.title("🔐 Face Login & Daily Streak Portal")
st.markdown("A secure, production-ready facial biometric authentication portal linked to a cloud database.")

menu = ["Log In", "Register New Profile"]
choice = st.sidebar.selectbox("Navigation Portal", menu)

# --- WORKFLOW A: USER LOG IN ---
if choice == "Log In":
    st.subheader("Verify Identity to Continue Streak")
    login_user = st.text_input("Enter Username").strip().lower()
    
    camera_img = st.camera_input("Look at the camera for biometric scan")
    
    if camera_img and login_user:
        with st.spinner("Analyzing facial metrics..."):
            current_features = extract_blendshapes_from_bytes(camera_img)
            
            if current_features is None:
                st.error("❌ No face detected. Please adjust lighting and face the camera directly.")
            else:
                # Query user record from cloud
                records = table.all(formula=f"{{username}}='{login_user}'")
                
                if not records:
                    st.warning("⚠️ Username not registered in biometric database.")
                else:
                    record = records[0]
                    record_id = record['id']
                    saved_features = json.loads(record['fields']['blendshapes'])
                    
                    # Compute Euclidean Match Distance
                    distance = np.linalg.norm(np.array(current_features) - np.array(saved_features))
                    MATCH_THRESHOLD = 0.16 # Strict threshold
                    
                    if distance < MATCH_THRESHOLD:
                        st.success(f"🔓 Identity Verified! Distance Match Score: {distance:.4f}")
                        
                        # --- STREAK MANAGEMENT ALGORITHM ---
                        last_login_str = record['fields'].get('last_login')
                        current_streak = int(record['fields'].get('streak', 0))
                        today = date.today()
                        
                        if last_login_str:
                            last_login_date = datetime.strptime(last_login_str, "%Y-%m-%d").date()
                            days_diff = (today - last_login_date).days
                            
                            if days_diff == 1:
                                current_streak += 1
                                st.balloons()
                                st.success(f"🔥 Streak maintained! You are now at a **{current_streak} Day Streak**!")
                            elif days_diff == 0:
                                st.info(f"✅ Already logged in today. Current Streak: **{current_streak} Days**.")
                            else:
                                current_streak = 1
                                st.warning("💔 Day missed! Streak reset back to 1 day.")
                        else:
                            current_streak = 1
                            st.success("🎉 First login tracked! Streak initialized to 1 day.")
                        
                        # Sync updates back to cloud database
                        table.update(record_id, {
                            "streak": current_streak,
                            "last_login": today.strftime("%Y-%m-%d")
                        })
                    else:
                        st.error(f"🔒 Access Denied. Biometric structural mismatch (Diff: {distance:.4f}).")

# --- WORKFLOW B: NEW REGISTRATION ---
elif choice == "Register New Profile":
    st.subheader("Enroll New Face Biometrics")
    new_user = st.text_input("Create Unique Username").strip().lower()
    
    st.info("💡 Pro-Tip: Neutral facial expression ensures a highly stable baseline registration matrix.")
    reg_img = st.camera_input("Take baseline profile picture")
    
    if st.button("Register Identity Profile") and reg_img and new_user:
        with st.spinner("Registering biometric vector data..."):
            # Check for existing profile conflict
            existing = table.all(formula=f"{{username}}='{new_user}'")
            if existing:
                st.error("❌ Username already taken! Choose a different handle.")
            else:
                features = extract_blendshapes_from_bytes(reg_img)
                if features is None:
                    st.error("❌ Registration failed: Could not map facial landmarks. Try again.")
                else:
                    # Write clean registration profile payload to Airtable
                    table.create({
                        "username": new_user,
                        "blendshapes": json.dumps(features),
                        "streak": 0,
                        "last_login": ""
                    })
                    st.success(f"🎯 Successfully created biometric model profile for **{new_user}**!")