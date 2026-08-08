# OpenCV Detection

[![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/opencv-4.9.0-blue?logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35.0-orange?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/numpy-1.26-informational?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pillow](https://img.shields.io/badge/pillow-10.0.0-lightgrey?logo=python&logoColor=white)](https://python-pillow.org/)


A simple, clean, and professional demo project using OpenCV and Streamlit for realtime detection of faces, eyes, hands, and full body. The app can run using a webcam or process uploaded images and demonstrates common OpenCV cascades and detection pipelines with a friendly Streamlit UI.

Live demo: https://opencv-uitqdgpaljq7fnzdsfudcg.streamlit.app/

---

## Features

- Realtime face, eye, hand and full-body detection using OpenCV.
- Streamlit-based user interface for quick interaction and deployment.
- Support for webcam stream and image uploads.
- Lightweight and easy to extend for custom detection models.

---

## Tech Stack

- Python 3.8+
- OpenCV (opencv-python-headless)
- Streamlit
- NumPy
- Pillow

Badges (clickable):

- [Python](https://www.python.org/): ![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)
- [OpenCV](https://opencv.org/): ![OpenCV](https://img.shields.io/badge/opencv-4.9.0-blue?logo=opencv&logoColor=white)
- [Streamlit](https://streamlit.io/): ![Streamlit](https://img.shields.io/badge/streamlit-1.35.0-orange?logo=streamlit&logoColor=white)
- [NumPy](https://numpy.org/): ![NumPy](https://img.shields.io/badge/numpy-1.26-informational?logo=numpy&logoColor=white)
- [Pillow](https://python-pillow.org/): ![Pillow](https://img.shields.io/badge/pillow-10.0.0-lightgrey?logo=python&logoColor=white)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/mayank-gariya/Opencv.git
cd Opencv
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r Opencv-Detection/requirements.txt
```

---

## Running the App

Start the Streamlit app (adjust the path if your main script is named differently):

```bash
streamlit run Opencv-Detection/app.py
```

- Choose the detection mode (Face / Eyes / Hands / Full Body) from the sidebar.
- Select Webcam or Upload an image to see detections in realtime.

---

## Configuration

- The detection uses OpenCV cascade classifiers by default. You can replace or extend these with your own trained models (Haar, LBP, or DNN-based detectors) by modifying the detection module in `Opencv-Detection`.

---

## Live Demo

A deployed live demo is available here:

https://opencv-uitqdgpaljq7fnzdsfudcg.streamlit.app/

Placeholders for demo screenshots (replace with your images):

- ![Live Demo - Webcam](docs/demo-webcam.png)
- ![Live Demo - Upload Image](docs/demo-upload.png)

Replace the above image files in `docs/` (create the folder if it doesn't exist) and update their filenames if necessary.

---

## Contributing

Contributions are welcome. If you find issues or want to add features (for example: pose estimation, improved hand detection, or DNN-based face detectors), please open an issue or submit a pull request.

Suggested workflow:

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes and push
4. Open a Pull Request

---

## License

If this project does not yet have a LICENSE file, add one (for example MIT) or update this section to reflect the correct license.

---

## Acknowledgements

- Built with OpenCV and Streamlit. Thanks to the open-source community for the cascade classifiers and tooling.

