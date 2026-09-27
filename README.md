# Smart Analysis of Football First Touch 

An AI-powered system designed to objectively evaluate a football player's first-touch quality using Computer Vision and Machine Learning. By analyzing 2D video footage, the system tracks the ball and player to compute mechanical physics metrics and classify the touch as Good, Poor, or Under Pressure.

---

## Features
- **Object Detection:** Real-time tracking of the football and receiving player using **YOLOv8**.
- **Trajectory Smoothing:** Mathematical spatial smoothing using **Linear Interpolation** to handle dropped frames and motion blur.
- **Physics Engine:** Custom extraction of 4 key kinematic metrics:
  - **Speed Ratio:** Velocity change upon contact.
  - **Control Distance:** How far the ball bounces away after the touch.
  - **Direction Change:** Angular deflection of the ball path.
  - **Pressure Factor:** Proximity and density of nearby defenders.
- **Explainable AI:** Classification using a **Random Forest** ensemble (100 trees) with a "Glass-Box" approach for coaching feedback.
- **Interactive Dashboard:** Premium UI built with **Streamlit** for video uploads and real-time metric visualization.

---

##  Technology Stack
- **Core:** Python 3.9+
- **Computer Vision:** Ultralytics (YOLOv8), OpenCV
- **Machine Learning:** Scikit-Learn, Pandas, NumPy
- **Interface:** Streamlit
- **Training Environment:** Google Colab (Nvidia Tesla T4 GPU)

---

##  Installation


1. **Create a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

##  Usage

To launch the analysis dashboard, run:
```bash
streamlit run demo_app.py
```
1. Upload an MP4 football clip.
2. Click **"Run AI Analysis"**.
3. View the neon-annotated video, trajectory lines, and the final **First-Touch Score (0-10)**.



##  Dataset & Accuracy
- **Dataset:** 200 custom annotated clips (70 Good, 65 Poor, 65 Pressure).
- **Detection Accuracy:** 0.941 mAP50.
- **Classification Accuracy:** 90.8% on a 98-sample holdout test set.

