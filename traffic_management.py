import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import os

# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title="Traffic Management System",
    layout="wide"
)

# Initialize session state
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False
if "cleared" not in st.session_state:
    st.session_state.cleared = False

# App title
st.title("🚦 Traffic Management System")
st.markdown("""
Upload a traffic video to detect and track vehicles using YOLO.
""")

# ----------------------
# Sidebar: Model Settings
# ----------------------
st.sidebar.header("Model Settings")

# Fixed model path
model_path = r"C:\Users\kirol\OneDrive - Arab Open University - AOU\Desktop\project 2\yolov8n.pt"

# Confidence slider with recommended range
confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Recommended: 0.2–0.5. Higher values may filter out real detections."
)

# Show warning if confidence is too high
if confidence > 0.5:
    st.sidebar.warning(
        "⚠️ High confidence thresholds (>0.5) may cause the model to miss vehicles."
    )

# ----------------------
# Load model
# ----------------------
@st.cache_resource
def load_model(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return YOLO(path)

try:
    model = load_model(model_path)
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.stop()

# ----------------------
# Video Upload
# ----------------------
uploaded_video = st.file_uploader(
    "Upload traffic video",
    type=["mp4", "avi", "mov"]
)

frame_placeholder = st.empty()  # Placeholder for video frames

# Start button
start = st.button("Start Processing")

# Clear button
clear = st.button("Clear")

# ----------------------
# Clear logic
# ----------------------
if clear:
    frame_placeholder.empty()
    st.session_state.processing_done = False
    st.session_state.cleared = True

if st.session_state.cleared:
    st.success("🙏 Thank you for using the platform!")
    st.session_state.cleared = False  # Reset for next run

# ----------------------
# Processing logic
# ----------------------
if uploaded_video is not None and start:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    st.write(f"Video resolution: {width}x{height} | FPS: {round(fps,2)}")

    total_frames = 0
    detected_vehicles = 0

    # Status metrics
    col1, col2 = st.columns(2)
    fps_text = col1.empty()
    count_text = col2.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Run YOLO detection
        results = model.predict(
            frame,
            conf=confidence,
            classes=[2, 3, 5, 7]  # car, motorcycle, bus, truck
        )

        annotated_frame = results[0].plot()

        # Count detections
        detected_vehicles += len(results[0].boxes)

        # Update UI
        frame_placeholder.image(
            annotated_frame,
            channels="BGR",
            use_container_width=True
        )

        fps_text.metric("Frames Processed", total_frames)
        count_text.metric("Total Vehicles Detected", detected_vehicles)

    cap.release()
    st.session_state.processing_done = True

# ----------------------
# Show completion message
# ----------------------
if st.session_state.processing_done:
    st.success("✅ Processing completed!")
    st.session_state.processing_done = False  # Reset for next run

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown(
    "Built for: **Traffic Management System** | Powered by YOLO & Streamlit , By Eng Kirollos Ashraf"
)
