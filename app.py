import io
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# CONSTANTS
# -----------------------------
MODEL_PATH = Path("runs/detect/train-2/weights/best.pt")
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]

# -----------------------------
# STYLES
# -----------------------------
PAGE_STYLE = """
<style>
html, body, [class*="css"] {
    background: #0F172A;
    color: white;
}

.main-title {
    font-size: 44px;
    font-weight: 800;
    color: #38BDF8;
}

.subtitle {
    color: #CBD5E1;
    font-size: 18px;
    line-height: 1.6;
}

.sidebar .stSidebar {
    background-color: #111827;
}

.stButton>button {
    background-color: #2563EB;
    color: white;
}

</style>
"""

st.markdown(PAGE_STYLE, unsafe_allow_html=True)

# -----------------------------
# UTILITIES
# -----------------------------

@st.cache_resource
def load_model(model_path: Path) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return YOLO(str(model_path))


def get_device() -> str:
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def build_detection_table(results) -> List[dict]:
    boxes = results[0].boxes
    class_names = results[0].names
    rows = []

    for index, box in enumerate(boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        rows.append(
            {
                "Tumor ID": index + 1,
                "Class": class_names[class_id],
                "Confidence (%)": round(confidence * 100, 2),
            }
        )

    return rows


def get_annotated_image(results) -> Image.Image:
    annotated_array = results[0].plot()
    return Image.fromarray(annotated_array)


def create_download_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()

# -----------------------------
# HEADER
# -----------------------------

st.markdown(
    """
    <div class='main-title'>
        
         Brain Tumor Detection System
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='subtitle'>
        Upload an MRI image and run YOLOv8 brain tumor detection with a professional,
        intuitive interface.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("Brain Tumor Detector")

st.sidebar.info(
    """
    YOLOv8 Brain Tumor Detection

    Developer: Supriya Devkota
    Framework: Ultralytics YOLOv8
    """
)

st.sidebar.success("Model ready to use")

# -----------------------------
# MODEL LOAD
# -----------------------------

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

# -----------------------------
# IMAGE UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=SUPPORTED_IMAGE_TYPES,
)

if uploaded_file is None:
    st.info("Upload a brain MRI image to begin tumor detection.")
    st.stop()

original_image = Image.open(uploaded_file).convert("RGB")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Original Image")
    st.image(original_image, use_column_width=True)

with col2:
    st.subheader(" Image Details")
    st.markdown(f"**Filename:** {uploaded_file.name}")
    st.markdown(f"**Format:** {original_image.format}")
    st.markdown(f"**Dimensions:** {original_image.width} x {original_image.height} px")

# -----------------------------
# PREDICTION
# -----------------------------

device = get_device()

with st.spinner(" Detecting brain tumor..."):
    with tempfile.NamedTemporaryFile(suffix=f".{uploaded_file.name.split('.')[-1]}", delete=False) as temp_file:
        original_image.save(temp_file.name)
        results = model.predict(
            source=temp_file.name,
            conf=0.25,
            device=device,
        )

annotated_image = get_annotated_image(results)

detection_rows = build_detection_table(results)
confidence_score = max((float(box.conf[0]) for box in results[0].boxes), default=0.0)

st.success("Detection completed")

# -----------------------------
# RESULTS
# -----------------------------

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction")
    st.image(annotated_image, use_column_width=True)

with col2:
    st.subheader("Detection Summary")

    if len(results[0].boxes) == 0:
        st.success("No brain tumor detected")
        st.info("The model did not find any suspicious regions in the uploaded image.")
    else:
        st.metric("Detected Tumors", len(results[0].boxes))
        st.metric("Highest Confidence", f"{confidence_score * 100:.2f}%")
        st.warning(
            "⚠️ This tool is for educational purposes only. Please consult a medical professional for diagnosis."
        )

        st.divider()
        st.subheader("Detection Details")
        st.dataframe(detection_rows, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Model Confidence")
st.progress(float(confidence_score))
st.write(f"Confidence Score: **{confidence_score * 100:.2f}%**")

st.divider()

st.subheader("Download Result")
result_bytes = create_download_bytes(annotated_image, fmt="JPEG")
st.download_button(
    label="Download Predicted Image",
    data=result_bytes,
    file_name="brain_tumor_prediction.jpg",
    mime="image/jpeg",
)

st.divider()

st.markdown(
    """
    ## About this Project

    This application uses a **YOLOv8 object detection model** to detect brain tumors
    from MRI images.

    ### Features

    - Brain Tumor Detection
    - MRI Image Upload
    - YOLOv8 Prediction
    - Confidence Score
    - Detection Summary
    - Download Results

    ---

    ### Medical Disclaimer

    This application is intended **for educational and research purposes only**.

    It **must not** be used as a substitute for professional medical diagnosis.

    Always consult a qualified healthcare professional.

    ---

    ### Developed By

    **Supriya Devkota**

    AI / ML Engineer
    YOLOv8 • Streamlit • Python
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='text-align:center; padding:20px; font-size:16px; color:gray;'>
        Made with using Streamlit & YOLOv8
    </div>
    """,
    unsafe_allow_html=True,
)
