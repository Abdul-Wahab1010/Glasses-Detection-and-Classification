import streamlit as st
import cv2
import numpy as np
import torch
import torchvision
import tensorflow as tf

from PIL import Image

from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Glasses AI System",
    page_icon="🕶️",
    layout="wide"
)

# ======================================
# CUSTOM CSS
# ======================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
    }

    h1 {
        color: white;
        text-align: center;
        font-size: 45px;
    }

    h3 {
        color: cyan;
    }

    .stButton>button {
        background-color: #00ADB5;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 200px;
        font-size: 18px;
    }

    .css-1d391kg {
        background-color: #161A23;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ======================================
# TITLE
# ======================================

st.title("🕶️ Glasses Detection & Classification System")

st.write(
    "Real-Time Detection using Faster R-CNN and Classification using CNN"
)

# ======================================
# DEVICE
# ======================================

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

st.sidebar.write(f"Device: {device}")

# ======================================
# BUILD CLASSIFICATION MODEL
# ======================================

base_model = MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

classifier = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

classifier.build((None,224,224,3))

# Load trained weights
classifier.load_weights("glasses_classifier.h5")

# ======================================
# LOAD DETECTION MODEL
# ======================================

num_classes = 2

model = fasterrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=None
)

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    num_classes
)

model.load_state_dict(
    torch.load(
        "faster_rcnn_glasses.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

# ======================================
# SIDEBAR
# ======================================

st.sidebar.title("⚙️ Settings")

mode = st.sidebar.selectbox(
    "Choose Mode",
    [
        "Classification",
        "Detection"
    ]
)

input_type = st.sidebar.selectbox(
    "Choose Input",
    [
        "Upload Image",
        "Webcam"
    ]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.7
)

# ======================================
# CLASSIFICATION FUNCTION
# ======================================

def classify_image(image_np):

    img = cv2.resize(image_np, (224, 224))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = classifier.predict(img, verbose=0)

    confidence = prediction[0][0]

    if confidence > 0.5:
        label = "No Glasses"
    else:
        label = "Glasses"

    return label, confidence

# ======================================
# DETECTION FUNCTION
# ======================================

def detect_glasses(frame):

    # Resize for speed
    frame = cv2.resize(frame, (640, 480))

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_tensor = F.to_tensor(image_rgb)

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        predictions = model(image_tensor)

    boxes = predictions[0]['boxes']

    scores = predictions[0]['scores']

    for i in range(len(boxes)):

        score = scores[i].item()

        if score > confidence_threshold:

            x1, y1, x2, y2 = boxes[i].int().cpu().numpy()

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Glasses {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    return frame

# ======================================
# IMAGE UPLOAD
# ======================================

if input_type == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        image_np = np.array(image)

        st.subheader("Original Image")

        st.image(image_np, use_container_width=True)

        # ==============================
        # CLASSIFICATION MODE
        # ==============================

        if mode == "Classification":

            label, confidence = classify_image(image_np)

            st.subheader("Prediction")

            st.success(
                f"Result: {label} | Confidence: {confidence:.2f}"
            )

        # ==============================
        # DETECTION MODE
        # ==============================

        elif mode == "Detection":

            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            output = detect_glasses(image_bgr)

            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            st.subheader("Detection Result")

            st.image(output, use_container_width=True)

# ======================================
# WEBCAM MODE
# ======================================

elif input_type == "Webcam":

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    # Smaller webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0

    while run:

        ret, frame = cap.read()

        if not ret:
            st.error("Webcam Not Working")
            break

        frame_count += 1

        # Skip frames to reduce lag
        if frame_count % 3 != 0:
            continue

        # ==============================
        # CLASSIFICATION MODE
        # ==============================

        if mode == "Classification":

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            label, confidence = classify_image(rgb_frame)

            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # ==============================
        # DETECTION MODE
        # ==============================

        else:

            frame = detect_glasses(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        FRAME_WINDOW.image(frame)

    cap.release()