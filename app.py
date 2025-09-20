import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

# ----------------------------
# Load models
# ----------------------------
detector = YOLO("yolov8n.pt")  # pretrained model, replace with cattle-specific
breed_model = torch.load("cattle_breed_model.pth", map_location="cpu")
breed_model.eval()

# Dummy label list for demonstration
BREEDS = ["Gir", "Sahiwal", "Murrah", "Red Sindhi", "Holstein"]

# ----------------------------
# Helper functions
# ----------------------------
def preprocess_image(image):
    image_resized = cv2.resize(image, (224, 224))
    tensor = torch.tensor(image_resized).permute(2,0,1).unsqueeze(0).float()
    return tensor

def predict_breed(crop):
    tensor = preprocess_image(crop)
    with torch.no_grad():
        pred = breed_model(tensor)
        idx = pred.argmax(dim=1).item()
        return BREEDS[idx], torch.softmax(pred, dim=1).max().item()

def detect_and_classify(image):
    results = detector(image)
    annotated = image.copy()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = image[y1:y2, x1:x2]

        # Breed prediction
        breed, conf = predict_breed(crop)

        # Draw box + label
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(annotated, f"{breed} ({conf:.2f})",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0,255,0), 2)
    return annotated

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🐄 Cattle & Buffalo Breed Recognition")
st.write("Upload a cattle/buffalo photo or capture from webcam.")

option = st.radio("Choose input source:", ["Upload Image", "Use Webcam"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        annotated = detect_and_classify(img_np)
        st.image(annotated, caption="Prediction", channels="BGR")

elif option == "Use Webcam":
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image")
            break
        annotated = detect_and_classify(frame)
        FRAME_WINDOW.image(annotated, channels="BGR")

    cap.release()
