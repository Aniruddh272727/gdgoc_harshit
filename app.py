import cv2
from ultralytics import YOLO

# Load YOLO model trained for cattle detection
detector = YOLO("yolov8n.pt")  # replace with cattle-specific model

# Load breed classification model
import torch
breed_model = torch.load("cattle_breed_model.pth")
breed_model.eval()

# Capture image from camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Detect animal
results = detector(frame)
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = frame[y1:y2, x1:x2]

    # Preprocess for classifier
    crop_resized = cv2.resize(crop, (224, 224))
    crop_tensor = torch.tensor(crop_resized).permute(2,0,1).unsqueeze(0).float()

    # Predict breed
    with torch.no_grad():
        pred = breed_model(crop_tensor)
        breed = pred.argmax(dim=1).item()
        print("Predicted breed:", breed)
