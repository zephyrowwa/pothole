import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

def load_labels(path=r'D:\ery\School\Jupyter Notebooks\yolov11_streamlit\labels.txt'):
    with open(path,'r') as f:
        return [line.strip() for line in f.readlines()]
    
@st.cache_resource
def load_model():
    model = YOLO(r'D:\ery\School\Jupyter Notebooks\yolov11_streamlit\best.pt')
    model.eval()
    return model

def detect_objects(model,image,labels,con_th=0.5):
    results = model(image)

    boxes = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= con_th:
                cls = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                label = labels[cls] if cls < len(labels) else str(cls)
                boxes.append({
                    'label': label,
                    'confidence': conf,
                    'bbox': xyxy.tolist()
                })
    return boxes

def draw_boxes(image, boxes):
    if isinstance(image, Image.Image):
        image = np.array(image)

    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for box in boxes:
        x1, y1, x2, y2 = box['bbox']
        conf = box['confidence']
        label = box['label']
        
        text = f"{label} {conf:.2f}"
        color = (0, 255, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

st.title('yolo 11 vehicle and pedestrian detection app')
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    labels = load_labels()

    if st.button("Detect Objects"):
        boxes = detect_objects(model, image, labels)
        st.success(f"Detected {len(boxes)} object(s)")
        result_img = draw_boxes(image, boxes)
        st.image(result_img, caption="Detection Result", use_column_width=True)