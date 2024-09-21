import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
# Load YOLOv8 model (ensure to specify the correct model path or name)
model = YOLO('best.pt')  # You can use 'yolov8n.pt', 'yolov8s.pt', etc.
# Title and description
st.title("YOLOv8 Object Detection")
st.write("Upload an image, and YOLOv8 will predict objects in the image.")


# Image uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image from the uploaded file
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Convert image to NumPy array (for YOLO processing)
    image_np = np.array(image)
    
    # Run YOLOv8 model prediction for classification
    results = model.predict(image_np)
    
    # Extract the predicted class and confidence
    top_result = results[0]  # Get the first result
    predicted_class = top_result.names[top_result.probs.top1]  # Get the class name using the top1 index
    confidence = top_result.probs.top1conf * 100  # Get the confidence score
    
    # Display the classification result
    st.write(f"**Predicted Class**: {predicted_class}")
    st.write(f"**Confidence**: {confidence:.2f}%")