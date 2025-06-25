import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np

# Placeholder function for loading model
def load_model():
    pass

# Load YOLOv5 model
@st.cache_resource()
def get_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/content/drive/MyDrive/Isnartech_internship/yolov5_folder_final/runs/train/exp/weights/best.pt', force_reload=True)
    return model

# Function to make predictions
def predict(image, model):
    results = model(image)
    return results

# Streamlit app
def main():
    st.set_page_config(page_title="Tumour Detector", page_icon="🧠")
    st.title("Brain Tumour Detector")
    st.write("This app detects brain tumors in MRI images.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image for the model
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Load YOLOv5 model
        model = get_model()

        # Detect button
        detect_button = st.button('Detect')

        if detect_button:
            # Make prediction
            results = predict(image, model)

            # Display the predicted image
            predicted_image = results.render()[0]
            st.image(predicted_image, caption='Predicted Image.', use_column_width=True)

            # Display the results
            st.subheader("Detected Tumors:")
            for result in results.xyxy[0]:
                label = result[-1]
                confidence = result[-2]
                st.write(f"Label: {label}, Confidence: {confidence}")

if __name__ == '__main__':
    main()
