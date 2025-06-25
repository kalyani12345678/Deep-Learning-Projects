import streamlit as st
import torch
from torchvision.transforms import ToTensor
from PIL import Image



# Load the trained model
model_path = r"C:\Users\RELINCE\Documents\streamlit-demo\best_model_1(99.79%).pth"  # Use raw string literal (r"") to avoid Unicode escape error
model = torch.load(model_path)
model.eval()
from model import ResNet9

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to 256x256
    image = image.resize((256, 256))
    # Convert image to tensor and normalize
    image = ToTensor()(image)
    # Add batch dimension
    image = image.unsqueeze(0)
    return image

# Function to make predictions
def predict_image(image):
    with torch.no_grad():
        # Make prediction
        outputs = model(image)
        # Get predicted class
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Streamlit app
def main():
    st.title("Plant Disease Classifier")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and make prediction
        processed_image = preprocess_image(image)
        prediction = predict_image(processed_image)

        # Display prediction
        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
