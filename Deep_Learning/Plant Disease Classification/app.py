import streamlit as st
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision import models
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data directories
data_dir = "/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "valid")
train = ImageFolder(train_dir, transform=transforms.ToTensor())

# Define transformation to resize images to 256x256
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
#prediction
def predict_image(img, model):
    """Converts image to array and return the predicted class
        with highest probability"""
    # Convert to a batch of 1
    xb = img.unsqueeze(0).to(device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return train.classes[preds[0].item()]

# Load trained model
class PlantDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseClassifier, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.densenet(x)
        return x
num_classes = len(os.listdir(train_dir))
model = PlantDiseaseClassifier(num_classes)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

st.set_page_config(page_title="Plant Disease Classifier", page_icon=":seedling:")
st.title('Plant Disease Classifier')
st.write('Upload an image of a plant and click on "Classify" to get the predicted class.')

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.subheader("Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Resize the image to 256x256 and preprocess
    image_tensor = preprocess(image)

    # Make predictions
    if st.button('Classify', key='classify_button'):
        predicted_class = predict_image(image_tensor, model)
        

        # Display prediction
        st.subheader("Prediction")
        st.success(f"Predicted Class: {predicted_class}")
        

