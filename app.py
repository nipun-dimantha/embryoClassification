import streamlit as st
import torch
from torchvision import models, transforms
import cv2
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# Define the CNN model class as used previously
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)

    def forward(self, x):
        return self.base_model(x)

# Load the saved models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
model.load_state_dict(torch.load('best_trained_model.pth'))

# Define preprocessing for uploaded images
def preprocess_jpeg(jpeg_path):
    img = cv2.imread(jpeg_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return transforms.ToTensor()(img).unsqueeze(0).to(device)

# Prediction function
def predict_image(image_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

# Streamlit interface
st.title('Embryo Image Classification Dashboard')


uploaded_file = st.file_uploader("Choose an embryo image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_path = f"temp_image.{uploaded_file.type.split('/')[1]}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(image_path, caption='Uploaded Image.', use_column_width=True)
    image_tensor = preprocess_jpeg(image_path)
    if st.button('Predict'):
        prediction = predict_image(image_tensor)
        if prediction == 1:
            st.markdown("<span style='color:green; font-size:24px'>Prediction: **Good Embryo**</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:red; font-size:24px'>Prediction: **Bad Embryo**</span>",
                        unsafe_allow_html=True)