# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 13:21:31 2025

@author: gowtham.balachan
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os

# Function to display files in the project directory
def display_project_files():
    with st.expander("📁 Show Project Directory Files"):
        project_dir = os.path.dirname(__file__)
        files = os.listdir(project_dir)
        for file in files:
            st.markdown(f"- {file}")

# Display the project directory files first
st.title("Chest X-Ray Classification: Pneumonia Detection")
st.write("Upload a chest X-ray image to determine if it indicates Pneumonia or is Normal.")

display_project_files()

# Load the trained model
@st.cache_resource
def load_trained_model():
    model_path = os.path.join(os.path.dirname(__file__), 'mobilenet_model.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = load_model(model_path)
    return model

# Try to load model with error handling
try:
    model = load_trained_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model_loaded = False

# Define class labels
class_labels = ['Normal', 'Pneumonia']

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
    return img_array

# Function to generate Grad-CAM heatmap
def generate_gradcam(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
