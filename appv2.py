# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 13:21:31 2025

@author: gowtham.balachan
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os

# Function to display files in the project directory
def display_project_files():
    with st.expander("📁 Show Project Directory Files"):
        # Use current working directory
        project_dir = os.getcwd()
        
        # If you have a specific folder for your models, you can specify it here:
        # project_dir = "/app/models"  # Example path for models
        
        files = os.listdir(project_dir)
        for file in files:
            st.markdown(f"- {file}")
    return project_dir, files


# Load the TensorFlow Lite model (only if file exists)
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

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
def generate_gradcam(img_array, interpreter, last_conv_layer_index, pred_index=None):
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    input_index = input_details[0]['index']
    interpreter.set_tensor(input_index, img_array)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_index = output_details[0]['index']
    predictions = interpreter.get_tensor(output_index)

    if pred_index is None:
        pred_index = np.argmax(predictions[0])

    # Get the gradients from the last convolutional layer
    grad_model = tf.keras.models.Model([interpreter.get_input_details()[0]['index']], 
                                       [interpreter.get_output_details()[0]['index']])
    grads = tf.gradients(predictions, grad_model.output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    return pooled_grads

# Function to overlay heatmap on image
def overlay_heatmap(heatmap, img):
    heatmap = cv2.resize(heatmap, (img.width, img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_array = np.array(img)
    superimposed_img = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(superimposed_img)

# --- Streamlit app interface ---
st.title("Chest X-Ray Classification: Pneumonia Detection")
st.write("Upload a chest X-ray image to determine if it indicates Pneumonia or is Normal.")

# First display the project directory files
project_dir, files = display_project_files()

# Check if model file is present
model_filename = 'mobilenet_model_quantized.tflite'
model_path = os.path.join(project_dir, model_filename)

if model_filename in files:
    interpreter = load_tflite_model(model_path)
    model_loaded = True
    st.success(f"✅ Model '{model_filename}' loaded successfully.")
else:
    model_loaded = False
    st.error(f"❌ Model file '{model_filename}' not found in project directory. Please upload the model file.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    # Read image from uploaded file
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_array = preprocess_image(img)

    # Predict
    # Set input tensor
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    interpreter.set_tensor(input_index, img_array)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_details = interpreter.get_output_details()
    output_index = output_details[0]['index']
    predictions = interpreter.get_tensor(output_index)
    probability = predictions[0][0]
    predicted_class = class_labels[int(round(probability))]

    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Probability of Pneumonia:** {probability:.2f}")
    st.write(f"**Probability of Normal:** {1 - probability:.2f}")

    # Generate and display Grad-CAM
    heatmap = generate_gradcam(img_array, interpreter, last_conv_layer_index=-1)
    superimposed_img = overlay_heatmap(heatmap, img)
    st.image(superimposed_img, caption='Grad-CAM', use_column_width=True)
elif uploaded_file is not None and not model_loaded:
    st.warning("⚠️ Cannot make predictions because model is not loaded.")
