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
import os

# Display files in the current working directory
def display_project_files():
    with st.expander("📁 Show Project Directory Files"):
        project_dir = os.getcwd()
        files = os.listdir(project_dir)
        for file in files:
            st.markdown(f"- {file}")
    return project_dir, files

# Load the TFLite model
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Class labels
class_labels = ['Normal', 'Pneumonia']

# Preprocess uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
    return img_array

# --- Streamlit UI ---
st.title("Chest X-Ray Classification: Pneumonia Detection")
st.write("Upload a chest X-ray image to determine if it indicates Pneumonia or is Normal.")

# Show project files
project_dir, files = display_project_files()

# Load model if found
model_filename = 'mobilenet_model_quantized.tflite'
model_path = os.path.join(project_dir, model_filename)

if model_filename in files:
    interpreter = load_tflite_model(model_path)
    model_loaded = True
    st.success(f"✅ Model '{model_filename}' loaded successfully.")
else:
    model_loaded = False
    st.error(f"❌ Model file '{model_filename}' not found. Please upload it.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img_array = preprocess_image(img)

    # Run prediction
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_index, img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    
    probability = predictions[0][0]
    predicted_class = class_labels[int(round(probability))]

    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Probability of Pneumonia:** {probability:.2f}")
    st.write(f"**Probability of Normal:** {1 - probability:.2f}")
elif uploaded_file is not None and not model_loaded:
    st.warning("⚠️ Cannot make predictions because the model is not loaded.")
