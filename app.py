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
import io

# Load the trained model
@st.cache_resource
def load_trained_model():

    model = load_model('mobilenet_model.keras')

    return model

model = load_trained_model()

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
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to overlay heatmap on image
def overlay_heatmap(heatmap, img):
    heatmap = cv2.resize(heatmap, (img.width, img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_array = np.array(img)
    superimposed_img = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(superimposed_img)

# Streamlit app interface
st.title("Chest X-Ray Classification: Pneumonia Detection")
st.write("Upload a chest X-ray image to determine if it indicates Pneumonia or is Normal.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image from uploaded file
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_array = preprocess_image(img)

    # Predict
    predictions = model.predict(img_array)
    probability = predictions[0][0]
    predicted_class = class_labels[int(round(probability))]

    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Probability of Pneumonia:** {probability:.2f}")
    st.write(f"**Probability of Normal:** {1 - probability:.2f}")

    # Generate and display Grad-CAM
    heatmap = generate_gradcam(img_array, model, last_conv_layer_name='conv_pw_13_relu')
    superimposed_img = overlay_heatmap(heatmap, img)
    st.image(superimposed_img, caption='Grad-CAM', use_column_width=True)
