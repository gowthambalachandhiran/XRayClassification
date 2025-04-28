FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY appv2.py .
# Copy the model file
COPY mobilenet_model_quantized.tflite .

EXPOSE 8501

CMD ["streamlit", "run", "appv2.py", "--server.port=8501", "--server.enableCORS", "false"]
