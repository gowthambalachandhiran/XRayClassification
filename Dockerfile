FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY appv2.py .
COPY mobilenet_model_quantized.tflite .

EXPOSE 8501

CMD ["streamlit", "run", "appv2.py", "--server.port=8501", "--server.enableCORS", "false"]
