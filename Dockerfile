FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY appv1.py .
COPY mobilenet_model.keras .

EXPOSE 8501

CMD ["streamlit", "run", "appv2.py", "--server.port=8501", "--server.enableCORS", "false"]
