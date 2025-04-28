FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY appv1.py .
# Copy the model file into the project
COPY mobilenet_model.keras .

EXPOSE 8501

CMD ["streamlit", "run", "appv1.py", "--server.port=8501", "--server.enableCORS", "false"]
