# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model file into the container
COPY appv1.py /app/appv1.py

COPY mobilenet_model.keras /app/mobilenet_model.keras


# Expose the port for Streamlit
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "appv1.py", "--server.port=8501", "--server.enableCORS=false"]
