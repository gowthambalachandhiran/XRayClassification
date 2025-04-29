# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        libpq-dev \
        git \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with version specified
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model file into the container
COPY appv1.py /app/appv1.py
COPY mobilenet_model.h5 /app/mobilenet_model.h5

# Expose the port for Streamlit
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "appv1.py", "--server.port=8501", "--server.enableCORS=false"]
