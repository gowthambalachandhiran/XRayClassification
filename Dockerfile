FROM python:3.11-slim

# Install dependencies required for building packages like `pickle5`
RUN apt-get update && \
    apt-get install -y gcc python3-dev libffi-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app



# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS", "false"]