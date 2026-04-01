# Using python:3.11-slim-bullseye instead of plain slim
# This is a more stable base image that unpacks more reliably
FROM python:3.11-slim-bullseye

# System dependency for librosa .wav file reading
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure uploads directory exists
RUN mkdir -p data/uploads

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]