# Base image có sẵn Python 3.11 + cần thiết cho TensorFlow CPU
FROM python:3.11-slim

# Cài thư viện hệ thống cần thiết cho librosa, tensorflow
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Thư mục làm việc
WORKDIR /app

# Copy code và requirements
COPY . /app

# Cài thư viện python
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 (fly.io default)
EXPOSE 8080

# Chạy Flask app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]