# Base image: resmi PyTorch + CUDA (CPU için istersen plain python:3.10 da olur)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Çalışma dizini oluştur
WORKDIR /app

# Sisteme temel paketleri yükle
RUN apt-get update && apt-get install -y \
    git wget unzip ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını yükle
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Kodları kopyala
COPY . .


# Varsayılan komut (örnek: preprocessing scripti çalıştırır)
CMD ["python", "scripts/preprocess.py", "--help"]
