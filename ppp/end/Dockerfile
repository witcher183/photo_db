# Dockerfile для Face Repair (Debian Trixie совместимый)
FROM python:3.12-slim

# Системные зависимости для OpenCV/GFPGAN (актуальные для Trixie)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем и устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Копируем код
COPY FaceRapairClass.py .
COPY FRIter.py .

# Создаем папки
RUN mkdir -p /app/inputs /app/results /app/weights /app/experiments/pretrained_models \
    && chmod -R 777 /app/inputs /app/results

# Volumes
VOLUME ["/app/inputs", "/app/results"]

# Запуск
