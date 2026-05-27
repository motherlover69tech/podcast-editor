FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

# ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY pyproject.toml .
RUN pip3 install --no-cache-dir -e ".[web,diarization]" && \
    pip3 install --no-cache-dir ctranslate2>=4.0

# faster-whisper auto-detects CUDA via ctranslate2 when device="cuda"

COPY . .

EXPOSE 8890

ENV PODCAST_EDITOR_DATA=/data
ENV HF_TOKEN=

CMD ["python3", "-m", "uvicorn", "src.web:app", "--host", "0.0.0.0", "--port", "8890"]
