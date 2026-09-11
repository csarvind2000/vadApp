FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# libGL/glib are needed by OpenCV even in headless builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --no-deps \
        pytorchvideo fvcore iopath portalocker termcolor tabulate yacs

COPY . .

# Ultralytics writes settings on import and needs a writable location; without
# this it falls back to /tmp and logs a warning on every start.
RUN mkdir -p /config/Ultralytics
ENV PYTHONUNBUFFERED=1 \
    RESEARCH_ROOT=/research \
    YOLO_CONFIG_DIR=/config/Ultralytics \
    VAD_YOLO_WEIGHTS=/app/yolo11n.pt \
    PORT=5000
EXPOSE 5000

# One worker with threads: the GPU model is loaded once and shared. Multiple
# processes would each load their own copy and exhaust GPU memory.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", \
     "--timeout", "3600", "app:app"]
