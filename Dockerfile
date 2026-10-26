FROM python:3.11-slim

WORKDIR /app

# System deps required at runtime by pytesseract (OCR), pdf2image/unstructured
# (poppler), opencv-python (libgl/libglib), and pydub/librosa (ffmpeg).
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "unstructured[pdf]" redis cachetools optimum onnxruntime psutil scikit-learn flask-socketio python-magic

COPY . .

EXPOSE 5001
ENV PORT=5001

CMD ["python", "app.py"]
