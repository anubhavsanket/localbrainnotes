FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyMuPDF + Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY vaults/ /app/vaults/

WORKDIR /app/backend

EXPOSE 8000

CMD ["python", "main.py"]
