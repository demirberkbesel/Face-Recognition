# =============================================================================
#  Face Recognition API – Dockerfile
#  Hedef: FastAPI + InsightFace model dosyalari imaja gomulu, 
#         container internete ihtiyac duymadan calissin.
# =============================================================================

FROM python:3.12-slim

# --------------------------------------------------------------------------
# 1. Sistem bagimliliklari
#    opencv-python-headless icin libgl1, libglib2.0 gerekir. 
#    build-essential, cmake ise insightface/onnxruntime derlemesi icin.
# --------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# 2. Python paketlerini yukle
# --------------------------------------------------------------------------
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------------
# 3. InsightFace model dosyalarini build asamasinda indir
#    Boylece container ilk calistiginda internet aramaz.
# --------------------------------------------------------------------------
COPY download_models.py .
RUN python download_models.py

# --------------------------------------------------------------------------
# 4. Uygulama kodunu kopyala
# --------------------------------------------------------------------------
COPY . .

# --------------------------------------------------------------------------
# 5. Container calisma ayarlari
# --------------------------------------------------------------------------
EXPOSE 8000

# healthcheck: uygulamanin ayakta oldugunu kontrol eder
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/docs')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
