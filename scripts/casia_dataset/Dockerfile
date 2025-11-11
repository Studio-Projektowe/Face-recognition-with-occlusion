# Użyj oficjalnego obrazu Python
FROM python:3.10-slim

# Ustaw katalog roboczy
WORKDIR /app

# 1. Zainstaluj zależności systemowe
# - 'unzip' do rozpakowywania datasetu
# - 'curl', 'gnupg', 'lsb-release' do instalacji Google Cloud SDK
# - 'libgl1' jest wymagane przez OpenCV
RUN apt-get update && apt-get install -y \
    unzip \
    curl \
    gnupg \
    lsb-release \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 2. Zainstaluj Google Cloud SDK (dla 'gsutil')
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y google-cloud-sdk

# 3. Skopiuj i zainstaluj zależności Pythona
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Skopiuj pliki z uwierzytelnieniem (WAŻNE!)
# Musisz mieć plik `kaggle.json` lokalnie obok Dockerfile
# Musisz mieć plik `gcloud-key.json` (service account key) lokalnie, jeśli Docker nie będzie miał dostępu do gcloud z hosta
COPY kaggle.json /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json

# Opcjonalnie: Uwierzytelnianie GCS za pomocą klucza serwisowego
# COPY gcloud-key.json /app/gcloud-key.json
# ENV GOOGLE_APPLICATION_CREDENTIALS=/app/gcloud-key.json

# 5. Skopiuj resztę kodu aplikacji
COPY . .

# 6. Ustaw domyślną komendę
# Uruchomi główny skrypt, gdy kontener wystartuje
CMD ["python", "main.py"]