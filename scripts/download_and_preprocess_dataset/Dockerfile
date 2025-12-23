FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    unzip \
    curl \
    gnupg \
    lsb-release \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
# instalacja Google Cloud SDK (gsutil)
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y google-cloud-sdk
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY kaggle.json /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json
COPY . .
CMD ["python", "main.py"]