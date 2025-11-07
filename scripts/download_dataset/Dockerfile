FROM python:3.11-slim

WORKDIR /app

RUN pip install kaggle
RUN apt-get update && apt-get install -y \
    gnupg \
    curl \
    apt-transport-https \
    ca-certificates \
    unzip \
    --no-install-recommends \
 && rm -rf /var/lib/apt/lists/*

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

RUN apt-get update && apt-get install -y \
    google-cloud-cli \
 && rm -rf /var/lib/apt/lists/*

COPY script.sh .
 
RUN chmod +x script.sh

ENTRYPOINT ["./script.sh"]