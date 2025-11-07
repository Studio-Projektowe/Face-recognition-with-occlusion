FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install kaggle
RUN apt-get update && apt-get install -y google-cloud-sdk unzip
RUN chmod +x script.sh

COPY script.sh .

ENTRYPOINT ["./script.sh"]