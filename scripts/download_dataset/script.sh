#!/bin/bash
set -e

echo "--- Konfigurowanie Kaggle API ---"

mkdir -p /app/.kaggle
echo "${KAGGLE_JSON_CONTENT}" > /app/.kaggle/kaggle.json
chmod 600 /app/.kaggle/kaggle.json

echo "Kaggle API skonfigurowane."

DATA_DIR="temp_data"
mkdir -p $DATA_DIR
echo "Stworzono folder roboczy: ${DATA_DIR}"

echo "Pobieranie i rozpakowywanie ${KAGGLE_DATASET} do ${DATA_DIR}..."
kaggle datasets download -d ${KAGGLE_DATASET} -p $DATA_DIR --unzip

echo "Pobieranie i rozpakowywanie zakończone."
echo "Rozpoczęto wysyłanie do GCS (metodą 'TIR-a')..."

gsutil -m cp -r ${DATA_DIR}/* gs://${BUCKET_NAME}/

echo "Wysyłanie do GCS zakończone."

echo "Sprzątanie dysku tymczasowego..."
rm -rf $DATA_DIR
rm -rf ~/.kaggle

echo "--- ZROBIONE! ---"