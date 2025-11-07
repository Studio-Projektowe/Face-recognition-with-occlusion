#!/bin/bash
# Zatrzymuje skrypt, jeśli jakakolwiek komenda zawiedzie
# (Bardzo ważne, żeby Job się wywalił, jak coś pójdzie nie tak)
set -e

echo "--- Konfigurowanie Kaggle API ---"

# Krok 1: Poprawne wstrzyknięcie sekretu
# (W kontenerze /root to folder domowy, więc ~/.kaggle to /root/.kaggle)
mkdir -p /root/.kaggle
# Poprawka: Używamy cudzysłowów, aby zachować formatowanie JSON i uniknąć wykonania
echo "${KAGGLE_JSON_CONTENT}" > /root/.kaggle/kaggle.json
# Zabezpieczamy plik (wymagane przez Kaggle)
chmod 600 /root/.kaggle/kaggle.json

echo "Kaggle API skonfigurowane."

# Krok 2: Przygotowanie folderu roboczego
# (Jesteśmy w /app, zgodnie z Dockerfile)
DATA_DIR="casia_temp_data"
mkdir -p $DATA_DIR
echo "Stworzono folder roboczy: ${DATA_DIR}"

# Krok 3: Pobieranie i ROZPAKOWYWANIE (o wiele czystsza metoda)
# $KAGGLE_DATASET jest wstrzykiwane przez Terraform
echo "Pobieranie i rozpakowywanie ${KAGGLE_DATASET} do ${DATA_DIR}..."
# Używamy -p (path), aby pobrać dane do naszego folderu roboczego
# i magicznej flagi --unzip, która robi całą robotę za nas!
kaggle datasets download -d ${KAGGLE_DATASET} -p $DATA_DIR --unzip

echo "Pobieranie i rozpakowywanie zakończone."
echo "Rozpoczęto wysyłanie do GCS (metodą 'TIR-a')..."

# Krok 4: Kopiowanie rozpakowanych plików do GCS (metoda "TIR-a")
# Poprawka: Używamy zmiennej $BUCKET_NAME (wstrzykniętej z Terraforma)
# `/*` na końcu kopiuje ZAWARTOŚĆ folderu, a nie sam folder
gsutil -m -q cp -r ${DATA_DIR}/* gs://${BUCKET_NAME}/

echo "Wysyłanie do GCS zakończone."

# Krok 5: Sprzątanie (zwalnianie dysku tymczasowego 15GB)
echo "Sprzątanie dysku tymczasowego..."
rm -rf $DATA_DIR
rm -rf /root/.kaggle # Sprzątamy też klucz API

echo "--- ZROBIONE! ---"