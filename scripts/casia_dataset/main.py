import os
import sys
import time
import logging
from config import KAGGLE_DATASET, BUCKET_NAME, DEVICE

import s_01_download
import s_02_prepare
import s_02b_restructure
import s_03_process
import s_04_upload

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    start_time = time.time()
    logging.info("Rozpoczynanie całego procesu przetwarzania WebFace.")
    
    if not KAGGLE_DATASET or not BUCKET_NAME:
        logging.error("BŁĄD KRYTYCZNY: Zmienne KAGGLE_DATASET i BUCKET_NAME muszą być ustawione.")
        sys.exit(1)
        
    logging.info(f"Dataset: {KAGGLE_DATASET}")
    logging.info(f"Bucket GCS: {BUCKET_NAME}")
    logging.info(f"Urządzenie do przetwarzania: {DEVICE}")

    try:
        # Krok 1: Pobieranie
        logging.info("="*50)
        s_01_download.run()
        
        # Krok 2: Przygotowanie (train/val/test)
        logging.info("="*50)
        s_02_prepare.run()
        
        # Krok 2b: Restrukturyzacja (przeniesienie do podfolderów)
        logging.info("="*50)
        s_02b_restructure.run()

        # Krok 3: Przetwarzanie (RetinaFace)
        logging.info("="*50)
        s_03_process.run()
        
        # Krok 4: Wysyłanie
        logging.info("="*50)
        s_04_upload.run()
        
    except Exception as e:
        logging.error(f"Wystąpił nieoczekiwany błąd podczas procesu: {e}")
        sys.exit(1)
    
    end_time = time.time()
    total_duration = end_time - start_time
    logging.info("="*50)
    logging.info(f"Cały proces zakończony pomyślnie!")
    logging.info(f"Całkowity czas: {total_duration:.2f} sekund ({total_duration/3600:.2f} godzin).")

if __name__ == "__main__":
    main()