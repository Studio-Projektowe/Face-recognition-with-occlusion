import os
import sys
import subprocess
from config import BASE_DATA_DIR, BUCKET_NAME

def run():
    print("--- Etap 4: Wysyłanie do Google Cloud Storage ---")

    if not BUCKET_NAME:
        print("BŁĄD: Zmienna środowiskowa BUCKET_NAME nie jest ustawiona.")
        sys.exit(1)
        
    if not os.path.exists(BASE_DATA_DIR):
        print(f"BŁĄD: Folder '{BASE_DATA_DIR}' nie istnieje. Nic do wysłania.")
        sys.exit(1)

    # 1. Weryfikacja dostępności gsutil
    try:
        subprocess.run(["gsutil", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        print("BŁĄD: Komenda 'gsutil' nie znaleziona.")
        print("Upewnij się, że Google Cloud SDK jest zainstalowane w kontenerze Docker.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"BŁĄD: Problem z gsutil: {e.stderr}")
        sys.exit(1)

    # 2. Przygotowanie komendy
    # gsutil -m cp -r webface-112x112 gs://my-awesome-bucket/
    source_path = BASE_DATA_DIR
    destination_path = f"gs://{BUCKET_NAME}/"
    
    command = [
        "gsutil",
        "-m",       # Użyj wielowątkowości do kopiowania
        "cp",       # Kopiuj
        "-r",       # Rekursywnie
        source_path,
        destination_path
    ]

    print(f"Uruchamianie komendy: {' '.join(command)}")

    # 3. Uruchomienie wysyłania
    try:
        subprocess.run(command, check=True)
        print("Wysyłanie zakończone pomyślnie.")
    except subprocess.CalledProcessError as e:
        print(f"BŁĄD podczas wysyłania do GCS: {e}")
        sys.exit(1)

    print("--- Etap 4: Zakończony Pomyślnie ---")

if __name__ == "__main__":
    # Wymaga ręcznego ustawienia zmiennych do testów:
    # os.environ["BUCKET_NAME"] = "your-test-bucket"
    # Oraz uwierzytelnienia gcloud (np. `gcloud auth login`)
    run()