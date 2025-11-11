import os
import sys
import zipfile
import subprocess
from config import KAGGLE_DATASET, BASE_DATA_DIR

def run():
    print("--- Etap 1: Pobieranie Datasetu ---")
    
    if not KAGGLE_DATASET:
        print("BŁĄD: Zmienna środowiskowa KAGGLE_DATASET nie jest ustawiona.")
        sys.exit(1)
        
    if os.path.exists(BASE_DATA_DIR):
        print(f"Folder '{BASE_DATA_DIR}' już istnieje. Pomijanie pobierania.")
        return

    print(f"Pobieranie datasetu: {KAGGLE_DATASET}...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"BŁĄD podczas pobierania z Kaggle: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("BŁĄD: Komenda 'kaggle' nie znaleziona. Upewnij się, że jest zainstalowana.")
        sys.exit(1)

    # 2. Znalezienie pliku .zip
    # Nazwa pliku zip to zazwyczaj 'nazwa-datasetu.zip'
    zip_filename = KAGGLE_DATASET.split('/')[-1] + ".zip"
    
    if not os.path.exists(zip_filename):
        print(f"BŁĄD: Nie znaleziono pobranego pliku {zip_filename}")
        sys.exit(1)

    # 3. Rozpakowanie
    print(f"Rozpakowywanie {zip_filename}...")
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(".")
        print(f"Pomyślnie rozpakowano do bieżącego katalogu.")
    except zipfile.BadZipFile:
        print(f"BŁĄD: Plik {zip_filename} nie jest poprawnym plikiem ZIP.")
        sys.exit(1)

    # 4. Sprzątanie
    os.remove(zip_filename)
    print(f"Usunięto archiwum {zip_filename}.")
    
    # 5. Weryfikacja
    if not os.path.exists(BASE_DATA_DIR):
        print(f"BŁĄD: Po rozpakowaniu nie znaleziono oczekiwanego folderu '{BASE_DATA_DIR}'.")
        print("Sprawdź, czy nazwa BASE_DATA_DIR w config.py zgadza się z zawartością archiwum.")
        sys.exit(1)
        
    print("--- Etap 1: Zakończony Pomyślnie ---")

if __name__ == "__main__":
    # Umożliwia testowe uruchomienie tylko tego skryptu
    # Wymaga ręcznego ustawienia zmiennych:
    # os.environ["KAGGLE_DATASET"] = "larryfreeman/webface-112x112"
    run()