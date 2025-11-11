import os
import sys
import shutil
import random
import math
from config import BASE_DATA_DIR, SPLIT_RATIOS

def run():
    print("--- Etap 2: Przygotowanie Struktury Plików (Train/Val/Test) ---")
    
    if not os.path.exists(BASE_DATA_DIR):
        print(f"BŁĄD: Folder źródłowy '{BASE_DATA_DIR}' nie istnieje. Uruchom najpierw 01_download.py.")
        sys.exit(1)

    # Sprawdzenie, czy podział już został wykonany
    train_dir = os.path.join(BASE_DATA_DIR, 'train')
    if os.path.exists(train_dir):
        print("Foldery train/val/test już istnieją. Pomijanie etapu przygotowania.")
        return

    # 1. Pobranie listy wszystkich folderów tożsamości
    try:
        identity_folders = [
            f for f in os.listdir(BASE_DATA_DIR) 
            if os.path.isdir(os.path.join(BASE_DATA_DIR, f))
        ]
        if not identity_folders:
            print(f"BŁĄD: Nie znaleziono folderów tożsamości w '{BASE_DATA_DIR}'.")
            sys.exit(1)
        print(f"Znaleziono {len(identity_folders)} folderów tożsamości.")
    except Exception as e:
        print(f"BŁĄD podczas listowania folderów w '{BASE_DATA_DIR}': {e}")
        sys.exit(1)

    # 2. Wymieszanie listy
    random.shuffle(identity_folders)

    # 3. Obliczenie punktów podziału
    total_identities = len(identity_folders)
    train_count = math.floor(total_identities * SPLIT_RATIOS['train'])
    val_count = math.floor(total_identities * SPLIT_RATIOS['val'])
    
    # test_count to reszta, aby uniknąć błędów zaokrągleń
    train_split = identity_folders[:train_count]
    val_split = identity_folders[train_count : train_count + val_count]
    test_split = identity_folders[train_count + val_count :]

    print(f"Podział: {len(train_split)} train, {len(val_split)} val, {len(test_split)} test.")

    # 4. Stworzenie docelowych folderów i przeniesienie
    splits = {
        "train": train_split,
        "val": val_split,
        "test": test_split
    }

    for split_name, identities in splits.items():
        target_dir = os.path.join(BASE_DATA_DIR, split_name)
        os.makedirs(target_dir, exist_ok=True)
        print(f"Przenoszenie {len(identities)} folderów do {target_dir}...")
        
        for identity in identities:
            source_path = os.path.join(BASE_DATA_DIR, identity)
            dest_path = os.path.join(target_dir, identity)
            try:
                shutil.move(source_path, dest_path)
            except Exception as e:
                print(f"BŁĄD podczas przenoszenia {source_path} do {dest_path}: {e}")

    print("--- Etap 2: Zakończony Pomyślnie ---")

if __name__ == "__main__":
    run()