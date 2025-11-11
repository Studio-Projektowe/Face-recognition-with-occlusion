import os
import sys
import glob
import shutil
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import BASE_DATA_DIR, PROCESSING_ORDER, NUM_WORKERS, IMAGE_EXTENSIONS

logging.basicConfig(level=logging.INFO)

def move_image(image_path):
    """
    Przenosi obraz ze struktury .../id/img.jpg do .../id/img/img.jpg
    """
    try:
        # 1. Wyodrębnij części ścieżki
        file_dir = os.path.dirname(image_path)    # .../train/id_0001
        file_name = os.path.basename(image_path) # 0.jpg
        base_name = os.path.splitext(file_name)[0] # 0

        # 2. Stwórz nową ścieżkę docelową
        # Nowy folder: .../train/id_0001/0
        new_dir = os.path.join(file_dir, base_name)
        
        # Nowa pełna ścieżka pliku: .../train/id_0001/0/0.jpg
        new_image_path = os.path.join(new_dir, file_name)

        # 3. Jeśli już jest na miejscu, pomiń
        if os.path.exists(new_image_path):
            return image_path, "Skipped (already moved)"

        # 4. Stwórz nowy folder
        os.makedirs(new_dir, exist_ok=True)

        # 5. Przenieś plik
        shutil.move(image_path, new_image_path)
        
        return new_image_path, "Success"

    except Exception as e:
        return image_path, f"Error: {str(e)}"

def run():
    print(f"--- Etap 2b: Restrukturyzacja plików (tworzenie podfolderów) ---")
    print(f"Liczba procesów roboczych: {NUM_WORKERS}")

    for split in PROCESSING_ORDER:
        split_dir = os.path.join(BASE_DATA_DIR, split)
        if not os.path.exists(split_dir):
            print(f"Folder podziału {split_dir} nie istnieje. Pomijanie.")
            continue
        
        print(f"\nRozpoczynanie restrukturyzacji podziału: '{split}'...")
        
        # 1. Znalezienie wszystkich obrazów w danym podziale
        # WAŻNE: Szukamy tylko na pierwszym poziomie podfolderów (id_xxx),
        # a nie rekursywnie, aby nie przetwarzać plików, które już przenieśliśmy.
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            # Wzór: .../split_dir/id_folder/*.jpg
            pattern = os.path.join(split_dir, "*", f"*{ext}")
            image_files.extend(glob.glob(pattern, recursive=False))
        
        if not image_files:
            print(f"Nie znaleziono obrazów do przeniesienia w {split_dir} (możliwe, że już są przeniesione).")
            continue
            
        print(f"Znaleziono {len(image_files)} obrazów do przeniesienia.")

        # 2. Uruchomienie puli procesów
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(move_image, img_path) for img_path in image_files]
            
            pbar = tqdm(total=len(futures), desc=f"Przenoszenie {split}")
            
            for future in as_completed(futures):
                pbar.update(1)
                img_path, status = future.result()
                if status != "Success" and status != "Skipped (already moved)":
                    logging.warning(f"Problem z {img_path}: {status}")
            
            pbar.close()

    print("--- Etap 2b: Zakończony Pomyślnie ---")

if __name__ == "__main__":
    run()