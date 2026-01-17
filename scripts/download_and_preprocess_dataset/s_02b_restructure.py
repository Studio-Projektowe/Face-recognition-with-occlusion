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
    try:
        file_dir = os.path.dirname(image_path)                       
        file_name = os.path.basename(image_path)        
        base_name = os.path.splitext(file_name)[0]    

        new_dir = os.path.join(file_dir, base_name)
        
        new_image_path = os.path.join(new_dir, file_name)

        if os.path.exists(new_image_path):
            return image_path, "Skipped (already moved)"

        os.makedirs(new_dir, exist_ok=True)

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
        
                                                                          
                                                                                 
        image_files = []
        for ext in IMAGE_EXTENSIONS:
                                                 
            pattern = os.path.join(split_dir, "*", f"*{ext}")
            image_files.extend(glob.glob(pattern, recursive=False))
        
        if not image_files:
            print(f"Nie znaleziono obrazów do przeniesienia w {split_dir} (możliwe, że już są przeniesione).")
            continue
            
        print(f"Znaleziono {len(image_files)} obrazów do przeniesienia.")

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