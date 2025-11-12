import os
import sys
import glob
import json
import cv2
import logging
import numpy as np  # Musimy to zaimportować
from retinaface import RetinaFace
from tqdm import tqdm
# MUSI BYĆ WERSJA Z WĄTKAMI, ABY GPU DZIAŁAŁO
from concurrent.futures import ThreadPoolExecutor, as_completed 
from config import BASE_DATA_DIR, PROCESSING_ORDER, DEVICE, NUM_WORKERS, IMAGE_EXTENSIONS

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logging.getLogger('RetinaFace').setLevel(logging.WARNING)


def process_image(image_path, model):
    """
    Przetwarza pojedynczy obraz: wykrywa twarz i zapisuje plik JSON.
    Model (funkcja) jest przekazywany jako argument.
    """
    try:
        # 1. Obliczenie ścieżki wyjściowej JSON
        base_name = os.path.splitext(image_path)[0]
        json_path = base_name + ".json"

        if os.path.exists(json_path):
            return image_path, "Skipped (JSON exists)"

        # 2. Wczytanie obrazu (BGR)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return image_path, "Failed to read image"
        
        # Poprawka: Konwersja BGR -> RGB (aby widział twarze)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Poprawka: Przygotowanie obrazu dla surowego modelu TF

        # 3. Detekcja twarzy (WŁAŚCIWA METODA)
        # Wywołujemy surową funkcję tf.function załadowaną w run()
        # To jest bezpieczne dla wątków.
        faces = RetinaFace.detect_faces(img_path=img_rgb, model=model)
        
        if not isinstance(faces, dict) or not faces:
            return image_path, "No face detected"

        # 4. Wybór najlepszej twarzy (z najwyższym 'score')
        best_face = None
        best_score = -1.0
        
        for face_data in faces.values():
            if face_data['score'] > best_score:
                best_score = face_data['score']
                best_face = face_data

        if best_face is None:
             return image_path, "Detection parsing error"

        # 5. Poprawka: Konwersja typów numpy na float dla JSON
        converted_landmarks = {
            key: [float(coord[0]), float(coord[1])] 
            for key, coord in best_face["landmarks"].items()
        }
        
        output_data = {
            "bbox": [float(val) for val in best_face["facial_area"]],
            "landmarks": converted_landmarks,
            "confidence": float(best_face["score"])
        }

        # 6. Zapis pliku JSON
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        return image_path, "Success"
        
    except Exception as e:
        return image_path, f"Error: {str(e)}"

# Ta funkcja jest poprawna
def run():
    print(f"--- Etap 3: Przetwarzanie Obrazów (Detekcja Twarzy) ---")
    print(f"Używane urządzenie: {DEVICE}")
    print(f"Liczba wątków roboczych: {NUM_WORKERS}") # ZAUWAŻ, ŻE TO WĄTKI

    # 1. Ładujemy model JEDEN RAZ
    print("Ładowanie modelu RetinaFace... (to może potrwać chwilę)")
    try:
        if DEVICE == 'cpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        # Poprawka: Ustawiamy próg (threshold) przy budowaniu modelu
        # To zwróci surową funkcję tf.function
        model = RetinaFace.build_model()
        print("Model załadowany pomyślnie.") # ZOBACZYSZ TEN LOG
        
    except Exception as e:
        print(f"BŁĄD KRYTYCZNY: Nie udało się załadować modelu RetinaFace: {e}")
        sys.exit(1)

    # Iterujemy zgodnie z wymaganą kolejnością
    for split in PROCESSING_ORDER:
        split_dir = os.path.join(BASE_DATA_DIR, split)
        if not os.path.exists(split_dir):
            print(f"Folder podziału {split_dir} nie istnieje. Pomijanie.")
            continue
        
        print(f"\nRozpoczynanie przetwarzania podziału: '{split}'...")
        
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            pattern = os.path.join(split_dir, "**", f"*{ext}")
            image_files.extend(glob.glob(pattern, recursive=True))
        
        if not image_files:
            print(f"Nie znaleziono obrazów w {split_dir}.")
            continue
            
        print(f"Znaleziono {len(image_files)} obrazów do przetworzenia.")

        # 2. Używamy ThreadPoolExecutor (puli WĄTKÓW)
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            
            # Przekazujemy ten JEDEN model do wszystkich wątków
            futures = [
                executor.submit(process_image, img_path, model) 
                for img_path in image_files
            ]
            
            pbar = tqdm(total=len(futures), desc=f"Przetwarzanie {split}")
            
            # Zbieranie wyników
            for future in as_completed(futures):
                pbar.update(1)
                img_path, status = future.result()
                if status != "Success" and status != "Skipped (JSON exists)":
                    logging.warning(f"Problem z {img_path}: {status}")
            
            pbar.close()

    print("--- Etap 3: Zakończony Pomyślnie ---")

if __name__ == "__main__":
    run()