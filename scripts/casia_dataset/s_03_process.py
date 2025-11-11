import os
import sys
import glob
import json
import cv2
import logging
from retinaface import RetinaFace
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import BASE_DATA_DIR, PROCESSING_ORDER, DEVICE, NUM_WORKERS, IMAGE_EXTENSIONS

# Konfiguracja logowania, aby wyciszyć komunikaty z retinaface (opcjonalnie)
logging.basicConfig(level=logging.INFO)
logging.getLogger('RetinaFace').setLevel(logging.WARNING)


# Zmienna globalna dla modelu, aby zainicjować go raz na proces roboczy
model = None

def init_worker():
    """Inicjalizuje model RetinaFace w każdym procesie roboczym."""
    global model
    try:
        # Używamy `None` dla CPU (zgodnie z dokumentacją retina-face)
        # lub 0 dla pierwszego GPU
        gpu_id = 0 if DEVICE == 'cuda' else -1
        model = RetinaFace.build_model(gpu_id=gpu_id)
        logging.info(f"Worker (PID: {os.getpid()}) loaded model on {DEVICE}.")
    except Exception as e:
        logging.error(f"Worker (PID: {os.getpid()}) failed to load model: {e}")
        model = None # Upewnij się, że jest None, jeśli ładowanie się nie powiedzie

def process_image(image_path):
    """Przetwarza pojedynczy obraz: wykrywa twarz i zapisuje plik JSON."""
    global model
    
    if model is None:
        return image_path, "Model not loaded"

    try:
        # 1. Obliczenie ścieżki wyjściowej JSON
        base_name = os.path.splitext(image_path)[0]
        json_path = base_name + ".json"

        # Pomiń, jeśli plik JSON już istnieje
        if os.path.exists(json_path):
            return image_path, "Skipped (JSON exists)"

        # 2. Wczytanie obrazu
        # Używamy cv2.imread, ponieważ tego oczekuje retina-face
        img = cv2.imread(image_path)
        if img is None:
            return image_path, "Failed to read image"

        # 3. Detekcja twarzy
        # `model.detect_faces` zwraca słownik, gdzie klucze to np. 'face_1', 'face_2'
        faces = model.detect_faces(img)
        
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

        # 5. Przygotowanie danych do zapisu (bbox i landmarks)
        # BBox to [x1, y1, x2, y2]
        # Landmarks to słownik: 'right_eye', 'left_eye', 'nose', 'mouth_right', 'mouth_left'
        output_data = {
            "bbox": [float(val) for val in best_face["facial_area"]],
            "landmarks": best_face["landmarks"],
            "confidence": float(best_face["score"])
        }

        # 6. Zapis pliku JSON
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        return image_path, "Success"
        
    except Exception as e:
        return image_path, f"Error: {str(e)}"

def run():
    print(f"--- Etap 3: Przetwarzanie Obrazów (Detekcja Twarzy) ---")
    print(f"Używane urządzenie: {DEVICE}")
    print(f"Liczba procesów roboczych: {NUM_WORKERS}")

    # Iterujemy zgodnie z wymaganą kolejnością
    for split in PROCESSING_ORDER:
        split_dir = os.path.join(BASE_DATA_DIR, split)
        if not os.path.exists(split_dir):
            print(f"Folder podziału {split_dir} nie istnieje. Pomijanie.")
            continue
        
        print(f"\nRozpoczynanie przetwarzania podziału: '{split}'...")
        
        # 1. Znalezienie wszystkich obrazów w danym podziale
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            # Szukamy rekursywnie we wszystkich podfolderach (id_0, id_1, ...)
            pattern = os.path.join(split_dir, "**", f"*{ext}")
            image_files.extend(glob.glob(pattern, recursive=True))
        
        if not image_files:
            print(f"Nie znaleziono obrazów w {split_dir}.")
            continue
            
        print(f"Znaleziono {len(image_files)} obrazów do przetworzenia.")

        # 2. Uruchomienie puli procesów
        # `initializer=init_worker` uruchomi funkcję `init_worker` raz w każdym procesie
        with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker) as executor:
            
            # Utworzenie zadań
            futures = [executor.submit(process_image, img_path) for img_path in image_files]
            
            # Pasek postępu TQDM
            pbar = tqdm(total=len(futures), desc=f"Przetwarzanie {split}")
            
            # Zbieranie wyników
            for future in as_completed(futures):
                pbar.update(1)
                img_path, status = future.result()
                if status != "Success" and status != "Skipped (JSON exists)":
                    # Logujemy tylko błędy, aby nie zaśmiecać konsoli
                    logging.warning(f"Problem z {img_path}: {status}")
            
            pbar.close()

    print("--- Etap 3: Zakończony Pomyślnie ---")

if __name__ == "__main__":
    run()