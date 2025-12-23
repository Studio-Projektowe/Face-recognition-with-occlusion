import os
import sys
import glob
import json
import cv2
import logging
from retinaface import RetinaFace
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed 
from config import BASE_DATA_DIR, PROCESSING_ORDER, DEVICE, NUM_WORKERS, IMAGE_EXTENSIONS

DETECTION_THRESHOLD = 0.5 

logging.basicConfig(level=logging.INFO)
logging.getLogger('RetinaFace').setLevel(logging.WARNING)

def process_missing_json(image_path, model):
    """
    Przetwarza obraz: wykrywa twarz i zapisuje plik JSON, ale tylko jeśli JSON nie istnieje.
    Zastosowano obniżony próg ufności.
    """
    try:
        base_name = os.path.splitext(image_path)[0]
        json_path = base_name + ".json"

        if os.path.exists(json_path):
            return image_path, "Skipped (JSON exists)"

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return image_path, "Failed to read image"
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        faces = RetinaFace.detect_faces(
            img_path=img_rgb, 
            model=model,
            threshold=DETECTION_THRESHOLD
        )
        
        if not isinstance(faces, dict) or not faces:
            return image_path, "No face detected (even with reduced threshold)"

        best_face = None
        best_score = -1.0
        
        for face_data in faces.values():
            if face_data['score'] > best_score:
                best_score = face_data['score']
                best_face = face_data

        if best_face is None:
            return image_path, "Detection parsing error"

        converted_landmarks = {
            key: [float(coord[0]), float(coord[1])] 
            for key, coord in best_face["landmarks"].items()
        }
        
        output_data = {
            "bbox": [float(val) for val in best_face["facial_area"]],
            "landmarks": converted_landmarks,
            "confidence": float(best_face["score"])
        }

        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        return image_path, "Success (New JSON created)"
        
    except Exception as e:
        return image_path, f"Error: {str(e)}"

def run():
    print(f"--- Etap 3.1: Ponowna Detekcja Twarzy (Tylko Brakujące JSON) ---")
    print(f"Używany próg ufności: {DETECTION_THRESHOLD}")
    print(f"Liczba wątków roboczych: {NUM_WORKERS}")

    print("Ładowanie modelu RetinaFace...")
    try:
        if DEVICE == 'cpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        model = RetinaFace.build_model()
        print("Model załadowany pomyślnie.")
        
    except Exception as e:
        print(f"BŁĄD KRYTYCZNY: Nie udało się załadować modelu RetinaFace: {e}")
        sys.exit(1)

    for split in PROCESSING_ORDER:
        split_dir = os.path.join(BASE_DATA_DIR, split)
        if not os.path.exists(split_dir):
            continue
        
        print(f"\nRozpoczynanie przetwarzania podziału: '{split}'...")
        
        files_to_process = []
        for ext in IMAGE_EXTENSIONS:
            pattern = os.path.join(split_dir, "**", f"*{ext}")
            
            all_image_files = glob.glob(pattern, recursive=True)

            for img_path in all_image_files:
                base_name = os.path.splitext(img_path)[0]
                json_path = base_name + ".json"
                
                if not os.path.exists(json_path):
                    files_to_process.append(img_path)

        if not files_to_process:
            print(f"Wszystkie obrazy w {split} mają już pliki JSON. Pomijanie.")
            continue
            
        print(f"Znaleziono {len(files_to_process)} obrazów do ponownego przetworzenia.")

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            
            futures = [
                executor.submit(process_missing_json, img_path, model) 
                for img_path in files_to_process
            ]
            
            pbar = tqdm(total=len(futures), desc=f"Ponowna Detekcja {split} (T={DETECTION_THRESHOLD})")
            
            for future in as_completed(futures):
                pbar.update(1)
                img_path, status = future.result()
                if status.startswith("Error") or status.startswith("Warning"):
                    logging.warning(f"Problem z {img_path}: {status}")
                
            pbar.close()

    print("--- Etap Ponownej Detekcji Zakończony Pomyślnie ---")

if __name__ == "__main__":
    run()