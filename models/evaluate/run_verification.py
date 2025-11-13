import os
import sys
import glob
import json
import csv
import random
import numpy as np
import cv2
import faiss
import insightface
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed 
from config import (
    BASE_FOLDER_LOCAL, 
    FAISS_INDEX_FILE, FAISS_MAPPING_FILE, OCCLUSION_SIZE,
    NUM_WORKERS
)

# Importujemy funkcje pomocnicze z poprzedniego skryptu
# (Zakładam, że są w tym samym folderze lub zaimportowane poprawnie)
# Dla prostoty, kopiuję je tutaj:

def initialize_services():
    print("Ładowanie modelu InsightFace (ArcFace)... (to może potrwać chwilę)")
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        model = insightface.app.FaceAnalysis(
            name="buffalo_s", # Używamy 'buffalo_s'
            root='./insightface_models', 
            providers=providers
        )
        model.prepare(ctx_id=0, det_size=(112, 112)) # Dla WebFace
    except Exception as e:
        print(f"BŁĄD: Nie udało się załadować modelu InsightFace: {e}")
        sys.exit(1)
    print("Inicjalizacja zakończona pomyślnie.")
    return model

def discover_file_structure(local_test_path):
    print(f"Wykrywanie struktury plików w {local_test_path}...")
    search_pattern = os.path.join(local_test_path, "*", "*", "*.jpg")
    all_jpg_files = list(glob.glob(search_pattern))
    if not all_jpg_files:
        print(f"BŁĄD: Nie znaleziono plików .jpg: {search_pattern}")
        return None, None
    print(f"Znaleziono łącznie {len(all_jpg_files)} plików .jpg.")
    identity_to_imgfolders = {} 
    image_pairs = {}       
    for jpg_path in tqdm(all_jpg_files, desc="Skanowanie plików"):
        jpg_path_norm = os.path.normpath(jpg_path)
        base_name = os.path.splitext(jpg_path_norm)[0]
        json_path = base_name + ".json"
        if not os.path.exists(json_path): continue 
        image_folder_path = os.path.dirname(jpg_path_norm)
        identity_path = os.path.dirname(image_folder_path)
        if identity_path not in identity_to_imgfolders:
            identity_to_imgfolders[identity_path] = set()
        if image_folder_path not in image_pairs:
            image_pairs[image_folder_path] = {'jpg': None, 'json': None}
        identity_to_imgfolders[identity_path].add(image_folder_path)
        image_pairs[image_folder_path]['jpg'] = jpg_path_norm
        image_pairs[image_folder_path]['json'] = json_path
    print(f"Wykryto {len(identity_to_imgfolders)} folderów tożsamości z parami JPG/JSON.")
    return identity_to_imgfolders, image_pairs

def get_embedding(model, image_bgr):
    try:
        faces = model.get(image_bgr)
        if faces and len(faces) > 0:
            return faces[0].normed_embedding
    except Exception as e:
        tqdm.write(f"Warning: Błąd podczas pobierania embeddingu: {e}")
    return None

def apply_occlusion(image, landmarks_dict, bbox):
    occluded_image = image.copy()
    try:
        left_eye_y = landmarks_dict["left_eye"][1]
        right_eye_y = landmarks_dict["right_eye"][1]
        eye_y_center = int((left_eye_y + right_eye_y) / 2)
        bar_height_half = OCCLUSION_SIZE // 2
        face_x1 = int(bbox[0])
        face_x2 = int(bbox[2])
        x1 = face_x1
        y1 = max(0, eye_y_center - bar_height_half)
        x2 = face_x2
        y2 = min(image.shape[0], eye_y_center + bar_height_half)
        cv2.rectangle(occluded_image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    except Exception as e:
        tqdm.write(f"Warning: Błąd podczas nakładania okluzji: {e}")
        return image.copy() 
    return occluded_image

# --- NOWA FUNKCJA ROBOCZA ---

def process_verification_query(args):
    """
    Funkcja robocza: bierze JEDEN obrazek, nakłada okluzję, 
    i porównuje go z CAŁĄ galerią, zwracając listę wyników (score, label).
    """
    img_folder_path, ground_truth_id, image_pairs, model, gallery_matrix, id_to_index_map = args

    local_img_path = image_pairs.get(img_folder_path, {}).get('jpg')
    local_json_path = image_pairs.get(img_folder_path, {}).get('json')

    if not local_img_path or not local_json_path:
        return f"Warning: Błąd mapowania dla {img_folder_path}"

    img = cv2.imread(local_img_path)
    json_data = None
    try:
        with open(local_json_path, 'r') as jf:
            json_data = json.load(jf)
    except Exception as e:
        return f"Warning: Błąd odczytu JSON {local_json_path}: {e}"
    
    if (img is None or json_data is None or 
        "landmarks" not in json_data or "bbox" not in json_data):
        return f"Warning: Brak pełnych danych dla {local_img_path}"

    occluded_img = apply_occlusion(img, json_data["landmarks"], json_data["bbox"])
    query_embedding = get_embedding(model, occluded_img)
    
    if query_embedding is None:
        return f"Warning: Nie udało się uzyskać embeddingu dla {local_img_path}"
        
    query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
    
    # --- GŁÓWNA ZMIANA ---
    # Porównaj z CAŁĄ galerią, a nie tylko Top-3
    all_scores = np.dot(gallery_matrix, query_embedding_normalized)
    
    try:
        ground_truth_index = id_to_index_map[ground_truth_id]
    except KeyError:
        return f"Warning: Nie znaleziono ID {ground_truth_id} w mapie galerii."

    verification_results = []
    for i in range(len(all_scores)):
        score = all_scores[i]
        if i == ground_truth_index:
            label = "genuine"
        else:
            label = "imposter"
        verification_results.append((score, label))
        
    return verification_results # Zwraca listę wyników


# --- GŁÓWNA FUNKCJA EWALUACYJNA (ZMODYFIKOWANA) ---

def run_verification_test(model, identity_to_imgfolders, image_pairs):
    """
    Uruchamia test weryfikacji 1:1 i zapisuje wyniki do 'verification_scores.csv'.
    """
    print("Wczytywanie galerii FAISS i mapowania ID...")
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(FAISS_MAPPING_FILE, 'r') as f:
            # Wczytujemy {'0': 'id_3', ...}
            index_to_id_map_str = json.load(f)
            # Odwracamy mapę, aby mieć {'id_3': 0, ...} dla szybkiego dostępu
            id_to_index_map = {v: int(k) for k, v in index_to_id_map_str.items()}
            
        # Wyciągamy surowe wektory z galerii
        gallery_matrix = index.reconstruct_n(0, index.ntotal)
        
    except Exception as e:
        print(f"BŁĄD: Nie udało się wczytać plików FAISS. Uruchom najpierw budowanie galerii.")
        print(f"Error: {e}")
        return

    output_csv = "verification_scores.csv"
    print(f"Rozpoczynanie testu weryfikacji (równolegle z {NUM_WORKERS} workerami)...")
    print(f"Wyniki będą zapisane w: {output_csv}")
    
    identity_paths = list(identity_to_imgfolders.keys())
    
    # Zbieramy wszystkie zadania do wykonania
    tasks = []
    for id_path in identity_paths:
        ground_truth_id = os.path.basename(id_path)
        image_folder_paths = sorted(list(identity_to_imgfolders[id_path]))
        
        split_point = max(1, len(image_folder_paths) // 2)
        query_folders = image_folder_paths[split_point:] # Bierzemy DRUGĄ połowę

        for img_folder_path in query_folders:
            tasks.append(
                (img_folder_path, ground_truth_id, image_pairs, model, gallery_matrix, id_to_index_map)
            )
            
    if not tasks:
        print("\n--- Ewaluacja Zakończona ---")
        print("Nie znaleziono żadnych zapytań do przetworzenia.")
        return

    # Otwieramy plik CSV i uruchamiamy pulę wątków
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["score", "label"]) # Prosta struktura: (wynik, genuine/imposter)
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(tqdm(
                executor.map(process_verification_query, tasks), 
                total=len(tasks), 
                desc="Testowanie Weryfikacji (równolegle)"
            ))

        # Zbieranie wyników i zapisywanie do CSV
        total_pairs = 0
        for result in results:
            if isinstance(result, list): # Jeśli to lista wyników
                for score, label in result:
                    writer.writerow([score, label])
                    total_pairs += 1
            else: # Jeśli to string z błędem
                tqdm.write(str(result))
                
    print(f"\n--- Test Weryfikacji Zakończony ---")
    print(f"Zapisano łącznie {total_pairs} par genuine/imposter do {output_csv}.")


def main():
    model = initialize_services()
    local_test_path = os.path.join(BASE_FOLDER_LOCAL, "test")
    
    identity_to_imgfolders, image_pairs = discover_file_structure(local_test_path)
    if not identity_to_imgfolders:
        print("Zatrzymanie, nie znaleziono plików.")
        return

    # Ten skrypt wymaga, aby Krok 1 (budowanie galerii) został już wykonany
    # przez 'run_evaluation.py'
    if not os.path.exists(FAISS_INDEX_FILE):
        print(f"BŁĄD: Nie znaleziono pliku {FAISS_INDEX_FILE}.")
        print("Proszę najpierw uruchomić 'run_evaluation.py', aby zbudować galerię.")
        sys.exit(1)

    # Uruchamiamy tylko test weryfikacji
    run_verification_test(model, identity_to_imgfolders, image_pairs)
    
    print("Gotowe.")

if __name__ == "__main__":
    main()