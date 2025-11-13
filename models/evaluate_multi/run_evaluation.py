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
# NOWE IMPORTY
from concurrent.futures import ThreadPoolExecutor, as_completed 
from config import (
    BASE_FOLDER_LOCAL, 
    FAISS_INDEX_FILE, FAISS_MAPPING_FILE, RESULTS_CSV, OCCLUSION_SIZE,
    NUM_WORKERS # NOWY IMPORT
)

# --- 1. INICJALIZACJA MODELU ---

def initialize_services():
    """Ładuje model InsightFace."""
    print("Ładowanie modelu InsightFace (ArcFace)... (to może potrwać chwilę)")
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        model = insightface.app.FaceAnalysis(
            name="buffalo_s", # Używamy 'buffalo_s', który jest bardziej stabilny
            root='./insightface_models', 
            providers=providers
        )
        # Upewnij się, że to pasuje do Twojego datasetu (112x112 dla WebFace)
        model.prepare(ctx_id=0, det_size=(112, 112)) 
    except Exception as e:
        print(f"BŁĄD: Nie udało się załadować modelu InsightFace.")
        print(f"Error: {e}")
        sys.exit(1)
        
    print("Inicjalizacja zakończona pomyślnie.")
    return model

def get_embedding(model, image_bgr):
    """Pobiera embedding dla pojedynczego obrazu."""
    try:
        faces = model.get(image_bgr)
        if faces and len(faces) > 0:
            return faces[0].normed_embedding
    except Exception as e:
        tqdm.write(f"Warning: Błąd podczas pobierania embeddingu: {e}")
    return None

# --- 2. FUNKCJA POMOCNICZA (LOKALNA) ---

def discover_file_structure(local_test_path):
    """Mapuje lokalną strukturę plików."""
    print(f"Wykrywanie struktury plików w {local_test_path}...")
    search_pattern = os.path.join(local_test_path, "*", "*", "*.jpg")
    all_jpg_files = list(glob.glob(search_pattern))
    
    if not all_jpg_files:
        print(f"BŁĄD: Nie znaleziono plików .jpg pasujących do wzorca: {search_pattern}")
        return None, None
        
    print(f"Znaleziono łącznie {len(all_jpg_files)} plików .jpg.")
    identity_to_imgfolders = {} 
    image_pairs = {}       

    for jpg_path in tqdm(all_jpg_files, desc="Skanowanie plików"):
        jpg_path_norm = os.path.normpath(jpg_path)
        base_name = os.path.splitext(jpg_path_norm)[0]
        json_path = base_name + ".json"
        
        if not os.path.exists(json_path):
            continue 
        
        image_folder_path = os.path.dirname(jpg_path_norm)
        identity_path = os.path.dirname(image_folder_path)
        
        if identity_path not in identity_to_imgfolders:
            identity_to_imgfolders[identity_path] = set()
        if image_folder_path not in image_pairs:
            image_pairs[image_folder_path] = {'jpg': None, 'json': None}
        
        identity_to_imgfolders[identity_path].add(image_folder_path)
        image_pairs[image_folder_path]['jpg'] = jpg_path_norm
        image_pairs[image_folder_path]['json'] = json_path

    print(f"Wykryto {len(identity_to_imgfolders)} folderów tożsamości z kompletnymi parami JPG/JSON.")
    return identity_to_imgfolders, image_pairs

# --- 3. NOWA FUNKCJA POMOCNIKA DLA GALERII ---

def process_identity_for_gallery(args):
    """
    Funkcja robocza dla workera. Przetwarza jedno ID.
    Przyjmuje krotkę (tuple) argumentów, aby działać z 'executor.map'.
    """
    id_path, identity_to_imgfolders, image_pairs, model = args
    
    identity_id = os.path.basename(id_path)
    image_folder_paths = sorted(list(identity_to_imgfolders[id_path]))
    
    split_point = max(1, len(image_folder_paths) // 2)
    gallery_folders = image_folder_paths[:split_point]
    
    if not gallery_folders:
        return (identity_id, None)

    id_embeddings = []
    for img_folder_path in gallery_folders:
        local_path = image_pairs.get(img_folder_path, {}).get('jpg')
        if not local_path:
            continue
            
        img = cv2.imread(local_path)
        if img is None:
            continue
            
        embedding = get_embedding(model, img)
        if embedding is not None:
            id_embeddings.append(embedding)

    if id_embeddings:
        avg_embedding = np.mean(id_embeddings, axis=0)
        avg_embedding /= np.linalg.norm(avg_embedding)
        return (identity_id, avg_embedding)
        
    return (identity_id, None)

# --- 4. BUDOWANIE GALERII FAISS (WERSJA RÓWNOLEGŁA) ---

def build_faiss_gallery(model, identity_to_imgfolders, image_pairs):
    """
    Tworzy galerię FAISS równolegle używając ThreadPoolExecutor.
    """
    print(f"--- ROZPOCZYNAM Budowanie Galerii FAISS (równolegle z {NUM_WORKERS} workerami) ---")
    
    identity_paths = list(identity_to_imgfolders.keys())
    if not identity_paths:
        print("BŁĄD: Lista folderów tożsamości jest pusta.")
        return False
    
    gallery_embeddings = []
    index_to_id_map = {}
    faiss_index_counter = 0

    # Przygotowanie zadań dla puli wątków
    tasks = [(id_path, identity_to_imgfolders, image_pairs, model) for id_path in identity_paths]

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Używamy executor.map, aby przetwarzać zadania równolegle
        # i zachować kolejność (dzięki tqdm)
        results = list(tqdm(
            executor.map(process_identity_for_gallery, tasks), 
            total=len(tasks), 
            desc="Tworzenie galerii ID (równolegle)"
        ))

    # Zbieranie wyników (już po zakończeniu pracy wątków)
    for identity_id, avg_embedding in results:
        if avg_embedding is not None:
            gallery_embeddings.append(avg_embedding)
            index_to_id_map[faiss_index_counter] = identity_id
            faiss_index_counter += 1
        else:
            tqdm.write(f"Warning: Nie udało się wygenerować embeddingu dla {identity_id}")

    print(f"Zakończono. Znaleziono {len(gallery_embeddings)} unikalnych tożsamości.")
    
    if not gallery_embeddings:
        print("BŁĄD: Galeria jest pusta, nie można zbudować indeksu FAISS.")
        return False
        
    dimension = gallery_embeddings[0].shape[0] 
    gallery_matrix = np.array(gallery_embeddings).astype('float32')
    
    index = faiss.IndexFlatIP(dimension)
    index.add(gallery_matrix)
    
    print(f"Zapisywanie indeksu FAISS do {FAISS_INDEX_FILE}...")
    faiss.write_index(index, FAISS_INDEX_FILE)
    
    print(f"Zapisywanie mapowania ID do {FAISS_MAPPING_FILE}...")
    with open(FAISS_MAPPING_FILE, 'w') as f:
        json.dump(index_to_id_map, f)
        
    return True

# --- 5. TESTOWANIE Z OKLUZJĄ (WERSJA RÓWNOLEGŁA) ---

# Ta funkcja pomocnicza jest potrzebna, aby przekazać wiele argumentów do puli wątków
def process_occlusion_query(args):
    """
    Funkcja robocza dla workera. Przetwarza jedno zapytanie okluzji.
    """
    img_folder_path, ground_truth_id, image_pairs, model, index, index_to_id_map, output_occlusion_dir = args

    local_img_path = image_pairs.get(img_folder_path, {}).get('jpg')
    local_json_path = image_pairs.get(img_folder_path, {}).get('json')

    if not local_img_path or not local_json_path:
        return f"Warning: Wewnętrzny błąd mapowania dla {img_folder_path}"

    img = cv2.imread(local_img_path)
    json_data = None
    try:
        with open(local_json_path, 'r') as jf:
            json_data = json.load(jf)
    except Exception as e:
        return f"Warning: Błąd odczytu JSON {local_json_path}: {e}"
    
    if (img is None or json_data is None or 
        "landmarks" not in json_data or "bbox" not in json_data):
        return f"Warning: Brak pełnych danych (JPG/JSON/Landmarks/BBox) dla {local_img_path}"

    occluded_img = apply_occlusion(img, json_data["landmarks"], json_data["bbox"])
    
    try:
        original_filename = os.path.basename(local_img_path)
        save_path = os.path.join(output_occlusion_dir, f"occluded_{ground_truth_id}_{original_filename}")
        cv2.imwrite(save_path, occluded_img)
    except Exception as e:
        tqdm.write(f"Warning: Nie udało się zapisać obrazu okluzji {save_path}: {e}")
    
    query_embedding = get_embedding(model, occluded_img)
    
    if query_embedding is None:
        return f"Warning: Nie udało się uzyskać embeddingu dla {local_img_path}"
        
    query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
    query_vector = np.expand_dims(query_embedding_normalized, axis=0).astype('float32')
    
    D, I = index.search(query_vector, 3) # Szukaj Top 3
    
    top1_idx, top2_idx, top3_idx = I[0]
    top1_sim, top2_sim, top3_sim = D[0]
    
    top1_id = index_to_id_map.get(str(top1_idx), "N/A")
    top2_id = index_to_id_map.get(str(top2_idx), "N/A")
    top3_id = index_to_id_map.get(str(top3_idx), "N/A")
    
    is_correct = (top1_id == ground_truth_id)
    
    # Zwracamy gotowy wiersz do zapisu
    return [ground_truth_id, top1_id, f"{top1_sim:.4f}", top2_id, f"{top2_sim:.4f}", top3_id, f"{top3_sim:.4f}", is_correct]


def apply_occlusion(image, landmarks_dict, bbox):
    """Nakłada pasek okluzji na wysokości oczu o szerokości twarzy."""
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
        tqdm.write(f"Warning: Błąd podczas nakładania okluzji (np. brak landmarków): {e}")
        return image.copy() 
    return occluded_image

def run_occlusion_evaluation(model, identity_to_imgfolders, image_pairs):
    """
    Testuje drugą połowę zdjęć równolegle.
    """
    print("Wczytywanie galerii FAISS i mapowania ID...")
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(FAISS_MAPPING_FILE, 'r') as f:
            index_to_id_map = json.load(f)
    except Exception as e:
        print(f"BŁĄD: Nie udało się wczytać plików FAISS. Uruchom najpierw budowanie galerii.")
        print(f"Error: {e}")
        return

    print(f"Rozpoczynanie ewaluacji z okluzją (równolegle z {NUM_WORKERS} workerami)...")
    
    identity_paths = list(identity_to_imgfolders.keys())
    total_queries = 0
    correct_top1 = 0
    
    output_occlusion_dir = "occlusion_photos"
    os.makedirs(output_occlusion_dir, exist_ok=True)
    print(f"Obrazy z okluzją będą zapisywane w: {output_occlusion_dir}")
    
    # Zbieramy wszystkie zadania do wykonania
    tasks = []
    for id_path in identity_paths:
        ground_truth_id = os.path.basename(id_path)
        image_folder_paths = sorted(list(identity_to_imgfolders[id_path]))
        
        split_point = max(1, len(image_folder_paths) // 2)
        query_folders = image_folder_paths[split_point:] # Bierzemy DRUGĄ połowę

        for img_folder_path in query_folders:
            tasks.append(
                (img_folder_path, ground_truth_id, image_pairs, model, index, index_to_id_map, output_occlusion_dir)
            )
            
    if not tasks:
        print("\n--- Ewaluacja Zakończona ---")
        print("Nie znaleziono żadnych zapytań do przetworzenia.")
        return

    # Otwieramy plik CSV i uruchamiamy pulę wątków
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "top1_id", "top1_similarity", "top2_id", "top2_similarity", "top3_id", "top3_similarity", "is_correct_top1"])
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Używamy executor.map do przetwarzania równoległego
            results = list(tqdm(
                executor.map(process_occlusion_query, tasks), 
                total=len(tasks), 
                desc="Testowanie okluzji (równolegle)"
            ))

        # Zbieranie wyników i zapisywanie do CSV
        for result in results:
            if isinstance(result, list): # Jeśli to poprawny wiersz
                writer.writerow(result)
                is_correct = result[-1] # Ostatni element to True/False
                if is_correct:
                    correct_top1 += 1
                total_queries += 1
            else: # Jeśli to string z błędem
                tqdm.write(str(result))


    if total_queries > 0:
        accuracy = (correct_top1 / total_queries) * 100
        print(f"\n--- Ewaluacja Zakończona ---")
        print(f"Całkowita liczba zapytań: {total_queries}")
        print(f"Poprawne trafienia Top-1: {correct_top1}")
        print(f"Celność Top-1: {accuracy:.2f}%")
    else:
        print("\n--- Ewaluacja Zakończona ---")
        print("Nie przetworzono żadnych zapytań.")

# --- 6. GŁÓWNA FUNKCJA URUCHAMIAJĄCA ---

def main():
    model = initialize_services()
    
    local_test_path = os.path.join(BASE_FOLDER_LOCAL, "test")
    
    identity_to_imgfolders, image_pairs = discover_file_structure(local_test_path)
    if not identity_to_imgfolders:
        print("Zatrzymanie, nie znaleziono plików.")
        return

    # Krok 1: Zbuduj galerię (indeks FAISS)
    print("--- ROZPOCZYNAM KROK 1: Budowanie Galerii FAISS ---")
    if not build_faiss_gallery(model, identity_to_imgfolders, image_pairs):
        print("Zatrzymanie skryptu z powodu błędu budowania galerii.")
        return
    print("\n" + "="*50 + "\n")
    print("--- KROK 1: Zakończony Pomyślnie ---")
    
    
    # Krok 2: Uruchom ewaluację z okluzją
    print("--- ROZPOCZYNAM KROK 2: Ewaluacja Okluzji ---")
    run_occlusion_evaluation(model, identity_to_imgfolders, image_pairs)
    
    print("Gotowe.")

if __name__ == "__main__":
    main()