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
# Usunięto import GCS
from config import (
    BASE_FOLDER_LOCAL, # Nowa zmienna
    FAISS_INDEX_FILE, FAISS_MAPPING_FILE, RESULTS_CSV, OCCLUSION_SIZE
)

# --- 1. INICJALIZACJA MODELU ---

def initialize_services():
    """Ładuje model InsightFace."""
    # Usunięto logikę GCS
    print("Ładowanie modelu InsightFace (ArcFace)... (to może potrwać chwilę)")
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        model = insightface.app.FaceAnalysis(
            name="buffalo_l", 
            root='./insightface_models', 
            providers=providers
        )
        # Ustaw (640, 640) dla datasetu testowego lub (112, 112) dla WebFace
        model.prepare(ctx_id=0, det_size=(224,224)) 
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
        print(f"Warning: Błąd podczas pobierania embeddingu: {e}")
    return None

# --- 2. NOWA FUNKCJA POMOCNICZA (WERSJA LOKALNA) ---

def discover_file_structure(local_test_path):
    """
    Mapuje lokalną strukturę plików na logiczne foldery.
    Zwraca te same słowniki co wersja GCS, ale z lokalnymi ścieżkami.
    """
    print(f"Wykrywanie struktury plików w {local_test_path}...")
    
    # Używamy glob do znalezienia wszystkich plików .jpg na odpowiedniej głębokości
    # Wzór: local_test_path / [id_folder] / [img_folder] / [img.jpg]
    search_pattern = os.path.join(local_test_path, "*", "*", "*.jpg")
    all_jpg_files = list(glob.glob(search_pattern))
    
    if not all_jpg_files:
        print("BŁĄD: Nie znaleziono ŻADNYCH plików .jpg pasujących do wzorca.")
        print(f"Szukany wzorzec: {search_pattern}")
        return None, None
        
    print(f"Znaleziono łącznie {len(all_jpg_files)} plików .jpg.")

    identity_to_imgfolders = {} 
    image_pairs = {}       

    for jpg_path in tqdm(all_jpg_files, desc="Skanowanie plików"):
        # Normalizuj ścieżki dla spójności
        jpg_path_norm = os.path.normpath(jpg_path)
        
        # Sprawdź, czy istnieje pasujący plik .json
        base_name = os.path.splitext(jpg_path_norm)[0]
        json_path = base_name + ".json"
        
        if not os.path.exists(json_path):
            #tqdm.write(f"Warning: Brak pliku .json dla {jpg_path_norm}")
            continue # Pomiń, jeśli nie ma pary
        
        # Odtwórz ścieżki
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

# --- 3. BUDOWANIE GALERII FAISS (ZMODYFIKOWANE) ---

def build_faiss_gallery(model, identity_to_imgfolders, image_pairs):
    """
    Tworzy galerię FAISS z pierwszej połowy zdjęć dla każdego ID.
    """
    print(f"--- ROZPOCZYNAM Budowanie Galerii FAISS ---")
    
    identity_paths = list(identity_to_imgfolders.keys())
    
    if not identity_paths:
        print("BŁĄD: Lista folderów tożsamości jest pusta.")
        return False
    
    gallery_embeddings = []
    index_to_id_map = {} 
    faiss_index_counter = 0

    for id_path in tqdm(identity_paths, desc="Tworzenie galerii ID"):
        identity_id = os.path.basename(id_path) # Pobiera 'id_3' ze ścieżki
        
        image_folder_paths = sorted(list(identity_to_imgfolders[id_path]))
        
        split_point = max(1, len(image_folder_paths) // 2)
        gallery_folders = image_folder_paths[:split_point]
        
        if not gallery_folders:
            continue

        id_embeddings = []
        for img_folder_path in gallery_folders:
            # Pobierz lokalną ścieżkę .jpg z naszej mapy
            local_path = image_pairs.get(img_folder_path, {}).get('jpg')
            
            if not local_path:
                tqdm.write(f"Warning: Wewnętrzny błąd mapowania dla {img_folder_path}")
                continue
                
            # Nie pobieramy, tylko czytamy
            img = cv2.imread(local_path)
            if img is None:
                tqdm.write(f"Warning: Błąd odczytu obrazu {local_path}")
                continue
                
            embedding = get_embedding(model, img)
            if embedding is not None:
                id_embeddings.append(embedding)
            
            # Nie usuwamy pliku

        if id_embeddings:
            avg_embedding = np.mean(id_embeddings, axis=0)
            avg_embedding /= np.linalg.norm(avg_embedding) 
            
            gallery_embeddings.append(avg_embedding)
            index_to_id_map[faiss_index_counter] = identity_id
            faiss_index_counter += 1

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

# --- 4. TESTOWANIE Z OKLUZJĄ (ZMODYFIKOWANE) ---

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
        print(f"Warning: Błąd podczas nakładania okluzji (np. brak landmarków): {e}")
        return image.copy() 
    return occluded_image

def run_occlusion_evaluation(model, identity_to_imgfolders, image_pairs):
    """
    Testuje drugą połowę zdjęć z okluzją i zapisuje wyniki do CSV.
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

    print(f"Rozpoczynanie ewaluacji z okluzją. Wyniki w {RESULTS_CSV}...")
    
    identity_paths = list(identity_to_imgfolders.keys())
    
    total_queries = 0
    correct_top1 = 0
    
    output_occlusion_dir = "occlusion_photos"
    os.makedirs(output_occlusion_dir, exist_ok=True)
    print(f"Obrazy z okluzją będą zapisywane w: {output_occlusion_dir}")
    
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "top1_id", "top1_similarity", "top2_id", "top2_similarity", "top3_id", "top3_similarity", "is_correct_top1"])
        
        for id_path in tqdm(identity_paths, desc="Testowanie okluzji"):
            ground_truth_id = os.path.basename(id_path)
            
            image_folder_paths = sorted(list(identity_to_imgfolders[id_path]))
            
            split_point = max(1, len(image_folder_paths) // 2)
            query_folders = image_folder_paths[split_point:] # Bierzemy DRUGĄ połowę

            for img_folder_path in query_folders:
                # Pobierz lokalne ścieżki
                local_img_path = image_pairs.get(img_folder_path, {}).get('jpg')
                local_json_path = image_pairs.get(img_folder_path, {}).get('json')

                if not local_img_path or not local_json_path:
                    tqdm.write(f"Warning: Wewnętrzny błąd mapowania dla {img_folder_path}")
                    continue
                
                # Nie pobieramy, tylko czytamy
                img = cv2.imread(local_img_path)
                json_data = None
                try:
                    with open(local_json_path, 'r') as jf:
                        json_data = json.load(jf)
                except Exception as e:
                    tqdm.write(f"Warning: Błąd odczytu JSON {local_json_path}: {e}")
                    img = None 
                
                if (img is None or json_data is None or 
                    "landmarks" not in json_data or "bbox" not in json_data):
                    tqdm.write(f"Warning: Brak pełnych danych (JPG/JSON/Landmarks/BBox) dla {local_img_path}")
                    continue

                # 1. Nałóż okluzję
                occluded_img = apply_occlusion(img, json_data["landmarks"], json_data["bbox"])
                
                try:
                    original_filename = os.path.basename(local_img_path)
                    save_path = os.path.join(output_occlusion_dir, f"occluded_{ground_truth_id}_{original_filename}")
                    cv2.imwrite(save_path, occluded_img)
                except Exception as e:
                    tqdm.write(f"Warning: Nie udało się zapisać obrazu okluzji {save_path}: {e}")
                
                # 2. Pobierz embedding
                query_embedding = get_embedding(model, occluded_img)
                
                if query_embedding is None:
                    continue 
                    
                # 3. Przeszukaj FAISS
                query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
                query_vector = np.expand_dims(query_embedding_normalized, axis=0).astype('float32')
                
                D, I = index.search(query_vector, 3) # Szukaj Top 3
                
                top1_idx = I[0][0]
                top2_idx = I[0][1]
                top3_idx = I[0][2]
                
                top1_sim = D[0][0]
                top2_sim = D[0][1]
                top3_sim = D[0][2]
                
                top1_id = index_to_id_map.get(str(top1_idx), "N/A")
                top2_id = index_to_id_map.get(str(top2_idx), "N/A")
                top3_id = index_to_id_map.get(str(top3_idx), "N/A")
                
                # 4. Zapisz wyniki
                is_correct = (top1_id == ground_truth_id)
                writer.writerow([ground_truth_id, top1_id, f"{top1_sim:.4f}", top2_id, f"{top2_sim:.4f}", top3_id, f"{top3_sim:.4f}", is_correct])
                
                if is_correct:
                    correct_top1 += 1
                total_queries += 1

                # Nie ma potrzeby sprzątania plików

    if total_queries > 0:
        accuracy = (correct_top1 / total_queries) * 100
        print(f"\n--- Ewaluacja Zakończona ---")
        print(f"Całkowita liczba zapytań: {total_queries}")
        print(f"Poprawne trafienia Top-1: {correct_top1}")
        print(f"Celność Top-1: {accuracy:.2f}%")
    else:
        print("\n--- Ewaluacja Zakończona ---")
        print("Nie przetworzono żadnych zapytań.")

# --- 5. GŁÓWNA FUNKCJA URUCHAMIAJĄCA (ZMODYFIKOWANA) ---

def main():
    model = initialize_services()
    
    # Krok 0: Zmapuj strukturę plików RAZ
    # Składamy ścieżkę do folderu 'test' wewnątrz folderu bazowego
    local_test_path = os.path.join(BASE_FOLDER_LOCAL, "test")
    
    identity_to_imgfolders, image_pairs = discover_file_structure(local_test_path)
    if not identity_to_imgfolders:
        print("Zatrzymanie, nie znaleziono plików.")
        return

    # Krok 1: Zbuduj galerię (indeks FAISS)
    # Odkomentuj to, jeśli robisz to pierwszy raz
    
    print("--- ROZPOCZYNAM KROK 1: Budowanie Galerii FAISS ---")
    if not build_faiss_gallery(model, identity_to_imgfolders, image_pairs):
        print("Zatrzymanie skryptu z powodu błędu budowania galerii.")
        return
    print("\n" + "="*50 + "\n")
    print("--- KROK 1: Zakończony Pomyślnie ---")
    
    
    # Krok 2: Uruchom ewaluację z okluzją
    # Zakładamy, że pliki FAISS już istnieją
    print("--- ROZPOCZYNAM KROK 2: Ewaluacja Okluzji ---")
    run_occlusion_evaluation(model, identity_to_imgfolders, image_pairs)
    
    # Usunięto sprzątanie folderu cache
    print("Gotowe.")

if __name__ == "__main__":
    main()