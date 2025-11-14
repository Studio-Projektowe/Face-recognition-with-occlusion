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
from google.cloud import storage
from config import (
    BUCKET_NAME, BASE_FOLDER_GCS, LOCAL_DATA_DIR, 
    FAISS_INDEX_FILE, FAISS_MAPPING_FILE, RESULTS_CSV, OCCLUSION_SIZE
)

# --- 1. INICJALIZACJA MODELU I GCS ---

def initialize_services():
    """Ładuje model InsightFace i klienta GCS."""
    print("Ładowanie klienta Google Cloud Storage...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
    except Exception as e:
        print(f"BŁĄD: Nie udało się połączyć z GCS. Sprawdź uwierzytelnienie.")
        print(f"Error: {e}")
        sys.exit(1)

    print("Ładowanie modelu InsightFace (ArcFace)... (to może potrwać chwilę)")
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        model = insightface.app.FaceAnalysis(
            name="buffalo_l", 
            root='./insightface_models', 
            providers=providers
        )
        # Używamy (112, 112) dla datasetu WebFace
        model.prepare(ctx_id=0, det_size=(640, 640)) 
    except Exception as e:
        print(f"BŁĄD: Nie udało się załadować modelu InsightFace.")
        print(f"Error: {e}")
        sys.exit(1)
        
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    print("Inicjalizacja zakończona pomyślnie.")
    return model, bucket

def get_embedding(model, image_bgr):
    """Pobiera embedding dla pojedynczego obrazu."""
    try:
        faces = model.get(image_bgr)
        if faces and len(faces) > 0:
            return faces[0].normed_embedding
    except Exception as e:
        # Błąd (18,) (32,) pojawia się, gdy det_size jest źle ustawione
        print(f"Warning: Błąd podczas pobierania embeddingu: {e}")
    return None

def download_blob(blob, destination_folder):
    """Pobiera plik z GCS do folderu tymczasowego."""
    local_path = os.path.join(destination_folder, blob.name.replace("/", "_"))
    try:
        blob.download_to_filename(local_path)
        return local_path
    except Exception as e:
        print(f"Warning: Nie udało się pobrać {blob.name}: {e}")
        return None

# --- 2. NOWA FUNKCJA POMOCNICZA ---

def discover_file_structure(bucket):
    """
    Mapuje "płaską" strukturę plików GCS na logiczne foldery.
    Zwraca dwa słowniki:
    1. identity_to_imgfolders: mapuje ID (np. '.../id_3') na ZBIÓR podfolderów obrazów (np. {.../dog-1, .../dog-2})
    2. image_pairs: mapuje podfolder obrazu (np. '.../dog-1') na {'jpg': blob, 'json': blob}
    """
    print(f"Wykrywanie struktury plików w gs://{BUCKET_NAME}/{BASE_FOLDER_GCS}/...")
    target_prefix = f"{BASE_FOLDER_GCS}/"
    all_blobs = list(bucket.list_blobs(prefix=target_prefix))
    
    if not all_blobs:
        print("BŁĄD: Nie znaleziono ŻADNYCH plików pasujących do prefixu.")
        return None, None
        
    print(f"Znaleziono łącznie {len(all_blobs)} plików pasujących do prefixu.")

    identity_to_imgfolders = {} 
    image_pairs = {}       

    for blob in all_blobs:
        # blob.name to np: 'photos_no_class/test/id_3/dog.../dog....jpg'
        parts = blob.name.split('/')
        if len(parts) < 5: 
            continue
        
        identity_path = "/".join(parts[:3]) # np. 'photos_no_class/test/id_3'
        image_folder_path = "/".join(parts[:4]) # np. '.../id_3/dog-g0973e56d5_640'
        
        if identity_path not in identity_to_imgfolders:
            identity_to_imgfolders[identity_path] = set()
        if image_folder_path not in image_pairs:
            image_pairs[image_folder_path] = {'jpg': None, 'json': None}
        
        if blob.name.endswith(".jpg"):
            identity_to_imgfolders[identity_path].add(image_folder_path)
            image_pairs[image_folder_path]['jpg'] = blob
        elif blob.name.endswith(".json"):
            image_pairs[image_folder_path]['json'] = blob
    
    print(f"Wykryto {len(identity_to_imgfolders)} folderów tożsamości.")
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
        identity_id = id_path.split('/')[-1]
        
        image_folder_paths = sorted(list(identity_to_imgfolders[id_path]))
        
        split_point = max(1, len(image_folder_paths) // 2)
        gallery_folders = image_folder_paths[:split_point]
        
        if not gallery_folders:
            continue

        id_embeddings = []
        for img_folder_path in gallery_folders:
            jpg_blob = image_pairs.get(img_folder_path, {}).get('jpg')
            
            if not jpg_blob:
                tqdm.write(f"Warning: Brak pliku .jpg w {img_folder_path}")
                continue
                
            local_path = download_blob(jpg_blob, LOCAL_DATA_DIR)
            if not local_path: continue

            img = cv2.imread(local_path)
            if img is None:
                tqdm.write(f"Warning: Błąd odczytu obrazu {local_path}")
                os.remove(local_path)
                continue
                
            embedding = get_embedding(model, img)
            if embedding is not None:
                id_embeddings.append(embedding)
            
            os.remove(local_path)

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
        # 1. Znajdź środek wysokości oczu (oś Y)
        # Bierzemy średnią wysokość lewego i prawego oka
        left_eye_y = landmarks_dict["left_eye"][1]
        right_eye_y = landmarks_dict["right_eye"][1]
        eye_y_center = int((left_eye_y + right_eye_y) / 2)
        
        # 2. Użyj OCCLUSION_SIZE jako CAŁKOWITEJ wysokości paska
        bar_height_half = OCCLUSION_SIZE // 2
        
        # 3. Znajdź szerokość twarzy (oś X) z bbox
        # bbox to [x1, y1, x2, y2]
        face_x1 = int(bbox[0])
        face_x2 = int(bbox[2])
        
        # 4. Oblicz współrzędne finalnego paska
        x1 = face_x1
        y1 = max(0, eye_y_center - bar_height_half) # Upewnij się, że nie wyjdziesz poza górę obrazu
        x2 = face_x2
        y2 = min(image.shape[0], eye_y_center + bar_height_half) # Upewnij się, że nie wyjdziesz poza dół
        
        # 5. Narysuj czarny prostokąt
        cv2.rectangle(occluded_image, (x1, y1), (x2, y2), (0, 0, 0), -1) # -1 = wypełniony
    
    except Exception as e:
        # Jeśli brakuje klucza "left_eye" lub coś jest nie tak, logujemy błąd
        print(f"Warning: Błąd podczas nakładania okluzji (np. brak landmarków): {e}")
        # Zwróć oryginał, jeśli coś pójdzie nie tak
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
    
    # === KOD DO TWORZENIA FOLDERU ===
    output_occlusion_dir = "occlusion_photos"
    os.makedirs(output_occlusion_dir, exist_ok=True)
    print(f"Obrazy z okluzją będą zapisywane w: {output_occlusion_dir}")
    
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "top1_id", "top1_similarity", "top2_id", "top2_similarity", "top3_id", "top3_similarity", "is_correct_top1"])
        
        for id_path in tqdm(identity_paths, desc="Testowanie okluzji"):
            ground_truth_id = id_path.split('/')[-1]
            
            image_folder_paths = sorted(list(identity_to_imgfolders[id_path]))
            
            split_point = max(1, len(image_folder_paths) // 2)
            query_folders = image_folder_paths[split_point:] # Bierzemy DRUGĄ połowę

            for img_folder_path in query_folders:
                jpg_blob = image_pairs.get(img_folder_path, {}).get('jpg')
                json_blob = image_pairs.get(img_folder_path, {}).get('json')

                if not jpg_blob or not json_blob:
                    tqdm.write(f"Warning: Brak pary JPG/JSON dla {img_folder_path}")
                    continue
                
                local_img_path = download_blob(jpg_blob, LOCAL_DATA_DIR)
                local_json_path = download_blob(json_blob, LOCAL_DATA_DIR)
                
                if not local_img_path or not local_json_path:
                    continue 

                img = cv2.imread(local_img_path)
                json_data = None
                try:
                    with open(local_json_path, 'r') as jf:
                        json_data = json.load(jf)
                except Exception as e:
                    tqdm.write(f"Warning: Błąd odczytu JSON {local_json_path}: {e}")
                    img = None # Ustawiamy na None, aby sprzątać
                
                # Sprawdzamy, czy mamy wszystko: obraz, landmarki ORAZ bbox
                if (img is None or json_data is None or 
                    "landmarks" not in json_data or "bbox" not in json_data):
                    
                    tqdm.write(f"Warning: Brak pełnych danych (JPG/JSON/Landmarks/BBox) dla {local_img_path}")
                    if local_img_path and os.path.exists(local_img_path): os.remove(local_img_path)
                    if local_json_path and os.path.exists(local_json_path): os.remove(local_json_path)
                    continue

                # 1. Nałóż okluzję
                # Przekazujemy teraz landmarki ORAZ bbox
                occluded_img = apply_occlusion(img, json_data["landmarks"], json_data["bbox"])
                
                # === KOD DO ZAPISU OBRAZU OKLUZJI ===
                try:
                    # Tworzymy unikalną nazwę pliku, np. "occluded_id_3_dog-g097...jpg"
                    original_filename = os.path.basename(local_img_path)
                    save_path = os.path.join(output_occlusion_dir, f"occluded_{ground_truth_id}_{original_filename}")
                    cv2.imwrite(save_path, occluded_img)
                except Exception as e:
                    tqdm.write(f"Warning: Nie udało się zapisać obrazu okluzji {save_path}: {e}")
                
                # 2. Pobierz embedding
                query_embedding = get_embedding(model, occluded_img)
                
                if query_embedding is None:
                    # Sprzątanie przed 'continue'
                    os.remove(local_img_path)
                    os.remove(local_json_path)
                    continue 
                    
                # 3. Przeszukaj FAISS
                
                # Musimy ręcznie znormalizować wektor zapytania, aby FAISS
                # zwrócił poprawne podobieństwo kosinusowe.
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

                # Sprzątanie
                os.remove(local_img_path)
                os.remove(local_json_path)

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
    model, bucket = initialize_services()
    
    # Krok 0: Zmapuj strukturę plików RAZ
    identity_to_imgfolders, image_pairs = discover_file_structure(bucket)
    if not identity_to_imgfolders:
        print("Zatrzymanie, nie znaleziono plików.")
        return

    # Krok 1: Zbuduj galerię (indeks FAISS)
    # Odkomentuj to, jeśli robisz to pierwszy raz
    
    # print("--- ROZPOCZYNAM KROK 1: Budowanie Galerii FAISS ---")
    # if not build_faiss_gallery(model, identity_to_imgfolders, image_pairs):
    #     print("Zatrzymanie skryptu z powodu błędu budowania galerii.")
    #     return
    # print("\n" + "="*50 + "\n")
    # print("--- KROK 1: Zakończony Pomyślnie ---")
    
    
    # Krok 2: Uruchom ewaluację z okluzją
    # Zakładamy, że pliki FAISS już istnieją
    print("--- ROZPOCZYNAM KROK 2: Ewaluacja Okluzji ---")
    run_occlusion_evaluation(model, identity_to_imgfolders, image_pairs)
    
    # Sprzątanie folderu cache
    print("Sprzątanie folderu cache...")
    for f in glob.glob(os.path.join(LOCAL_DATA_DIR, "*")):
        try:
            os.remove(f)
        except OSError:
            pass # Ignoruj błędy, jeśli plik został już usunięty
    os.rmdir(LOCAL_DATA_DIR)
    print("Gotowe.")

if __name__ == "__main__":
    main()