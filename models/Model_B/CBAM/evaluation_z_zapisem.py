# -*- coding: utf-8 -*-

import os
import sys
import glob
import json
import csv
import random
import numpy as np
import cv2
import faiss
from tqdm import tqdm
import torch

# Importujemy architekturę (upewnij się, że to wersja z CBAM!)
from backbone import iresnet50

# --- KONFIGURACJA ---
BASE_DIR = '../../../../webface_112x112'
TEST_DIR = os.path.join(BASE_DIR, 'test') # Ewaluacja na zbiorze testowym

# Ścieżka do modelu CBAM (wynik ostatniego treningu)
MODEL_PATH = 'best_model_cbam_occlusion.pth'

# Pliki wyjściowe
FAISS_INDEX_FILE = "gallery_cbam_3.index"
FAISS_MAPPING_FILE = "gallery_cbam_map_3.json"
RESULTS_CSV = "results_cbam_occlusion_top3.csv"
OUTPUT_OCCLUSION_DIR = "evaluation_photos_cbam"

OCCLUSION_SIZE = 20 # Dopasowane do treningu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 1. Inicjalizacja Modelu
# ---------------------------
def initialize_model():
    print("Ładowanie modelu IResNet50...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"BŁĄD: Nie znaleziono pliku wag: {MODEL_PATH}")
        sys.exit(1)

    try:
        print("Inicjalizacja")
        # Inicjalizacja pustego modelu (architektura z pliku backbone_iresnet.py)
        model = iresnet50(weights_path=None) 
        print("ładowanie wag")
        # Ładowanie wag
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        print("Załadowano")
        # Obsługa formatu state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
            
        model.load_state_dict(checkpoint)
        print(f" Pomyślnie załadowano wagi: {MODEL_PATH}")

        model.to(DEVICE)
        model.eval() # Tryb ewaluacji (wyłącza Dropout/Batch Norm update)
        return model

    except Exception as e:
        print(f"BŁĄD krytyczny modelu: {e}")
        sys.exit(1)

# ---------------------------
# 2. Funkcje Przetwarzania Obrazu
# ---------------------------
def preprocess_face_from_bbox(img, bbox, output_size=112, pad_ratio=0.2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    # Margines
    bw = x2 - x1
    bh = y2 - y1
    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    x1c = max(0, x1 - pad_w)
    y1c = max(0, y1 - pad_h)
    x2c = min(w, x2 + pad_w)
    y2c = min(h, y2 + pad_h)

    crop = img[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return cv2.resize(img, (output_size, output_size))

    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

def image_to_embedding(model, img_bgr, bbox=None):
    try:
        if bbox is not None:
            face = preprocess_face_from_bbox(img_bgr, bbox, output_size=112)
        else:
            face = cv2.resize(img_bgr, (112, 112))

        # Preprocessing zgodny z treningiem:
        # transforms.ToTensor() -> [0,1], Normalize(0.5, 0.5) -> (x - 0.5)/0.5
        # To matematycznie to samo co (x - 127.5) / 127.5 na pikselach [0, 255]
        face = face.astype(np.float32)
        face = (face - 127.5) / 127.5
        
        # Konwersja BGR -> RGB (ważne, bo trening był na RGB!)
        face = face[:, :, ::-1] 
        
        # HWC -> CHW
        face = np.transpose(face, (2, 0, 1))
        tensor = torch.from_numpy(face.copy()).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = model(tensor)
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            emb = emb.cpu().numpy().reshape(-1)
            # Normalizacja L2 wektora
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            return emb.astype('float32')
            
    except Exception as e:
        print(f"Warning: Błąd embeddingu: {e}")
        return None

# ---------------------------
# 3. Odkrywanie Plików Lokalnych
# ---------------------------
def discover_file_structure(local_root):
    print(f"Skanowanie folderu: {local_root}...")
    
    # Wzorzec: root/id_xxx/subfolder/img.jpg
    search_pattern = os.path.join(local_root, "*", "*", "*.jpg")
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        print(f"Nie znaleziono plików JPG w {local_root}")
        return None, None

    identity_to_folders = {}
    image_pairs = {}

    for jpg_path in tqdm(all_files, desc="Indeksowanie plików"):
        jpg_path = os.path.normpath(jpg_path)
        
        # Struktura: .../test / id_001 / 001 / image.jpg
        # Wyciągamy 'id_001' (parent parent) i folder obrazu (parent)
        img_dir = os.path.dirname(jpg_path)
        id_dir = os.path.dirname(img_dir)
        identity_id = os.path.basename(id_dir)
        
        # Szukamy JSON
        json_path = jpg_path.replace('.jpg', '.json')
        
        if identity_id not in identity_to_folders:
            identity_to_folders[identity_id] = set()
        
        identity_to_folders[identity_id].add(img_dir)
        
        if img_dir not in image_pairs:
            image_pairs[img_dir] = []
            
        image_pairs[img_dir].append({'jpg': jpg_path, 'json': json_path})

    print(f"Znaleziono {len(all_files)} zdjęć w {len(identity_to_folders)} tożsamościach.")
    return identity_to_folders, image_pairs

# ---------------------------
# 4. Budowanie Galerii (FAISS)
# ---------------------------
def build_gallery(model, identity_to_folders, image_pairs):
    print("\n--- KROK 1: Budowanie Galerii FAISS (Czyste zdjęcia) ---")
    
    gallery_embeddings = []
    index_to_id_map = {}
    idx_counter = 0
    
    for identity_id in tqdm(identity_to_folders.keys(), desc="Przetwarzanie ID"):
        folders = sorted(list(identity_to_folders[identity_id]))
        
        # Bierzemy PIERWSZĄ POŁOWĘ folderów jako galerię (wzorzec)
        split_point = max(1, len(folders) // 2)
        gallery_folders = folders[:split_point]
        
        id_vectors = []
        
        for f_path in gallery_folders:
            items = image_pairs.get(f_path, [])
            for item in items:
                img = cv2.imread(item['jpg'])
                if img is None: continue
                
                bbox = None
                if os.path.exists(item['json']):
                    try:
                        with open(item['json'], 'r') as jf:
                            data = json.load(jf)
                            bbox = data.get('bbox')
                    except: pass
                
                emb = image_to_embedding(model, img, bbox)
                if emb is not None:
                    id_vectors.append(emb)
        
        # Uśredniamy wektory dla jednej osoby (tworzymy prototyp)
        if id_vectors:
            avg_emb = np.mean(np.stack(id_vectors), axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb) # Renormalizacja
            
            gallery_embeddings.append(avg_emb)
            index_to_id_map[str(idx_counter)] = identity_id
            idx_counter += 1
            
    # Tworzenie indeksu FAISS
    if not gallery_embeddings:
        print("Błąd: Pusta galeria.")
        return False
        
    matrix = np.array(gallery_embeddings).astype('float32')
    dim = matrix.shape[1]
    
    index = faiss.IndexFlatIP(dim) # Inner Product (Cosine Similarity dla znormalizowanych)
    index.add(matrix)
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_MAPPING_FILE, 'w') as f:
        json.dump(index_to_id_map, f)
        
    print(f"Zbudowano galerię dla {idx_counter} osób.")
    return True

# ---------------------------
# 5. Funkcja Nakładania Okluzji
# ---------------------------
def apply_occlusion(image, landmarks_dict, bbox):
    occ_img = image.copy()
    try:
        # Obsługa różnych formatów kluczy w JSON
        leye = landmarks_dict.get('left_eye') or landmarks_dict.get('left')
        reye = landmarks_dict.get('right_eye') or landmarks_dict.get('right')
        
        if not leye or not reye: return image
        
        cy = int((leye[1] + reye[1]) / 2)
        h_half = OCCLUSION_SIZE // 2
        
        x1, _, x2, _ = [int(v) for v in bbox]
        
        y_start = max(0, cy - h_half)
        y_end = min(image.shape[0], cy + h_half)
        
        # Czarny pasek (symulacja okularów VR/opaski)
        cv2.rectangle(occ_img, (x1, y_start), (x2, y_end), (0, 0, 0), -1)
        
    except Exception:
        return image
    return occ_img

# ---------------------------
# 6. Ewaluacja (Query z Okluzją) - ZMODYFIKOWANA
# ---------------------------
def run_evaluation(model, identity_to_folders, image_pairs):
    print("\n--- KROK 2: Ewaluacja (Zdjęcia z Okluzją) ---")
    
    # Ładowanie FAISS
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(FAISS_MAPPING_FILE, 'r') as f:
            idx_map = json.load(f)
    except Exception as e:
        print(f"Nie znaleziono plików FAISS. Uruchom budowanie galerii. Błąd: {e}")
        return
        
    os.makedirs(OUTPUT_OCCLUSION_DIR, exist_ok=True)
    
    csv_file = open(RESULTS_CSV, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    # ZAKTUALIZOWANY NAGŁÓWEK CSV
    writer.writerow(["query_id", "top1_id", "top1_similarity", "top2_id", "top2_similarity", "top3_id", "top3_similarity", "is_correct_top1"])
    
    correct = 0
    total = 0
    
    for identity_id in tqdm(identity_to_folders.keys(), desc="Testowanie"):
        folders = sorted(list(identity_to_folders[identity_id]))
        
        # DRUGA POŁOWA folderów to zestaw testowy (Query)
        split_point = max(1, len(folders) // 2)
        query_folders = folders[split_point:]
        
        for f_path in query_folders:
            items = image_pairs.get(f_path, [])
            for item in items:
                img = cv2.imread(item['jpg'])
                if img is None: continue
                
                # Wczytaj metadane
                landmarks = None
                bbox = None
                if os.path.exists(item['json']):
                    try:
                        with open(item['json'], 'r') as jf:
                            jd = json.load(jf)
                            landmarks = jd.get('landmarks')
                            bbox = jd.get('bbox')
                    except: pass
                
                if landmarks and bbox:
                    # 1. Nałóż okluzję
                    img_occ = apply_occlusion(img, landmarks, bbox)
                    
                    # 2. Zapisz próbkę (dla weryfikacji wizualnej)
                    if random.random() < 0.05: # Zapisz 5% zdjęć
                        fname = os.path.basename(item['jpg'])
                        cv2.imwrite(os.path.join(OUTPUT_OCCLUSION_DIR, f"occ_{fname}"), img_occ)
                    
                    # 3. Zrób embedding
                    emb = image_to_embedding(model, img_occ, bbox)
                    
                    if emb is not None:
                        # 4. Szukaj w FAISS (Szukamy 3 sąsiadów)
                        q_vec = np.expand_dims(emb, axis=0)
                        
                        # Pobierz 3 najlepsze wyniki
                        # Zabezpieczenie: jeśli w bazie jest mniej niż 3 osoby, szukamy k=liczba_osób
                        k_neighbors = min(3, index.ntotal)
                        dists, idxs = index.search(q_vec, k_neighbors)
                        
                        # Top 1
                        top1_idx = str(idxs[0][0])
                        top1_id = idx_map.get(top1_idx, "Unknown")
                        top1_sim = dists[0][0]
                        
                        # Top 2 (jeśli istnieje)
                        top2_id = "N/A"
                        top2_sim = 0.0
                        if k_neighbors >= 2:
                            top2_idx = str(idxs[0][1])
                            top2_id = idx_map.get(top2_idx, "Unknown")
                            top2_sim = dists[0][1]
                            
                        # Top 3 (jeśli istnieje)
                        top3_id = "N/A"
                        top3_sim = 0.0
                        if k_neighbors >= 3:
                            top3_idx = str(idxs[0][2])
                            top3_id = idx_map.get(top3_idx, "Unknown")
                            top3_sim = dists[0][2]
                        
                        # Weryfikacja
                        is_ok = (top1_id == identity_id)
                        if is_ok: correct += 1
                        total += 1
                        
                        # Zapis do CSV
                        writer.writerow([identity_id, top1_id, f"{top1_sim:.4f}", top2_id, f"{top2_sim:.4f}", top3_id, f"{top3_sim:.4f}", is_ok])

    csv_file.close()
    
    if total > 0:
        acc = (correct / total) * 100
        print(f"\n WYNIKI KOŃCOWE:")
        print(f"   Przetworzono zapytań: {total}")
        print(f"   Poprawne rozpoznania: {correct}")
        print(f"   ACCURACY (Top-1):     {acc:.2f}%")
        print(f"   Szczegóły w pliku:    {RESULTS_CSV}")
    else:
        print(" Nie przetworzono żadnych zdjęć.")

# ---------------------------
# MAIN
# ---------------------------
def main():
    # 1. Model
    model = initialize_model()
    
    # 2. Pliki
    id_map, img_pairs = discover_file_structure(TEST_DIR)
    if not id_map: return
    
    # 3. Galeria (Indeksowanie czystych twarzy)
    if build_gallery(model, id_map, img_pairs):
        # 4. Test (Szukanie twarzy z okluzją)
        run_evaluation(model, id_map, img_pairs)

if __name__ == "__main__":
    main()