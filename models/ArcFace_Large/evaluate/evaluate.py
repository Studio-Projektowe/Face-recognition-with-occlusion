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
import insightface
from insightface.app import FaceAnalysis

BASE_DIR = '../../../../webface_112x112' 
TEST_DIR = os.path.join(BASE_DIR, 'test') 

FAISS_INDEX_FILE = "gallery_buffalo.index"
FAISS_MAPPING_FILE = "gallery_buffalo_map.json"
RESULTS_CSV = "results_buffalo_occlusion.csv"
OUTPUT_OCCLUSION_DIR = "evaluation_photos_buffalo"

OCCLUSION_SIZE = 20
PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

def initialize_insightface():
    print("Ładowanie oryginalnego modelu InsightFace (buffalo_l)...")
    
    app = FaceAnalysis(name='buffalo_l', providers=PROVIDERS)
    
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    rec_model = app.models.get('recognition')
    
    if rec_model is None:
        print("Błąd: Nie znaleziono modelu 'recognition' w pakiecie buffalo_l.")
        sys.exit(1)
        
    print(f"Załadowano model: {rec_model.input_shape} -> {rec_model.output_shape}")
    return rec_model

def preprocess_face_from_bbox(img, bbox, output_size=112, pad_ratio=0.2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
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

def image_to_embedding(rec_model, img_bgr, bbox=None):
    try:
        if bbox is not None:
            face = preprocess_face_from_bbox(img_bgr, bbox, output_size=112)
        else:
            face = cv2.resize(img_bgr, (112, 112))

        # Biblioteka InsightFace (handler ONNX) oczekuje czystego obrazu BGR (numpy).
        # Sama robi normalizację (div 127.5) i transpozycję (HWC->CHW).
        # Metoda get_feat zwraca od razu embedding.
        
        emb = rec_model.get_feat(face)
        
        if isinstance(emb, list):
            emb = emb[0]
            
        emb = emb.flatten()
        
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
            
        return emb.astype('float32')
            
    except Exception as e:
        print(f"Warning: Błąd embeddingu: {e}")
        return None

def discover_file_structure(local_root):
    print(f"Skanowanie folderu: {local_root}...")
    search_pattern = os.path.join(local_root, "*", "*", "*.jpg")
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        print(f"Nie znaleziono plików JPG w {local_root}")
        return None, None

    identity_to_folders = {}
    image_pairs = {}

    for jpg_path in tqdm(all_files, desc="Indeksowanie plików"):
        jpg_path = os.path.normpath(jpg_path)
        img_dir = os.path.dirname(jpg_path)
        id_dir = os.path.dirname(img_dir)
        identity_id = os.path.basename(id_dir)
        
        json_path = jpg_path.replace('.jpg', '.json')
        
        if identity_id not in identity_to_folders:
            identity_to_folders[identity_id] = set()
        
        identity_to_folders[identity_id].add(img_dir)
        
        if img_dir not in image_pairs:
            image_pairs[img_dir] = []
            
        image_pairs[img_dir].append({'jpg': jpg_path, 'json': json_path})

    print(f"Znaleziono {len(all_files)} zdjęć w {len(identity_to_folders)} tożsamościach.")
    return identity_to_folders, image_pairs

def build_gallery(rec_model, identity_to_folders, image_pairs):
    print("\n--- KROK 1: Budowanie Galerii FAISS (Buffalo_L) ---")
    
    gallery_embeddings = []
    index_to_id_map = {}
    idx_counter = 0
    
    for identity_id in tqdm(identity_to_folders.keys(), desc="Przetwarzanie ID"):
        folders = sorted(list(identity_to_folders[identity_id]))
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
                
                emb = image_to_embedding(rec_model, img, bbox)
                if emb is not None:
                    id_vectors.append(emb)
        
        if id_vectors:
            avg_emb = np.mean(np.stack(id_vectors), axis=0)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-10)
            
            gallery_embeddings.append(avg_emb)
            index_to_id_map[str(idx_counter)] = identity_id
            idx_counter += 1
            
    if not gallery_embeddings:
        print("Błąd: Pusta galeria.")
        return False
        
    matrix = np.array(gallery_embeddings).astype('float32')
    dim = matrix.shape[1]
    
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_MAPPING_FILE, 'w') as f:
        json.dump(index_to_id_map, f)
        
    print(f"Zbudowano galerię dla {idx_counter} osób.")
    return True

def apply_occlusion(image, landmarks_dict, bbox):
    occ_img = image.copy()
    try:
        leye = landmarks_dict.get('left_eye') or landmarks_dict.get('left')
        reye = landmarks_dict.get('right_eye') or landmarks_dict.get('right')
        
        if not leye or not reye: return image
        
        cy = int((leye[1] + reye[1]) / 2)
        h_half = OCCLUSION_SIZE // 2
        
        x1, _, x2, _ = [int(v) for v in bbox]
        
        y_start = max(0, cy - h_half)
        y_end = min(image.shape[0], cy + h_half)
        
        cv2.rectangle(occ_img, (x1, y_start), (x2, y_end), (0, 0, 0), -1)
        
    except Exception:
        return image
    return occ_img

def run_evaluation(rec_model, identity_to_folders, image_pairs):
    print("\n--- KROK 2: Ewaluacja Buffalo_L (Zdjęcia z Okluzją) ---")
    
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(FAISS_MAPPING_FILE, 'r') as f:
        idx_map = json.load(f)
        
    os.makedirs(OUTPUT_OCCLUSION_DIR, exist_ok=True)
    
    csv_file = open(RESULTS_CSV, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["query_id", "pred_id", "similarity", "correct"])
    
    correct = 0
    total = 0
    
    for identity_id in tqdm(identity_to_folders.keys(), desc="Testowanie"):
        folders = sorted(list(identity_to_folders[identity_id]))
        split_point = max(1, len(folders) // 2)
        query_folders = folders[split_point:]
        
        for f_path in query_folders:
            items = image_pairs.get(f_path, [])
            for item in items:
                img = cv2.imread(item['jpg'])
                if img is None: continue
                
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
                    img_occ = apply_occlusion(img, landmarks, bbox)
                    
                    if random.random() < 0.05:
                        fname = os.path.basename(item['jpg'])
                        cv2.imwrite(os.path.join(OUTPUT_OCCLUSION_DIR, f"occ_{fname}"), img_occ)
                    
                    emb = image_to_embedding(rec_model, img_occ, bbox)
                    
                    if emb is not None:
                        q_vec = np.expand_dims(emb, axis=0)
                        dists, idxs = index.search(q_vec, 1)
                        
                        pred_idx = str(idxs[0][0])
                        pred_id = idx_map.get(pred_idx, "Unknown")
                        score = dists[0][0]
                        
                        is_ok = (pred_id == identity_id)
                        if is_ok: correct += 1
                        total += 1
                        
                        writer.writerow([identity_id, pred_id, f"{score:.4f}", is_ok])

    csv_file.close()
    
    if total > 0:
        acc = (correct / total) * 100
        print(f"\nWYNIKI KOŃCOWE (Buffalo_L Original):")
        print(f"   Przetworzono zapytań: {total}")
        print(f"   Poprawne rozpoznania: {correct}")
        print(f"   ACCURACY (Top-1):     {acc:.2f}%")
        print(f"   Szczegóły w pliku:    {RESULTS_CSV}")
    else:
        print("Nie przetworzono żadnych zdjęć.")

def main():
    model = initialize_insightface()
    
    id_map, img_pairs = discover_file_structure(TEST_DIR)
    if not id_map: return
    
    if build_gallery(model, id_map, img_pairs):
        run_evaluation(model, id_map, img_pairs)

if __name__ == "__main__":
    main()
