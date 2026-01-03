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
import threading

import tensorflow as tf
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

from concurrent.futures import ThreadPoolExecutor, as_completed 
from config import (
    BASE_FOLDER_LOCAL, 
    FAISS_INDEX_FILE, FAISS_MAPPING_FILE, RESULTS_CSV, OCCLUSION_SIZE,
    NUM_WORKERS
)

def initialize_services():
    """Ładuje model VGGFace i tworzy blokadę GPU."""
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Znaleziono i skonfigurowano {len(gpus)} kart GPU.")
        except RuntimeError as e:
            print(e)
    else:
        print("OSTRZEŻENIE: Nie znaleziono GPU. Skrypt będzie działał wolno na CPU.")

    print("Ładowanie modelu VGGFace (RESNET-50)...")
    try:
        vgg_model = VGGFace(model='resnet50', 
                            include_top=False, 
                            input_shape=(224, 224, 3), 
                            pooling='avg')
    except Exception as e:
        print(f"BŁĄD: Nie udało się załadować modelu VGGFace: {e}")
        sys.exit(1)
        
    gpu_lock = threading.Lock()
    print("Inicjalizacja zakończona pomyślnie.")
    return vgg_model, gpu_lock

def preprocess_face_from_bbox(img, bbox, output_size=224, pad_ratio=0.2):
    """
    Wycina twarz na podstawie bbox [x1, y1, x2, y2] z marginesem i skaluje.
    VGGFace wymaga 224x224.
    """
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

def get_embedding(face_bgr, vgg_model, gpu_lock):
    """
    Pobiera embedding dla wyciętej twarzy (224x224).
    """
    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        
        face = np.expand_dims(face_rgb, axis=0)
        face = face.astype('float32') 
        
        face = preprocess_input(face, version=2) 
        
        with gpu_lock:
            embedding = vgg_model.predict(face, verbose=0)
            
        return embedding.flatten()
        
    except Exception as e:
        return None

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

    print(f"Wykryto {len(identity_to_imgfolders)} folderów tożsamości z parami JPG/JSON.")
    return identity_to_imgfolders, image_pairs

def process_identity_for_gallery(args):
    id_path, identity_to_imgfolders, image_pairs, vgg_model, gpu_lock = args
    
    identity_id = os.path.basename(id_path)
    image_folder_paths = sorted(list(identity_to_imgfolders[id_path]))
    
    split_point = max(1, len(image_folder_paths) // 2)
    gallery_folders = image_folder_paths[:split_point]
    
    if not gallery_folders:
        return (identity_id, None)

    id_embeddings = []
    for img_folder_path in gallery_folders:
        local_img = image_pairs.get(img_folder_path, {}).get('jpg')
        local_json = image_pairs.get(img_folder_path, {}).get('json')
        
        if not local_img or not local_json: continue
            
        img = cv2.imread(local_img)
        if img is None: continue
        
        bbox = None
        try:
            with open(local_json, 'r') as f:
                data = json.load(f)
                bbox = data.get('bbox')
        except: pass
        
        if bbox:
            face_img = preprocess_face_from_bbox(img, bbox, output_size=224)
        else:
            face_img = cv2.resize(img, (224, 224))
            
        embedding = get_embedding(face_img, vgg_model, gpu_lock)
        if embedding is not None:
            id_embeddings.append(embedding)

    if id_embeddings:
        avg_embedding = np.mean(id_embeddings, axis=0)
        avg_embedding /= np.linalg.norm(avg_embedding)
        return (identity_id, avg_embedding)
        
    return (identity_id, None)

def build_faiss_gallery(vgg_model, gpu_lock, identity_to_imgfolders, image_pairs):
    print(f"--- ROZPOCZYNAM Budowanie Galerii FAISS (równolegle z {NUM_WORKERS} workerami) ---")
    
    identity_paths = list(identity_to_imgfolders.keys())
    if not identity_paths: return False
    
    gallery_embeddings = []
    index_to_id_map = {}
    faiss_index_counter = 0

    tasks = [(id_path, identity_to_imgfolders, image_pairs, vgg_model, gpu_lock) for id_path in identity_paths]

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(
            executor.map(process_identity_for_gallery, tasks), 
            total=len(tasks), 
            desc="Tworzenie galerii"
        ))

    for identity_id, avg_embedding in results:
        if avg_embedding is not None:
            gallery_embeddings.append(avg_embedding)
            index_to_id_map[faiss_index_counter] = identity_id
            faiss_index_counter += 1

    if not gallery_embeddings:
        print("BŁĄD: Galeria jest pusta.")
        return False
        
    dimension = gallery_embeddings[0].shape[0] 
    print(f"Wymiar embeddingu: {dimension}")
    
    gallery_matrix = np.array(gallery_embeddings).astype('float32')
    index = faiss.IndexFlatIP(dimension)
    index.add(gallery_matrix)
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_MAPPING_FILE, 'w') as f:
        json.dump(index_to_id_map, f)
        
    return True

def apply_occlusion(image, landmarks_dict, bbox):
    """Nakłada pasek okluzji na pełne zdjęcie."""
    occluded_image = image.copy()
    try:
        left_eye_y = landmarks_dict["left_eye"][1]
        right_eye_y = landmarks_dict["right_eye"][1]
        eye_y_center = int((left_eye_y + right_eye_y) / 2)
        bar_height_half = OCCLUSION_SIZE // 2
        
        face_x1 = int(bbox[0])
        face_x2 = int(bbox[2])
        
        x1 = max(0, face_x1)
        y1 = max(0, eye_y_center - bar_height_half)
        x2 = min(image.shape[1], face_x2)
        y2 = min(image.shape[0], eye_y_center + bar_height_half)
        
        cv2.rectangle(occluded_image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    except Exception:
        return image.copy() 
    return occluded_image

def process_occlusion_query(args):
    img_folder_path, ground_truth_id, image_pairs, vgg_model, gpu_lock, index, index_to_id_map, output_occlusion_dir = args

    local_img_path = image_pairs.get(img_folder_path, {}).get('jpg')
    local_json_path = image_pairs.get(img_folder_path, {}).get('json')

    if not local_img_path or not local_json_path: return None

    img = cv2.imread(local_img_path)
    json_data = None
    try:
        with open(local_json_path, 'r') as jf:
            json_data = json.load(jf)
    except: return None
    
    if (img is None or json_data is None or 
        "landmarks" not in json_data or "bbox" not in json_data):
        return None
    occluded_full_img = apply_occlusion(img, json_data["landmarks"], json_data["bbox"])
    
    try:
        if random.random() < 0.01:
            original_filename = os.path.basename(local_img_path)
            save_path = os.path.join(output_occlusion_dir, f"occ_{ground_truth_id}_{original_filename}")
            cv2.imwrite(save_path, occluded_full_img)
    except: pass
    
    bbox = json_data["bbox"]
    face_img = preprocess_face_from_bbox(occluded_full_img, bbox, output_size=224)
    
    query_embedding = get_embedding(face_img, vgg_model, gpu_lock)
    
    if query_embedding is None: return None
        
    query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
    query_vector = np.expand_dims(query_embedding_normalized, axis=0).astype('float32')
    
    D, I = index.search(query_vector, 3)
    
    top1_idx, top2_idx, top3_idx = I[0]
    top1_sim, top2_sim, top3_sim = D[0]
    
    top1_id = index_to_id_map.get(str(top1_idx), "N/A")
    top2_id = index_to_id_map.get(str(top2_idx), "N/A")
    top3_id = index_to_id_map.get(str(top3_idx), "N/A")
    
    is_correct = (top1_id == ground_truth_id)
    
    return [ground_truth_id, top1_id, f"{top1_sim:.4f}", top2_id, f"{top2_sim:.4f}", top3_id, f"{top3_sim:.4f}", is_correct]

def run_occlusion_evaluation(vgg_model, gpu_lock, identity_to_imgfolders, image_pairs):
    print("\n--- KROK 2: Ewaluacja z Okluzją ---")
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(FAISS_MAPPING_FILE, 'r') as f:
            index_to_id_map = json.load(f)
    except Exception as e:
        print(f"BŁĄD ładowania FAISS: {e}")
        return

    identity_paths = list(identity_to_imgfolders.keys())
    output_occlusion_dir = "occlusion_photos"
    os.makedirs(output_occlusion_dir, exist_ok=True)
    
    tasks = []
    for id_path in identity_paths:
        ground_truth_id = os.path.basename(id_path)
        image_folder_paths = sorted(list(identity_to_imgfolders[id_path]))
        
        split_point = max(1, len(image_folder_paths) // 2)
        query_folders = image_folder_paths[split_point:] 

        for img_folder_path in query_folders:
            tasks.append(
                (img_folder_path, ground_truth_id, image_pairs, vgg_model, gpu_lock, index, index_to_id_map, output_occlusion_dir)
            )
            
    if not tasks:
        print("Brak zapytań.")
        return

    total_queries = 0
    correct_top1 = 0
    
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "top1_id", "top1_similarity", "top2_id", "top2_similarity", "top3_id", "top3_similarity", "is_correct_top1"])
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(tqdm(
                executor.map(process_occlusion_query, tasks), 
                total=len(tasks), 
                desc="Testowanie"
            ))

        for result in results:
            if result:
                writer.writerow(result)
                if result[-1]: correct_top1 += 1
                total_queries += 1

    if total_queries > 0:
        accuracy = (correct_top1 / total_queries) * 100
        print(f"\nWYNIKI KOŃCOWE (VGGFace):")
        print(f"  Zapytań: {total_queries}")
        print(f"  Accuracy Top-1: {accuracy:.2f}%")
        print(f"  Plik: {RESULTS_CSV}")
    else:
        print("Brak wyników.")


def main():
    vgg_model, gpu_lock = initialize_services()
    
    local_test_path = os.path.join(BASE_FOLDER_LOCAL, "test")
    identity_to_imgfolders, image_pairs = discover_file_structure(local_test_path)
    
    if not identity_to_imgfolders: return

    if build_faiss_gallery(vgg_model, gpu_lock, identity_to_imgfolders, image_pairs):
        run_occlusion_evaluation(vgg_model, gpu_lock, identity_to_imgfolders, image_pairs)
    
    print("Gotowe.")

if __name__ == "__main__":
    main()