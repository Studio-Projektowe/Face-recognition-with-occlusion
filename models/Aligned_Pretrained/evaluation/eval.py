import os
import sys
import glob
import json
import csv
import numpy as np
import cv2
import faiss
import torch
from torchvision import transforms
from tqdm import tqdm

try:
    from load import load_clean_model, DEVICE
except ImportError:
    print("BŁĄD: Nie znaleziono pliku load_and_test.py w tym folderze.")
    sys.exit(1)

MODEL_PATH = 'baseline.pth'

BASE_TEST_FOLDER = '../webface_112x112/test' 

METRICS_DIR = 'metrics'
FAISS_INDEX_FILE = os.path.join(METRICS_DIR, 'gallery.index')
FAISS_MAPPING_FILE = os.path.join(METRICS_DIR, 'gallery_map.json')
RESULTS_CSV = os.path.join(METRICS_DIR, 'evaluation_results.csv')
OCCLUSION_OUTPUT_DIR = 'occlusion_photos_eval'

OCCLUSION_SIZE = 20
K_NEIGHBORS = 3

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def initialize_custom_model():
    """Ładuje Twój wytrenowany backbone."""
    print(f"Budowanie architektury iResNet...")
    model = load_clean_model()
    
    print(f"Wczytywanie Twoich wag z: {MODEL_PATH}...")
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "").replace("backbone.", "")
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            print("Wagi Stage 2 załadowane pomyślnie!")
        except Exception as e:
            print(f"BŁĄD ładowania wag: {e}")
            sys.exit(1)
    else:
        print(f"BŁĄD: Nie znaleziono pliku {MODEL_PATH}")
        sys.exit(1)
        
    model.to(DEVICE)
    model.eval()
    return model

def get_embedding(model, image_bgr):
    """
    Pobiera embedding używając Twojego modelu.
    Wejście: Obraz BGR (cv2)
    Wyjście: Znormalizowany wektor numpy (512,)
    """
    try:
        if image_bgr.shape[0] != 112 or image_bgr.shape[1] != 112:
            image_bgr = cv2.resize(image_bgr, (112, 112))
            
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        img_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            embedding = model(img_tensor)
            embedding = embedding.cpu().numpy().flatten()
            
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
            
        return embedding
        
    except Exception as e:
        print(f"Warning: Błąd podczas pobierania embeddingu: {e}")
        return None


def discover_file_structure(local_test_path):
    print(f"Skanowanie plików w {local_test_path}...")
    
    # Wzór: [dataset]/[id]/[session]/[img.jpg]
    search_pattern = os.path.join(local_test_path, "*", "*", "*.jpg")
    all_jpg_files = list(glob.glob(search_pattern))
    
    if not all_jpg_files:
        print("BŁĄD: Nie znaleziono plików .jpg.")
        return None, None
        
    print(f"Znaleziono {len(all_jpg_files)} plików .jpg.")

    identity_to_imgfolders = {} 
    image_pairs = {}        

    for jpg_path in tqdm(all_jpg_files, desc="Indeksowanie"):
        jpg_path_norm = os.path.normpath(jpg_path)
        base_name = os.path.splitext(jpg_path_norm)[0]
        json_path = base_name + ".json"
        
        has_json = os.path.exists(json_path)
        
        image_folder_path = os.path.dirname(jpg_path_norm)
        identity_path = os.path.dirname(image_folder_path)
        
        if identity_path not in identity_to_imgfolders:
            identity_to_imgfolders[identity_path] = set()
        
        identity_to_imgfolders[identity_path].add(image_folder_path)
        
        if image_folder_path not in image_pairs:
            image_pairs[image_folder_path] = {'jpg': jpg_path_norm, 'json': json_path if has_json else None}

    print(f"Zmapowano {len(identity_to_imgfolders)} tożsamości.")
    return identity_to_imgfolders, image_pairs


def build_faiss_gallery(model, identity_to_imgfolders, image_pairs):
    print(f"Budowanie Galerii FAISS...")
    
    identity_paths = list(identity_to_imgfolders.keys())
    gallery_embeddings = []
    index_to_id_map = {} 
    faiss_index_counter = 0

    for id_path in tqdm(identity_paths, desc="Przetwarzanie ID"):
        identity_id = os.path.basename(id_path)
        image_folder_paths = sorted(list(identity_to_imgfolders[id_path]))
        
        split_point = max(1, len(image_folder_paths) // 2)
        gallery_folders = image_folder_paths[:split_point]
        
        id_embeddings = []
        for img_folder_path in gallery_folders:
            local_path = image_pairs.get(img_folder_path, {}).get('jpg')
            if not local_path: continue
                
            img = cv2.imread(local_path)
            if img is None: continue
                
            embedding = get_embedding(model, img)
            if embedding is not None:
                id_embeddings.append(embedding)
            
        if id_embeddings:
            avg_embedding = np.mean(id_embeddings, axis=0)
            avg_embedding /= np.linalg.norm(avg_embedding) 
            
            gallery_embeddings.append(avg_embedding)
            index_to_id_map[faiss_index_counter] = identity_id
            faiss_index_counter += 1

    if not gallery_embeddings:
        print("BŁĄD: Galeria pusta.")
        return False
        
    dimension = gallery_embeddings[0].shape[0] 
    gallery_matrix = np.array(gallery_embeddings).astype('float32')
    
    # Indeks FlatIP (Inner Product) = Cosine Similarity dla znormalizowanych wektorów
    index = faiss.IndexFlatIP(dimension)
    index.add(gallery_matrix)
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_MAPPING_FILE, 'w') as f:
        json.dump(index_to_id_map, f)
        
    print(f"Zapisano indeks FAISS ({len(gallery_embeddings)} osób) do {FAISS_INDEX_FILE}")
    return True


def apply_occlusion(image, landmarks_dict, bbox):
    """Nakłada pasek okluzji na oczy."""
    occluded_image = image.copy()
    try:
        if landmarks_dict:
            left_eye_y = landmarks_dict["left_eye"][1]
            right_eye_y = landmarks_dict["right_eye"][1]
            eye_y_center = int((left_eye_y + right_eye_y) / 2)
        else:
            eye_y_center = image.shape[0] // 2 - 10

        bar_height_half = OCCLUSION_SIZE // 2
        
        if bbox:
            x1 = int(max(0, bbox[0]))
            x2 = int(min(image.shape[1], bbox[2]))
        else:
            x1 = 0
            x2 = image.shape[1]

        y1 = max(0, eye_y_center - bar_height_half)
        y2 = min(image.shape[0], eye_y_center + bar_height_half)
        
        cv2.rectangle(occluded_image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    except Exception as e:
        h, w = image.shape[:2]
        cv2.rectangle(occluded_image, (0, h//2 - 10), (w, h//2 + 10), (0,0,0), -1)
        
    return occluded_image

def run_occlusion_evaluation(model, identity_to_imgfolders, image_pairs):
    print(f"Wczytywanie FAISS (szukanie Top-{K_NEIGHBORS})...")
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(FAISS_MAPPING_FILE, 'r') as f:
            index_to_id_map = json.load(f)
    except Exception as e:
        print(f"BŁĄD FAISS: {e}")
        return

    identity_paths = list(identity_to_imgfolders.keys())
    total_queries = 0
    correct_top1 = 0
    correct_top3 = 0
    
    os.makedirs(OCCLUSION_OUTPUT_DIR, exist_ok=True)
    
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["query_id", "top1_id", "sim1", "top2_id", "sim2", "top3_id", "sim3", "found_in_top3"]
        writer.writerow(header)
        
        for id_path in tqdm(identity_paths, desc="Test Okluzji"):
            ground_truth_id = os.path.basename(id_path)
            image_folder_paths = sorted(list(identity_to_imgfolders[id_path]))
            
            split_point = max(1, len(image_folder_paths) // 2)
            query_folders = image_folder_paths[split_point:]

            for img_folder_path in query_folders:
                data = image_pairs.get(img_folder_path, {})
                local_img_path = data.get('jpg')
                local_json_path = data.get('json')

                if not local_img_path: continue
                
                img = cv2.imread(local_img_path)
                if img is None: continue

                landmarks = None
                bbox = None
                if local_json_path and os.path.exists(local_json_path):
                    try:
                        with open(local_json_path, 'r') as jf:
                            jd = json.load(jf)
                            landmarks = jd.get("landmarks")
                            bbox = jd.get("bbox")
                    except: pass

                occluded_img = apply_occlusion(img, landmarks, bbox)
                
                if total_queries % 100 == 0:
                    save_path = os.path.join(OCCLUSION_OUTPUT_DIR, f"{ground_truth_id}_{total_queries}.jpg")
                    cv2.imwrite(save_path, occluded_img)
                
                query_embedding = get_embedding(model, occluded_img)
                if query_embedding is None: continue
                
                query_vector = np.expand_dims(query_embedding, axis=0).astype('float32')
                
                D, I = index.search(query_vector, K_NEIGHBORS)
                
                top_identities = []
                top_similarities = []
                found_in_top3 = False
                
                for k in range(K_NEIGHBORS):
                    if I[0][k] == -1:
                        top_identities.append("N/A")
                        top_similarities.append("0.0000")
                        continue
                        
                    idx = str(I[0][k])
                    sim = D[0][k]
                    pred_id = index_to_id_map.get(idx, "N/A")
                    
                    top_identities.append(pred_id)
                    top_similarities.append(f"{sim:.4f}")
                    
                    if pred_id == ground_truth_id:
                        found_in_top3 = True

                if found_in_top3:
                    correct_top3 += 1
                
                if top_identities[0] == ground_truth_id:
                    correct_top1 += 1
                
                total_queries += 1

                csv_row = [ground_truth_id]
                for i in range(K_NEIGHBORS):
                    csv_row.append(top_identities[i])
                    csv_row.append(top_similarities[i])
                csv_row.append(found_in_top3)
                
                writer.writerow(csv_row)
                
    if total_queries > 0:
        acc_top1 = (correct_top1 / total_queries) * 100
        acc_top3 = (correct_top3 / total_queries) * 100
        print(f"\nWYNIKI (Okluzja):")
        print(f"Liczba zapytań: {total_queries}")
        print(f"Poprawne (Rank-1): {correct_top1} -> Accuracy @ Top-1: {acc_top1:.2f}%")
        print(f"Poprawne (Rank-3): {correct_top3} -> Accuracy @ Top-3: {acc_top3:.2f}%")
        print(f"Szczegóły w: {RESULTS_CSV}")
    else:
        print("Brak zapytań.")


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    model = initialize_custom_model()
    
    identity_to_imgfolders, image_pairs = discover_file_structure(BASE_TEST_FOLDER)
    if not identity_to_imgfolders:
        return

    if not build_faiss_gallery(model, identity_to_imgfolders, image_pairs):
        return
    
    run_occlusion_evaluation(model, identity_to_imgfolders, image_pairs)

if __name__ == "__main__":
    main()