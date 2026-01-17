import torch
import torch.nn as nn
import numpy as np
import sys
import os
import cv2
import json
import glob
import csv
import random
import faiss
from tqdm import tqdm
from collections import OrderedDict

try:
    from backbone_iresnet import iresnet50
except ImportError:
    print("BŁĄD: Brak pliku 'backbone_iresnet.py'.")
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_FOLDER_LOCAL = '../../../../webface_112x112'
LOCAL_WEIGHTS_PATH = 'w600k_r50_from_onnx.pth'
TEST_SUBDIR = 'test'

FAISS_INDEX_FILE = "gallery_custom.index"
FAISS_MAPPING_FILE = "gallery_custom_map.json"
RESULTS_CSV = "results_custom_occlusion.csv"
OUTPUT_OCCLUSION_DIR = "evaluation_photos_custom"

OCCLUSION_SIZE = 20

def normalize_name(name):
    name = name.replace('_initializer_', '')
    name = name.replace('module.', '')
    name = name.replace('.', '').replace('_', '').lower()
    return name

def reset_layer_params(layer, name):
                                           
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)
        layer.running_mean.zero_()
        layer.running_var.fill_(1)
    elif isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, 0, 0.01)
        nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.PReLU):
        nn.init.constant_(layer.weight, 0.25)

def sanitize_model(model):
    print("\nUruchamiam SZPITAL (Sanitize Model)...")
    fixes = 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.running_var is not None:
                if torch.isnan(m.running_var).any() or (m.running_var < 1e-5).any():
                    m.running_var.data.clamp_(min=1e-4)
                    m.running_var.data[torch.isnan(m.running_var.data)] = 1.0
                    fixes += 1
            if torch.isnan(m.weight).any():
                nn.init.ones_(m.weight)
                fixes += 1
        elif isinstance(m, nn.PReLU):
            if torch.isnan(m.weight).any():
                nn.init.constant_(m.weight, 0.25)
                fixes += 1
        elif hasattr(m, 'weight') and m.weight is not None:
            if torch.isnan(m.weight).any():
                reset_layer_params(m, name)
                fixes += 1
    print(f"Szpital zakończony. Wprowadzono {fixes} poprawek.")

def smart_load_hybrid_v4(model, state_dict_path):
    print(f"Ładowanie hybrydowe z: {state_dict_path}")
    
    try:
        source_state = torch.load(state_dict_path, map_location='cpu')
        if 'state_dict' in source_state:
            source_state = source_state['state_dict']
    except Exception as e:
        print(f"Błąd odczytu pliku: {e}")
        return False

    target_state = model.state_dict()
    new_state_dict = OrderedDict()
    
    source_keys = list(source_state.keys())
    all_target_keys = list(target_state.keys())
    target_keys = [k for k in all_target_keys if 'num_batches_tracked' not in k]
    
    used_source_keys = set()
    matched_target_keys = set()
    
    source_norm_map = {normalize_name(k): k for k in source_keys}
    for t_key in target_keys:
        t_shape = target_state[t_key].shape
        t_norm = normalize_name(t_key)
        if t_norm in source_norm_map:
            original_source_key = source_norm_map[t_norm]
            s_tensor = source_state[original_source_key]
            if s_tensor.shape == t_shape:
                new_state_dict[t_key] = s_tensor
                used_source_keys.add(original_source_key)
                matched_target_keys.add(t_key)

    remaining_source = []
    for k in source_keys:
        if k not in used_source_keys:
            remaining_source.append((k, source_state[k]))
    remaining_source.sort(key=lambda x: x[0])
    
    for t_key in target_keys:
        if t_key in matched_target_keys: continue
        t_shape = target_state[t_key].shape
        for idx, (s_key, s_tensor) in enumerate(remaining_source):
            if s_tensor.shape == t_shape:
                new_state_dict[t_key] = s_tensor
                used_source_keys.add(s_key)
                matched_target_keys.add(t_key)
                remaining_source.pop(idx)
                break

    model.load_state_dict(new_state_dict, strict=False)
    
    unassigned_target = set(target_keys) - matched_target_keys
    if unassigned_target:
        print(f"Brak wag dla {len(unassigned_target)} warstw - reinicjalizacja.")
        for name, module in model.named_modules():
            has_missing = False
            for p_name, _ in module.named_parameters(recurse=False):
                full = f"{name}.{p_name}" if name else p_name
                if full in unassigned_target:
                    has_missing = True
                    break
            if has_missing:
                reset_layer_params(module, name)

    sanitize_model(model)
    return True

def load_clean_model():
    print(f"Inicjalizacja modelu IResNet50 na {DEVICE}...")
    model = iresnet50(weights_path=None)
    
    if os.path.exists(LOCAL_WEIGHTS_PATH):
        success = smart_load_hybrid_v4(model, LOCAL_WEIGHTS_PATH)
        if not success:
            print("Nie udało się załadować wag, model losowy!")
    else:
        print(f"Brak pliku wag {LOCAL_WEIGHTS_PATH}, model losowy!")

    model.eval()
    model.to(DEVICE)
    return model


def get_embedding_raw(model, img_bgr):
    if img_bgr is None: return None

    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (112, 112))
        
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        
                                                                                     
        img_tensor = (img_tensor - 127.5) / 128.0
        
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            features = model(img_tensor)
            features = features.cpu().numpy().flatten()
            
        if np.isnan(features).any():
            return None
            
        feat_norm = np.linalg.norm(features)
        return features / (feat_norm + 1e-10)
        
    except Exception as e:
        print(f"Warning: Błąd w get_embedding_raw: {e}")
        return None

def discover_file_structure(local_test_path):
    print(f"Skanowanie struktury plików w: {local_test_path}")
    
    search_pattern = os.path.join(local_test_path, "*", "*", "*.jpg")
    all_jpg_files = list(glob.glob(search_pattern))
    
    if not all_jpg_files:
        print("BŁĄD: Nie znaleziono plików .jpg. Sprawdź ścieżkę.")
        return None, None
        
    print(f"   Znaleziono {len(all_jpg_files)} plików .jpg.")

    identity_to_imgfolders = {} 
    image_pairs = {}       

    for jpg_path in tqdm(all_jpg_files, desc="Indeksowanie"):
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

    print(f"Zindeksowano {len(identity_to_imgfolders)} tożsamości.")
    return identity_to_imgfolders, image_pairs

def build_faiss_gallery(model, identity_to_imgfolders, image_pairs):
    print(f"\nBudowanie Galerii FAISS...")
    
    identity_paths = list(identity_to_imgfolders.keys())
    gallery_embeddings = []
    index_to_id_map = {} 
    faiss_index_counter = 0

    for id_path in tqdm(identity_paths, desc="Tworzenie galerii"):
        identity_id = os.path.basename(id_path)
        
        image_folder_paths = sorted(list(identity_to_imgfolders[id_path]))
        
        split_point = max(1, len(image_folder_paths) // 2)
        gallery_folders = image_folder_paths[:split_point]
        
        if not gallery_folders: continue

        id_embeddings = []
        for img_folder_path in gallery_folders:
            local_path = image_pairs.get(img_folder_path, {}).get('jpg')
            if not local_path: continue
                
            img = cv2.imread(local_path)
            if img is None: continue
                
            embedding = get_embedding_raw(model, img)
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
    
    index = faiss.IndexFlatIP(dimension)
    index.add(gallery_matrix)
    
    print(f"Zapisywanie indeksu ({faiss_index_counter} wektorów)...")
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_MAPPING_FILE, 'w') as f:
        json.dump(index_to_id_map, f)
        
    return True

def apply_occlusion(image, landmarks_dict, bbox):
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

def run_occlusion_evaluation(model, identity_to_imgfolders, image_pairs):
    print(f"\nRozpoczynanie Ewaluacji Okluzji...")
    
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(FAISS_MAPPING_FILE, 'r') as f:
            index_to_id_map = json.load(f)
    except Exception as e:
        print(f"BŁĄD ładowania FAISS: {e}")
        return

    identity_paths = list(identity_to_imgfolders.keys())
    total_queries = 0
    correct_top1 = 0
    
    os.makedirs(OUTPUT_OCCLUSION_DIR, exist_ok=True)
    
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "top1_id", "score", "is_correct"])
        
        for id_path in tqdm(identity_paths, desc="Testowanie"):
            ground_truth_id = os.path.basename(id_path)
            
            image_folder_paths = sorted(list(identity_to_imgfolders[id_path]))
            
            split_point = max(1, len(image_folder_paths) // 2)
            query_folders = image_folder_paths[split_point:]

            for img_folder_path in query_folders:
                local_img_path = image_pairs.get(img_folder_path, {}).get('jpg')
                local_json_path = image_pairs.get(img_folder_path, {}).get('json')

                if not local_img_path or not local_json_path: continue
                
                img = cv2.imread(local_img_path)
                
                json_data = None
                try:
                    with open(local_json_path, 'r') as jf:
                        json_data = json.load(jf)
                except: pass
                
                if (img is None or json_data is None or 
                    "landmarks" not in json_data or "bbox" not in json_data):
                    continue

                occluded_img = apply_occlusion(img, json_data["landmarks"], json_data["bbox"])
                
                if random.random() < 0.01:
                    fname = os.path.basename(local_img_path)
                    cv2.imwrite(os.path.join(OUTPUT_OCCLUSION_DIR, f"occ_{fname}"), occluded_img)
                
                query_embedding = get_embedding_raw(model, occluded_img)
                
                if query_embedding is None: continue 
                    
                query_vector = np.expand_dims(query_embedding, axis=0).astype('float32')
                D, I = index.search(query_vector, 1)
                
                top1_idx = str(I[0][0])
                top1_score = D[0][0]
                top1_id = index_to_id_map.get(top1_idx, "N/A")
                
                is_correct = (top1_id == ground_truth_id)
                writer.writerow([ground_truth_id, top1_id, f"{top1_score:.4f}", is_correct])
                
                if is_correct:
                    correct_top1 += 1
                total_queries += 1

    if total_queries > 0:
        accuracy = (correct_top1 / total_queries) * 100
        print(f"\nWYNIKI KOŃCOWE:")
        print(f"   Liczba zapytań: {total_queries}")
        print(f"   Poprawne Top-1: {correct_top1}")
        print(f"   ACCURACY:       {accuracy:.2f}%")
        print(f"   Szczegóły w:    {RESULTS_CSV}")
    else:
        print("\nOstrzeżenie: Nie przetworzono żadnych zapytań.")

def main():
    model = load_clean_model()
    
    local_test_path = os.path.join(BASE_FOLDER_LOCAL, TEST_SUBDIR)
    
    identity_to_imgfolders, image_pairs = discover_file_structure(local_test_path)
    if not identity_to_imgfolders:
        print("Koniec pracy.")
        return

    if build_faiss_gallery(model, identity_to_imgfolders, image_pairs):
        run_occlusion_evaluation(model, identity_to_imgfolders, image_pairs)
    
    print("\nGotowe.")

if __name__ == "__main__":
    main()