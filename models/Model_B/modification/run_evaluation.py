# corrected_run_evaluation.py
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
from google.cloud import storage
import torch

from backbone_iresnet import iresnet50
from config import (
    BUCKET_NAME, BASE_FOLDER_GCS, LOCAL_DATA_DIR,
    FAISS_INDEX_FILE, FAISS_MAPPING_FILE, RESULTS_CSV, OCCLUSION_SIZE
)

# ---------------------------
# Helper: device
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 1. Inicjalizacja usług (GCS) i modelu PyTorch
# ---------------------------
def initialize_services():
    """Ładuje klienta GCS i model PyTorch (IResNet50). Zwraca (model, bucket)."""
    print("Ładowanie klienta Google Cloud Storage...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
    except Exception as e:
        print(f"BŁĄD: Nie udało się połączyć z GCS. Sprawdź uwierzytelnienie.")
        print(f"Error: {e}")
        sys.exit(1)

    print("Ładowanie modelu IResNet (PyTorch)...")
    try:
        state_path = 'w600k_r50_from_onnx.pth'
        model = iresnet50(weights_path=state_path)  # Używamy zmodyfikowanej funkcji z obsługą weights_path
        if not os.path.exists(state_path):
            print(f"BŁĄD: Nie znaleziono pliku wag: {state_path}")
            sys.exit(1)

        # state = torch.load(state_path, map_location="cpu")
        # Jeżeli plik zawiera klucz 'state_dict' -> dostosuj:
        # if isinstance(state, dict) and "state_dict" in state:
        #     state = state["state_dict"]

        # try:
        #     model.load_state_dict(state, strict=False)
        # except Exception as e:
        #     print("Warning: Nie udało się wczytać state_dict bez strict=False. Próba dalej z strict=False.")
        #     model.load_state_dict(state, strict=False)

        model.to(DEVICE)
        model.eval()

    except Exception as e:
        print(f"BŁĄD: Nie udało się załadować modelu IResNet.")
        print(f"Error: {e}")
        sys.exit(1)

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    print("Inicjalizacja zakończona pomyślnie.")
    return model, bucket

# ---------------------------
# 2. Funkcja pobierająca embedding dla obrazu
# ---------------------------
def preprocess_face_from_bbox(img, bbox, output_size=112, pad_ratio=0.2):
    """
    Przyjmuje obraz BGR i bbox [x1,y1,x2,y2].
    Zwraca 112x112 BGR (uint8) wycięty i wyrównany (prostokąt).
    pad_ratio -> dołożenie marginesu wokół bbox.
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
        # Fallback: center crop
        min_edge = min(h, w)
        cx = w // 2
        cy = h // 2
        crop = img[cy - min_edge // 2: cy + min_edge // 2, cx - min_edge // 2: cx + min_edge // 2]

    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return resized

def image_to_embedding(model, img_bgr, bbox=None, landmarks=None):
    """
    Z img_bgr (BGR uint8) zwraca embedding numpy (float32, znormalizowany).
    Jeśli bbox dostępny -> używamy crop+resize; jeśli nie -> próbujemy wykryć central crop.
    """
    try:
        if bbox is not None:
            face = preprocess_face_from_bbox(img_bgr, bbox, output_size=112)
        else:
            # fallback: center crop + resize
            h, w = img_bgr.shape[:2]
            min_edge = min(h, w)
            cx = w // 2
            cy = h // 2
            crop = img_bgr[max(0, cy - min_edge//2):min(h, cy + min_edge//2),
                           max(0, cx - min_edge//2):min(w, cx + min_edge//2)]
            face = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)

        # Preprocessing: ArcFace typical: (img - 127.5) / 127.5  and swap to CHW
        face = face.astype(np.float32)
        face = (face - 127.5) / 127.5
        # HWC -> CHW
        face = face[:, :, ::-1]  # BGR -> RGB if model expects RGB; many ArcFace expect BGR with swapRB=True earlier.
        # Note: if your iresnet50 expects BGR, remove the swap above.
        face = np.transpose(face, (2, 0, 1))
        tensor = torch.from_numpy(face.copy()).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = model(tensor)  # zakładamy, że model zwraca (N,512)
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            emb = emb.cpu().numpy().reshape(-1)
            # Normalize
            emb = emb / np.linalg.norm(emb + 1e-10)
            return emb.astype('float32')
    except Exception as e:
        print(f"Warning: image_to_embedding error: {e}")
        return None

# ---------------------------
# 3. Pobieranie pliku z GCS
# ---------------------------
def download_blob(blob, destination_folder):
    local_path = os.path.join(destination_folder, os.path.basename(blob.name))
    try:
        blob.download_to_filename(local_path)
        return local_path
    except Exception as e:
        print(f"Warning: Nie udało się pobrać {blob.name}: {e}")
        return None

# ---------------------------
# 4. Odkrywanie struktury plików w bucket
# ---------------------------
def discover_file_structure(bucket):
    print(f"Wykrywanie struktury plików w gs://{BUCKET_NAME}/{BASE_FOLDER_GCS}/...")
    target_prefix = f"{BASE_FOLDER_GCS}/"
    all_blobs = list(bucket.list_blobs(prefix=target_prefix))

    if not all_blobs:
        print("BŁĄD: Nie znaleziono plików.")
        return None, None

    identity_to_imgfolders = {}
    image_pairs = {}

    for blob in all_blobs:
        parts = blob.name.split('/')
        # oczekujemy struktury: {BASE_FOLDER}/{split}/{id}/{img_folder}/{file}
        if len(parts) < 5:
            continue
        identity_path = "/".join(parts[:3])
        image_folder_path = "/".join(parts[:4])

        if identity_path not in identity_to_imgfolders:
            identity_to_imgfolders[identity_path] = set()
        if image_folder_path not in image_pairs:
            image_pairs[image_folder_path] = {'jpg': None, 'json': None}

        if blob.name.lower().endswith(".jpg") or blob.name.lower().endswith(".jpeg") or blob.name.lower().endswith(".png"):
            identity_to_imgfolders[identity_path].add(image_folder_path)
            image_pairs[image_folder_path]['jpg'] = blob
        elif blob.name.lower().endswith(".json"):
            image_pairs[image_folder_path]['json'] = blob

    print(f"Wykryto {len(identity_to_imgfolders)} identity folders.")
    return identity_to_imgfolders, image_pairs

# ---------------------------
# 5. Budowanie galerii FAISS
# ---------------------------
def build_faiss_gallery(model, identity_to_imgfolders, image_pairs):
    print(f"--- ROZPOCZYNAM Budowanie Galerii FAISS ---")
    identity_paths = list(identity_to_imgfolders.keys())
    if not identity_paths:
        print("BŁĄD: brak identity.")
        return False

    gallery_embeddings = []
    index_to_id_map = {}
    faiss_index_counter = 0

    for id_path in tqdm(identity_paths, desc="Tworzenie galerii ID"):
        identity_id = id_path.split('/')[-1]
        image_folder_paths = sorted(list(identity_to_imgfolders[id_path]))
        # Weź połowę jako galeria
        split_point = max(1, len(image_folder_paths) // 2)
        gallery_folders = image_folder_paths[:split_point]

        if not gallery_folders:
            continue

        id_embeddings = []
        for img_folder_path in gallery_folders:
            jpg_blob = image_pairs.get(img_folder_path, {}).get('jpg')
            json_blob = image_pairs.get(img_folder_path, {}).get('json')

            if not jpg_blob:
                tqdm.write(f"Warning: brak JPG w {img_folder_path}")
                continue

            local_img = download_blob(jpg_blob, LOCAL_DATA_DIR)
            local_json = None
            if json_blob:
                local_json = download_blob(json_blob, LOCAL_DATA_DIR)

            img = None
            bbox = None
            try:
                img = cv2.imread(local_img)
            except Exception:
                img = None

            if img is None:
                tqdm.write(f"Warning: Nie odczytano obrazu {local_img}")
                if local_img and os.path.exists(local_img): os.remove(local_img)
                if local_json and os.path.exists(local_json): os.remove(local_json)
                continue

            if local_json and os.path.exists(local_json):
                try:
                    with open(local_json, 'r') as jf:
                        j = json.load(jf)
                        bbox = j.get("bbox", None)
                except Exception as e:
                    tqdm.write(f"Warning: Błąd odczytu JSON {local_json}: {e}")

            emb = image_to_embedding(model, img, bbox=bbox)
            if emb is not None:
                id_embeddings.append(emb)

            # cleanup
            if local_img and os.path.exists(local_img): os.remove(local_img)
            if local_json and os.path.exists(local_json): os.remove(local_json)

        if id_embeddings:
            avg_embedding = np.mean(np.stack(id_embeddings, axis=0), axis=0)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-10)
            gallery_embeddings.append(avg_embedding)
            index_to_id_map[str(faiss_index_counter)] = identity_id
            faiss_index_counter += 1

    print(f"Zakończono. Znaleziono {len(gallery_embeddings)} unikalnych tożsamości.")
    if not gallery_embeddings:
        print("BŁĄD: pusta galeria.")
        return False

    gallery_matrix = np.array(gallery_embeddings).astype('float32')
    d = gallery_matrix.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product on normalized vectors == cosine
    index.add(gallery_matrix)

    print(f"Zapisywanie indeksu FAISS do {FAISS_INDEX_FILE}...")
    faiss.write_index(index, FAISS_INDEX_FILE)

    print(f"Zapisywanie mapowania ID do {FAISS_MAPPING_FILE}...")
    with open(FAISS_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(index_to_id_map, f, ensure_ascii=False, indent=2)

    return True

# ---------------------------
# 6. Okluzja (twoja funkcja poprawiona)
# ---------------------------
def apply_occlusion(image, landmarks_dict, bbox):
    occluded_image = image.copy()
    try:
        left_eye = landmarks_dict.get("left_eye") or landmarks_dict.get("leftEye") or landmarks_dict.get("left")
        right_eye = landmarks_dict.get("right_eye") or landmarks_dict.get("rightEye") or landmarks_dict.get("right")
        if left_eye is None or right_eye is None:
            raise ValueError("Brak kluczy do oczu w landmarks")

        left_eye_y = int(left_eye[1])
        right_eye_y = int(right_eye[1])
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
        print(f"Warning: Błąd podczas nakładania okluzji: {e}")
        return image.copy()

    return occluded_image

# ---------------------------
# 7. Ewaluacja z okluzją
# ---------------------------
def run_occlusion_evaluation(model, identity_to_imgfolders, image_pairs):
    print("Wczytywanie galerii FAISS i mapowania ID...")
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(FAISS_MAPPING_FILE, 'r', encoding='utf-8') as f:
            index_to_id_map = json.load(f)
    except Exception as e:
        print(f"BŁĄD: Nie udało się wczytać plików FAISS. Uruchom budowanie galerii.")
        print(f"Error: {e}")
        return

    print(f"Rozpoczynam ewaluację z okluzją. Wyniki w {RESULTS_CSV}...")
    identity_paths = list(identity_to_imgfolders.keys())

    total_queries = 0
    correct_top1 = 0

    output_occlusion_dir = "occlusion_photos"
    os.makedirs(output_occlusion_dir, exist_ok=True)

    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "top1_id", "top1_similarity", "top2_id", "top2_similarity", "top3_id", "top3_similarity", "is_correct_top1"])

        for id_path in tqdm(identity_paths, desc="Testowanie okluzji"):
            ground_truth_id = id_path.split('/')[-1]
            image_folder_paths = sorted(list(identity_to_imgfolders[id_path]))
            split_point = max(1, len(image_folder_paths) // 2)
            query_folders = image_folder_paths[split_point:]

            for img_folder_path in query_folders:
                jpg_blob = image_pairs.get(img_folder_path, {}).get('jpg')
                json_blob = image_pairs.get(img_folder_path, {}).get('json')

                if not jpg_blob or not json_blob:
                    tqdm.write(f"Warning: Brak pary JPG/JSON dla {img_folder_path}")
                    continue

                local_img = download_blob(jpg_blob, LOCAL_DATA_DIR)
                local_json = download_blob(json_blob, LOCAL_DATA_DIR)

                if not local_img or not local_json:
                    continue

                img = cv2.imread(local_img)
                if img is None:
                    tqdm.write(f"Warning: Nie odczytano obrazu {local_img}")
                    try:
                        os.remove(local_img)
                    except: pass
                    try:
                        os.remove(local_json)
                    except: pass
                    continue

                try:
                    with open(local_json, 'r', encoding='utf-8') as jf:
                        j = json.load(jf)
                except Exception as e:
                    tqdm.write(f"Warning: Nie można odczytać JSON: {local_json}: {e}")
                    os.remove(local_img)
                    os.remove(local_json)
                    continue

                if "landmarks" not in j or "bbox" not in j:
                    tqdm.write(f"Warning: JSON nie zawiera landmarks/bbox w {local_json}")
                    os.remove(local_img)
                    os.remove(local_json)
                    continue

                occluded_img = apply_occlusion(img, j["landmarks"], j["bbox"])

                try:
                    original_filename = os.path.basename(local_img)
                    save_path = os.path.join(output_occlusion_dir, f"occluded_{ground_truth_id}_{original_filename}")
                    cv2.imwrite(save_path, occluded_img)
                except Exception as e:
                    tqdm.write(f"Warning: Nie udało się zapisać okluzji: {e}")

                query_embedding = image_to_embedding(model, occluded_img, bbox=j["bbox"], landmarks=j["landmarks"])
                if query_embedding is None:
                    os.remove(local_img)
                    os.remove(local_json)
                    continue

                # normalized query vector (we stored normalized gallery vectors)
                q = np.expand_dims(query_embedding.astype('float32'), axis=0)
                D, I = index.search(q, 3)

                top1_idx, top2_idx, top3_idx = I[0][0], I[0][1], I[0][2]
                top1_sim, top2_sim, top3_sim = D[0][0], D[0][1], D[0][2]

                top1_id = index_to_id_map.get(str(top1_idx), "N/A")
                top2_id = index_to_id_map.get(str(top2_idx), "N/A")
                top3_id = index_to_id_map.get(str(top3_idx), "N/A")

                is_correct = (top1_id == ground_truth_id)
                writer.writerow([ground_truth_id, top1_id, f"{top1_sim:.4f}", top2_id, f"{top2_sim:.4f}", top3_id, f"{top3_sim:.4f}", is_correct])

                if is_correct:
                    correct_top1 += 1
                total_queries += 1

                # cleanup
                try: os.remove(local_img)
                except: pass
                try: os.remove(local_json)
                except: pass

    if total_queries > 0:
        accuracy = (correct_top1 / total_queries) * 100.0
        print("\n--- Ewaluacja Zakończona ---")
        print(f"Total queries: {total_queries}")
        print(f"Correct Top-1: {correct_top1}")
        print(f"Top-1 Accuracy: {accuracy:.2f}%")
    else:
        print("\n--- Ewaluacja Zakończona ---")
        print("No queries processed.")

# ---------------------------
# 8. Main
# ---------------------------
def main():
    model, bucket = initialize_services()

    identity_to_imgfolders, image_pairs = discover_file_structure(bucket)
    if not identity_to_imgfolders:
        print("Brak plików - koniec.")
        return

    print("--- Krok 1: Budowanie galerii FAISS ---")
    ok = build_faiss_gallery(model, identity_to_imgfolders, image_pairs)
    if not ok:
        print("Błąd budowania galerii - koniec.")
        return
    print("--- Krok 1: Zakończono ---\n")

    print("--- Krok 2: Ewaluacja z okluzją ---")
    run_occlusion_evaluation(model, identity_to_imgfolders, image_pairs)
    print("--- Krok 2: Zakończono ---\n")

    # cleanup temp files but keep folder
    for f in glob.glob(os.path.join(LOCAL_DATA_DIR, "*")):
        try: os.remove(f)
        except: pass

if __name__ == "__main__":
    main()
