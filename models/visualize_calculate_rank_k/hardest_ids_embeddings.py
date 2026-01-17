import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import cv2
import faiss
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
HARD_IDS_CSV = os.path.join(CURRENT_DIR, "hardest_ids_summary.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


                           
               
                           

def _filter_and_load(model, state_dict, min_match_ratio=0.2, weights_path=""):
    new_state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in new_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v

    if not filtered_state:
        raise ValueError("Nie wczytano żadnych wag — sprawdź architekturę i ścieżkę.")

    match_ratio = len(filtered_state) / max(1, len(model_state))
    if match_ratio < min_match_ratio:
        raise ValueError(
            f"Wczytano zbyt mało wag ({match_ratio:.1%})."
        )

    model.load_state_dict(filtered_state, strict=False)
    print(f"Wczytano {len(filtered_state)}/{len(model_state)} warstw z: {weights_path}")


def load_model(weights_path: str, model_type: str):
    if model_type == "baseline":
        try:
            from Aligned_Pretrained.load import load_clean_model
            model = load_clean_model()
        except Exception:
            from Aligned_Pretrained.backbone_iresnet import iresnet50
            model = iresnet50(weights_path=None)
    elif model_type == "transformer":
        transformer_dir = os.path.join(MODELS_DIR, "Aligned_Pretrained_Transformers")
        if transformer_dir not in os.sys.path:
            os.sys.path.insert(0, transformer_dir)
        from Aligned_Pretrained_Transformers.load import load_clean_model as load_transformer
        model = load_transformer()
    elif model_type == "cbam":
        from Aligned_Pretrained_CBAM_Block.evaluation.run_evaluation import IResNetCBAM, CBAMBasicBlock
        model = IResNetCBAM(CBAMBasicBlock, [3, 4, 14, 3])
    else:
        raise ValueError("model_type musi być: baseline | cbam | transformer")

    state_dict = torch.load(weights_path, map_location=DEVICE)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    _filter_and_load(model, state_dict, min_match_ratio=0.2, weights_path=weights_path)

    model.to(DEVICE)
    model.eval()
    return model


                           
            
                           

def load_hardest_ids(csv_path: str, model_name: str, top_n: int) -> List[str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Brak pliku: {csv_path}")

    ids = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model") != model_name:
                continue
            qid = row.get("query_id", "").strip()
            if qid and qid not in ids:
                ids.append(qid)
            if len(ids) >= top_n:
                break
    return ids


def find_images_for_id(test_root: str, id_name: str) -> List[Tuple[str, str]]:
    id_dir = os.path.join(test_root, id_name)
    if not os.path.isdir(id_dir):
        return []

    results = []
    for root, _, files in os.walk(id_dir):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(root, fname)
            json_path = os.path.splitext(img_path)[0] + ".json"
            results.append((img_path, json_path))
    return results


def apply_eye_occlusion(img: np.ndarray, y_center: int = 52, height: int = 20) -> np.ndarray:
    h, w = img.shape[:2]
    y1 = max(0, int(y_center - height / 2))
    y2 = min(h, int(y_center + height / 2))
    occluded = img.copy()
    cv2.rectangle(occluded, (0, y1), (w, y2), (0, 0, 0), -1)
    return occluded


def preprocess_face(img: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def get_embedding(model, img: np.ndarray) -> np.ndarray:
    tensor = preprocess_face(img).to(DEVICE)
    with torch.no_grad():
        emb = model(tensor)
        if emb.dim() > 2:
            emb = torch.flatten(emb, 1)
        emb = torch.nn.functional.normalize(emb, dim=1)
    return emb.cpu().numpy().astype(np.float32)


def build_color_map(ids: List[str]) -> Dict[str, str]:
    palette = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
        "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
        "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
    ]
    color_map = {}
    for i, id_name in enumerate(ids):
        color_map[id_name] = palette[i % len(palette)]
    return color_map


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    x0 = x - mean
    cov = np.cov(x0, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order[:2]]
    return x0 @ vecs


def visualize_embeddings(embeddings: np.ndarray, labels: List[dict], color_map: Dict[str, str], output_path: str):
    if embeddings.shape[0] < 2:
        return
    coords = pca_2d(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for id_name, color in color_map.items():
        idx = [i for i, item in enumerate(labels) if item["id"] == id_name]
        if not idx:
            continue
        ax.scatter(coords[idx, 0], coords[idx, 1], s=10, c=color, label=id_name, alpha=0.85)
    ax.set_title("Embeddings (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=6, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Embed hardest IDs with eye occlusion and build FAISS index.")
    parser.add_argument("--test-root", required=True, help="Ścieżka do webface_112x112/test")
    parser.add_argument("--model-name", default="Aligned_Pretrained_CBAM_Block", help="Nazwa modelu z hardest_ids_summary.csv")
    parser.add_argument("--model-type", default="cbam", choices=["baseline", "cbam", "transformer"], help="Typ modelu")
    parser.add_argument("--weights", required=True, help="Ścieżka do wag .pth")
    parser.add_argument("--top-n", type=int, default=10, help="Ile hardest IDs użyć")
    parser.add_argument("--output-dir", default=os.path.join(CURRENT_DIR, "hardest_ids_embeddings"), help="Folder wyjściowy")
    parser.add_argument("--save-occluded", action=argparse.BooleanOptionalAction, default=False, help="Zapisuj obrazy z okluzją")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Zapisuj wizualizację 2D (PCA)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Wybór urządzenia")
    args = parser.parse_args()

    global DEVICE
    if args.device == "cpu":
        DEVICE = torch.device("cpu")
    elif args.device == "cuda":
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        else:
            print("CUDA niedostępne, przełączam na CPU.")
            DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    ids = load_hardest_ids(HARD_IDS_CSV, args.model_name, args.top_n)
    if not ids:
        raise ValueError(f"Brak hardest IDs dla modelu: {args.model_name}")

    model = load_model(args.weights, args.model_type)

    embeddings = []
    labels = []
    faiss_ids = []
    faiss_id = 0

    for id_name in ids:
        items = find_images_for_id(args.test_root, id_name)
        for img_path, _json_path in items:
            img = cv2.imread(img_path)
            if img is None:
                continue
            occluded = apply_eye_occlusion(img, y_center=52, height=20)
            emb = get_embedding(model, occluded)
            embeddings.append(emb[0])
            labels.append({
                "faiss_id": faiss_id,
                "id": id_name,
                "image_path": img_path,
            })
            faiss_ids.append(faiss_id)

            if args.save_occluded:
                out_path = os.path.join(args.output_dir, f"occ_{id_name}_{faiss_id}.jpg")
                cv2.imwrite(out_path, occluded)

            faiss_id += 1

    if not embeddings:
        raise ValueError("Nie znaleziono żadnych obrazów do przetworzenia.")

    embeddings = np.stack(embeddings).astype(np.float32)
    faiss_ids = np.array(faiss_ids, dtype=np.int64)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(index)
    index.add_with_ids(embeddings, faiss_ids)

    faiss.write_index(index, os.path.join(args.output_dir, "index.faiss"))
    np.save(os.path.join(args.output_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(args.output_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    color_map = build_color_map(ids)
    with open(os.path.join(args.output_dir, "id_colors.json"), "w", encoding="utf-8") as f:
        json.dump(color_map, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "used_ids.txt"), "w", encoding="utf-8") as f:
        for id_name in ids:
            f.write(f"{id_name}\n")

    if args.plot:
        plot_path = os.path.join(args.output_dir, "embeddings_plot.png")
        visualize_embeddings(embeddings, labels, color_map, plot_path)

    print(f"Gotowe: {len(embeddings)} embeddingów, zapisano w: {args.output_dir}")


if __name__ == "__main__":
    main()
