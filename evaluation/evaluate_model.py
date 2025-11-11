# evaluate_model.py

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import faiss
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# ========================
# 1. Ustawienia
# ========================
MODEL_PATH = "models/face_model.pt"          # ścieżka do wytrenowanego modelu
TEST_NO_OCCLUSION = "data/test_no_occlusion"
TEST_OCCLUSION = "data/test_occlusion"
TOP_K = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 512

# ========================
# 2. Transformacje
# ========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ========================
# 3. Ładowanie danych
# ========================
dataset_no = datasets.ImageFolder(TEST_NO_OCCLUSION, transform=transform)
dataset_occ = datasets.ImageFolder(TEST_OCCLUSION, transform=transform)

# Mapowanie indeks -> nazwa osoby
idx_to_class = {v: k for k, v in dataset_no.class_to_idx.items()}

# ========================
# 4. Ładowanie modelu
# ========================
from model import FaceModel   # Twój model, który zwraca embedding 512D

model = FaceModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ========================
# 5. Funkcja do liczenia embeddingów
# ========================
def get_embeddings(dataset):
    embeddings = []
    labels = []
    with torch.no_grad():
        for img, label in dataset:
            img = img.unsqueeze(0).to(DEVICE)
            emb = model(img)                 # [1, 512]
            emb = F.normalize(emb, p=2, dim=1)  # L2 normalization
            embeddings.append(emb.cpu().numpy())
            labels.append(label)
    return np.vstack(embeddings), np.array(labels)

emb_no, labels_no = get_embeddings(dataset_no)
emb_occ, labels_occ = get_embeddings(dataset_occ)

# ========================
# 6. Tworzenie indexu Faiss
# ========================
index = faiss.IndexFlatIP(EMBED_DIM)   # Inner Product = cosine similarity przy L2-normalized
faiss.normalize_L2(emb_no)
index.add(emb_no)

# ========================
# 7. Wyszukiwanie top-K
# ========================
faiss.normalize_L2(emb_occ)
D, I = index.search(emb_occ, TOP_K)

# ========================
# 8. Rank@1 i Rank@K
# ========================
correct_top1, correct_topk = 0, 0
for i in range(len(emb_occ)):
    true_label = labels_occ[i]
    if labels_no[I[i][0]] == true_label:
        correct_top1 += 1
    if true_label in labels_no[I[i]]:
        correct_topk += 1

rank1_acc = correct_top1 / len(emb_occ)
rankk_acc = correct_topk / len(emb_occ)
print(f"Rank@1: {rank1_acc:.3f}, Rank@{TOP_K}: {rankk_acc:.3f}")

# ========================
# 9. ROC-AUC dla top-1
# ========================
sims = D[:, 0]  # podobieństwo top-1
labels_binary = (labels_occ == labels_no[I[:,0]]).astype(int)
auc = roc_auc_score(labels_binary, sims)
print(f"ROC-AUC (top-1): {auc:.3f}")

# ========================
# 10. Wizualizacja t-SNE
# ========================
all_emb = np.vstack([emb_no, emb_occ])
all_labels = np.hstack([labels_no, labels_occ])
tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(all_emb)

plt.figure(figsize=(8,8))
scatter = plt.scatter(emb_2d[:,0], emb_2d[:,1], c=all_labels, cmap='tab20', alpha=0.7)
plt.legend(handles=scatter.legend_elements()[0], labels=[idx_to_class[i] for i in np.unique(all_labels)])
plt.title("t-SNE visualization of embeddings")
plt.savefig("results/tsne_plot.png", dpi=300)
plt.show()
