import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
import os
import json
import random
from tqdm import tqdm
from skimage import transform as trans

# Import modelu
from load import load_clean_model, DEVICE

# ================= CONFIG =================
BASE_DIR = '../webface_112x112'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'test')
SAVE_PATH = 'baseline.pth'

BATCH_SIZE = 32
EPOCHS = 25
OCCLUSION_PROB = 0.7
OCCLUSION_HEIGHT = 20
NUM_WORKERS = 4 if os.name != 'nt' else 0
PATIENCE = 5
NUM_VERIFY_PAIRS = 500

LR_HEAD = 0.01
# ==========================================

# ---------- ALIGNMENT ----------
arcface_dst = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def estimate_norm(lmk):
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, arcface_dst)
    return tform.params[0:2, :]

def norm_crop(img, landmark):
    M = estimate_norm(landmark)
    return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)

def get_landmarks_from_json(json_path):
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        lms = data.get('landmarks', data)
        if isinstance(lms, dict):
            return np.array([
                lms['right_eye'],
                lms['left_eye'],
                lms['nose'],
                lms['mouth_right'],
                lms['mouth_left']
            ], dtype=np.float32)
        return np.array(lms, dtype=np.float32)
    except:
        return None

# ---------- VERIFICATION DATASET ----------
class VerificationDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_pairs=500):
        self.transform = transform
        self.pairs = []

        identities = [d for d in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, d))]
        random.shuffle(identities)

        for identity in identities:
            if len(self.pairs) >= num_pairs:
                break

            imgs = []
            for root, _, files in os.walk(os.path.join(root_dir, identity)):
                for f in files:
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        imgs.append(os.path.join(root, f))

            if len(imgs) < 2:
                continue

            pair = []
            for ip in imgs[:2]:
                jp = os.path.splitext(ip)[0] + '.json'
                pair.append((ip, jp if os.path.exists(jp) else None))

            self.pairs.append({'clean': pair[0], 'query': pair[1]})

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]

        def load_img(path, json_path):
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((112,112,3), np.uint8)
            lms = get_landmarks_from_json(json_path) if json_path else None
            img = norm_crop(img, lms) if lms is not None else cv2.resize(img, (112,112))
            return img

        img_c = load_img(*item['clean'])
        img_o = load_img(*item['query'])

        cv2.rectangle(img_o, (0,42), (112,62),
                      np.random.randint(0,256,(3,)).tolist(), -1)

        img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
        img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)

        return self.transform(img_c), self.transform(img_o)

# ---------- TRAIN DATASET ----------
class OcclusionFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for root, _, files in os.walk(cls_dir):
                for f in files:
                    if f.lower().endswith(('.jpg','.png','.jpeg')):
                        self.samples.append(
                            (os.path.join(root,f), self.class_to_idx[cls])
                        )

    def __len__(self):
        return len(self.samples)

    def apply_occlusion(self, img):
        y = 52 + random.randint(-5,5)
        h2 = OCCLUSION_HEIGHT // 2
        cv2.rectangle(img, (0,y-h2), (112,y+h2),
                      np.random.randint(0,256,(3,)).tolist(), -1)
        return img

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((112,112,3), np.uint8)

        lms = get_landmarks_from_json(os.path.splitext(path)[0] + '.json')
        img = norm_crop(img, lms) if lms is not None else cv2.resize(img,(112,112))

        if random.random() < OCCLUSION_PROB:
            img = self.apply_occlusion(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), label

# ---------- MODEL ----------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_f, out_f, s=30.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, x, y):
        cosine = nn.functional.linear(
            nn.functional.normalize(x),
            nn.functional.normalize(self.weight)
        )
        sine = torch.sqrt(1 - cosine**2)
        phi = cosine * np.cos(self.m) - sine * np.sin(self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, y.view(-1,1), 1)
        return self.s * (one_hot * phi + (1-one_hot) * cosine)

class FaceModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.arc = ArcMarginProduct(512, num_classes)

    def forward(self, x, y=None):
        f = self.backbone(x).view(x.size(0), -1)
        return self.arc(f,y) if y is not None else f

# ---------- MAIN ----------
def main():
    backbone = load_clean_model()
    model = FaceModel(backbone, len(os.listdir(TRAIN_DIR))).to(DEVICE)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_ds = OcclusionFaceDataset(TRAIN_DIR, transform)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_ds = VerificationDataset(VAL_DIR, transform, NUM_VERIFY_PAIRS)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False)

    optimizer = optim.SGD(model.parameters(), lr=LR_HEAD, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Zmienne do śledzenia najlepszego modelu
    best_sim = -1.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        # --- TRENING ---
        model.train()
        train_loss = 0.0
        for img, lbl in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            output = model(img, lbl)
            loss = criterion(output, lbl)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # --- WALIDACJA ---
        model.eval()
        sims = []
        with torch.no_grad():
            for a, b in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                a, b = a.to(DEVICE), b.to(DEVICE)
                ea, eb = model(a), model(b)
                ea = nn.functional.normalize(ea, dim=1)
                eb = nn.functional.normalize(eb, dim=1)
                sims.extend((ea*eb).sum(1).cpu().numpy())
        
        val_sim = np.mean(sims)
        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Ver Sim: {val_sim:.4f}")

        # --- ZAPISYWANIE NAJLEPSZEGO MODELU I EARLY STOPPING ---
        if val_sim > best_sim:
            best_sim = val_sim
            patience_counter = 0
            # Zapisujemy cały state_dict (backbone + head)
            torch.save(model.state_dict(), SAVE_PATH)
            # Opcjonalnie: zapisz sam backbone, jeśli tylko jego użyjesz na produkcji:
            # torch.save(model.backbone.state_dict(), 'backbone_only.pth')
            print(f"--> Saved best model with sim: {best_sim:.4f}")
        else:
            patience_counter += 1
            print(f"--> No improvement. Patience: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print("Early stopping triggered!")
                break

if __name__ == "__main__":
    main()