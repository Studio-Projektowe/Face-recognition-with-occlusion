import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
import cv2
import os
import json
import glob
import random
import csv
from datetime import datetime
from tqdm import tqdm

# Twoje importy
from load_and_test import load_clean_model, DEVICE
import face_align

# --- KONFIGURACJA ---
BASE_DIR = '../../../../webface_112x112'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
DEBUG_DIR = 'training_photos_stage2'
METRICS_DIR = 'metrics'
METRICS_CSV = None  # Will be set with timestamp at runtime

# ŚCIEŻKA DO TWOJEGO POPRZEDNIEGO MODELU
PRETRAINED_PATH = 'best_model_stage2.pth' 

BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 2
EPOCHS = 20           # Jedziemy dalej!
OCCLUSION_PROB = 0.7
OCCLUSION_HEIGHT = 20
NUM_WORKERS = 4
PATIENCE = 5
RANK_K_SAMPLES = 1000 # Ile próbek z walidacji użyć do liczenia Rank-1 (żeby nie trwało wieki)

# --- LEARNING RATES (Stage 2 - trochę mniejszy start dla Backbone) ---
LR_BACKBONE = 1e-5   # Było 5e-4, zmniejszamy, żeby nie "szarpać" wag
LR_HEAD = 0.001       # Głowa ArcFace uczy się od nowa (bo nie była zapisana)
AUX_LOSS_WEIGHT = 0.01

# --- 1. ZAMRAŻANIE (Możemy odmrozić więcej lub zostawić tak samo) ---
def freeze_layers(model):
    print("\n❄️ Konfiguracja zamrażania warstw...")
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        param.requires_grad = False 
        
        # Odmrażamy kluczowe bloki (możesz tu dodać layer2 jeśli chcesz agresywniej)
        if (
            'se' in name or       
            'layer1' in name or   
            'layer2' in name or   
            'layer3' in name or   
            'layer4' in name or   
            'bn' in name or       
            'fc' in name or       
            'prelu' in name       
        ):
            param.requires_grad = True
            
        if param.requires_grad:
            trainable += 1
        else:
            frozen += 1
            
    print(f"✅ Zamrożono: {frozen} parametrów. Do treningu: {trainable} parametrów.")

# --- NOWOŚĆ: OBLICZANIE RANK-1 ACCURACY ---
def calculate_rank_k(model, val_loader, num_samples=1000):
    model.eval()
    
    embeddings = []
    labels = []
    
    # 1. Pobieranie embeddingów
    with torch.no_grad():
        samples_collected = 0
        for img, lbl, _ in val_loader:
            if samples_collected >= num_samples:
                break
                
            img = img.to(DEVICE)
            # Pobieramy same cechy (features), bez ArcFace Head
            features, _ = model(img, labels=None)
            
            embeddings.append(features.cpu())
            labels.append(lbl)
            samples_collected += len(lbl)
    
    if len(embeddings) == 0: return 0.0, 0.0
    
    # Złączamy wszystko w jeden tensor
    embeddings = torch.cat(embeddings, dim=0)[:num_samples]
    labels = torch.cat(labels, dim=0)[:num_samples]
    
    # 2. Normalizacja (Face Recognition opiera się na cosinusach)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # 3. Macierz podobieństwa (każdy z każdym)
    # [N, 512] x [512, N] = [N, N]
    similarity = torch.mm(embeddings, embeddings.t())
    
    # Wykluczamy samo-dopasowanie (przekątna = -inf)
    similarity.fill_diagonal_(-float('inf'))
    
    # 4. Znajdź Top-K
    _, top3_indices = similarity.topk(3, dim=1)
    
    rank1_correct = 0
    rank3_correct = 0
    total = len(labels)
    
    for i in range(total):
        query_label = labels[i].item()
        
        # Sprawdzamy Top 1
        top1_idx = top3_indices[i, 0]
        if labels[top1_idx].item() == query_label:
            rank1_correct += 1
            
        # Sprawdzamy Top 3
        top3_found = False
        for k in range(3):
            idx = top3_indices[i, k]
            if labels[idx].item() == query_label:
                top3_found = True
        if top3_found:
            rank3_correct += 1
            
    rank1_acc = (rank1_correct / total) * 100
    rank3_acc = (rank3_correct / total) * 100
    
    return rank1_acc, rank3_acc

# --- NOWOŚĆ: ZAPIS DO CSV ---
def save_metrics_to_csv(filepath, epoch, train_loss, val_loss, rank1, rank3):
    file_exists = os.path.exists(filepath)
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Nagłówek
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'rank1_acc', 'rank3_acc'])
        writer.writerow([epoch, f'{train_loss:.4f}', f'{val_loss:.4f}', f'{rank1:.2f}', f'{rank3:.2f}'])

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='best_model_stage2.pth'): # Zmiana nazwy pliku
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'   ⚠️ EarlyStopping: {self.counter}/{self.patience} (Val Loss nie spada)')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(f'   ✅ Val Loss spadł ({self.best_loss:.6f} --> {val_loss:.6f}). Zapis modelu...')
        torch.save(model.backbone.state_dict(), self.path)

# --- KLASY MODELU (BEZ ZMIAN) ---
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, input, label):
        cosine = torch.nn.functional.linear(
            torch.nn.functional.normalize(input), 
            torch.nn.functional.normalize(self.weight)
        )
        eps = 1e-7
        cosine = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output *= self.s
        return output

class FaceModelWithAux(nn.Module):
    def __init__(self, backbone, num_classes):
        super(FaceModelWithAux, self).__init__()
        self.backbone = backbone
        self.arcface = ArcMarginProduct(512, num_classes, easy_margin=True)
        self.aux_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 7*7), nn.Sigmoid()
        )

    def forward(self, x, labels=None):
        features = self.backbone(x)
        features_flat = features.view(features.size(0), -1)
        mask_pred = self.aux_head(features_flat)
        if labels is not None:
            arcface_out = self.arcface(features_flat, labels)
            return arcface_out, mask_pred
        return features_flat, mask_pred

# --- DATASET (BEZ ZMIAN) ---
class OcclusionFaceDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.root_dirs = root_dirs
        self.transform = transform
        self.image_paths = []
        print(f"🔄 Skanowanie folderów: {root_dirs}...")
        for d in root_dirs:
            if os.path.exists(d):
                files = glob.glob(os.path.join(d, "*", "*", "*.jpg"))
                self.image_paths.extend(files)
            else:
                print(f"⚠️ Uwaga: Folder {d} nie istnieje!")
        self.classes = sorted(list(set([path.split(os.sep)[-3] for path in self.image_paths])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"✅ Znaleziono łącznie {len(self.image_paths)} zdjęć w {len(self.classes)} tożsamościach.")

    def __len__(self): return len(self.image_paths)

    def align_face(self, img, landmarks):
        try:
            kps = np.array([
                landmarks['right_eye'], landmarks['left_eye'], landmarks['nose'],
                landmarks['mouth_right'], landmarks['mouth_left']
            ], dtype=np.float32)
            return face_align.norm_crop(img, landmark=kps, image_size=112)
        except: return cv2.resize(img, (112, 112))

    def apply_random_occlusion(self, img):
        h, w, _ = img.shape
        mask = np.zeros((h, w), dtype=np.float32)
        center_y = 52 + random.randint(-5, 5)
        bar_h_half = int(OCCLUSION_HEIGHT / 2)
        y1 = max(0, center_y - bar_h_half)
        y2 = min(h, center_y + bar_h_half)
        cv2.rectangle(img, (0, y1), (w, y2), np.random.randint(0,256,(3,)).tolist(), -1)
        cv2.rectangle(mask, (0, y1), (w, y2), 1.0, -1)
        return img, mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        class_name = img_path.split(os.sep)[-3]
        label = self.class_to_idx.get(class_name, -1)
        img = cv2.imread(img_path)
        if img is None: img = np.zeros((112, 112, 3), dtype=np.uint8)
        
        json_path = img_path.replace('.jpg', '.json')
        final_img = cv2.resize(img, (112, 112))
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    final_img = self.align_face(img, data.get('landmarks', data))
            except: pass

        mask_target = np.zeros((7, 7), dtype=np.float32)
        if random.random() < OCCLUSION_PROB:
            final_img, full_mask = self.apply_random_occlusion(final_img)
            mask_target = cv2.resize(full_mask, (7, 7), interpolation=cv2.INTER_NEAREST)

        img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = transforms.ToTensor()(img_rgb)
        return img_tensor, label, torch.from_numpy(mask_target).flatten()

# --- MAIN ---
def main():
    global METRICS_CSV
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # Tworzymy nazwę pliku CSV z datą
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    METRICS_CSV = os.path.join(METRICS_DIR, f'training_metrics_{timestamp}.csv')
    print(f"📊 Metryki będą zapisywane do: {METRICS_CSV}")
    
    # 1. Ładowanie pustej architektury
    backbone = load_clean_model()
    
    # 2. Wczytanie wag z Twojego poprzedniego treningu
    print(f"🔄 Wczytuję wagi z: {PRETRAINED_PATH}...")
    if os.path.exists(PRETRAINED_PATH):
        backbone.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
        print("✅ Wagi wczytane pomyślnie!")
    else:
        print("❌ BŁĄD: Nie znaleziono pliku wag! Upewnij się, że best_model_merged.pth jest w folderze.")
        return

    freeze_layers(backbone)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    full_dataset = OcclusionFaceDataset(root_dirs=[TRAIN_DIR, VAL_DIR], transform=transform)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    model = FaceModelWithAux(backbone, len(full_dataset.classes)).to(DEVICE)
    
    # Optimizer (Stage 2)
    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
            
    optimizer = optim.SGD([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params, 'lr': LR_HEAD}
    ], momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    early_stopping = EarlyStopping(patience=PATIENCE) # Zapisuje do best_model_stage2.pth
    criterion_cls = nn.CrossEntropyLoss()
    criterion_aux = nn.MSELoss()
    
    print(f"\n🚀 START STAGE 2: Backbone LR={LR_BACKBONE}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        loop = tqdm(train_loader, desc=f"Epoka {epoch+1}")
        for i, (img, lbl, msk) in enumerate(loop):
            img, lbl, msk = img.to(DEVICE), lbl.to(DEVICE), msk.to(DEVICE)
            
            logits, m_pred = model(img, lbl)
            loss = criterion_cls(logits, lbl) + AUX_LOSS_WEIGHT * criterion_aux(m_pred, msk)
            
            if torch.isnan(loss):
                print(f"❌ NaN detected!")
                optimizer.zero_grad()
                continue
            
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * GRAD_ACCUM_STEPS
            loop.set_postfix({'loss': loss.item() * GRAD_ACCUM_STEPS})
            
        # --- Walidacja ---
        model.eval()
        val_total_loss = 0.0
        
        with torch.no_grad():
            for img, lbl, msk in val_loader:
                img, lbl, msk = img.to(DEVICE), lbl.to(DEVICE), msk.to(DEVICE)
                logits, m_pred = model(img, lbl)
                v_loss_cls = criterion_cls(logits, lbl)
                v_loss_aux = criterion_aux(m_pred, msk)
                val_total_loss += (v_loss_cls + AUX_LOSS_WEIGHT * v_loss_aux).item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_total_loss / len(val_loader)
        
        # --- OBLICZANIE RANK-1 (UWU) ---
        # Używamy próbki danych, żeby nie trwało to wieki
        rank1_acc, rank3_acc = calculate_rank_k(model, val_loader, num_samples=RANK_K_SAMPLES)
        
        print(f"📊 Stage 2 Epoka {epoch+1}: Train: {avg_train:.4f} | Val: {avg_val:.4f} | Rank-1: {rank1_acc:.2f}% | Rank-3: {rank3_acc:.2f}%")
        
        # Zapis do CSV
        save_metrics_to_csv(METRICS_CSV, epoch+1, avg_train, avg_val, rank1_acc, rank3_acc)
        
        scheduler.step(avg_val)
        early_stopping(avg_val, model)
        
        if early_stopping.early_stop:
            print("🛑 Early Stopping zadziałał.")
            break

    print(f"✅ Koniec treningu. Wyniki zapisane w: {METRICS_CSV}")

if __name__ == "__main__":
    main()