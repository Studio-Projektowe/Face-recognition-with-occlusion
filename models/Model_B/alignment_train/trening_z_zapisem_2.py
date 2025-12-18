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

from load_and_test import load_clean_model, DEVICE
import face_align


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)      # Channel attention
        out = out * self.sa(out)  # Spatial attention
        return out


class FrontEndCBAM(nn.Module):
    def __init__(self, backbone, cbam_channels=64):
        super(FrontEndCBAM, self).__init__()
        self.backbone = backbone
        # CBAM po pierwszej konwolucji (64 kanały w iResNet50)
        self.cbam_front = CBAM(cbam_channels, ratio=16, kernel_size=7)
        
    def forward(self, x):
        # Pierwsza część backbone: conv1 -> bn1 -> prelu
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.prelu(x)
        
        # === CBAM NA POCZĄTKU ===
        x = self.cbam_front(x)
        
        # Reszta backbone
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.bn2(x)
        x = torch.flatten(x, 1)
        x = self.backbone.dropout(x)
        x = self.backbone.fc(x)
        x = self.backbone.features(x)
        return x
    
    # Przekazywanie named_parameters dla freeze_layers
    def named_parameters(self, prefix='', recurse=True):
        # Parametry CBAM
        for name, param in self.cbam_front.named_parameters(prefix='cbam_front', recurse=recurse):
            yield name, param
        # Parametry backbone
        for name, param in self.backbone.named_parameters(prefix='backbone', recurse=recurse):
            yield name, param
    
    def state_dict(self, *args, **kwargs):
        # Zwracamy tylko backbone state_dict dla kompatybilności z zapisem
        return self.backbone.state_dict(*args, **kwargs)

# --- KONFIGURACJA ---
BASE_DIR = '../../../../webface_112x112'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
DEBUG_DIR = 'training_photos'
METRICS_DIR = 'metrics'
METRICS_CSV = None  # Will be set with timestamp at runtime

BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 2   # Symulacja Batch=64
EPOCHS = 25
OCCLUSION_PROB = 0.7
OCCLUSION_HEIGHT = 20
NUM_WORKERS = 4
PATIENCE = 5
RANK_K_SAMPLES = 500  # Number of samples to use for Rank-K calculation (for speed)

# --- CBAM CONFIGURATION ---
USE_CBAM = True        # Włącz/wyłącz CBAM (True = włączony)
CBAM_RATIO = 16        # Ratio dla Channel Attention (mniejszy = więcej parametrów)
CBAM_KERNEL = 7        # Kernel size dla Spatial Attention (7 lub 3)

# --- LEARNING RATES (POWRÓT DO MOCY) ---
LR_BACKBONE = 5e-4   # 0.0005 (Kompromis między szybkością a stabilnością)
LR_HEAD = 0.01       # ArcFace wymaga dużego LR na starcie
AUX_LOSS_WEIGHT = 0.01

# --- 1. ZAMRAŻANIE WARSTW ---
def freeze_layers(model):
    print("\nConfiguring layer freezing...")
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        param.requires_grad = False 
        
        # Unfreeze only key blocks
        if (
            'cbam' in name or     # CBAM - ALWAYS TRAINED (new module)
            'se' in name or       
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
            
    print(f"Frozen: {frozen} params. Trainable: {trainable} params.")


# --- RANK-K CALCULATION ---
def calculate_rank_k(model, val_loader, num_samples=500):
    model.eval()
    
    embeddings = []
    labels = []
    
    with torch.no_grad():
        samples_collected = 0
        for img, lbl, _ in val_loader:
            if samples_collected >= num_samples:
                break
                
            img = img.to(DEVICE)
            # Get embeddings (without ArcFace head)
            features, _ = model(img, labels=None)
            
            embeddings.append(features.cpu())
            labels.append(lbl)
            samples_collected += len(lbl)
    
    if len(embeddings) == 0:
        return 0.0, 0.0
    
    embeddings = torch.cat(embeddings, dim=0)[:num_samples]
    labels = torch.cat(labels, dim=0)[:num_samples]
    
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    similarity = torch.mm(embeddings, embeddings.t())
    
    similarity.fill_diagonal_(-float('inf'))
    
    _, top3_indices = similarity.topk(3, dim=1)
    
    rank1_correct = 0
    rank3_correct = 0
    total = len(labels)
    
    for i in range(total):
        query_label = labels[i].item()
        top1_label = labels[top3_indices[i, 0]].item()
        top3_labels = [labels[top3_indices[i, j]].item() for j in range(3)]
        
        if query_label == top1_label:
            rank1_correct += 1
        
        if query_label in top3_labels:
            rank3_correct += 1
    
    rank1_acc = rank1_correct / total * 100
    rank3_acc = rank3_correct / total * 100
    
    return rank1_acc, rank3_acc


def save_metrics_to_csv(filepath, epoch, train_loss, val_loss, rank1, rank3):
    file_exists = os.path.exists(filepath)
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'rank1_acc', 'rank3_acc'])
        writer.writerow([epoch, f'{train_loss:.4f}', f'{val_loss:.4f}', f'{rank1:.2f}', f'{rank3:.2f}'])

# --- 2. EARLY STOPPING ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='best_model_merged.pth'):
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
            print(f'   EarlyStopping: {self.counter}/{self.patience} (Val Loss not improving)')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(f'   Val Loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.backbone.state_dict(), self.path)

# --- 3. MODEL ARCFACE (Z FIXEM NA NaN) ---
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
        # Normalizacja wektorów i wag
        cosine = torch.nn.functional.linear(
            torch.nn.functional.normalize(input), 
            torch.nn.functional.normalize(self.weight)
        )
        
        # --- FIX: ZABEZPIECZENIE PRZED NaN ---
        eps = 1e-7
        cosine = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)
        # -------------------------------------

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
        # KLUCZOWE: easy_margin=True dla stabilnego startu
        self.arcface = ArcMarginProduct(512, num_classes, easy_margin=True)
        self.aux_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7*7),
            nn.Sigmoid()
        )

    def forward(self, x, labels=None):
        features = self.backbone(x)
        features_flat = features.view(features.size(0), -1)
        
        mask_pred = self.aux_head(features_flat)
        
        if labels is not None:
            arcface_out = self.arcface(features_flat, labels)
            return arcface_out, mask_pred
        return features_flat, mask_pred

# --- 4. DATASET (POPRAWIONE INDEKSOWANIE [-3]) ---
class OcclusionFaceDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.root_dirs = root_dirs
        self.transform = transform
        self.image_paths = []
        
        print(f"Scanning folders: {root_dirs}...")
        for d in root_dirs:
            if os.path.exists(d):
                # Deep search: id/session/image.jpg
                files = glob.glob(os.path.join(d, "*", "*", "*.jpg"))
                self.image_paths.extend(files)
            else:
                print(f"Warning: Folder {d} does not exist!")

        # Extract labels from index [-3] (ID folder)
        self.classes = sorted(list(set([path.split(os.sep)[-3] for path in self.image_paths])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Found {len(self.image_paths)} images in {len(self.classes)} identities.")

    def __len__(self): return len(self.image_paths)

    def align_face(self, img, landmarks):
        try:
            kps = np.array([
                landmarks['right_eye'], landmarks['left_eye'], landmarks['nose'],
                landmarks['mouth_right'], landmarks['mouth_left']
            ], dtype=np.float32)
            return face_align.norm_crop(img, landmark=kps, image_size=112)
        except: 
            return cv2.resize(img, (112, 112))

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
        
        # Wyciągamy ID (folder [-3])
        class_name = img_path.split(os.sep)[-3]
        label = self.class_to_idx.get(class_name, -1)
        
        img = cv2.imread(img_path)
        if img is None: img = np.zeros((112, 112, 3), dtype=np.uint8)
        
        # ALIGNMENT: Sprawdzamy czy jest JSON z landmarkami
        json_path = img_path.replace('.jpg', '.json')
        final_img = cv2.resize(img, (112, 112)) # Domyślny resize
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # Używamy face_align do wyprostowania twarzy
                    final_img = self.align_face(img, data.get('landmarks', data))
            except: pass

        # OKLUZJA
        mask_target = np.zeros((7, 7), dtype=np.float32)
        if random.random() < OCCLUSION_PROB:
            final_img, full_mask = self.apply_random_occlusion(final_img)
            mask_target = cv2.resize(full_mask, (7, 7), interpolation=cv2.INTER_NEAREST)

        # TRANSFORMACJA
        img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = transforms.ToTensor()(img_rgb)
            
        return img_tensor, label, torch.from_numpy(mask_target).flatten()

# --- 5. MAIN ---
def main():
    global METRICS_CSV
    
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # Generate timestamped metrics filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    METRICS_CSV = os.path.join(METRICS_DIR, f'training_metrics_{timestamp}.csv')
    print(f"Metrics will be saved to: {METRICS_CSV}")
    
    backbone_raw = load_clean_model()
    
    if USE_CBAM:
        backbone = FrontEndCBAM(backbone_raw, cbam_channels=64)
    else:
        backbone = backbone_raw
    
    # Zamrażanie warstw (CBAM będzie zawsze trenowany)
    freeze_layers(backbone)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 1. Dataset (Jeden duży, potem dzielony)
    full_dataset = OcclusionFaceDataset(root_dirs=[TRAIN_DIR, VAL_DIR], transform=transform)
    
    # Random 90/10 split (ensures val has same IDs as train)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    print(f"Splitting dataset: {train_size} (Train) + {val_size} (Val)")
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    model = FaceModelWithAux(backbone, len(full_dataset.classes)).to(DEVICE)
    
    # 2. OPTIMIZER Z GRUPAMI (Klucz do sukcesu)
    # CBAM params będą w 'cbam' group razem z backbone (ale trenowane)
    backbone_params = [p for n, p in model.named_parameters() if ('backbone' in n or 'cbam' in n) and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if ('backbone' not in n and 'cbam' not in n) and p.requires_grad]
            
    optimizer = optim.SGD([
        {'params': backbone_params, 'lr': LR_BACKBONE}, # 5e-4
        {'params': head_params, 'lr': LR_HEAD}          # 0.01
    ], momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    early_stopping = EarlyStopping(patience=PATIENCE)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_aux = nn.MSELoss()
    
    print(f"\nSTART: CBAM={'ON' if USE_CBAM else 'OFF'}, Backbone LR={LR_BACKBONE}, Head LR={LR_HEAD}, Batch={BATCH_SIZE}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for i, (img, lbl, msk) in enumerate(loop):
            img, lbl, msk = img.to(DEVICE), lbl.to(DEVICE), msk.to(DEVICE)
            
            # Podgląd zdjęć
            if i == 0: 
                vutils.save_image(img[:20]*0.5+0.5, f"{DEBUG_DIR}/ep{epoch}.jpg", nrow=5)
            
            logits, m_pred = model(img, lbl)
            loss = criterion_cls(logits, lbl) + AUX_LOSS_WEIGHT * criterion_aux(m_pred, msk)
            
            # NaN Check
            if torch.isnan(loss):
                print(f"NaN detected at epoch {epoch}, step {i}! Skipping step.")
                optimizer.zero_grad()
                continue
            
            # Gradient Accumulation
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()
            
            current_loss = loss.item() * GRAD_ACCUM_STEPS
            train_loss += current_loss
            loop.set_postfix({'loss': current_loss})
            
        # --- Walidacja ---
        model.eval()
        val_total_loss = 0.0
        
        with torch.no_grad():
            for img, lbl, msk in val_loader:
                img, lbl, msk = img.to(DEVICE), lbl.to(DEVICE), msk.to(DEVICE)
                
                logits, m_pred = model(img, lbl)
                
                # WAŻNE: Mierzymy loss ArcFace + Aux
                v_loss_cls = criterion_cls(logits, lbl)
                v_loss_aux = criterion_aux(m_pred, msk)
                
                val_total_loss += (v_loss_cls + AUX_LOSS_WEIGHT * v_loss_aux).item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_total_loss / len(val_loader)
        
        # --- Calculate Rank-K ---
        rank1_acc, rank3_acc = calculate_rank_k(model, val_loader, num_samples=RANK_K_SAMPLES)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Rank-1: {rank1_acc:.2f}% | Rank-3: {rank3_acc:.2f}%")
        
        # Save metrics to CSV
        save_metrics_to_csv(METRICS_CSV, epoch+1, avg_train, avg_val, rank1_acc, rank3_acc)
        
        scheduler.step(avg_val)
        early_stopping(avg_val, model)
        
        if early_stopping.early_stop:
            print("Early Stopping triggered.")
            break

    print(f"Done. Metrics saved to {METRICS_CSV}")

if __name__ == "__main__":
    main()