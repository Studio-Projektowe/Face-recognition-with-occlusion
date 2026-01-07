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
# Import modelu - UWAGA: Backbone musi zwracać tensor (B, C, H, W)
from load import load_clean_model, DEVICE

# ================= CONFIG =================
BASE_DIR = '../webface_112x112'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'test')
SAVE_PATH = 'best_model_transformer.pth'
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 2
EPOCHS = 25
OCCLUSION_PROB = 0.7
OCCLUSION_HEIGHT = 20
NUM_WORKERS = 4 if os.name != 'nt' else 0
PATIENCE = 5
NUM_VERIFY_PAIRS = 500
LR_BACKBONE = 5e-4
LR_HEAD = 0.01

# Parametry z artykułu
ALPHA = 0.4  # Loss balancing factor (Sec IV.C) [cite: 148]
TRANSFORMER_LAYERS = 6  # (Sec III) [cite: 71]
TRANSFORMER_HEADS = 8   # (Sec IV.B) [cite: 140]
# ==========================================

# ... [Funkcje ALIGNMENT i DATASET pozostają bez zmian] ...

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

# ---------- DATASETS ----------
class VerificationDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_pairs=500):
        self.transform = transform
        self.pairs = []
        identities = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        random.shuffle(identities)
        for identity in identities:
            if len(self.pairs) >= num_pairs: break
            imgs = []
            for root, _, files in os.walk(os.path.join(root_dir, identity)):
                for f in files:
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        imgs.append(os.path.join(root, f))
            if len(imgs) < 2: continue
            pair = []
            for ip in imgs[:2]:
                jp = os.path.splitext(ip)[0] + '.json'
                pair.append((ip, jp if os.path.exists(jp) else None))
            self.pairs.append({'clean': pair[0], 'query': pair[1]})

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        item = self.pairs[idx]
        def load_img(path, json_path):
            img = cv2.imread(path)
            if img is None: img = np.zeros((112,112,3), np.uint8)
            lms = get_landmarks_from_json(json_path) if json_path else None
            img = norm_crop(img, lms) if lms is not None else cv2.resize(img, (112,112))
            return img
        img_c = load_img(*item['clean'])
        img_o = load_img(*item['query'])
        cv2.rectangle(img_o, (0,42), (112,62), np.random.randint(0,256,(3,)).tolist(), -1)
        return self.transform(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)), self.transform(cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB))

class OcclusionFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir): continue
            for root, _, files in os.walk(cls_dir):
                for f in files:
                    if f.lower().endswith(('.jpg','.png','.jpeg')):
                        self.samples.append((os.path.join(root,f), self.class_to_idx[cls]))

    def __len__(self): return len(self.samples)
    def apply_occlusion(self, img):
        y = 52 + random.randint(-5,5)
        h2 = OCCLUSION_HEIGHT // 2
        cv2.rectangle(img, (0,y-h2), (112,y+h2), np.random.randint(0,256,(3,)).tolist(), -1)
        return img
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None: img = np.zeros((112,112,3), np.uint8)
        lms = get_landmarks_from_json(os.path.splitext(path)[0] + '.json')
        img = norm_crop(img, lms) if lms is not None else cv2.resize(img,(112,112))
        if random.random() < OCCLUSION_PROB: img = self.apply_occlusion(img)
        return self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), label

# ---------- MODEL COMPONENTS ----------

class ArcMarginProduct(nn.Module):
    def __init__(self, in_f, out_f, s=30.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
    def forward(self, x, y):
        cosine = nn.functional.linear(nn.functional.normalize(x), nn.functional.normalize(self.weight))
        sine = torch.sqrt(1 - cosine**2)
        phi = cosine * np.cos(self.m) - sine * np.sin(self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, y.view(-1,1), 1)
        return self.s * (one_hot * phi + (1-one_hot) * cosine)

# Implementacja "Transformer Loss" (Branch-2) [cite: 9]
class TransformerHead(nn.Module):
    def __init__(self, in_channels, num_classes, dim_feedforward=2048):
        super().__init__()
        # Input: (Batch, Sequence, Features)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels, 
            nhead=TRANSFORMER_HEADS, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=TRANSFORMER_LAYERS)
        
        # Linear layer for classification directly from transformer embedding [cite: 91]
        # Uwaga: Branch 2 używa standardowego CrossEntropy, nie ArcFace 
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x is spatial feature map: (Batch, C, H, W)
        b, c, h, w = x.shape
        
        # 1. Reshape to sequence of vectors (Contextual representation) [cite: 66]
        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x = x.view(b, c, h*w).permute(0, 2, 1)
        
        # 2. Transformer Encoder [cite: 72]
        x = self.transformer(x)
        
        # 3. Mean pooling along sequence length (Eq. 7) [cite: 78]
        # T_epsilon calculation
        x = x.mean(dim=1) 
        
        # 4. Linear Projection (Eq. 8) [cite: 89]
        out = self.fc(x)
        return out

class FaceModelWithTransformer(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        
        # Branch 1: Standard Metric Learning Head (ArcFace)
        # Nie tworzymy tutaj nowych warstw 'flatten' czy 'bn', 
        # bo użyjemy tych, które są już w self.backbone (wczytane z pliku wag)
        self.arc = ArcMarginProduct(512, num_classes)
        
        # Branch 2: Transformer Auxiliary Head [cite: 9]
        # Wejście: 512 kanałów (głębokość mapy cech)
        self.transformer_head = TransformerHead(in_channels=512, num_classes=num_classes)

    def forward(self, x, y=None):
        # 1. Pobieramy mapę przestrzenną (B, 512, 7, 7) z backbone
        # To jest wyjście potrzebne dla Transformera (Branch-2) [cite: 66]
        features_spatial = self.backbone(x) 
        
        # --- BRANCH 1 (Standard Metric Learning) ---
        # Musimy "ręcznie" dokończyć to, co usunęłaś z forward backbone'a,
        # żeby uzyskać embedding 512 zgodny z wytrenowanymi wagami.
        
        # A. Spłaszczenie (Flatten)
        features_flat = torch.flatten(features_spatial, 1)
        
        # B. Dropout (jeśli istnieje w backbone)
        if hasattr(self.backbone, 'dropout'):
            features_flat = self.backbone.dropout(features_flat)
            
        # C. Oryginalna warstwa FC (Linear) - kluczowa dla dopasowania wymiarów
        if hasattr(self.backbone, 'fc'):
            features_flat = self.backbone.fc(features_flat)
        
        # D. Oryginalna warstwa BN (Features) - to jest finalny embedding
        embedding_512 = features_flat
        if hasattr(self.backbone, 'features'):
            embedding_512 = self.backbone.features(features_flat)
            
        # Obliczamy ArcFace Loss lub zwracamy sam embedding (dla walidacji) [cite: 96]
        metric_out = self.arc(embedding_512, y) if y is not None else embedding_512
        
        # --- BRANCH 2 (Transformer) ---
        # Obliczamy tylko podczas treningu [cite: 122]
        trans_out = None
        if self.training and y is not None:
            # Transformer bierze mapę przestrzenną (spatial features)
            trans_out = self.transformer_head(features_spatial)
            
        return metric_out, trans_out

# ---------- MAIN ----------
def main():
    # WAŻNE: Upewnij się, że load_clean_model zwraca model bez ostatniej warstwy pooling/flatten
    # np. return_features=True w timm lub forward_features()
    backbone = load_clean_model() 
    
    num_classes = len(os.listdir(TRAIN_DIR))
    model = FaceModelWithTransformer(backbone, num_classes).to(DEVICE)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    train_ds = OcclusionFaceDataset(TRAIN_DIR, transform)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_ds = VerificationDataset(VAL_DIR, transform, NUM_VERIFY_PAIRS)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False)
    
    # Optymalizacja obu gałęzi
    optimizer = optim.SGD(model.parameters(), lr=LR_HEAD, momentum=0.9, weight_decay=5e-4) # [cite: 134]
    criterion = nn.CrossEntropyLoss()
    
    best_sim = -1.0
    patience_counter = 0
    
    print(f"Starting training with Transformer Auxiliary Loss (Alpha={ALPHA})...")
    
    for epoch in range(EPOCHS):
        # --- TRENING ---
        model.train()
        train_loss = 0.0
        
        # Pasek postępu
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")
        for img, lbl in pbar:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            
            # Forward zwraca dwa wyjścia
            metric_out, trans_out = model(img, lbl)
            
            # Loss 1: Metric Loss (ArcFace) [cite: 96]
            loss_metric = criterion(metric_out, lbl)
            
            # Loss 2: Transformer Loss (Standard CrossEntropy) [cite: 114, 116]
            loss_trans = criterion(trans_out, lbl)
            
            # Final combined loss (Eq. 10) 
            # L_F = (1 - alpha) * L_Metric + alpha * L_Trans
            loss = (1 - ALPHA) * loss_metric + ALPHA * loss_trans
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'L_Main': loss_metric.item(), 'L_Trans': loss_trans.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- WALIDACJA ---
        model.eval()
        sims = []
        with torch.no_grad():
            for a, b in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                a, b = a.to(DEVICE), b.to(DEVICE)
                
                # Do inferencji używamy TYLKO gałęzi standardowej (Branch-1) [cite: 121]
                # Branch-2 (Transformer) służy tylko do aktualizacji wag podczas treningu
                ea, _ = model(a) 
                eb, _ = model(b)
                
                ea = nn.functional.normalize(ea, dim=1)
                eb = nn.functional.normalize(eb, dim=1)
                sims.extend((ea*eb).sum(1).cpu().numpy())
        
        val_sim = np.mean(sims)
        print(f"Epoch {epoch+1} | Combined Loss: {avg_train_loss:.4f} | Ver Sim: {val_sim:.4f}")
        
        if val_sim > best_sim:
            best_sim = val_sim
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"--> Saved best model with sim: {best_sim:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered!")
                break

if __name__ == "__main__":
    main()