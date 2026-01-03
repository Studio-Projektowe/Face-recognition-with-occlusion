import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
import os
import glob
import json
import random
from tqdm import tqdm
import cv2

from load_and_test import load_clean_model, DEVICE
import face_align

BASE_DIR = '../../../../webface_112x112'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
DEBUG_DIR = 'training_photos_stage2'

PRETRAINED_PATH = 'aligned_pretrained_v1.pth' 

BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 2
EPOCHS = 15          
NUM_WORKERS = 4
OCCLUSION_PROB = 0.7
OCCLUSION_HEIGHT = 20
PATIENCE = 3

LR_BACKBONE = 1e-5
LR_HEAD = 1e-3
AUX_LOSS_WEIGHT = 0.1

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
        # UWAGA: Tu wyłączamy easy_margin na starcie (False), bo to Stage 2
        self.arcface = ArcMarginProduct(512, num_classes, easy_margin=False)
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
            print(f'EarlyStopping: {self.counter}/{self.patience} (Val Loss nie spada)')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(f'Val Loss spadł ({self.best_loss:.6f} --> {val_loss:.6f}). Zapis modelu...')
        torch.save(model.backbone.state_dict(), self.path)

class OcclusionFaceDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.root_dirs = root_dirs
        self.transform = transform
        self.image_paths = []
        
        print(f"Skanowanie folderów: {root_dirs}...")
        for d in root_dirs:
            if os.path.exists(d):
                files = glob.glob(os.path.join(d, "*", "*", "*.jpg"))
                self.image_paths.extend(files)
            else:
                print(f"Uwaga: Folder {d} nie istnieje!")

        self.classes = sorted(list(set([path.split(os.sep)[-3] for path in self.image_paths])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Znaleziono łącznie {len(self.image_paths)} zdjęć w {len(self.classes)} tożsamościach.")

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


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    print(f"Inicjalizacja architektury IResNet...")
    backbone = load_clean_model() 
    
    print(f"Wczytywanie Twoich wag z: {PRETRAINED_PATH}...")
    if os.path.exists(PRETRAINED_PATH):
        state_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "").replace("backbone.", "")
            new_state_dict[name] = v
            
        msg = backbone.load_state_dict(new_state_dict, strict=False)
        print(f"Wagi załadowane! Status: {msg}")
    else:
        print(f"BŁĄD: Nie znaleziono pliku {PRETRAINED_PATH}!")
        return

    for param in backbone.parameters():
        param.requires_grad = True
    print("Wszystkie warstwy odmrożone (Ready for Fine-Tuning).")
    
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
    
    model = FaceModelWithAux(backbone, len(full_dataset.classes))
    model.to(DEVICE)
    
    optimizer = optim.SGD([
        {'params': model.backbone.parameters(), 'lr': LR_BACKBONE},
        {'params': model.arcface.parameters(), 'lr': LR_HEAD},
        {'params': model.aux_head.parameters(), 'lr': LR_HEAD}
    ], momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    early_stopping = EarlyStopping(patience=PATIENCE, path='aligned_pretrained_v2.pth') 
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_aux = nn.MSELoss()
    
    print(f"\nSTART ETAPU 2. Epochs: {EPOCHS}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        loop = tqdm(train_loader, desc=f"Epoka {epoch+1}")
        for i, (img, lbl, msk) in enumerate(loop):
            img, lbl, msk = img.to(DEVICE), lbl.to(DEVICE), msk.to(DEVICE)
            
            if i == 0: vutils.save_image(img[:20]*0.5+0.5, f"{DEBUG_DIR}/ep{epoch}_s2.jpg", nrow=5)

            logits, m_pred = model(img, lbl)
            # W Stage 2 waga AUX może być większa (np. 0.1 lub 0.5), żeby mocniej karać za błędy maski
            loss = criterion_cls(logits, lbl) + AUX_LOSS_WEIGHT * criterion_aux(m_pred, msk)

            if torch.isnan(loss):
                print("NaN loss!")
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
            
        model.eval()
        val_total_loss = 0.0
        
        with torch.no_grad():
            for img, lbl, msk in val_loader:
                img, lbl, msk = img.to(DEVICE), lbl.to(DEVICE), msk.to(DEVICE)
                logits, m_pred = model(img, lbl)
                v_loss_cls = criterion_cls(logits, lbl)
                v_loss_aux = criterion_aux(m_pred, msk)
                val_total_loss += (v_loss_cls +AUX_LOSS_WEIGHT * v_loss_aux).item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_total_loss / len(val_loader)
        
        print(f"Stage 2 Epoka {epoch+1}: Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        
        scheduler.step(avg_val)
        early_stopping(avg_val, model)
        if early_stopping.early_stop: break

if __name__ == "__main__":
    main()