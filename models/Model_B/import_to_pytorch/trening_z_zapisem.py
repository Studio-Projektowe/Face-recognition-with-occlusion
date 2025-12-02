import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils  # Do zapisywania siatki zdjęć
import numpy as np
import cv2
import os
import json
import glob
import random
from tqdm import tqdm

# Importujemy Twój loader i architekturę
from models.Model_B.modification.load_and_test import load_clean_model, DEVICE

# --- KONFIGURACJA ---
BASE_DIR = 'webface_112x112'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
DEBUG_DIR = 'training_photos'  # Folder na podgląd zdjęć

BATCH_SIZE = 32
LR_HEAD = 0.01
LR_BACKBONE = 0.001
EPOCHS = 25
OCCLUSION_PROB = 0.7
OCCLUSION_HEIGHT = 20
NUM_WORKERS = 4
PATIENCE = 5

# --- KLASA EARLY STOPPING ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='best_model_se_occlusion.pth'):
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
            print(f'   ⚠️ EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(f'   ✅ Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.backbone.state_dict(), self.path)

# --- 1. MODUŁ ARCFACE (Tylko do treningu) ---
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
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(input), torch.nn.functional.normalize(self.weight))
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

# --- 2. DATASET Z OKLUZJĄ ---
class OcclusionFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_validation=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_validation = is_validation
        
        print(f"Skanowanie datasetu: {root_dir}...")
        self.image_paths = glob.glob(os.path.join(root_dir, "*", "*", "*.jpg"))
        
        if not self.image_paths:
            print(f"⚠️ Nie znaleziono plików .jpg w {root_dir} (sprawdź strukturę katalogów!)")
        
        self.classes = sorted(list(set([path.split(os.sep)[-3] for path in self.image_paths])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Znaleziono {len(self.image_paths)} zdjęć w {len(self.classes)} tożsamościach.")

    def __len__(self):
        return len(self.image_paths)

    def apply_random_occlusion(self, img, landmarks):
        h, w, _ = img.shape
        mask = np.zeros((h, w), dtype=np.float32)
        try:
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            center_y = int((left_eye[1] + right_eye[1]) / 2)
            bar_h_half = int(OCCLUSION_HEIGHT / 2)
            y1 = max(0, center_y - bar_h_half)
            y2 = min(h, center_y + bar_h_half)
            x1 = 0
            x2 = w
            # Losowy kolor paska
            color = np.random.randint(0, 256, (3,), dtype=int).tolist()
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            # Maska binarna
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, -1)
        except Exception:
            pass
        return img, mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        class_name = img_path.split(os.sep)[-3]
        label = self.class_to_idx.get(class_name, -1)
        
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((112, 112, 3), dtype=np.uint8)
            
        json_path = img_path.replace('.jpg', '.json')
        landmarks = None
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    landmarks = data.get('landmarks')
            except:
                pass

        mask_target = np.zeros((7, 7), dtype=np.float32) 
        if random.random() < OCCLUSION_PROB and landmarks:
            img, full_mask = self.apply_random_occlusion(img, landmarks)
            mask_resized = cv2.resize(full_mask, (7, 7), interpolation=cv2.INTER_NEAREST)
            mask_target = mask_resized

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
            img_tensor = (img_tensor - 127.5) / 128.0

        mask_tensor = torch.from_numpy(mask_target).flatten()
        return img_tensor, label, mask_tensor

# --- 3. MODEL HYBRYDOWY ---
class FaceModelWithAux(nn.Module):
    def __init__(self, backbone, num_classes):
        super(FaceModelWithAux, self).__init__()
        self.backbone = backbone
        self.arcface = ArcMarginProduct(512, num_classes)
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
        else:
            return features_flat, mask_pred

# --- 4. ZAMRAŻANIE ---
def freeze_layers(model):
    print("\n❄️ Konfiguracja zamrażania warstw...")
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'se' in name or 'layer3' in name or 'layer4' in name or 'bn' in name or 'fc' in name:
            param.requires_grad = True
        if param.requires_grad:
            trainable += 1
        else:
            frozen += 1
    print(f"✅ Zamrożono: {frozen} parametrów. Do treningu: {trainable} parametrów.")

# --- 5. GŁÓWNA FUNKCJA ---
def main():
    # Przygotowanie folderu na podgląd
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    # Ładowanie Backbone
    backbone = load_clean_model()
    freeze_layers(backbone)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Ładowanie Datasetów
    if os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR):
        print(f"📂 Ładowanie danych treningowych z: {TRAIN_DIR}")
        train_dataset = OcclusionFaceDataset(TRAIN_DIR, transform=transform, is_validation=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        
        print(f"📂 Ładowanie danych walidacyjnych z: {VAL_DIR}")
        val_dataset = OcclusionFaceDataset(VAL_DIR, transform=transform, is_validation=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        num_classes = len(train_dataset.classes)
    else:
        print(f"⚠️ Błąd ścieżek! Sprawdź czy istnieją:\n {TRAIN_DIR}\n {VAL_DIR}")
        return

    # Inicjalizacja modelu
    full_model = FaceModelWithAux(backbone, num_classes)
    full_model.to(DEVICE)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_aux = nn.MSELoss()
    
    trainable_params = [p for p in full_model.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable_params, lr=LR_BACKBONE, momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=PATIENCE, path='best_model_se_occlusion.pth')

    print(f"\n🚀 Rozpoczynamy trening na {EPOCHS} epok...")
    
    for epoch in range(EPOCHS):
        # --- TRENING ---
        full_model.train()
        train_loss_cls = 0.0
        train_loss_aux = 0.0
        
        # Używamy enumerate, żeby złapać pierwszy batch w epoce
        progress_bar = tqdm(train_loader, desc=f"Epoka {epoch+1}/{EPOCHS} [Train]")
        
        for batch_idx, (imgs, labels, mask_targets) in enumerate(progress_bar):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            mask_targets = mask_targets.to(DEVICE)
            
            # --- ZAPISUJEMY PODGLĄD ZDJĘĆ (Raz na epokę) ---
            if batch_idx == 0:
                # Bierzemy max 20 zdjęć
                debug_imgs = imgs[:20].clone().cpu()
                # Denormalizacja: (x * 0.5) + 0.5, żeby wrócić do zakresu [0, 1]
                debug_imgs = debug_imgs * 0.5 + 0.5
                
                save_path = os.path.join(DEBUG_DIR, f"epoch_{epoch+1:02d}_sample.jpg")
                vutils.save_image(debug_imgs, save_path, nrow=5)
                # print(f"   [INFO] Zapisano próbkę zdjęć treningowych do: {save_path}") # Opcjonalne info w konsoli
            # -----------------------------------------------
            
            optimizer.zero_grad()
            logits, mask_pred = full_model(imgs, labels)
            
            loss_face = criterion_cls(logits, labels)
            loss_aux = criterion_aux(mask_pred, mask_targets)
            
            total_loss = loss_face + 0.1 * loss_aux
            total_loss.backward()
            optimizer.step()
            
            train_loss_cls += loss_face.item()
            train_loss_aux += loss_aux.item()
            progress_bar.set_postfix({'T_Face': loss_face.item(), 'T_Aux': loss_aux.item()})
            
        # --- WALIDACJA ---
        full_model.eval()
        val_loss_aux = 0.0
        
        with torch.no_grad():
            for imgs, _, mask_targets in tqdm(val_loader, desc=f"Epoka {epoch+1}/{EPOCHS} [Val]"):
                imgs = imgs.to(DEVICE)
                mask_targets = mask_targets.to(DEVICE)
                
                _, mask_pred = full_model(imgs, labels=None)
                
                v_loss_aux = criterion_aux(mask_pred, mask_targets)
                val_loss_aux += v_loss_aux.item()
        
        avg_train_loss = (train_loss_cls + 0.1 * train_loss_aux) / len(train_loader)
        avg_val_loss = val_loss_aux / len(val_loader)
        
        print(f"📊 Podsumowanie epoki {epoch+1}:")
        print(f"   Train Loss (Total): {avg_train_loss:.4f} | Val Loss (Aux Only): {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, full_model)
        
        if early_stopping.early_stop:
            print("🛑 Early Stopping zadziałał. Przerywamy trening.")
            break

    print("✅ Trening zakończony.")

if __name__ == "__main__":
    main()