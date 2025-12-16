import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
import cv2
import os
import json
import glob
import random
from tqdm import tqdm

# Importujemy architekturę ZMODYFIKOWANĄ o CBAM
from backbone import iresnet50

# --- KONFIGURACJA ---
BASE_DIR = '../../../../webface_112x112'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
DEBUG_DIR = 'training_photos_cbam_v2'

# Używamy tego samego modelu startowego (66%)
PRETRAINED_PATH = 'best_model_res50_occlusion.pth' 
NEW_MODEL_PATH = 'best_model_cbam_occlusion_v2.pth'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
# --- KLUCZOWA ZMIANA: RÓŻNE LR ---
LR_CBAM = 0.01      # Nowe warstwy uczą się szybko
LR_HEAD = 0.01      # Głowa uczy się szybko
LR_BACKBONE = 0.0001 # Stare warstwy tylko delikatnie drgają

EPOCHS = 25
OCCLUSION_PROB = 0.7
OCCLUSION_HEIGHT = 20
NUM_WORKERS = 4
PATIENCE = 6 # Dajemy mu chwilę więcej

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path=NEW_MODEL_PATH):
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

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, input, label):
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(input), torch.nn.functional.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        return output * self.s

class OcclusionFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_validation=False):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "*", "*", "*.jpg"))
        if not self.image_paths: print("⚠️ Brak plików")
        self.classes = sorted(list(set([path.split(os.sep)[-3] for path in self.image_paths])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"Znaleziono {len(self.image_paths)} zdjęć.")

    def __len__(self): return len(self.image_paths)

    def apply_random_occlusion(self, img, landmarks):
        h, w, _ = img.shape
        mask = np.zeros((h, w), dtype=np.float32)
        try:
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            center_y = int((left_eye[1] + right_eye[1]) / 2)
            bar_h_half = int(OCCLUSION_HEIGHT / 2)
            y1, y2 = max(0, center_y - bar_h_half), min(h, center_y + bar_h_half)
            color = np.random.randint(0, 256, (3,), dtype=int).tolist()
            cv2.rectangle(img, (0, y1), (w, y2), color, -1)
            cv2.rectangle(mask, (0, y1), (w, y2), 1.0, -1)
        except: pass
        return img, mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.class_to_idx.get(img_path.split(os.sep)[-3], -1)
        img = cv2.imread(img_path)
        if img is None: img = np.zeros((112,112,3), dtype=np.uint8)
        
        json_path = img_path.replace('.jpg', '.json')
        landmarks = None
        if os.path.exists(json_path):
            try:
                with open(json_path) as f: landmarks = json.load(f).get('landmarks')
            except: pass
            
        mask_target = np.zeros((7, 7), dtype=np.float32)
        if random.random() < OCCLUSION_PROB and landmarks:
            img, m = self.apply_random_occlusion(img, landmarks)
            mask_target = cv2.resize(m, (7, 7), interpolation=cv2.INTER_NEAREST)
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb) if self.transform else torch.from_numpy(img_rgb).permute(2,0,1).float()/255.0
        return img_tensor, label, torch.from_numpy(mask_target).flatten()

class FaceModelWithAux(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.arcface = ArcMarginProduct(512, num_classes)
        self.aux_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 49), nn.Sigmoid())

    def forward(self, x, labels=None):
        features = self.backbone(x).view(x.size(0), -1)
        mask_pred = self.aux_head(features)
        if labels is not None:
            return self.arcface(features, labels), mask_pred
        return features, mask_pred

def load_cbam_weights():
    print(f"🔧 Inicjalizacja modelu IResNet50 z CBAM...")
    model = iresnet50(weights_path=None) 
    if os.path.exists(PRETRAINED_PATH):
        print(f"📥 Ładowanie wag z: {PRETRAINED_PATH}")
        checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        # strict=False pozwala na wejście wag ResNet i pominięcie wag CBAM
        missing, _ = model.load_state_dict(state_dict, strict=False)
        print(f"✅ Wagi załadowane. Brakujące klucze (CBAM): {len(missing)}")
    else:
        print(f"⚠️ Nie znaleziono wag startowych!")
    return model

def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    backbone = load_cbam_weights()
    print("ok dalej")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])

    if not os.path.exists(TRAIN_DIR): return

    train_dataset = OcclusionFaceDataset(TRAIN_DIR, transform)
    val_dataset = OcclusionFaceDataset(VAL_DIR, transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    full_model = FaceModelWithAux(backbone, len(train_dataset.classes)).to(DEVICE)
    
    # --- KONFIGURACJA OPTYMALIZATORA (DIFFERENTIAL LEARNING RATES) ---
    cbam_params = []
    backbone_params = []
    head_params = []
    
    for name, param in full_model.named_parameters():
        if not param.requires_grad: continue
        
        if 'cbam' in name:
            cbam_params.append(param)
        elif 'arcface' in name or 'aux_head' in name or 'fc' in name:
            head_params.append(param)
        else:
            backbone_params.append(param) # Stare warstwy ResNet
            
    optimizer = optim.SGD([
        {'params': cbam_params, 'lr': LR_CBAM},      # CBAM uczy się szybko (0.01)
        {'params': head_params, 'lr': LR_HEAD},      # Głowa uczy się szybko (0.01)
        {'params': backbone_params, 'lr': LR_BACKBONE} # Reszta uczy się wolno (0.0001)
    ], momentum=0.9, weight_decay=5e-4)
    
    print(f"📊 Parametry: CBAM={len(cbam_params)}, Backbone={len(backbone_params)}, Head={len(head_params)}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    early_stopping = EarlyStopping(patience=PATIENCE, path=NEW_MODEL_PATH)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_aux = nn.MSELoss()

    print(f"\n🚀 Start Treningu CBAM (Smart LR)...")
    
    for epoch in range(EPOCHS):
        full_model.train()
        t_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoka {epoch+1}/{EPOCHS}")
        for i, (imgs, lbls, masks) in enumerate(pbar):
            imgs, lbls, masks = imgs.to(DEVICE), lbls.to(DEVICE), masks.to(DEVICE)
            
            if i==0: vutils.save_image(imgs[:20]*0.5+0.5, f"{DEBUG_DIR}/e{epoch}.jpg", nrow=5)
            
            optimizer.zero_grad()
            logits, m_pred = full_model(imgs, lbls)
            loss = criterion_cls(logits, lbls) + 0.1 * criterion_aux(m_pred, masks)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
            
        full_model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for imgs, _, masks in tqdm(val_loader, desc="Val"):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                _, m_pred = full_model(imgs)
                v_loss += criterion_aux(m_pred, masks).item()
        
        avg_val = v_loss / len(val_loader)
        print(f"📊 Epoka {epoch+1}: Val Loss: {avg_val:.4f}")
        
        scheduler.step(avg_val)
        early_stopping(avg_val, full_model)
        if early_stopping.early_stop: break

    print("✅ Koniec.")

if __name__ == "__main__":
    main()