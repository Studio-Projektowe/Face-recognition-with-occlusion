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

# Twoje importy
from load_and_test import load_clean_model, DEVICE
import face_align

# --- KONFIGURACJA ---
BASE_DIR = '../../../../webface_112x112'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
DEBUG_DIR = 'training_photos'

BATCH_SIZE = 32
LR_HEAD = 0.01
LR_BACKBONE = 0.001
EPOCHS = 25
OCCLUSION_PROB = 0.7
OCCLUSION_HEIGHT = 25
NUM_WORKERS = 4
PATIENCE = 5

# NOWE PARAMETRY
WARMUP_EPOCHS = 3      # Przez pierwsze 3 epoki powoli zwiększamy LR
GRAD_CLIP_VALUE = 5.0  # Maksymalna "siła" zmiany wag w jednym kroku

# --- 1. ZAMRAŻANIE (SAFE MODE) ---
def freeze_layers(model):
    print("\n❄️ Konfiguracja zamrażania warstw (SAFE MODE)...")
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        param.requires_grad = False 
        
        if (
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
            
    print(f"✅ Zamrożono: {frozen} parametrów. Do treningu: {trainable} parametrów.")

# --- 2. KLASA EARLY STOPPING ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='best_model_align_occlusion.pth'):
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

# --- 3. MODEL I MODUŁY ---
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

# --- 4. DATASET ---
class OcclusionFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_validation=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_validation = is_validation
        
        print(f"Skanowanie datasetu: {root_dir}...")
        self.image_paths = glob.glob(os.path.join(root_dir, "*", "*", "*.jpg"))
        
        if not self.image_paths:
            print(f"⚠️ Nie znaleziono plików .jpg w {root_dir}")
        
        self.classes = sorted(list(set([path.split(os.sep)[-3] for path in self.image_paths])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"Znaleziono {len(self.image_paths)} zdjęć w {len(self.classes)} tożsamościach.")

    def __len__(self):
        return len(self.image_paths)

    def align_face(self, img, landmarks):
        try:
            kps = np.array([
                landmarks['right_eye'], landmarks['left_eye'], landmarks['nose'],
                landmarks['mouth_right'], landmarks['mouth_left']
            ], dtype=np.float32)
            return face_align.norm_crop(img, landmark=kps, image_size=112)
        except Exception:
            return cv2.resize(img, (112, 112))

    def apply_random_occlusion(self, img):
        h, w, _ = img.shape
        mask = np.zeros((h, w), dtype=np.float32)
        center_y = 52 + random.randint(-5, 5)
        bar_h_half = int(OCCLUSION_HEIGHT / 2)
        y1 = max(0, center_y - bar_h_half)
        y2 = min(h, center_y + bar_h_half)
        x1, x2 = 0, w
        color = np.random.randint(0, 256, (3,), dtype=int).tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, -1)
        return img, mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        class_name = img_path.split(os.sep)[-3]
        label = self.class_to_idx.get(class_name, -1)
        
        img = cv2.imread(img_path)
        if img is None: 
            # Stwórz czarny obraz jeśli plik jest uszkodzony
            img = np.zeros((112, 112, 3), dtype=np.uint8)
            print(f"❌ BŁĄD: Nie można wczytać zdjęcia: {img_path}")
            
        json_path = img_path.replace('.jpg', '.json')
        
        # Domyślnie robimy resize (jeśli alignment się nie uda)
        final_img = cv2.resize(img, (112, 112))
        alignment_success = False # Flaga do sprawdzenia czy się udało

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # Obsługa różnych formatów JSONa
                    landmarks = data.get('landmarks', data)
                    
                    # Próba alignmentu
                    aligned = self.align_face(img, landmarks)
                    if aligned is not None and aligned.shape[0] > 0:
                        final_img = aligned
                        alignment_success = True
                    else:
                        print(f"⚠️ Alignment zwrócił puste zdjęcie dla: {img_path}")
            except Exception as e:
                # TERAZ WIDZISZ BŁĄD!
                print(f"⚠️ Błąd podczas alignmentu {img_path}: {e}")
        else:
            # Jeśli wchodzisz tutaj, to znaczy że nie masz plików JSON!
            # Odkomentuj linię niżej tylko do testów, żeby nie spamowało konsoli
            print(f"⚠️ Brak pliku JSON dla: {img_path}") 
            pass

        # ... reszta kodu z okluzją bez zmian ...
        
        mask_target = np.zeros((7, 7), dtype=np.float32) 
        if random.random() < OCCLUSION_PROB:
            final_img, full_mask = self.apply_random_occlusion(final_img)
            mask_resized = cv2.resize(full_mask, (7, 7), interpolation=cv2.INTER_NEAREST)
            mask_target = mask_resized

        img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
            img_tensor = (img_tensor - 127.5) / 128.0

        mask_tensor = torch.from_numpy(mask_target).flatten()
        return img_tensor, label, mask_tensor

# --- 5. GŁÓWNA FUNKCJA ---
def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    # Ładowanie modelu
    backbone = load_clean_model()
    freeze_layers(backbone)
    
    # Standardowa transformacja (BEZ udziwnień kolorystycznych)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Ładowanie Datasetów
    if not os.path.exists(TRAIN_DIR):
        print(f"⚠️ Błąd: Brak folderu {TRAIN_DIR}")
        return

    train_dataset = OcclusionFaceDataset(TRAIN_DIR, transform=transform, is_validation=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    val_dataset = OcclusionFaceDataset(VAL_DIR, transform=transform, is_validation=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    num_classes = len(train_dataset.classes)

    # Setup Treningu
    full_model = FaceModelWithAux(backbone, num_classes)
    full_model.to(DEVICE)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_aux = nn.MSELoss()
    
    trainable_params = [p for p in full_model.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable_params, lr=LR_BACKBONE, momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    early_stopping = EarlyStopping(patience=PATIENCE, path='best_model_align_occlusion.pth')

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n🚀 Rozpoczynamy trening na {EPOCHS} epok.")
    print(f"🔥 Warmup przez {WARMUP_EPOCHS} epok.")
    print(f"✂️ Gradient Clipping ustawiony na {GRAD_CLIP_VALUE}.")

    for epoch in range(EPOCHS):
        # === LEARNING RATE WARMUP ===
        if epoch < WARMUP_EPOCHS:
            # Liniowy wzrost LR od 0 do LR_BACKBONE
            warmup_factor = (epoch + 1) / WARMUP_EPOCHS
            current_lr = LR_BACKBONE * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"🔥 [Warmup] Epoka {epoch+1}: Ustawiono LR = {current_lr:.6f}")
        
        # --- TRENING ---
        full_model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoka {epoch+1}/{EPOCHS} [Train]")
        
        for batch_idx, (imgs, labels, mask_targets) in enumerate(progress_bar):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            mask_targets = mask_targets.to(DEVICE)
            
            # Podgląd zdjęć (raz na epokę)
            if batch_idx == 0:
                debug_imgs = imgs[:20].clone().cpu() * 0.5 + 0.5 
                vutils.save_image(debug_imgs, os.path.join(DEBUG_DIR, f"epoch_{epoch+1:02d}_sample.jpg"), nrow=5)
            
            optimizer.zero_grad()
            logits, mask_pred = full_model(imgs, labels)
            
            loss_face = criterion_cls(logits, labels)
            loss_aux = criterion_aux(mask_pred, mask_targets)
            total_loss = loss_face + 0.1 * loss_aux
            
            total_loss.backward()
            
            # === GRADIENT CLIPPING (NOWOŚĆ) ===
            # Zapobiega wybuchom gradientów
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), max_norm=GRAD_CLIP_VALUE)
            
            optimizer.step()
            
            # Statystyki
            train_loss_total += total_loss.item()
            _, preds = torch.max(logits, 1)
            train_correct += (preds == labels).sum().item()
            train_total_samples += labels.size(0)
            
            progress_bar.set_postfix({'Loss': total_loss.item()})
            
        # --- WALIDACJA ---
        full_model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for imgs, labels, mask_targets in tqdm(val_loader, desc=f"Epoka {epoch+1}/{EPOCHS} [Val]"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                mask_targets = mask_targets.to(DEVICE)
                
                logits, mask_pred = full_model(imgs, labels)
                
                v_loss_face = criterion_cls(logits, labels)
                v_loss_aux = criterion_aux(mask_pred, mask_targets)
                v_total_loss = v_loss_face + 0.1 * v_loss_aux
                val_loss_total += v_total_loss.item()

                _, preds = torch.max(logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total_samples += labels.size(0)
        
        # Średnie
        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_acc = train_correct / train_total_samples * 100
        avg_val_loss = val_loss_total / len(val_loader)
        avg_val_acc = val_correct / val_total_samples * 100
        
        print(f"📊 Epoka {epoch+1}:")
        print(f"   Train -> Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.2f}%")
        print(f"   Val   -> Loss: {avg_val_loss:.4f}   | Acc: {avg_val_acc:.2f}%")
        
        # Zapis historii
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        with open('history.json', 'w') as f: json.dump(history, f)

        # Scheduler działa dopiero po Warmupie
        if epoch >= WARMUP_EPOCHS:
            scheduler.step(avg_val_loss)
            
        early_stopping(avg_val_loss, full_model)
        
        if early_stopping.early_stop:
            print("🛑 Early Stopping zadziałał.")
            break

    print("✅ Trening zakończony. Historia zapisana w history.json")

main()
