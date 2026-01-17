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
                                                                  
from load import load_clean_model, DEVICE

                                            
BASE_DIR = '../webface_112x112'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'test')
SAVE_PATH = 'best_model_transformer.pth'
BATCH_SIZE = 32
EPOCHS = 25
OCCLUSION_PROB = 0.7
OCCLUSION_HEIGHT = 20
NUM_WORKERS = 4 if os.name != 'nt' else 0
PATIENCE = 5
NUM_VERIFY_PAIRS = 500
LR_HEAD = 0.01

                      
ALPHA = 0.4                                                
TRANSFORMER_LAYERS = 6                        
TRANSFORMER_HEADS = 8                           
                                            

                                                           

                                 
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

                                                       
class TransformerHead(nn.Module):
    def __init__(self, in_channels, num_classes, dim_feedforward=2048):
        super().__init__()
                                            
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels, 
            nhead=TRANSFORMER_HEADS, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=TRANSFORMER_LAYERS)
        
                                                                                        
                                                                        
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
                                                    
        b, c, h, w = x.shape
        
                                                                                  
                                                    
        x = x.view(b, c, h*w).permute(0, 2, 1)
        
                                           
        x = self.transformer(x)
        
                                                                  
                               
        x = x.mean(dim=1) 
        
                                                 
        out = self.fc(x)
        return out

class FaceModelWithTransformer(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        
                                                           
                                                               
                                                                              
        self.arc = ArcMarginProduct(512, num_classes)
        
                                                        
                                                    
        self.transformer_head = TransformerHead(in_channels=512, num_classes=num_classes)

    def forward(self, x, y=None):
                                                                  
                                                                          
        features_spatial = self.backbone(x) 
        
                                                     
                                                                          
                                                                   
        
                                   
        features_flat = torch.flatten(features_spatial, 1)
        
                                                
        if hasattr(self.backbone, 'dropout'):
            features_flat = self.backbone.dropout(features_flat)
            
                                                                               
        if hasattr(self.backbone, 'fc'):
            features_flat = self.backbone.fc(features_flat)
        
                                                                         
        embedding_512 = features_flat
        if hasattr(self.backbone, 'features'):
            embedding_512 = self.backbone.features(features_flat)
            
                                                                                      
        metric_out = self.arc(embedding_512, y) if y is not None else embedding_512
        
                                        
                                                      
        trans_out = None
        if self.training and y is not None:
                                                                     
            trans_out = self.transformer_head(features_spatial)
            
        return metric_out, trans_out

                            
def main():
                                                                                                
                                                            
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
    
                              
    optimizer = optim.SGD(model.parameters(), lr=LR_HEAD, momentum=0.9, weight_decay=5e-4)              
    criterion = nn.CrossEntropyLoss()
    
    best_sim = -1.0
    patience_counter = 0
    
    print(f"Starting training with Transformer Auxiliary Loss (Alpha={ALPHA})...")
    
    for epoch in range(EPOCHS):
                         
        model.train()
        train_loss = 0.0
        
                       
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")
        for img, lbl in pbar:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            
                                        
            metric_out, trans_out = model(img, lbl)
            
                                                      
            loss_metric = criterion(metric_out, lbl)
            
                                                                               
            loss_trans = criterion(trans_out, lbl)
            
                                           
                                                            
            loss = (1 - ALPHA) * loss_metric + ALPHA * loss_trans
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'L_Main': loss_metric.item(), 'L_Trans': loss_trans.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
                           
        model.eval()
        sims = []
        with torch.no_grad():
            for a, b in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                a, b = a.to(DEVICE), b.to(DEVICE)
                
                                                                                        
                                                                                         
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