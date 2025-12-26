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
from collections import OrderedDict

try:
    from backbone_iresnet import IBasicBlock, conv1x1
except ImportError:
    raise ImportError("Brak pliku backbone_iresnet.py! Upewnij się, że jest w katalogu roboczym.")

import face_align

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_PATH = 'aligned_pretrained_v2.pth'
BASE_DIR = '../../../../webface_112x112'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
DEBUG_DIR = 'training_photos_cbam'
METRICS_DIR = 'metrics_cbam'
METRICS_CSV = None

BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 2
EPOCHS = 25
OCCLUSION_PROB = 0.7
OCCLUSION_HEIGHT = 20
NUM_WORKERS = 4
PATIENCE = 5
RANK_K_SAMPLES = 500

LR_BACKBONE = 1e-3
LR_HEAD = 0.01
LR_CBAM = 5e-3
AUX_LOSS_WEIGHT = 0.01

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out)
        return out.view(b, c, 1, 1)

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
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + x

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)

class IResNetWithCBAM(nn.Module):
    fc_scale = 7 * 7
    
    def __init__(self, block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNetWithCBAM, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        self.cbam_layer = CBAM(256, ratio=16, kernel_size=7)
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for name, m in self.named_modules():
            if 'cbam' in name:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # (after layer3, before layer4)
        x = self.cbam_layer(x)
        
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x

def iresnet50_cbam(**kwargs):
    return IResNetWithCBAM(IBasicBlock, [3, 4, 14, 3], **kwargs)

def load_model_with_cbam(pretrained_path):
    print(f"Creating IResNet50 + CBAM (after layer3) model...")
    model = iresnet50_cbam()
    
    print(f"Loading pretrained weights from: {pretrained_path}")
    if not os.path.exists(pretrained_path):
        print(f"ERROR: File {pretrained_path} does not exist! Starting from scratch.")
        return model.to(DEVICE)
    
    pretrained_state = torch.load(pretrained_path, map_location='cpu')
    if 'state_dict' in pretrained_state:
        pretrained_state = pretrained_state['state_dict']
    
    model_state = model.state_dict()
    new_state_dict = OrderedDict()
    
    loaded = 0
    skipped = 0
    
    for k, v in pretrained_state.items():
        if k.startswith('backbone.'):
            name = k[9:]
        elif k.startswith('module.'):
            name = k[7:]
        else:
            name = k
            
        if name in model_state:
            if v.shape == model_state[name].shape:
                new_state_dict[name] = v
                loaded += 1
            else:
                print(f"Skipping {name} due to shape mismatch: {v.shape} vs {model_state[name].shape}")
                skipped += 1
        else:
            pass
            
    model.load_state_dict(new_state_dict, strict=False)
    
    print(f"Successfully loaded {loaded} layers.")
    if skipped > 0:
        print(f"Skipped {skipped} layers due to shape mismatch.")
    
    return model.to(DEVICE)

def freeze_layers(model):
    print("\nConfiguring layer freezing...")
    frozen = 0
    trainable = 0
    
    for name, param in model.named_parameters():
        param.requires_grad = False
        
        if (
            'cbam' in name or
            'layer3' in name or
            'layer4' in name or
            ('bn' in name and ('layer3' in name or 'layer4' in name)) or
            'fc' in name or
            ('prelu' in name and ('layer3' in name or 'layer4' in name))
        ):
            param.requires_grad = True
            
        if param.requires_grad:
            trainable += 1
        else:
            frozen += 1
            
    print(f"Frozen: {frozen}. Trainable: {trainable}.")

def calculate_rank_k(model, val_loader, num_samples=500):
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        samples_collected = 0
        for img, lbl, _ in val_loader:
            if samples_collected >= num_samples: break
            img = img.to(DEVICE)
            features, _ = model(img, labels=None)
            embeddings.append(features.cpu())
            labels.append(lbl)
            samples_collected += len(lbl)
    
    if len(embeddings) == 0: return 0.0, 0.0
    
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
        
        if query_label == top1_label: rank1_correct += 1
        if query_label in top3_labels: rank3_correct += 1
    
    return (rank1_correct/total)*100, (rank3_correct/total)*100

def save_metrics_to_csv(filepath, epoch, train_loss, val_loss, rank1, rank3):
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'rank1_acc', 'rank3_acc'])
        writer.writerow([epoch, f'{train_loss:.4f}', f'{val_loss:.4f}', f'{rank1:.2f}', f'{rank3:.2f}'])

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='best_model_cbam.pth'):
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
            print(f'  EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(f'  Saving model (Val Loss: {self.best_loss:.6f} -> {val_loss:.6f})')
        torch.save(model.backbone.state_dict(), self.path)

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
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7*7),
            nn.Sigmoid()
        )

    def forward(self, x, labels=None):
        features = self.backbone(x)
        
        mask_pred = self.aux_head(features)
        
        if labels is not None:
            arcface_out = self.arcface(features, labels)
            return arcface_out, mask_pred
        return features, mask_pred

class OcclusionFaceDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.root_dirs = root_dirs
        self.transform = transform
        self.image_paths = []
        
        print(f"Scanning folders: {root_dirs}...")
        for d in root_dirs:
            if os.path.exists(d):
                # root / identity / image_folder / image.jpg
                files = glob.glob(os.path.join(d, "*", "*", "*.jpg"))
                self.image_paths.extend(files)
            else:
                print(f"WARNING: Folder {d} does not exist!")

        self.classes = sorted(list(set([p.split(os.sep)[-3] for p in self.image_paths])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Found {len(self.image_paths)} images in {len(self.classes)} identities.")

    def __len__(self): 
        return len(self.image_paths)

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
        cv2.rectangle(img, (0, y1), (w, y2), np.random.randint(0, 256, (3,)).tolist(), -1)
        cv2.rectangle(mask, (0, y1), (w, y2), 1.0, -1)
        return img, mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        class_name = img_path.split(os.sep)[-3]
        label = self.class_to_idx.get(class_name, -1)
        
        if label == -1:
            print(f"ERROR: Invalid label for {img_path}")
            label = 0 

        img = cv2.imread(img_path)
        
        if img is None:
            print(f"ERROR: Corrupt image: {img_path}")
            img = np.zeros((112, 112, 3), dtype=np.uint8)
        
        json_path = img_path.replace('.jpg', '.json')
        final_img = cv2.resize(img, (112, 112))
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    final_img = self.align_face(img, data.get('landmarks', data))
            except: 
                pass

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
    global METRICS_CSV
    
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    METRICS_CSV = os.path.join(METRICS_DIR, f'training_metrics_cbam_{timestamp}.csv')
    print(f"Metrics: {METRICS_CSV}")
    
    backbone = load_model_with_cbam(PRETRAINED_PATH)
    freeze_layers(backbone)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    full_dataset = OcclusionFaceDataset(root_dirs=[TRAIN_DIR, VAL_DIR], transform=transform)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    model = FaceModelWithAux(backbone, len(full_dataset.classes)).to(DEVICE)
    
    backbone_params = [p for n, p in model.named_parameters() 
                       if 'backbone' in n and 'cbam' not in n and p.requires_grad]
    cbam_params = [p for n, p in model.named_parameters() 
                   if 'cbam' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() 
                   if 'backbone' not in n and p.requires_grad]
            
    optimizer = optim.SGD([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': cbam_params, 'lr': LR_CBAM},
        {'params': head_params, 'lr': LR_HEAD}
    ], momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    early_stopping = EarlyStopping(patience=PATIENCE, path='best_model_cbam.pth')
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_aux = nn.MSELoss()
    
    print(f"\nSTARTING TRAINING...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for i, (img, lbl, msk) in enumerate(loop):
            img, lbl, msk = img.to(DEVICE), lbl.to(DEVICE), msk.to(DEVICE)
            
            if i == 0: vutils.save_image(img[:16]*0.5+0.5, f"{DEBUG_DIR}/ep{epoch}.jpg", nrow=4)
            
            logits, m_pred = model(img, lbl)
            loss = criterion_cls(logits, lbl) + AUX_LOSS_WEIGHT * criterion_aux(m_pred, msk)
            
            if torch.isnan(loss):
                print(f"NaN loss detected! Step {i}")
                optimizer.zero_grad()
                continue
            
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()
            
            current_loss = loss.item() * GRAD_ACCUM_STEPS
            train_loss += current_loss
            loop.set_postfix({'loss': f"{current_loss:.4f}"})
            
        avg_train = train_loss / len(train_loader)
        
        rank1_acc, rank3_acc = calculate_rank_k(model, val_loader, num_samples=RANK_K_SAMPLES)
        
        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for img, lbl, msk in val_loader:
                img, lbl, msk = img.to(DEVICE), lbl.to(DEVICE), msk.to(DEVICE)
                logits, m_pred = model(img, lbl) 
                v_loss = criterion_cls(logits, lbl) + AUX_LOSS_WEIGHT * criterion_aux(m_pred, msk)
                val_total_loss += v_loss.item()
                
        avg_val = val_total_loss / len(val_loader)

        print(f"Ep {epoch+1}: Train: {avg_train:.4f} | Val: {avg_val:.4f} | Rank-1: {rank1_acc:.2f}%")
        save_metrics_to_csv(METRICS_CSV, epoch+1, avg_train, avg_val, rank1_acc, rank3_acc)
        
        scheduler.step(avg_val)
        early_stopping(avg_val, model)
        if early_stopping.early_stop:
            print("Early Stopping!")
            break

    print("Done.")

if __name__ == "__main__":
    main()