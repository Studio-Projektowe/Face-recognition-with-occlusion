import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
import glob
import json
import random
from tqdm import tqdm
from skimage import transform as trans

PRETRAINED_PATH = 'checkpoints_repair_v1/Aligned_Pretrained_Aux_v2.pth' 
TRAIN_DIR = 'webface_112x112/train' 
VAL_DIR = 'webface_112x112/test'      

BATCH_SIZE = 64
LR_CBAM = 0.01
LR_HEAD = 0.1
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = 'checkpoints_repair'

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], 
    dtype=np.float32)

def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, arcface_dst)
    M = tform.params[0:2, :]
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def get_landmarks_from_json(json_path):
    """Parsuje JSON do formatu (5, 2) dla norm_crop."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            lms = data.get('landmarks')
            
        if not lms: return None

        
        kps = np.array([
            lms['right_eye'],
            lms['left_eye'],
            lms['nose'],
            lms['mouth_right'],
            lms['mouth_left']
        ], dtype=np.float32)
        
        return kps
    except Exception as e:
        return None


class OcclusionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "*", "*", "*.jpg"))
        
        if len(self.image_paths) == 0:
            print(f" BŁĄD: Pusto w {root_dir}")
            sys.exit(1)
            
        self.classes = sorted(list(set([
            os.path.basename(os.path.dirname(os.path.dirname(p))) 
            for p in self.image_paths
        ])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"Train Dataset: {len(self.image_paths)} zdjęć.")

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        json_path = path.rsplit('.', 1)[0] + ".json"
        
        person_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        label = self.class_to_idx[person_id]
        
        img = cv2.imread(path)
        if img is None: img = np.zeros((112, 112, 3), dtype=np.uint8)
        
        landmarks = get_landmarks_from_json(json_path)
        if landmarks is not None:
            img = norm_crop(img, landmarks)
        else:
            if img.shape[0] != 112: img = cv2.resize(img, (112, 112))

        h_bar = 20
        center_y = 52
        center_y += random.randint(-5, 5) 
        
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, (0, center_y - h_bar), (112, center_y + h_bar), color, -1)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform: img_tensor = self.transform(img_rgb)
        else: img_tensor = transforms.ToTensor()(img_rgb)
        
        return img_tensor, label

class VerificationDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_identities=100):
        self.transform = transform
        self.pairs = [] 
        all_identity_folders = glob.glob(os.path.join(root_dir, "*"))
        random.shuffle(all_identity_folders)
        
        print(f"Szukanie par walidacyjnych...")
        count = 0
        for id_folder in all_identity_folders:
            if count >= num_identities: break
            imgs = glob.glob(os.path.join(id_folder, "*", "*.jpg"))
            if len(imgs) < 2: continue
            
            valid_pair = []
            for img_path in imgs:
                json_path = img_path.rsplit('.', 1)[0] + ".json"
                if os.path.exists(json_path):
                    valid_pair.append((img_path, json_path))
                if len(valid_pair) == 2: break
            
            if len(valid_pair) == 2:
                self.pairs.append({
                    'clean': valid_pair[0],
                    'query': valid_pair[1],
                    'label': count
                })
                count += 1
                
        print(f"Verification Set: {len(self.pairs)} par (Aligned).")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        
        img_c = cv2.imread(item['clean'][0])
        lm_c = get_landmarks_from_json(item['clean'][1])
        if lm_c is not None: img_c = norm_crop(img_c, lm_c)
        else: img_c = cv2.resize(img_c, (112, 112))
        img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
        
        img_o = cv2.imread(item['query'][0])
        lm_o = get_landmarks_from_json(item['query'][1])
        
        if lm_o is not None: img_o = norm_crop(img_o, lm_o) 
        else: img_o = cv2.resize(img_o, (112, 112))
        
        h_bar = 10
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img_o, (0, 52 - h_bar), (112, 52 + h_bar), color, -1)
        
        img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            ten_c = self.transform(img_c)
            ten_o = self.transform(img_o)
        else:
            ten_c = transforms.ToTensor()(img_c)
            ten_o = transforms.ToTensor()(img_o)
            
        return ten_c, ten_o, item['label']


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

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
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x * self.ca(x)
        return out * self.sa(out)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, use_cbam=False):
        super(IBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.bn1(x); out = self.conv1(out); out = self.bn2(out); out = self.prelu(out)
        out = self.conv2(out); out = self.bn3(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return out

class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, block, layers, dropout=0, num_features=512, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False, use_cbam=False):
        super(IResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None: replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, self.dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.prelu(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.bn2(x); x = torch.flatten(x, 1); x = self.dropout(x); x = self.fc(x); x = self.features(x)
        return x

class FrontEndCBAM(nn.Module):
    def __init__(self, backbone, cbam_channels=64):
        super(FrontEndCBAM, self).__init__()
        self.backbone = backbone
        self.cbam1 = CBAM(cbam_channels, ratio=16, kernel_size=7)
    
    def forward(self, x):
        x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.prelu(x)
        x = self.cbam1(x) 
        x = self.backbone.layer1(x); x = self.backbone.layer2(x); x = self.backbone.layer3(x); x = self.backbone.layer4(x)
        x = self.backbone.bn2(x); x = torch.flatten(x, 1); x = self.backbone.dropout(x); x = self.backbone.fc(x); x = self.backbone.features(x)
        return x

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features; self.out_features = out_features; self.s = s; self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = np.cos(m); self.sin_m = np.sin(m); self.th = np.cos(np.pi - m); self.mm = np.sin(np.pi - m) * m

    def forward(self, input, label):
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(input), torch.nn.functional.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin: phi = torch.where(cosine > 0, phi, cosine)
        else: phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


def load_repair_model():
    print("Budowanie modelu...")
    backbone = IResNet(IBasicBlock, [3, 4, 14, 3], use_cbam=False)
    model = FrontEndCBAM(backbone, cbam_channels=64)
    
    print(f"Wczytywanie wag z: {PRETRAINED_PATH}")
    if os.path.exists(PRETRAINED_PATH):
        checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        
        is_resume = any("cbam1" in k for k in state_dict.keys())
        
        if is_resume:
            print(">>> Wykryto checkpoint RESUME (kontynuacja treningu). Ładuję 1:1...")
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
        else:
            print(">>> Wykryto czysty BACKBONE. Dodaję prefix 'backbone.'...")
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                if k.startswith("backbone."):
                    new_state_dict[k] = v
                else:
                    new_state_dict[f"backbone.{k}"] = v
        
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        
        if len(missing) > 0:
            print(f"Brakujące klucze (to OK dla CBAM przy pierwszym uruchomieniu): {missing[:5]}...")
        if len(unexpected) > 0:
            print(f"Nieoczekiwane klucze: {unexpected[:5]}...")
            
        print("Wagi załadowane pomyślnie.")
    else:
        print(f"BŁĄD: Nie znaleziono pliku {PRETRAINED_PATH}")
        sys.exit(1)
        
    return model

def freeze_backbone_keep_cbam(model):
    for param in model.backbone.parameters(): param.requires_grad = False
    for param in model.cbam1.parameters(): param.requires_grad = True
    for name, module in model.backbone.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for param in module.parameters(): param.requires_grad = True

def run_verification_test(model, val_loader, device):
    model.eval()
    gallery_feats, query_feats, labels = [], [], []
    with torch.no_grad():
        for img_clean, img_occ, label in val_loader:
            img_clean, img_occ = img_clean.to(device), img_occ.to(device)
            emb_c = model(img_clean)
            emb_o = model(img_occ)
            emb_c = torch.nn.functional.normalize(emb_c, p=2, dim=1)
            emb_o = torch.nn.functional.normalize(emb_o, p=2, dim=1)
            gallery_feats.append(emb_c.cpu())
            query_feats.append(emb_o.cpu())
            labels.append(label)
    if len(gallery_feats) == 0: return 0.0
    gallery_feats = torch.cat(gallery_feats, dim=0)
    query_feats = torch.cat(query_feats, dim=0)
    labels = torch.cat(labels, dim=0)
    sim_matrix = torch.mm(query_feats, gallery_feats.t())
    correct = 0
    total = labels.size(0)
    for i in range(total):
        best_match_idx = torch.argmax(sim_matrix[i]).item()
        if labels[best_match_idx] == labels[i]: correct += 1
    return (correct / total) * 100

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    print("Inicjalizacja Datasetów (z Alignmentem)...")
    train_ds = OcclusionDataset(TRAIN_DIR, transform=transform)
    verification_ds = VerificationDataset(VAL_DIR, transform=transform, num_identities=200)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    verify_loader = DataLoader(verification_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = load_repair_model().to(DEVICE)
    freeze_backbone_keep_cbam(model)
    
    metric_fc = ArcMarginProduct(512, len(train_ds.classes)).to(DEVICE)
    trainables = [p for p in model.parameters() if p.requires_grad] + list(metric_fc.parameters())
    optimizer = optim.SGD(trainables, lr=LR_CBAM, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print(f"\n START NAPRAWY (Startowy Rank-1: sprawdzimy za chwilę...)")
    
    acc_0 = run_verification_test(model, verify_loader, DEVICE)
    print(f" Stan zerowy (Random CBAM + Alignment) -> Rank-1 Accuracy: {acc_0:.2f}%")

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        metric_fc.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoka {epoch+1}/{EPOCHS}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            embeddings = model(imgs)
            outputs = metric_fc(embeddings, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        
        current_acc = run_verification_test(model, verify_loader, DEVICE)
        print(f" Epoka {epoch+1}: Train Loss: {avg_loss:.4f} | Rank-1 Accuracy: {current_acc:.2f}%")
        
        if current_acc > best_acc:
            best_acc = current_acc
            save_path = os.path.join(OUTPUT_DIR, 'repaired_cbam_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f" Nowy rekord! Zapisano: {save_path}")

    print(f"\nKoniec. Najlepszy wynik z CBAM: {best_acc:.2f}%")

if __name__ == "__main__":
    main()