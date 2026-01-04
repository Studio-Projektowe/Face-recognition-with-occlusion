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
from collections import OrderedDict

PRETRAINED_PATH = './Aligned_Pretrained_L3_v0.pth' 
TRAIN_DIR = '../webface_112x112/train' 
VAL_DIR = '../webface_112x112/test'       

BATCH_SIZE = 64
LR_CBAM = 0.01 
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = 'checkpoints_repair_layer3_fixed'

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
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            lms = data.get('landmarks')
        if not lms: return None
        kps = np.array([
            lms['right_eye'], lms['left_eye'], lms['nose'],
            lms['mouth_right'], lms['mouth_left']
        ], dtype=np.float32)
        return kps
    except Exception:
        return None


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
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
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

class IResNetWithCBAM(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNetWithCBAM, self).__init__()
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
        
        self.cbam_layer = CBAM(256, ratio=16, kernel_size=7)
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
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
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.cbam_layer(x)
        x = self.layer4(x); x = self.bn2(x); x = torch.flatten(x, 1)
        x = self.dropout(x); x = self.fc(x); x = self.features(x)
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


def load_exact_model(pretrained_path):
    print(f"Tworzenie modelu IResNetWithCBAM...")
    model = IResNetWithCBAM(IBasicBlock, [3, 4, 14, 3])
    
    print(f"Wczytywanie wag z: {pretrained_path}")
    if not os.path.exists(pretrained_path):
        print(f" BŁĄD: Plik nie istnieje!")
        sys.exit(1)
    
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    
    new_state_dict = OrderedDict()
    
    for k, v in checkpoint.items():
        name = k.replace("module.", "")
        
        if name.startswith("backbone."):
            name = name.replace("backbone.", "")
        
        if "cbam1" in name:
            name = name.replace("cbam1", "cbam_layer")
            
        new_state_dict[name] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    print("-" * 50)
    cbam_missing = [k for k in missing if "cbam" in k]
    if len(cbam_missing) == 0:
        print(" SUKCES: Wczytano wagi CBAM (nie inicjalizujemy od zera!)")
    else:
        print(f" UWAGA: Brakuje wag CBAM: {cbam_missing}")
        print("Czy na pewno plik .pth zawiera wytrenowany moduł CBAM?")
    
    print("-" * 50)
    return model

def freeze_setup(model):
    print(" Konfiguracja mrożenia (Freeze 1-3, Unfreeze 4 + CBAM)...")
    
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.cbam_layer.parameters():
        param.requires_grad = True
        
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    for module in [model.bn2, model.fc, model.features]:
        for param in module.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Liczba trenowalnych parametrów: {trainable:,}")

class OcclusionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "*", "*", "*.jpg"))
        if len(self.image_paths) == 0: sys.exit(1)
        self.classes = sorted(list(set([os.path.basename(os.path.dirname(os.path.dirname(p))) for p in self.image_paths])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        json_path = path.rsplit('.', 1)[0] + ".json"
        person_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        label = self.class_to_idx[person_id]
        img = cv2.imread(path)
        if img is None: img = np.zeros((112, 112, 3), dtype=np.uint8)
        landmarks = get_landmarks_from_json(json_path)
        if landmarks is not None: img = norm_crop(img, landmarks)
        else:
             if img.shape[0] != 112: img = cv2.resize(img, (112, 112))
        h_bar = 20
        center_y = 52 + random.randint(-5, 5)
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
        all_ids = glob.glob(os.path.join(root_dir, "*"))
        random.shuffle(all_ids)
        count = 0
        for id_f in all_ids:
            if count >= num_identities: break
            imgs = glob.glob(os.path.join(id_f, "*", "*.jpg"))
            if len(imgs) < 2: continue
            pair = []
            for ip in imgs:
                jp = ip.rsplit('.', 1)[0] + ".json"
                if os.path.exists(jp): pair.append((ip, jp))
                if len(pair)==2: break
            if len(pair)==2:
                self.pairs.append({'clean': pair[0], 'query': pair[1], 'label': count})
                count += 1
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
            tc = self.transform(img_c); to = self.transform(img_o)
        else:
            tc = transforms.ToTensor()(img_c); to = transforms.ToTensor()(img_o)
        return tc, to, item['label']

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
        if labels[torch.argmax(sim_matrix[i]).item()] == labels[i]: correct += 1
    return (correct / total) * 100

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_ds = OcclusionDataset(TRAIN_DIR, transform=transform)
    verify_loader = DataLoader(VerificationDataset(VAL_DIR, transform=transform), batch_size=BATCH_SIZE, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = load_exact_model(PRETRAINED_PATH).to(DEVICE)
    
    freeze_setup(model)
    
    metric_fc = ArcMarginProduct(512, len(train_ds.classes)).to(DEVICE)
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad] + list(metric_fc.parameters()), 
                          lr=LR_CBAM, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    acc_0 = run_verification_test(model, verify_loader, DEVICE)
    print(f" Start: Rank-1: {acc_0:.2f}% (Powinno być > 20% jeśli wczytano CBAM)")
    
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train(); metric_fc.train()
        loop = tqdm(train_loader, desc=f"Epoka {epoch+1}")
        total_loss = 0
        for imgs, lbls in loop:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(metric_fc(model(imgs), lbls), lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        scheduler.step()
        acc = run_verification_test(model, verify_loader, DEVICE)
        print(f" Epoka {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'fixed_cbam_best.pth'))

if __name__ == "__main__":
    main()