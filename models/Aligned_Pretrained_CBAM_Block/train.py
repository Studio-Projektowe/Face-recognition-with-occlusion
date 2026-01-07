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

BACKBONE_PATH = './Aligned_Pretrained_Aux_v2.pth'
TRAIN_DIR = 'webface_112x112/train' 
VAL_DIR = 'webface_112x112/test'       

BATCH_SIZE = 64
LR_START = 0.1
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = 'checkpoints_cbam_inside_v2'

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
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
        return self.sigmoid(self.conv1(x))

class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class CBAMBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(CBAMBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        
        self.cbam = CBAM(planes, ratio=16, kernel_size=7)

    def forward(self, x):
        identity = x
        out = self.bn1(x); out = self.conv1(out); out = self.bn2(out); out = self.prelu(out)
        out = self.conv2(out); out = self.bn3(out)
        
        out = self.cbam(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class IResNetCBAM(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, block, layers, dropout=0, num_features=512, groups=1, width_per_group=64):
        super(IResNetCBAM, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05); self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)

    def _make_layer(self, block, planes, blocks, stride=1):
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

arcface_dst = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

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
        with open(json_path, 'r') as f: data = json.load(f)
        lms = data.get('landmarks')
        if not lms: return None
        return np.array([lms['right_eye'], lms['left_eye'], lms['nose'], lms['mouth_right'], lms['mouth_left']], dtype=np.float32)
    except:
        return None

class MixedOcclusionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, "*", "*", "*.jpg"))
        self.transform = transform
        self.classes = sorted(list(set([os.path.basename(os.path.dirname(os.path.dirname(p))) for p in self.image_paths])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        json_path = path.rsplit('.', 1)[0] + ".json"
        
        person_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        label = self.class_to_idx[person_id]
        
        img = cv2.imread(path)
        if img is None: 
            img = np.zeros((112, 112, 3), dtype=np.uint8)
        
        lms = get_landmarks_from_json(json_path)
        if lms is not None: 
            img = norm_crop(img, lms)
        else:
            if img.shape[0] != 112 or img.shape[1] != 112:
                img = cv2.resize(img, (112, 112))
        
        if random.random() > 0.5:
            center_y = 52
            h_bar = 20
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(img, (0, center_y - h_bar//2), (112, center_y + h_bar//2), color, -1)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform: return self.transform(img), label
        return transforms.ToTensor()(img), label

class VerificationDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_pairs=500):
        self.transform = transform
        self.pairs = []
        all_ids = glob.glob(os.path.join(root_dir, "*"))
        random.shuffle(all_ids)
        
        count = 0
        for id_f in all_ids:
            if count >= num_pairs: break
            imgs = glob.glob(os.path.join(id_f, "*", "*.jpg"))
            if len(imgs) < 2: continue
            
            pair_data = []
            for ip in imgs:
                jp = ip.rsplit('.', 1)[0] + ".json"
                if os.path.exists(jp): pair_data.append((ip, jp))
                if len(pair_data) == 2: break
            
            if len(pair_data) == 2:
                self.pairs.append({'clean': pair_data[0], 'query': pair_data[1]})
                count += 1
        print(f"Walidacja na {len(self.pairs)} parach.")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        
        img_c = cv2.imread(item['clean'][0])
        lm_c = get_landmarks_from_json(item['clean'][1])
        if lm_c is not None: img_c = norm_crop(img_c, lm_c)
        else: img_c = cv2.resize(img_c, (112,112))
        
        img_o = cv2.imread(item['query'][0])
        lm_o = get_landmarks_from_json(item['query'][1])
        if lm_o is not None: img_o = norm_crop(img_o, lm_o)
        else: img_o = cv2.resize(img_o, (112,112))
        
        cv2.rectangle(img_o, (0, 42), (112, 62), (random.randint(0,255), random.randint(0,255), random.randint(0,255)), -1)
        
        if self.transform: 
            return self.transform(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)), \
                   self.transform(cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB))
        return transforms.ToTensor()(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)), \
               transforms.ToTensor()(cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB))

def run_verification_test(model, val_loader, device):
    model.eval()
    sims = []
    with torch.no_grad():
        for img_clean, img_occ in val_loader:
            img_clean = img_clean.to(device)
            img_occ = img_occ.to(device)
            emb_c = torch.nn.functional.normalize(model(img_clean), p=2, dim=1)
            emb_o = torch.nn.functional.normalize(model(img_occ), p=2, dim=1)
            cosine = (emb_c * emb_o).sum(dim=1)
            sims.extend(cosine.cpu().numpy())
    return np.mean(sims)

def load_backbone_weights(model, path):
    print(f"Ładowanie Backbone'u z: {path}")
    if not os.path.exists(path):
        print(" BŁĄD: Brak pliku backbone!"); sys.exit(1)
    ckpt = torch.load(path, map_location='cpu')
    if 'state_dict' in ckpt: ckpt = ckpt['state_dict']
    
    model_state = model.state_dict()
    new_state = OrderedDict()
    loaded_count = 0
    for k, v in ckpt.items():
        name = k.replace("module.", "")
        if name.startswith("backbone."): name = name.replace("backbone.", "")
        if name in model_state and v.shape == model_state[name].shape:
            new_state[name] = v
            loaded_count += 1
            
    model.load_state_dict(new_state, strict=False)
    print(f" Załadowano {loaded_count} warstw backbone'u.")
    return model

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    
    train_ds = MixedOcclusionDataset(TRAIN_DIR, transform)
    verify_loader = DataLoader(VerificationDataset(VAL_DIR, transform, num_pairs=500), batch_size=BATCH_SIZE, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = IResNetCBAM(CBAMBasicBlock, [3, 4, 14, 3]).to(DEVICE)
    model = load_backbone_weights(model, BACKBONE_PATH)
    
    metric_fc = ArcMarginProduct(512, len(train_ds.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    print(" MROŻENIE BACKBONE'U NA 1. EPOKĘ...")
    for name, param in model.named_parameters():
        if "cbam" in name: param.requires_grad = True
        else: param.requires_grad = False
            
    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(metric_fc.parameters())
    optimizer = optim.SGD(trainable_params, lr=LR_START, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    best_sim = -1.0
    print(" START TRENINGU (Final Config)")
    
    for epoch in range(EPOCHS):
        if epoch == 1:
            print(" ODMRAŻANIE CAŁEGO MODELU...")
            for param in model.parameters():
                param.requires_grad = True
            
            optimizer = optim.SGD([
                {'params': model.parameters()},
                {'params': metric_fc.parameters()}
            ], lr=LR_START, momentum=0.9, weight_decay=5e-4)
            
            for group in optimizer.param_groups:
                group['initial_lr'] = LR_START

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=epoch)

        model.train(); metric_fc.train()
        loop = tqdm(train_loader, desc=f"Epoka {epoch+1}")
        total_loss = 0
        
        for imgs, lbls in loop:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            out = metric_fc(model(imgs), lbls)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        scheduler.step()
        
        avg_sim = run_verification_test(model, verify_loader, DEVICE)
        print(f" Epoka {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Avg Sim (500 par): {avg_sim:.4f}")
        
        if avg_sim > best_sim:
            best_sim = avg_sim
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_cbam_final.pth'))
            print(" Zapisano najlepszy model!")

if __name__ == "__main__":
    main()