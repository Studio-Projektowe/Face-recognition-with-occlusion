import os
import sys
import json
import csv
import numpy as np
import cv2
import faiss
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from skimage import transform as trans
from collections import defaultdict, OrderedDict

BASE_DATA_DIR = '../webface_112x112/test'  
MODEL_PATH = 'best_cbam_final.pth' 
METRICS_DIR = 'metrics_final_correct'

FAISS_INDEX_FILE = os.path.join(METRICS_DIR, 'gallery.index')
FAISS_MAPPING_FILE = os.path.join(METRICS_DIR, 'gallery_map.json')
RESULTS_CSV = os.path.join(METRICS_DIR, 'evaluation_results.csv')
OCCLUSION_OUTPUT_DIR = os.path.join(METRICS_DIR, 'occlusion_debug')

BATCH_SIZE = 32
OCCLUSION_SIZE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REFERENCE_POINTS_112 = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]
], dtype=np.float32)


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
        
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return out

class IResNetCBAM(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, block, layers, dropout=0, num_features=512, groups=1, width_per_group=64):
        super(IResNetCBAM, self).__init__()
        self.inplanes = 64; self.dilation = 1; self.groups = groups; self.base_width = width_per_group
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


transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_model():
    print(f"Ładowanie modelu z {MODEL_PATH}...")
    model = IResNetCBAM(CBAMBasicBlock, [3, 4, 14, 3])
    model.to(DEVICE)
    model.eval()

    if not os.path.exists(MODEL_PATH):
        print(f"BŁĄD: Nie znaleziono pliku wag: {MODEL_PATH}")
        sys.exit(1)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    
    new_state = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace("module.", "")
        new_state[name] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    
    print("-" * 30)
    cbam_loaded = False
    for k in new_state.keys():
        if "cbam" in k:
            cbam_loaded = True
            break
            
    if cbam_loaded:
        print("SUKCES: Wagi CBAM (wewnątrz bloków) zostały wykryte.")
    else:
        print("ALARM: Plik .pth nie zawiera warstw CBAM wewnątrz bloków!")
        
    if len(missing) == 0:
        print(" Wszystkie wagi załadowane idealnie.")
    else:
        print(f"Brakujące klucze: {missing[:3]}...")
    print("-" * 30)

    return model

def align_face(img, landmarks_dict):
    if img is None: return None
    try:
        if not landmarks_dict:
            return cv2.resize(img, (112, 112))
        src = np.array([
            landmarks_dict['right_eye'], landmarks_dict['left_eye'],
            landmarks_dict['nose'],
            landmarks_dict['mouth_right'], landmarks_dict['mouth_left']
        ], dtype=np.float32)
        dst = REFERENCE_POINTS_112
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        return cv2.warpAffine(img, tform.params[0:2, :], (112, 112), borderValue=0.0)
    except:
        return cv2.resize(img, (112, 112))

def apply_occlusion(img):
    if img is None: return None
    occ = img.copy()
    y = 52; h_bar = OCCLUSION_SIZE // 2
    cv2.rectangle(occ, (0, y - h_bar), (112, y + h_bar), (0,0,0), -1)
    return occ

def get_embeddings(model, images):
    if not images: return [], []
    tensors = []
    valid_indices = []
    for i, img in enumerate(images):
        if img is None: continue
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(transform_pipeline(rgb))
            valid_indices.append(i)
        except: pass
    if not tensors: return [], []
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        feats = model(batch).cpu().numpy()
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / (norms + 1e-10), valid_indices

def discover_local_data():
    print(f"Skanowanie {BASE_DATA_DIR}...")
    identities_data = defaultdict(list)
    id_folders = [d for d in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, d))]
    for id_name in tqdm(id_folders):
        id_path = os.path.join(BASE_DATA_DIR, id_name)
        for p_folder in os.listdir(id_path):
            p_path = os.path.join(id_path, p_folder)
            if not os.path.isdir(p_path): continue
            jpg_file = next((f for f in os.listdir(p_path) if f.lower().endswith('.jpg')), None)
            if jpg_file:
                json_file = os.path.splitext(jpg_file)[0] + ".json"
                full_json = os.path.join(p_path, json_file)
                identities_data[id_name].append({
                    'jpg': os.path.join(p_path, jpg_file),
                    'json': full_json if os.path.exists(full_json) else None
                })
    return identities_data

def build_gallery_and_query(model, identities_data):
    if os.path.exists(FAISS_INDEX_FILE):
        print("Wykryto starą galerię - USUWANIE (Wymagane dla nowej architektury)...")
        import shutil
        shutil.rmtree(METRICS_DIR, ignore_errors=True)
        os.makedirs(METRICS_DIR, exist_ok=True)

    index = None; mapping = {}; query_tasks = []; gallery_vectors = []
    sorted_ids = sorted(identities_data.keys())
    print("Przetwarzanie danych...")
    
    for id_name in tqdm(sorted_ids):
        images = sorted(identities_data[id_name], key=lambda x: x['jpg'])
        split = max(1, len(images) // 2)
        gallery_files = images[:split]
        query_files = images[split:]
        
        batch_imgs = []
        for item in gallery_files:
            img = cv2.imread(item['jpg'])
            lms = None
            if item['json']:
                try: 
                    with open(item['json']) as f: lms = json.load(f).get('landmarks')
                except: pass
            batch_imgs.append(align_face(img, lms))
        
        embs, _ = get_embeddings(model, batch_imgs)
        if len(embs) > 0:
            mean_emb = np.mean(embs, axis=0)
            mean_emb /= (np.linalg.norm(mean_emb) + 1e-10)
            gallery_vectors.append(mean_emb)
            mapping[str(len(gallery_vectors)-1)] = id_name
            
        for item in query_files:
            query_tasks.append({'gt_id': id_name, 'jpg': item['jpg'], 'json': item['json']})

    gallery_vectors = np.array(gallery_vectors).astype('float32')
    index = faiss.IndexFlatIP(512)
    index.add(gallery_vectors)
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_MAPPING_FILE, 'w') as f: json.dump(mapping, f)
    return index, mapping, query_tasks

def run_evaluation_csv(model, index, mapping, query_tasks):
    print("Ewaluacja...")
    top1 = 0; top3 = 0; total = 0
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "top1", "sim1", "top2", "sim2", "top3", "sim3", "found"])
        
        for i in tqdm(range(0, len(query_tasks), BATCH_SIZE)):
            batch = query_tasks[i:i+BATCH_SIZE]
            imgs = []
            for task in batch:
                img = cv2.imread(task['jpg'])
                lms = None
                if task['json']:
                    try: 
                        with open(task['json']) as jf: lms = json.load(jf).get('landmarks')
                    except: pass
                aligned = align_face(img, lms)
                imgs.append(apply_occlusion(aligned))
            
            embs, valid_map = get_embeddings(model, imgs)
            if len(embs) == 0: continue
            
            D, I = index.search(embs.astype('float32'), 3)
            
            for k, original_idx in enumerate(valid_map):
                task = batch[original_idx]
                gt = task['gt_id']
                found = False
                res_row = [gt]
                
                for n in range(3):
                    pred = mapping.get(str(I[k][n]), "Unknown")
                    res_row.extend([pred, f"{D[k][n]:.4f}"])
                    if pred == gt: found = True
                
                res_row.append(found)
                writer.writerow(res_row)
                
                if mapping.get(str(I[k][0])) == gt: top1 += 1
                if found: top3 += 1
                total += 1

    print(f"\nRank-1: {top1/total:.2%}")
    print(f"Rank-3: {top3/total:.2%}")

if __name__ == "__main__":
    os.makedirs(METRICS_DIR, exist_ok=True)
    model = load_model()
    data = discover_local_data()
    idx, map, q = build_gallery_and_query(model, data)
    run_evaluation_csv(model, idx, map, q)