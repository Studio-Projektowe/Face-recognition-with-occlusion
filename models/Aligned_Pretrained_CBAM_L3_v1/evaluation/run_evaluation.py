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

BASE_DATA_DIR = '../../webface_112x112/test'
MODEL_PATH = './fixed_cbam_best.pth'
METRICS_DIR = 'metrics_local_occlusion_dobicie'

FAISS_INDEX_FILE = os.path.join(METRICS_DIR, 'gallery.index')
FAISS_MAPPING_FILE = os.path.join(METRICS_DIR, 'gallery_map.json')
RESULTS_CSV = os.path.join(METRICS_DIR, 'evaluation_results.csv')
OCCLUSION_OUTPUT_DIR = os.path.join(METRICS_DIR, 'occlusion_debug')

BATCH_SIZE = 32
OCCLUSION_SIZE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REFERENCE_POINTS_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)



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
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
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
    def __init__(self, block, layers, dropout=0, num_features=512, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        if dilate: self.dilation *= stride; stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion, eps=1e-05))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, self.dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks): layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.prelu(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.cbam_layer(x)
        x = self.layer4(x); x = self.bn2(x); x = torch.flatten(x, 1)
        x = self.dropout(x); x = self.fc(x); x = self.features(x)
        return x


transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_model():
    print(f"Ładowanie modelu z {MODEL_PATH}...")
    model = IResNetWithCBAM(IBasicBlock, [3, 4, 14, 3])
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
        
        if name.startswith("backbone."):
            name = name.replace("backbone.", "")
            
        if "cbam1" in name:
            name = name.replace("cbam1", "cbam_layer")
            
        new_state[name] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    
    if len(missing) > 0:
        print(f"⚠️ Brakujące klucze (sprawdź czy to nie core): {missing[:5]}...")
    
    cbam_in = any("cbam_layer" in k for k in new_state.keys())
    if cbam_in:
        print("✅ Wykryto i załadowano wagi CBAM.")
    else:
        print("⚠️ OSTRZEŻENIE: Brak wag CBAM w pliku .pth!")

    return model

def align_face(img, landmarks_dict):
    """Wyrównuje twarz na podstawie słownika landmarków."""
    if img is None: return None
    try:
        if not landmarks_dict:
            return cv2.resize(img, (112, 112))
        
        src = np.array([
            landmarks_dict['right_eye'],
            landmarks_dict['left_eye'],
            landmarks_dict['nose'],
            landmarks_dict['mouth_right'],
            landmarks_dict['mouth_left']
        ], dtype=np.float32)

        dst = REFERENCE_POINTS_112
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]
        return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    except Exception as e:
        print(f"Align Error: {e}")
        return cv2.resize(img, (112, 112))

def apply_occlusion(img):
    """Rysuje czarny pasek na oczach (dla obrazu 112x112)."""
    if img is None: return None
    occ = img.copy()
    y = 52
    h_bar = OCCLUSION_SIZE // 2
    cv2.rectangle(occ, (0, y - h_bar), (112, y + h_bar), (0,0,0), -1)
    return occ

def get_embeddings(model, images):
    """Batch processing embeddingów."""
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
    feats = feats / (norms + 1e-10)
    return feats, valid_indices

def discover_local_data():
    """Skanuje folder test/ i zwraca strukturę danych."""
    print(f"Skanowanie folderu {BASE_DATA_DIR}...")
    if not os.path.exists(BASE_DATA_DIR):
        print(f"BŁĄD: Folder {BASE_DATA_DIR} nie istnieje.")
        sys.exit(1)

    identities_data = defaultdict(list)
    
    id_folders = [d for d in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, d))]
    
    for id_name in tqdm(id_folders, desc="Indeksowanie"):
        id_path = os.path.join(BASE_DATA_DIR, id_name)
        
        photo_folders = [d for d in os.listdir(id_path) if os.path.isdir(os.path.join(id_path, d))]
        
        for p_folder in photo_folders:
            p_path = os.path.join(id_path, p_folder)
            
            files = os.listdir(p_path)
            jpg_file = next((f for f in files if f.lower().endswith('.jpg')), None)
            
            if jpg_file:
                full_jpg_path = os.path.join(p_path, jpg_file)
                base_name = os.path.splitext(jpg_file)[0]
                json_file = base_name + ".json"
                full_json_path = os.path.join(p_path, json_file)
                
                if not os.path.exists(full_json_path):
                    full_json_path = None
                    
                identities_data[id_name].append({
                    'jpg': full_jpg_path,
                    'json': full_json_path
                })

    print(f"Znaleziono {len(identities_data)} tożsamości.")
    return identities_data

def build_gallery_and_query(model, identities_data):
    """Tworzy galerię (mean embedding) i listę zapytań (occlusion)."""
    
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_MAPPING_FILE):
        print("Galeria już istnieje. Wczytywanie...")
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(FAISS_MAPPING_FILE, 'r') as f:
            mapping = json.load(f)
        gallery_ready = True
    else:
        index = None
        mapping = {}
        gallery_ready = False

    query_tasks = []
    gallery_vectors = []
    gallery_ids = []

    sorted_ids = sorted(identities_data.keys())
    
    print("Przetwarzanie danych (Split 50/50)...")
    for idx, id_name in enumerate(tqdm(sorted_ids, desc="Processing IDs")):
        images_list = sorted(identities_data[id_name], key=lambda x: x['jpg'])
        
        split_idx = max(1, len(images_list) // 2)
        gallery_files = images_list[:split_idx]
        query_files = images_list[split_idx:]
        
        if not gallery_ready:
            batch_imgs = []
            for item in gallery_files:
                img = cv2.imread(item['jpg'])
                lms = None
                if item['json']:
                    with open(item['json'], 'r') as f:
                        data = json.load(f)
                        lms = data.get('landmarks')
                
                aligned = align_face(img, lms)
                batch_imgs.append(aligned)
            
            embs, _ = get_embeddings(model, batch_imgs)
            
            if len(embs) > 0:
                mean_emb = np.mean(embs, axis=0)
                mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
                
                gallery_vectors.append(mean_emb)
                mapping[str(len(gallery_vectors)-1)] = id_name
        
        for item in query_files:
            query_tasks.append({
                'gt_id': id_name,
                'jpg': item['jpg'],
                'json': item['json']
            })

    if not gallery_ready and gallery_vectors:
        gallery_vectors = np.array(gallery_vectors).astype('float32')
        index = faiss.IndexFlatIP(512)
        index.add(gallery_vectors)
        faiss.write_index(index, FAISS_INDEX_FILE)
        with open(FAISS_MAPPING_FILE, 'w') as f:
            json.dump(mapping, f)
        print(f"Zbudowano galerię: {gallery_vectors.shape[0]} wektorów.")

    return index, mapping, query_tasks

def run_evaluation_csv(model, index, mapping, query_tasks):
    print(f"Rozpoczynanie ewaluacji na {len(query_tasks)} zdjęciach...")
    os.makedirs(OCCLUSION_OUTPUT_DIR, exist_ok=True)
    
    top1 = 0
    top3 = 0
    total = 0
    
    with open(RESULTS_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["query_id", "top1_id", "sim1", "top2_id", "sim2", "top3_id", "sim3", "found_in_top3?"])
        
        for i in tqdm(range(0, len(query_tasks), BATCH_SIZE), desc="Testowanie"):
            batch = query_tasks[i:i+BATCH_SIZE]
            
            batch_imgs = []
            valid_indices = []
            
            for b_idx, task in enumerate(batch):
                img = cv2.imread(task['jpg'])
                if img is None: continue
                
                lms = None
                if task['json']:
                    try:
                        with open(task['json'], 'r') as f:
                            jd = json.load(f)
                            lms = jd.get('landmarks')
                    except: pass
                
                aligned = align_face(img, lms)
                
                occ_img = apply_occlusion(aligned)
                
                if (total + b_idx) % 100 == 0:
                     cv2.imwrite(f"{OCCLUSION_OUTPUT_DIR}/debug_{task['gt_id']}_{total+b_idx}.jpg", occ_img)
                
                batch_imgs.append(occ_img)
                valid_indices.append(b_idx)
            
            embs, valid_emb_map = get_embeddings(model, batch_imgs)
            
            if len(embs) == 0: continue
            
            D, I = index.search(embs.astype('float32'), 3)
            
            for k, emb_idx in enumerate(valid_emb_map):
                original_task_idx = valid_indices[emb_idx]
                task = batch[original_task_idx]
                gt_id = task['gt_id']
                
                res_ids = []
                res_sims = []
                
                found_in_top3 = False
                
                for n in range(3):
                    faiss_id = str(I[k][n])
                    pred_id = mapping.get(faiss_id, "Unknown")
                    score = D[k][n]
                    
                    res_ids.append(pred_id)
                    res_sims.append(f"{score:.4f}")
                    
                    if pred_id == gt_id:
                        found_in_top3 = True
                
                if res_ids[0] == gt_id:
                    top1 += 1
                if found_in_top3:
                    top3 += 1
                total += 1
                
                row = [
                    gt_id,
                    res_ids[0], res_sims[0],
                    res_ids[1], res_sims[1],
                    res_ids[2], res_sims[2],
                    found_in_top3
                ]
                writer.writerow(row)

    print("\n" + "="*30)
    if total > 0:
        print(f"Rank-1 Accuracy: {top1/total:.2%}")
        print(f"Rank-3 Accuracy: {top3/total:.2%}")
    else:
        print("Brak danych do testowania.")
    print("="*30)
    print(f"Wyniki zapisano w: {RESULTS_CSV}")

def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    model = load_model()
    
    data = discover_local_data()
    
    if not data:
        print("Nie znaleziono danych w folderze test.")
        return

    index, mapping, queries = build_gallery_and_query(model, data)
    
    if queries:
        run_evaluation_csv(model, index, mapping, queries)
    else:
        print("Brak zdjęć do testowania (Query set is empty).")

if __name__ == "__main__":
    main()