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
from concurrent.futures import ThreadPoolExecutor

try:
    from google.cloud import storage
    from skimage import transform as trans
except ImportError:
    print("BŁĄD: Brak bibliotek.")
    print("pip install google-cloud-storage scikit-image")
    sys.exit(1)

MODEL_PATH = 'Aligned_pretrained_CBAM_L1.pth'
GCS_BUCKET_NAME = 'face-recognition-476110_cloudbuild'
GCS_PREFIX = 'test'
METRICS_DIR = 'metrics_cbam_gcs'

FAISS_INDEX_FILE = os.path.join(METRICS_DIR, 'gallery.index')
FAISS_MAPPING_FILE = os.path.join(METRICS_DIR, 'gallery_map.json')
RESULTS_CSV = os.path.join(METRICS_DIR, 'evaluation_results.csv')
OCCLUSION_OUTPUT_DIR = os.path.join(METRICS_DIR, 'occlusion_photos_eval')

BATCH_SIZE = 32
DOWNLOAD_THREADS = 16
OCCLUSION_SIZE = 20
K_NEIGHBORS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REFERENCE_POINTS_112 = np.array([
    [30.2946, 51.6963],  # left eye
    [65.5318, 51.5014],  # right eye
    [48.0252, 71.7366],  # nose
    [33.5493, 92.3655],  # left mouth
    [62.7299, 92.2041]   # right mouth
], dtype=np.float32)


def norm_crop(img, landmark, image_size=112):
    """
    Wyrównuje twarz tak, aby oczy, nos i usta były w standardowych pozycjach.
    landmark: lista 5 punktów [lewe_oko, prawe_oko, nos, lewe_usta, prawe_usta]
    """
    M = None
    if image_size == 112:
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041] ], dtype=np.float32)
    else:
        src = REFERENCE_POINTS_112 # fallback
    
    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]
    
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def align_face_wrapper(img, landmarks_dict):
    """Wrapper parsujący słownik landmarków z JSONa do formatu numpy."""
    try:
        if landmarks_dict is None: return cv2.resize(img, (112, 112))
        
        kps = np.array([
            landmarks_dict['left_eye'],
            landmarks_dict['right_eye'],
            landmarks_dict['nose'],
            landmarks_dict['mouth_left'],
            landmarks_dict['mouth_right']
        ], dtype=np.float32)
        
        return norm_crop(img, kps, 112)
    except Exception as e:
        return cv2.resize(img, (112, 112))


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
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, use_cbam=False):
        super(IBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride
        self.use_cbam = use_cbam
        if self.use_cbam: self.cbam = CBAM(planes, 16)
    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.use_cbam: out = self.cbam(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return out

class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False, use_cbam=False):
        super(IResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None: replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2, use_cbam=use_cbam)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], use_cbam=use_cbam)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], use_cbam=use_cbam)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], use_cbam=use_cbam)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, use_cbam=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, self.dilation, use_cbam=use_cbam))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                use_cbam=use_cbam))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x

class FrontEndCBAM(nn.Module):
    def __init__(self, backbone, cbam_channels=64):
        super(FrontEndCBAM, self).__init__()
        self.backbone = backbone
        self.cbam1 = CBAM(cbam_channels, ratio=16, kernel_size=7)
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.prelu(x)
        x = self.cbam1(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.bn2(x)
        x = torch.flatten(x, 1)
        x = self.backbone.dropout(x)
        x = self.backbone.fc(x)
        x = self.backbone.features(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_gcs_bucket():
    storage_client = storage.Client()
    return storage_client.bucket(GCS_BUCKET_NAME)

def download_file_content(bucket, blob_path):
    try: return bucket.blob(blob_path).download_as_bytes()
    except: return None

def initialize_custom_model():
    print(f"Budowanie modelu FrontEndCBAM...")
    backbone = IResNet(IBasicBlock, [3, 4, 14, 3], use_cbam=False)
    model = FrontEndCBAM(backbone, cbam_channels=64)
    model.to(DEVICE)
    model.eval()

    if not os.path.exists(MODEL_PATH):
        print(f"BŁĄD: Brak pliku {MODEL_PATH}")
        sys.exit(1)

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            if "cbam1.ca.fc" in k and "weight" in k and v.dim() == 2:
                v = v.unsqueeze(-1).unsqueeze(-1)
            if k.startswith("cbam1"): new_state_dict[k] = v
            elif k.startswith("backbone."): new_state_dict[k] = v
            else: new_state_dict[f"backbone.{k}"] = v

        model.load_state_dict(new_state_dict, strict=True)
        print("SUKCES: Wagi załadowane! (strict=True)")
    except Exception as e:
        print(f"BŁĄD wag: {e}")
        sys.exit(1)
    return model

def get_embeddings_batch(model, aligned_images_list):
    """Oblicza embeddingi dla listy JUŻ WYRÓWNANYCH obrazów 112x112"""
    if not aligned_images_list: return []
    tensors = []
    valid_indices = []
    for i, img in enumerate(aligned_images_list):
        if img is None: continue
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(transform(img_rgb))
            valid_indices.append(i)
        except: pass
    if not tensors: return []
    
    batch_tensor = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        embeddings = model(batch_tensor).cpu().numpy()
        
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)
    return embeddings, valid_indices

def discover_gcs_file_structure(bucket):
    print(f"Skanowanie GCS...")
    blobs = list(bucket.list_blobs(prefix=GCS_PREFIX))
    jpg_files = [b.name for b in blobs if b.name.lower().endswith('.jpg')]
    all_blob_names = set(b.name for b in blobs)
    
    if not jpg_files: return None, None
    print(f"Znaleziono {len(jpg_files)} plików.")
    
    identity_to_imgfolders = {}
    image_pairs = {} # key: folder_path, val: {jpg, json}

    for jpg_path in tqdm(jpg_files, desc="Indeksowanie"):
        base = jpg_path.rsplit('.', 1)[0]
        json_path = base + ".json"
        
        parts = jpg_path.split('/')
        if len(parts) < 3: continue
        
        folder_path = "/".join(parts[:-1])
        identity = "/".join(parts[:-2])
        
        unique_key = jpg_path 
        
        if identity not in identity_to_imgfolders:
            identity_to_imgfolders[identity] = []
        identity_to_imgfolders[identity].append(unique_key)
        
        image_pairs[unique_key] = {'jpg': jpg_path, 'json': json_path if json_path in all_blob_names else None}

    return identity_to_imgfolders, image_pairs

def build_faiss_gallery_gcs(model, bucket, identity_to_imgfolders, image_pairs):
    if os.path.exists(FAISS_INDEX_FILE):
        print("Galeria istnieje.")
        return True

    print(f"Budowanie Galerii (z Align)...")
    gallery_embeddings = []
    index_to_id_map = {}
    cnt = 0
    all_tasks = []

    for id_path, keys in identity_to_imgfolders.items():
        keys = sorted(keys)
        gallery_keys = keys[:max(1, len(keys) // 2)] # 50%
        for k in gallery_keys:
            data = image_pairs.get(k, {})
            all_tasks.append((id_path.split('/')[-1], data['jpg'], data['json']))

    for i in tqdm(range(0, len(all_tasks), BATCH_SIZE), desc="Galeria"):
        batch = all_tasks[i:i+BATCH_SIZE]
        images, metadatas = [None]*len(batch), [None]*len(batch)

        with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as executor:
            f_jpg = {executor.submit(download_file_content, bucket, t[1]): i for i, t in enumerate(batch)}
            f_json = {executor.submit(download_file_content, bucket, t[2]): i for i, t in enumerate(batch) if t[2]}
            
            for f in f_jpg: 
                d = f.result()
                if d: images[f_jpg[f]] = cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR)
            for f in f_json:
                d = f.result()
                if d: metadatas[f_json[f]] = json.loads(d)

        aligned_imgs = []
        for idx, img in enumerate(images):
            if img is None: 
                aligned_imgs.append(None)
                continue
            
            md = metadatas[idx]
            lms = md.get('landmarks') if md else None
            aligned = align_face_wrapper(img, lms)
            aligned_imgs.append(aligned)

        embs_res = get_embeddings_batch(model, aligned_imgs)
        if not embs_res: continue
        embs, valid_indices = embs_res
        
        for k, v_idx in enumerate(valid_indices):
            gallery_embeddings.append(embs[k])
            index_to_id_map[cnt] = batch[v_idx][0]
            cnt += 1

    if not gallery_embeddings: return False
    index = faiss.IndexFlatIP(gallery_embeddings[0].shape[0])
    index.add(np.array(gallery_embeddings).astype('float32'))
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_MAPPING_FILE, 'w') as f: json.dump(index_to_id_map, f)
    return True

def apply_occlusion_aligned(aligned_img):
    """Nakłada pasek na JUŻ WYRÓWNANY obraz 112x112."""
    occ = aligned_img.copy()
    h, w = 112, 112
    y = 52
    h_bar = OCCLUSION_SIZE // 2
    cv2.rectangle(occ, (0, y - h_bar), (w, y + h_bar), (0,0,0), -1)
    return occ

def run_evaluation(model, bucket, identity_to_imgfolders, image_pairs):
    print("Testowanie (z Align)...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(FAISS_MAPPING_FILE, 'r') as f: index_map = json.load(f)
    os.makedirs(OCCLUSION_OUTPUT_DIR, exist_ok=True)
    
    query_tasks = []
    for id_path, keys in identity_to_imgfolders.items():
        keys = sorted(keys)
        query_keys = keys[max(1, len(keys) // 2):]
        gt_id = id_path.split('/')[-1]
        for k in query_keys:
            data = image_pairs.get(k)
            query_tasks.append({'id': gt_id, 'jpg': data['jpg'], 'json': data['json']})

    total, top1, top3 = 0, 0, 0
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "top1", "sim1", "found_top3"])

        for i in tqdm(range(0, len(query_tasks), BATCH_SIZE), desc="Test"):
            batch = query_tasks[i:i+BATCH_SIZE]
            images, metadatas = [None]*len(batch), [None]*len(batch)
            
            with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as executor:
                f_jpg = {executor.submit(download_file_content, bucket, t['jpg']): idx for idx, t in enumerate(batch)}
                f_json = {executor.submit(download_file_content, bucket, t['json']): idx for idx, t in enumerate(batch) if t['json']}
                for f in f_jpg:
                    d = f.result()
                    if d: images[f_jpg[f]] = cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR)
                for f in f_json:
                    d = f.result()
                    if d: metadatas[f_json[f]] = json.loads(d)

            processed_imgs = []
            valid_batch_indices = []
            
            for idx, img in enumerate(images):
                if img is None: continue
                
                md = metadatas[idx]
                lms = md.get('landmarks') if md else None
                aligned = align_face_wrapper(img, lms)
                
                occ_img = apply_occlusion_aligned(aligned)
                
                processed_imgs.append(occ_img)
                valid_batch_indices.append(idx)
                
                if (total + idx) % 100 == 0:
                     cv2.imwrite(f"{OCCLUSION_OUTPUT_DIR}/{batch[idx]['id']}_{total+idx}.jpg", occ_img)

            embs_res = get_embeddings_batch(model, processed_imgs)
            if not embs_res: continue
            embs, valid_emb_indices = embs_res
            
            D, I = index.search(embs.astype('float32'), K_NEIGHBORS)
            
            for k, emb_idx in enumerate(valid_emb_indices):
                orig_idx = valid_batch_indices[emb_idx]
                gt_id = batch[orig_idx]['id']
                
                res_row = [gt_id]
                found = False
                batch_top1 = False
                
                for n in range(K_NEIGHBORS):
                    pid = index_map.get(str(I[k][n]), "N/A")
                    res_row.extend([pid, f"{D[k][n]:.4f}"])
                    if pid == gt_id:
                        found = True
                        if n == 0: batch_top1 = True
                
                if batch_top1: top1 += 1
                if found: top3 += 1
                total += 1
                res_row.append(found)
                writer.writerow(res_row)

    if total > 0:
        print(f"\nWYNIKI: Acc@1: {top1/total:.2%}, Acc@3: {top3/total:.2%}")

def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    bucket = get_gcs_bucket()
    model = initialize_custom_model()
    ids, pairs = discover_gcs_file_structure(bucket)
    if ids and build_faiss_gallery_gcs(model, bucket, ids, pairs):
        run_evaluation(model, bucket, ids, pairs)

if __name__ == "__main__":
    main()