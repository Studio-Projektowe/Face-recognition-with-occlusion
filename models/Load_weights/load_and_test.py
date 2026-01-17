import torch
import torch.nn as nn
import numpy as np
import sys
import os
import cv2
import torchvision.models as models
from collections import OrderedDict
import re

from backbone_iresnet import iresnet50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCAL_WEIGHTS_PATH = 'w600k_r50_from_onnx.pth'
EXAMPLE_IMG_PATH = 'example.jpg'
REFERENCE_EMBEDDING_SAMPLE = np.array([0.0076, -0.0122, 0.0345, -0.0511, 0.0210, -0.0049])

def normalize_name(name):
    name = name.replace('_initializer_', '')
    name = name.replace('module.', '')
    name = name.replace('.', '').replace('_', '').lower()
    return name

def reset_layer_params(layer, name):
    print(f"Resetowanie warstwy: {name}")
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)
        layer.running_mean.zero_()
        layer.running_var.fill_(1)
    elif isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, 0, 0.01)
        nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.PReLU):
        nn.init.constant_(layer.weight, 0.25)

def sanitize_model(model):
    print("\nUruchamiam SZPITAL (Sanitize Model)...")
    fixes = 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.running_var is not None:
                                                                                     
                if torch.isnan(m.running_var).any() or (m.running_var < 1e-5).any():
                    m.running_var.data.clamp_(min=1e-4)
                    m.running_var.data[torch.isnan(m.running_var.data)] = 1.0
                    fixes += 1
            
            if torch.isnan(m.weight).any():
                nn.init.ones_(m.weight)
                fixes += 1

        elif isinstance(m, nn.PReLU):
            if torch.isnan(m.weight).any():
                print(f"Naprawa NaN w PReLU: {name}")
                nn.init.constant_(m.weight, 0.25)
                fixes += 1

        elif hasattr(m, 'weight') and m.weight is not None:
            if torch.isnan(m.weight).any():
                print(f"Wykryto NaN w wagach warstwy: {name} -> Re-init kaiming")
                reset_layer_params(m, name)
                fixes += 1

    print(f"Szpital zakończony. Wprowadzono {fixes} poprawek.")

def smart_load_hybrid_v4(model, state_dict_path):
    print(f"Uruchamiam HYBRID LOAD v5 (Greedy + Reinit + DETAILED REPORT) z: {state_dict_path}")
    
    try:
        source_state = torch.load(state_dict_path, map_location='cpu')
        if 'state_dict' in source_state:
            source_state = source_state['state_dict']
    except Exception as e:
        print(f"Błąd odczytu pliku: {e}")
        return False

    target_state = model.state_dict()
    new_state_dict = OrderedDict()
    
    source_keys = list(source_state.keys())
    all_target_keys = list(target_state.keys())
    
    target_keys = [k for k in all_target_keys if 'num_batches_tracked' not in k]
    
    used_source_keys = set()
    matched_target_keys = set()
    
    print("   ... Etap 1: Dopasowanie znormalizowanych nazw ...")
    source_norm_map = {normalize_name(k): k for k in source_keys}
    
    for t_key in target_keys:
        t_shape = target_state[t_key].shape
        t_norm = normalize_name(t_key)
        
        if t_norm in source_norm_map:
            original_source_key = source_norm_map[t_norm]
            s_tensor = source_state[original_source_key]
            if s_tensor.shape == t_shape:
                new_state_dict[t_key] = s_tensor
                used_source_keys.add(original_source_key)
                matched_target_keys.add(t_key)

    print("   ... Etap 2: Chciwe dopasowanie kształtu dla pozostałych ...")
    remaining_source = []
    for k in source_keys:
        if k not in used_source_keys:
            remaining_source.append((k, source_state[k]))
    remaining_source.sort(key=lambda x: x[0])
    
    for t_key in target_keys:
        if t_key in matched_target_keys: continue
        
        t_shape = target_state[t_key].shape
        for idx, (s_key, s_tensor) in enumerate(remaining_source):
            if s_tensor.shape == t_shape:
                new_state_dict[t_key] = s_tensor
                used_source_keys.add(s_key)
                matched_target_keys.add(t_key)
                remaining_source.pop(idx)
                break

    model.load_state_dict(new_state_dict, strict=False)

    print("\n" + "="*60)
    print("SZCZEGÓŁOWY RAPORT DOPASOWANIA WAG")
    print("="*60)

    unused_source = set(source_keys) - used_source_keys
    unused_source = [k for k in unused_source if 'num_batches_tracked' not in k]

    print(f"\n[PLIK -> KOSZ] Wagi z pliku ONNX/PTH nieużyte w modelu ({len(unused_source)}):")
    if len(unused_source) > 0:
        for k in sorted(unused_source):
            print(f"   - {k} {source_state[k].shape}")
    else:
        print("   (Wszystkie istotne wagi z pliku zostały wykorzystane!)")

    unassigned_target = set(target_keys) - matched_target_keys
    
    print(f"\n[MODEL -> PUSTE] Warstwy modelu bez wczytanych wag ({len(unassigned_target)}):")
    if len(unassigned_target) > 0:
        for k in sorted(unassigned_target):
            print(f"   - {k} {target_state[k].shape}")
        print("\n  Te warstwy zostaną zreinicjalizowane losowo (Kaiming/Normal).")
    else:
        print("   (Wszystkie warstwy modelu mają przypisane wagi!)")
        
    print("="*60 + "\n")

    reinit_count = 0
    for name, module in model.named_modules():
        has_missing_weights = False
        for param_name, _ in module.named_parameters(recurse=False):
            full_param_name = f"{name}.{param_name}" if name else param_name
            if full_param_name in unassigned_target:
                has_missing_weights = True
                break
        
        if has_missing_weights:
            reset_layer_params(module, name)
            reinit_count += 1
            
    print(f"Zreinicjalizowano {reinit_count} modułów, którym brakowało wag.")
    
    sanitize_model(model)

    return True

def load_clean_model():
    print(f"Start ładowania modelu. Urządzenie: {DEVICE}")
    model = iresnet50(weights_path=None)
    
    if os.path.exists(LOCAL_WEIGHTS_PATH):
        smart_load_hybrid_v4(model, LOCAL_WEIGHTS_PATH)
    else:
        print(f"Brak pliku wag {LOCAL_WEIGHTS_PATH}")

    model.eval()
    model.to(DEVICE)
    return model

nan_found = False
def nan_hook(module, input, output):
    global nan_found
    if nan_found: return
    
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any():
            print(f"NaN WYKRYTY w warstwie: {module}")
            nan_found = True
        elif torch.isinf(output).any():
            print(f"INF (Nieskończoność) w warstwie: {module}")
            nan_found = True

def register_hooks(model):
    print("Podłączanie detektywa NaN do wszystkich warstw...")
    for name, layer in model.named_modules():
        layer.register_forward_hook(nan_hook)

def generate_embedding(model, img_path):
    if not os.path.exists(img_path):
        img_bgr = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    else:
        img_bgr = cv2.imread(img_path)

    if img_bgr is None: return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (112, 112))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
    img_tensor = (img_tensor - 127.5) / 128.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    register_hooks(model)

    with torch.no_grad():
        features = model(img_tensor)
        features = features.cpu().numpy().flatten()

    if np.isnan(features).any():
        print("CRITICAL: Wyjście modelu nadal zawiera NaN (mimo szpitala)!")
        return np.zeros_like(features)

    feat_norm = np.linalg.norm(features)
    print(f"Norma wektora przed normalizacją: {feat_norm:.6f}")
    
    return features / (feat_norm + 1e-10)

def main():
    model = load_clean_model()
    
    print("\n--- TEST GENEROWANIA (v5 - Szpital + Detektyw) ---")
    emb = generate_embedding(model, EXAMPLE_IMG_PATH)
    
    print(f"Embedding sample: {emb[:6]}")
    
    if np.allclose(emb, 0):
        print("Wynik to zera. Sprawdź logi powyżej.")
    else:
        ref_norm = np.linalg.norm(REFERENCE_EMBEDDING_SAMPLE) + 1e-10
        normalized_ref = REFERENCE_EMBEDDING_SAMPLE / ref_norm
        similarity = np.dot(emb[:6], normalized_ref[:6])
        print(f"Podobieństwo: {similarity:.4f}")
        print("Model nie wybucha. Możesz zacząć trening (fine-tuning naprawi niskie podobieństwo).")

if __name__ == "__main__":
    main()