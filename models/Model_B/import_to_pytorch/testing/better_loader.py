import torch
import torch.nn as nn
from collections import OrderedDict
import os
import re

# Import Twojej architektury
from backbone_iresnet import iresnet50

# Ustawienia
DEVICE = torch.device("cpu") # Do konwersji wystarczy CPU
RAW_WEIGHTS_PATH = 'w600k_r50_from_onnx.pth'
OUTPUT_CLEAN_PATH = 'iresnet50_clean.pth'

# --- NARZĘDZIA POMOCNICZE (z Twojego kodu) ---
def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def reset_layer_params(layer, name):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(layer.weight, 1) # Ważne: 1, żeby sygnał przechodził!
        nn.init.constant_(layer.bias, 0)
        layer.running_mean.zero_()
        layer.running_var.fill_(1)
    elif isinstance(layer, nn.PReLU):
        nn.init.constant_(layer.weight, 0.25)

def sanitize_model(model):
    print("🏥 Uruchamiam SZPITAL...")
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.running_var is not None:
                m.running_var.data.clamp_(min=1e-5)
                m.running_var.data[torch.isnan(m.running_var.data)] = 1.0
            if torch.isnan(m.weight).any(): nn.init.ones_(m.weight)
            if torch.isnan(m.bias).any(): nn.init.zeros_(m.bias)
        elif isinstance(m, nn.PReLU):
            if torch.isnan(m.weight).any(): nn.init.constant_(m.weight, 0.25)

def smart_load_sequential_and_save(model, state_dict_path):
    print(f"🔄 Ładowanie sekwencyjne z {state_dict_path}...")
    
    try:
        source_state = torch.load(state_dict_path, map_location='cpu')
        if 'state_dict' in source_state: source_state = source_state['state_dict']
        elif 'model' in source_state: source_state = source_state['model']
    except Exception as e:
        print(f"❌ Błąd: {e}")
        return

    target_state = model.state_dict()
    new_state_dict = OrderedDict()
    
    # Sortowanie kluczy źródłowych
    source_keys = list(source_state.keys())
    source_keys.sort(key=natural_keys)
    source_values = [source_state[k] for k in source_keys]
    
    source_idx = 0
    assigned = 0
    missing_layers = []

    # Mapowanie
    for t_key, t_param in target_state.items():
        if 'num_batches_tracked' in t_key: continue
        
        t_shape = t_param.shape
        found = False
        
        search_idx = source_idx
        while search_idx < len(source_values):
            s_tensor = source_values[search_idx]
            if s_tensor.shape == t_shape:
                new_state_dict[t_key] = s_tensor
                source_values.pop(search_idx) # Usuwamy zużyty
                found = True
                assigned += 1
                break
            search_idx += 1
        
        if not found:
            missing_layers.append(t_key)

    print(f"✅ Przypisano {assigned} warstw.")
    
    # Ładowanie tego co mamy
    model.load_state_dict(new_state_dict, strict=False)
    
    # Re-inicjalizacja brakujących (Kluczowe dla treningu!)
    print(f"⚠️ Re-inicjalizacja {len(missing_layers)} brakujących warstw...")
    for name, module in model.named_modules():
        has_missing = False
        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{name}.{param_name}" if name else param_name
            if full_name in missing_layers:
                has_missing = True
        
        if has_missing:
            reset_layer_params(module, name)

    # Szpital
    sanitize_model(model)
    
    # ZAPIS
    print(f"💾 Zapisywanie naprawionego modelu do: {OUTPUT_CLEAN_PATH}")
    torch.save(model.state_dict(), OUTPUT_CLEAN_PATH)
    print("🎉 Gotowe! Używaj teraz pliku 'iresnet50_clean.pth' w swoim skrypcie treningowym.")

def main():
    model = iresnet50(weights_path=None)
    smart_load_sequential_and_save(model, RAW_WEIGHTS_PATH)

if __name__ == "__main__":
    main()