import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from backbone_iresnet import iresnet50  # Twój plik backbone_iresnet.py

# --- KONFIGURACJA ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PTH_PATH = "best_model_res50_occlusion.pth"
ONNX_PATH = "best_model_res50_occlusion.pth.onnx"

# --- Funkcja do ładowania modelu ---
def load_model(pth_path):
    model = iresnet50(weights_path=None)
    if os.path.exists(pth_path):
        state = torch.load(pth_path, map_location="cpu")
        if 'state_dict' in state:
            state = state['state_dict']
        # Dostosowanie kluczy, jeśli są z "module."
        new_state = {}
        for k, v in state.items():
            new_key = k.replace("module.", "")
            new_state[new_key] = v
        model.load_state_dict(new_state, strict=False)
        print(f"✅ Załadowano model z {pth_path}")
    else:
        print(f"❌ Nie znaleziono pliku {pth_path}")
    model.eval().to(DEVICE)
    return model

# --- Funkcja konwertująca do ONNX ---
def export_to_onnx(model, onnx_path):
    # Przykładowy input 112x112 RGB
    dummy_input = torch.randn(1, 3, 112, 112, device=DEVICE)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ Model wyeksportowany do ONNX: {onnx_path}")

# --- MAIN ---
if __name__ == "__main__":
    model = load_model(PTH_PATH)
    export_to_onnx(model, ONNX_PATH)
