# load_model.py (Właściwe rozwiązanie)

import torch
from backbone_iresnet import iresnet50

# 1. Inicjalizacja pustej architektury
model_a = iresnet50() 

# 2. Ładowanie Twoich wag .pth
loaded_state_dict = torch.load('w600k_r50_from_onnx.pth')

# 3. KLUCZOWY MOMENT: Ładowanie wag, z ignorowaniem niezgodności
model_a.load_state_dict(loaded_state_dict, strict=False) # <--- TO JEST TO!

print("Model A (IResNet) załadowany i gotowy do fine-tuningu.")