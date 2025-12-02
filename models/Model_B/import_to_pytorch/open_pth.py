import torch
import os

path = 'w600k_r50_from_onnx.pth'
if os.path.exists(path):
    # Wyświetl klucze i kształty, aby sprawdzić, czy nie są puste
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    
    # Wyświetl pierwszy klucz i jego wartość (powinna być niezerowa)
    first_key = next(iter(state_dict))
    print(f"Pierwszy klucz w pliku: {first_key}")
    print(f"Kształt pierwszej wagi: {state_dict[first_key].shape}")
    print(f"Pierwsze 10 wartości (sprawdź, czy nie są zerami): {state_dict[first_key].flatten()[:10]}")