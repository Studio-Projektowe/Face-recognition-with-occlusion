import onnx
from onnx2pytorch import ConvertModel
import torch
import cv2
import numpy as np
import os
import sys

ONNX_PATH = "w600k_r50.onnx"
OUTPUT_PTH_PATH = "w600k_r50_from_onnx.pth"
EXAMPLE_IMG_PATH = "example.jpg"

if not os.path.exists(ONNX_PATH):
    print(f"BŁĄD: Nie znaleziono pliku ONNX pod ścieżką: {ONNX_PATH}")
    sys.exit(1)

print(f"Ładowanie modelu ONNX z: {ONNX_PATH}")
onnx_model_definition = onnx.load(ONNX_PATH)
pytorch_model = ConvertModel(onnx_model_definition)

print("Konwersja na model PyTorch zakończona pomyślnie.")

try:
    torch.save(pytorch_model.state_dict(), OUTPUT_PTH_PATH)
    print(f"Wagi zapisane do: {OUTPUT_PTH_PATH}")
except Exception as e:
    print(f"BŁĄD zapisu wag: {e}")

print("\nRozpoczynanie testu na skonwertowanym modelu...")

pytorch_model.eval() 

if not os.path.exists(EXAMPLE_IMG_PATH):
    print(f"BŁĄD: Nie znaleziono pliku {EXAMPLE_IMG_PATH} dla testu.")
    img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
else:
    img = cv2.imread(EXAMPLE_IMG_PATH)[:,:,::-1] # BGR->RGB
    if img.shape[0] != 112 or img.shape[1] != 112:
        img = cv2.resize(img, (112, 112))

inp = (img.astype(np.float32) - 127.5) / 127.5
inp = np.transpose(inp, (2, 0, 1))[None, ...].astype('float32')
inp_tensor = torch.from_numpy(inp)

print(f"Tensor wejściowy: {inp_tensor.shape}")

with torch.no_grad():
    out = pytorch_model(inp_tensor)
    
if len(out.shape) > 2:
    out = out.squeeze()

print(f"Wymiar wektora (embeddingu): {out.shape}")