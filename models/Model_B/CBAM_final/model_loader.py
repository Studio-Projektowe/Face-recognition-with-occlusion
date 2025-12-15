import torch
import os
import sys

# Importujemy naszą "naprawioną" architekturę
from backbone_iresnet import iresnet50

def get_ready_model(weights_path, device='cuda'):
    """
    Tworzy model iresnet50, ładuje wagi z ONNX->PTH i przygotowuje do pracy.
    """
    print(f"\n🚀 [Loader] Inicjalizacja modelu IResNet50...")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Nie znaleziono pliku wag: {weights_path}")

    # Tworzenie modelu i automatyczne ładowanie wag (dzięki logice w backbone_iresnet.py)
    model = iresnet50(weights_path=weights_path)
    
    # Przeniesienie na urządzenie
    target_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    
    # Ustawiamy w tryb eval (bezpieczniej na start)
    model.eval()
    
    print(f"✅ [Loader] Model gotowy na urządzeniu: {target_device}")
    return model

# --- TEST (Uruchom ten plik, żeby sprawdzić czy działa) ---
if __name__ == "__main__":
    # Ścieżka do Twojego pliku skonwertowanego z ONNX
    WEIGHTS_FILE = 'w600k_r50_from_onnx.pth' 
    
    try:
        model = get_ready_model(WEIGHTS_FILE)
        
        # Szybki test przelotu (Forward Pass)
        dummy_input = torch.randn(1, 3, 112, 112).to(next(model.parameters()).device)
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"🧪 [Test] Kształt wyjścia: {output.shape}")
        if output.shape == (1, 512):
            print("🎉 [Test] SUKCES! Model działa poprawnie.")
        else:
            print("⚠️ [Test] Dziwny kształt wyjścia (oczekiwano 1, 512).")
            
    except Exception as e:
        print(f"💀 [Test] BŁĄD: {e}")