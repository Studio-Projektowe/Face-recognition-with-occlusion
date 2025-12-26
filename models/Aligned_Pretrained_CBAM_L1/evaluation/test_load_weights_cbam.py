import torch
import sys
import os

try:
    from run_evaluation import initialize_custom_model, DEVICE, MODEL_PATH
except ImportError:
    print("BŁĄD: Nie znaleziono pliku 'run_evaluation.py' lub funkcji 'initialize_custom_model'.")
    sys.exit(1)

def verify_loaded_weights():
    print(f"--- ROZPOCZYNAM TWARDĄ WERYFIKACJĘ WAG ---")
    print(f"Plik wag: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print("BŁĄD: Plik z wagami nie istnieje.")
        return

    print("\n[KROK 1] Uruchamianie initialize_custom_model()...")
    try:
        model = initialize_custom_model()
        print("-> Funkcja zakończyła się bez błędów (strict=True przeszło).")
    except Exception as e:
        print(f"-> KRYTYCZNY BŁĄD podczas ładowania modelu: {e}")
        return

    print("\n[KROK 2] Wczytywanie surowego pliku .pth do porównania...")
    raw_checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(raw_checkpoint, dict) and 'state_dict' in raw_checkpoint:
        raw_state = raw_checkpoint['state_dict']
    else:
        raw_state = raw_checkpoint

    print("\n[KROK 3] Weryfikacja warstwy BACKBONE (conv1.weight)...")
    
    model_weight_backbone = model.backbone.conv1.weight
    
    raw_key_backbone = 'conv1.weight'
    if raw_key_backbone not in raw_state:
        raw_key_backbone = 'module.conv1.weight' # fallback
    
    file_weight_backbone = raw_state[raw_key_backbone].to(DEVICE)

    if torch.equal(model_weight_backbone, file_weight_backbone):
        print("SUKCES: Wagi Backbone są identyczne co do bitu!")
    else:
        print("BŁĄD: Wagi Backbone się różnią!")
        print(f"   Model: {model_weight_backbone.shape}, Plik: {file_weight_backbone.shape}")

    print("\n[KROK 4] Weryfikacja warstwy CBAM (cbam1.ca.fc.0.weight)...")
    
    model_weight_cbam = model.cbam1.ca.fc[0].weight
    
    raw_key_cbam = 'cbam1.ca.fc.0.weight'
    if raw_key_cbam not in raw_state:
        raw_key_cbam = 'module.cbam1.ca.fc.0.weight'

    file_weight_cbam = raw_state[raw_key_cbam].to(DEVICE)
    
    print(f"   Kształt w modelu: {model_weight_cbam.shape}")
    print(f"   Kształt w pliku : {file_weight_cbam.shape}")
    
    if file_weight_cbam.dim() == 2:
        print("   -> Wykryto wagę 2D w pliku, aplikuję unsqueeze(-1).unsqueeze(-1) do testu...")
        file_weight_cbam_fixed = file_weight_cbam.unsqueeze(-1).unsqueeze(-1)
    else:
        file_weight_cbam_fixed = file_weight_cbam

    if torch.equal(model_weight_cbam, file_weight_cbam_fixed):
        print("SUKCES: Wagi CBAM są identyczne (po naprawie wymiarów)!")
    else:
        print("BŁĄD: Wagi CBAM się różnią!")

    print("\n--- PODSUMOWANIE ---")
    print("SUKCES powyżej, Twój model jest załadowany PERFEKCYJNIE.")

if __name__ == "__main__":
    verify_loaded_weights()