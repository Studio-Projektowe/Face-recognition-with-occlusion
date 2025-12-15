import torch
import os
import sys

# --- KONFIGURACJA ---
# Tutaj wpisz nazwę swojego pliku z wagami, który chcesz sprawdzić
PTH_FILE = 'w600k_r50_from_onnx.pth' 
OUTPUT_FILE = 'inspekcja_wag.txt'

def inspect_pth(file_path, save_path):
    print(f"📂 Otwieranie pliku: {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"❌ BŁĄD: Nie znaleziono pliku {file_path}")
        return

    try:
        # Ładujemy na CPU, żeby nie zajmować pamięci GPU
        checkpoint = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"❌ BŁĄD krytyczny przy ładowaniu torch.load: {e}")
        return

    # Sprawdzamy strukturę (czy to słownik, czy czyste wagi)
    meta_info = []
    state_dict = None
    
    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        meta_info.append(f"Typ obiektu: dict")
        meta_info.append(f"Główne klucze: {keys}")
        
        # Szukamy właściwych wag
        if 'state_dict' in checkpoint:
            print("ℹ️ Znaleziono klucz 'state_dict' - wchodzę głębiej.")
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            print("ℹ️ Znaleziono klucz 'model' - wchodzę głębiej.")
            state_dict = checkpoint['model']
        else:
            print("ℹ️ Plik wygląda na płaski słownik wag (state_dict).")
            state_dict = checkpoint
    else:
        print("⚠️ Plik nie jest słownikiem! To może być cały model (niezalecane).")
        meta_info.append(f"Typ obiektu: {type(checkpoint)}")
        # Próba wyciągnięcia state_dict jeśli to model
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()

    # Zapis do pliku
    print(f"💾 Zapisywanie raportu do: {save_path}...")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"=== RAPORT INSPEKCJI PLIKU: {file_path} ===\n")
        for line in meta_info:
            f.write(line + "\n")
        f.write("\n" + "="*80 + "\n")
        f.write(f"{'NAZWA WARSTWY (KEY)':<50} | {'KSZTAŁT (SHAPE)':<20} | {'TYP':<10} | {'ŚREDNIA':<10}\n")
        f.write("="*80 + "\n")

        if state_dict is None:
            f.write("❌ Nie udało się wyodrębnić słownika wag (state_dict).\n")
        else:
            count = 0
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    shape_str = str(list(tensor.shape))
                    dtype_str = str(tensor.dtype).replace('torch.', '')
                    
                    # Proste statystyki, żeby wykryć "martwe" wagi (same zera)
                    if tensor.numel() > 0:
                        try:
                            mean_val = f"{tensor.float().mean().item():.4f}"
                            # Opcjonalnie: min/max
                            # min_val = f"{tensor.min().item():.4f}"
                            # max_val = f"{tensor.max().item():.4f}"
                        except:
                            mean_val = "N/A"
                    else:
                        mean_val = "EMPTY"

                    f.write(f"{key:<50} | {shape_str:<20} | {dtype_str:<10} | {mean_val:<10}\n")
                    count += 1
                else:
                    f.write(f"{key:<50} | {'NIE-TENSOR':<20} | {type(tensor)} | -\n")
            
            f.write("="*80 + "\n")
            f.write(f"Łącznie znaleziono tensorów: {count}\n")

    print("✅ Gotowe! Otwórz plik tekstowy i sprawdź nazwy.")

if __name__ == "__main__":
    inspect_pth(PTH_FILE, OUTPUT_FILE)