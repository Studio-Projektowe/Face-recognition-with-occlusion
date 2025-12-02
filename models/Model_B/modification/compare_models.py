import torch
import numpy as np
import cv2
import os
import insightface
from insightface.app import FaceAnalysis
import sys

# Importujemy Twój loader
from podejscie_2 import load_clean_model, DEVICE

# --- KONFIGURACJA ---
IMG_PATH = 'example.jpg'  # Upewnij się, że masz tu jakieś zdjęcie twarzy (najlepiej wycięte 112x112)
# Jeśli nie masz zdjęcia, skrypt wygeneruje losowy szum, 
# co pozwoli sprawdzić czy modele działają, ale podobieństwo będzie czysto matematyczne.

def prepare_image(img_path):
    """
    Przygotowuje obraz tak, aby pasował idealnie do obu modeli.
    Zakładamy input 112x112 RGB.
    """
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (112, 112))
    else:
        print(f"⚠️ Brak pliku {img_path}, generuję losowy szum.")
        img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    # Konwersja BGR -> RGB (InsightFace i nasz model to lubią)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Wersja dla InsightFace (zostawiamy numpy, biblioteka sama robi pre-process)
    # Ale uwaga: biblioteka 'get_feat' oczekuje zazwyczaj surowego obrazka.
    img_numpy = img_rgb

    # 2. Wersja dla PyTorch (Manualny pre-process: (x - 127.5) / 128.0)
    input_blob = (img_rgb.astype(np.float32) - 127.5) / 128.0
    input_tensor = torch.from_numpy(input_blob).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    
    return img_numpy, input_tensor

def get_insightface_embedding(img_numpy):
    """Wyciąga embedding z oficjalnej biblioteki InsightFace (buffalo_l)."""
    print("🔵 Ładowanie InsightFace (buffalo_l)...")
    
    # Inicjalizacja całego pakietu (detekcja + rozpoznawanie)
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Wyciągamy konkretnie model rozpoznawania twarzy (ArcFace r50)
    # Zazwyczaj jest to model pod indeksem w app.models, ale bezpieczniej tak:
    rec_model = app.models['recognition']
    
    if rec_model is None:
        print("❌ Nie udało się znaleźć modelu 'recognition' w pakiecie buffalo_l.")
        sys.exit(1)

    # InsightFace oczekuje zazwyczaj wykrytej twarzy (obiektu Face).
    # Ale my chcemy wymusić ekstrakcję cech z naszego wycinka 112x112.
    # Używamy wewnętrznej metody get_feat (jeśli obraz jest już wycięty/alignowany)
    # Model onnx oczekuje batcha, biblioteka to obsługuje.
    
    # Przekazujemy obraz. Biblioteka zrobi (img - 127.5) / 128.0 wewnątrz ONNX Runtime.
    embedding = rec_model.get_feat(img_numpy)
    return embedding.flatten()

def get_pytorch_embedding(img_tensor):
    """Wyciąga embedding z naszego modelu PyTorch."""
    print("🟠 Ładowanie modelu PyTorch (z loadera)...")
    model = load_clean_model()
    
    with torch.no_grad():
        embedding = model(img_tensor).cpu().numpy().flatten()
    return embedding

def compare_embeddings(emb_orig, emb_pt):
    """Liczy podobieństwo kosinusowe i dystans."""
    # Normalizacja
    norm_orig = np.linalg.norm(emb_orig)
    norm_pt = np.linalg.norm(emb_pt)
    
    emb_orig_norm = emb_orig / (norm_orig + 1e-10)
    emb_pt_norm = emb_pt / (norm_pt + 1e-10)
    
    # Cosine Similarity (Dot product znormalizowanych wektorów)
    similarity = np.dot(emb_orig_norm, emb_pt_norm)
    
    # Euclidean Distance
    diff = emb_orig_norm - emb_pt_norm
    dist = np.linalg.norm(diff)
    
    return similarity, dist

def main():
    print("--- ROZPOCZYNAMY PORÓWNANIE MODELI ---\n")
    
    # 1. Przygotowanie danych
    img_numpy, img_tensor = prepare_image(IMG_PATH)
    
    # 2. Pobranie wektorów
    try:
        emb_if = get_insightface_embedding(img_numpy)
        print(f"✅ InsightFace vector shape: {emb_if.shape}")
    except Exception as e:
        print(f"❌ Błąd InsightFace: {e}")
        return

    try:
        emb_pt = get_pytorch_embedding(img_tensor)
        print(f"✅ PyTorch vector shape: {emb_pt.shape}")
    except Exception as e:
        print(f"❌ Błąd PyTorch: {e}")
        return

    # 3. Porównanie
    print("\n--- WYNIKI ---")
    similarity, dist = compare_embeddings(emb_if, emb_pt)
    
    print(f"🔹 Podobieństwo Kosinusowe (Cosine Similarity): {similarity:.6f}")
    print(f"🔹 Dystans Euklidesowy (L2 Distance):          {dist:.6f}")
    print("-" * 30)
    
    # 4. Werdykt
    if similarity > 0.99:
        print("🚀 IDEALNIE! Modele są identyczne (cyfra w cyfrę).")
    elif similarity > 0.9:
        print("✅ BARDZO DOBRZE. Modele są prawie identyczne (różnice numeryczne float/double).")
    elif similarity > 0.5:
        print("⚠️ ŚREDNIO. Modele są semantycznie podobne, ale różnią się detalami.")
    else:
        print("❌ SŁABO / RÓŻNE. Modele dają zupełnie inne wyniki.")
        print("   PRZYCZYNA: Pamiętaj, że zresetowaliśmy 8 warstw w PyTorch.")
        print("   To oznacza, że model PyTorch jest 'częściowo nieświadomy'.")
        print("   Wymagany jest Fine-Tuning, aby odzyskać zbieżność!")

if __name__ == "__main__":
    main()