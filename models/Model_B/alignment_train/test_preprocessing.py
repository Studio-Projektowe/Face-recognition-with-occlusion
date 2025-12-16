import cv2
import numpy as np
import json
import os
import random
import sys

# Próbujemy zaimportować Twój moduł face_align
try:
    import face_align
except ImportError:
    print("❌ BŁĄD: Nie znaleziono pliku 'face_align.py' w tym katalogu.")
    print("Upewnij się, że uruchamiasz ten skrypt tam, gdzie jest Twój projekt.")
    sys.exit(1)

# --- KONFIGURACJA (musi być identyczna jak w treningu) ---
OCCLUSION_HEIGHT = 25
IMAGE_SIZE = 112

# --- FUNKCJE WYCIĘTE Z TWOJEGO DATASETU ---

def align_face_logic(img, landmarks):
    """
    Dokładnie ta sama logika co w OcclusionFaceDataset.align_face
    """
    try:
        kps = np.array([
            landmarks['right_eye'], 
            landmarks['left_eye'], 
            landmarks['nose'],
            landmarks['mouth_right'], 
            landmarks['mouth_left']
        ], dtype=np.float32)
        
        # Wywołanie Twojej biblioteki
        return face_align.norm_crop(img, landmark=kps, image_size=IMAGE_SIZE)
    except Exception as e:
        print(f"⚠️ Błąd wewnątrz align_face: {e}")
        # Fallback z datasetu
        return cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

def apply_random_occlusion_logic(img):
    """
    Dokładnie ta sama logika co w OcclusionFaceDataset.apply_random_occlusion
    """
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.float32)
    
    # Logika losowania pozycji paska
    center_y = 52 + random.randint(-5, 5)
    bar_h_half = int(OCCLUSION_HEIGHT / 2)
    
    y1 = max(0, center_y - bar_h_half)
    y2 = min(h, center_y + bar_h_half)
    x1, x2 = 0, w
    
    # Losowy kolor
    color = np.random.randint(0, 256, (3,), dtype=int).tolist()
    
    # Rysowanie na obrazie (oryginał jest modyfikowany!)
    img_copy = img.copy() # Kopiujemy żeby nie psuć oryginału w teście
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, -1)
    
    # Rysowanie maski (1.0 tam gdzie pasek)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, -1)
    
    return img_copy, mask

# --- GŁÓWNA PĘTLA TESTOWA ---

def main():
    # 1. Ustalanie plików wejściowych
    # Możesz zmienić nazwy tutaj, jeśli masz inne pliki testowe
    input_image = 'image.png' 
    input_json = 'image.json'
    
    # Obsługa .jpg jeśli .png nie ma
    if not os.path.exists(input_image) and os.path.exists('image.jpg'):
        input_image = 'image.jpg'

    print(f"📂 Wczytywanie: {input_image} oraz {input_json}")

    if not os.path.exists(input_image) or not os.path.exists(input_json):
        print("❌ Brakuje pliku obrazu lub JSON! Wgraj je do folderu ze skryptem.")
        return

    # 2. Wczytanie danych
    img = cv2.imread(input_image)
    if img is None:
        print("❌ Nie udało się wczytać obrazu przez cv2.")
        return

    with open(input_json, 'r') as f:
        data = json.load(f)
        # Logika wyciągania landmarks identyczna jak w Twoim __getitem__
        landmarks = data.get('landmarks', data)

    print("✅ Dane wczytane. Rozpoczynam przetwarzanie...")

    # 3. KROK 1: ALIGNMENT
    aligned_img = align_face_logic(img, landmarks)

    if aligned_img is None:
        print("❌ Funkcja align_face zwróciła None!")
        return

    print(f"   Wymiary po alignment: {aligned_img.shape}")
    cv2.imwrite('test_step1_aligned.png', aligned_img)
    print("💾 Zapisano: test_step1_aligned.png (Sama twarz, prosto)")

    # 4. KROK 2: OCCLUSION (Zasłonięcie oczu)
    occluded_img, mask = apply_random_occlusion_logic(aligned_img)

    cv2.imwrite('test_step2_final.png', occluded_img)
    print("💾 Zapisano: test_step2_final.png (To wchodzi do modelu)")

    # 5. KROK 3: MASKA (Dla pewności co widzi aux_head)
    # Skalujemy maskę do 0-255 żeby była widoczna jako obrazek
    mask_visible = (mask * 255).astype(np.uint8)
    cv2.imwrite('test_step3_mask.png', mask_visible)
    print("💾 Zapisano: test_step3_mask.png (Maska zasłonięcia)")

    print("\n🎉 Gotowe! Sprawdź wygenerowane pliki PNG.")

if __name__ == "__main__":
    main()