import cv2
import numpy as np
import json
import os
import argparse
import face_align  # Twój plik (musi być w tym samym folderze)

def get_aligned_face(image_path, json_path, output_size=112):
    """
    Wczytuje zdjęcie i landmarki, wykonuje alignment i zwraca gotowy obraz.
    """
    # 1. Wczytaj obraz
    if not os.path.exists(image_path):
        print(f"❌ Błąd: Nie znaleziono pliku obrazu: {image_path}")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Błąd: Nie udało się wczytać obrazu przez OpenCV.")
        return None

    # 2. Wczytaj JSON z landmarkami
    if not os.path.exists(json_path):
        print(f"❌ Błąd: Nie znaleziono pliku JSON: {json_path}")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Obsługa różnych formatów JSON (czasami jest klucz "landmarks", czasami płasko)
    lmk_data = data.get('landmarks', data) 

    # 3. Konwersja punktów do formatu (5, 2)
    # WAŻNE: face_align.norm_crop oczekuje kolejności punktów na obrazie:
    # [LeweOko, PraweOko, Nos, LewyKącikUst, PrawyKącikUst] (patrząc na ekran)
    # Twój JSON ma nazwy anatomiczne (right_eye osoby = lewa strona na zdjęciu).
    
    try:
        kps = np.array([
            lmk_data['right_eye'],    # Index 0: Lewe oko na zdjęciu
            lmk_data['left_eye'],     # Index 1: Prawe oko na zdjęciu
            lmk_data['nose'],         # Index 2: Nos
            lmk_data['mouth_right'],  # Index 3: Lewy kącik ust na zdjęciu
            lmk_data['mouth_left']    # Index 4: Prawy kącik ust na zdjęciu
        ], dtype=np.float32)
    except KeyError as e:
        print(f"❌ Błąd w strukturze JSON: Brakuje klucza {e}")
        return None

    # 4. ALIGNMENT (Kluczowa operacja z face_align.py)
    # To odpowiada linijce z Twojego przykładu:
    # aimg = face_align.norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
    aligned_img = face_align.norm_crop(img, landmark=kps, image_size=output_size)

    return aligned_img

def main():
    # --- KONFIGURACJA ---
    IMG_FILE = 'example.jpg'       # Twoje zdjęcie wejściowe
    JSON_FILE = 'example.json'     # Twój plik z punktami
    OUTPUT_FILE = 'aligned_112x112.jpg' # Gdzie zapisać wynik
    
    print(f"🔄 Przetwarzanie: {IMG_FILE}...")
    
    aligned_face = get_aligned_face(IMG_FILE, JSON_FILE, output_size=112)

    if aligned_face is not None:
        # Zapisz wynik na dysk
        cv2.imwrite(OUTPUT_FILE, aligned_face)
        print(f"✅ SUKCES! Zapisano wyrównane zdjęcie do: {OUTPUT_FILE}")
        print(f"   Wymiary wyjściowe: {aligned_face.shape}")
        
        # Opcjonalnie: wyświetl podgląd (jeśli masz ekran)
        # cv2.imshow("Aligned Face", aligned_face)
        # cv2.waitKey(0)
    else:
        print("❌ Operacja nieudana.")

if __name__ == "__main__":
    main()