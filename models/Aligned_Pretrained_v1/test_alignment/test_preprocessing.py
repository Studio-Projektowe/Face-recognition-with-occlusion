import cv2
import numpy as np
import json
import os
import random
import sys

try:
    import face_align
except ImportError:
    print("BŁĄD: Nie znaleziono pliku 'face_align.py' w tym katalogu.")
    print("Upewnij się, że uruchamiasz ten skrypt tam, gdzie jest Twój projekt.")
    sys.exit(1)

OCCLUSION_HEIGHT = 25
IMAGE_SIZE = 112


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
        
        return face_align.norm_crop(img, landmark=kps, image_size=IMAGE_SIZE)
    except Exception as e:
        print(f"Błąd wewnątrz align_face: {e}")
        return cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

def apply_random_occlusion_logic(img):
    """
    Dokładnie ta sama logika co w OcclusionFaceDataset.apply_random_occlusion
    """
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.float32)
    
    center_y = 52 + random.randint(-5, 5)
    bar_h_half = int(OCCLUSION_HEIGHT / 2)
    
    y1 = max(0, center_y - bar_h_half)
    y2 = min(h, center_y + bar_h_half)
    x1, x2 = 0, w
    
    color = np.random.randint(0, 256, (3,), dtype=int).tolist()
    
    img_copy = img.copy()
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, -1)
    
    cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, -1)
    
    return img_copy, mask


def main():
    input_image = 'example.jpg' 
    input_json = 'example.json'
    
    if not os.path.exists(input_image) and os.path.exists('example.jpg'):
        input_image = 'example.jpg'

    print(f"Wczytywanie: {input_image} oraz {input_json}")

    if not os.path.exists(input_image) or not os.path.exists(input_json):
        print("Brakuje pliku obrazu lub JSON! Wgraj je do folderu ze skryptem.")
        return

    img = cv2.imread(input_image)
    if img is None:
        print("Nie udało się wczytać obrazu przez cv2.")
        return

    with open(input_json, 'r') as f:
        data = json.load(f)
        landmarks = data.get('landmarks', data)

    print("Dane wczytane. Rozpoczynam przetwarzanie...")

    aligned_img = align_face_logic(img, landmarks)

    if aligned_img is None:
        print("Funkcja align_face zwróciła None!")
        return

    print(f"   Wymiary po alignment: {aligned_img.shape}")
    cv2.imwrite('test_step1_aligned.png', aligned_img)
    print("Zapisano: test_step1_aligned.png (Sama twarz, prosto)")

    occluded_img, mask = apply_random_occlusion_logic(aligned_img)

    cv2.imwrite('test_step2_final.png', occluded_img)
    print("Zapisano: test_step2_final.png (To wchodzi do modelu)")

    mask_visible = (mask * 255).astype(np.uint8)
    cv2.imwrite('test_step3_mask.png', mask_visible)
    print("Zapisano: test_step3_mask.png (Maska zasłonięcia)")

    print("\nGotowe! Sprawdź wygenerowane pliki PNG.")

if __name__ == "__main__":
    main()