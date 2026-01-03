import cv2
import numpy as np
import json
import os
import argparse
import face_align

def get_aligned_face(image_path, json_path, output_size=112):
    """
    Wczytuje zdjęcie i landmarki, wykonuje alignment i zwraca gotowy obraz.
    """
    if not os.path.exists(image_path):
        print(f"Błąd: Nie znaleziono pliku obrazu: {image_path}")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Błąd: Nie udało się wczytać obrazu przez OpenCV.")
        return None

    if not os.path.exists(json_path):
        print(f"Błąd: Nie znaleziono pliku JSON: {json_path}")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    lmk_data = data.get('landmarks', data) 
    
    try:
        kps = np.array([
            lmk_data['right_eye'],
            lmk_data['left_eye'],
            lmk_data['nose'],
            lmk_data['mouth_right'],
            lmk_data['mouth_left']
        ], dtype=np.float32)
    except KeyError as e:
        print(f"Błąd w strukturze JSON: Brakuje klucza {e}")
        return None

    # aimg = face_align.norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
    aligned_img = face_align.norm_crop(img, landmark=kps, image_size=output_size)

    return aligned_img

def main():
    IMG_FILE = 'example.jpg'
    JSON_FILE = 'example.json'
    OUTPUT_FILE = 'aligned_112x112.jpg'
    
    print(f"Przetwarzanie: {IMG_FILE}...")
    
    aligned_face = get_aligned_face(IMG_FILE, JSON_FILE, output_size=112)

    if aligned_face is not None:
        cv2.imwrite(OUTPUT_FILE, aligned_face)
        print(f"SUKCES! Zapisano wyrównane zdjęcie do: {OUTPUT_FILE}")
        print(f"   Wymiary wyjściowe: {aligned_face.shape}")
        
        # cv2.imshow("Aligned Face", aligned_face)
        # cv2.waitKey(0)
    else:
        print("Operacja nieudana.")

if __name__ == "__main__":
    main()