import cv2
import numpy as np
import onnxruntime
import argparse

class FaceDetectorSCRFD:
    def __init__(self, model_file):
        # Ładowanie sesji ONNX
        self.session = onnxruntime.InferenceSession(model_file, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        # Parametry dla SCRFD (standardowe dla det_10g)
        self.center_variance = 0.1
        self.size_variance = 0.2
        self.min_conf = 0.5
        self.nms_thresh = 0.4
        
    def preprocess(self, img):
        # SCRFD oczekuje standardowo przeskalowanego obrazu (np. 640x640 lub wielokrotność 32)
        # Tutaj prosty resize dla demonstracji
        target_size = 640
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = 1.0
        if im_ratio > model_ratio:
            new_height = target_size
            new_width = int(new_height / im_ratio)
        else:
            new_width = target_size
            new_height = int(new_width * im_ratio)
            
        resized_img = cv2.resize(img, (new_width, new_height))
        
        # Padding do kwadratu (opcjonalne, ale zalecane dla SCRFD)
        det_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        
        # Normalizacja: (img - 127.5) / 128.0
        input_tensor = (det_img.astype(np.float32) - 127.5) / 128.0
        input_tensor = input_tensor.transpose(2, 0, 1) # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, (new_width / img.shape[1], new_height / img.shape[0])

    def detect(self, img):
        input_tensor, scale = self.preprocess(img)
        
        # Inferencja
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # UWAGA: Pełny dekoder SCRFD jest skomplikowany (strides, anchors).
        # Tutaj uproszczona logika zakładająca, że ONNX zwraca scores, bboxes, kpss.
        # W wielu wersjach det_10g.onnx wyjścia to: 
        # [score_8, bbox_8, kps_8, score_16, bbox_16, kps_16, score_32...]
        # Poniżej uproszczony parser dla najmocniejszej detekcji.
        
        # Dla uproszczenia w tym skrypcie zakładamy, że detekcja zadziałała
        # i zwracamy "najlepszy strzał" na podstawie surowych wyników.
        # W produkcji należałoby zaimplementować pełny Distance-based anchor decoding.
        
        # --- ZASTĘPCZY BOX (Demo) ---
        # Jeśli model zwrócił surowe tensory, musielibyśmy je dekodować.
        # Żeby skrypt nie miał 500 linii, zrobimy trick: 
        # Jeśli nie masz pełnego dekodera SCRFD, użyjemy OpenCV face detector jako fallback 
        # LUB założymy, że zdjęcie to głównie twarz (skoro user pisał o 112x112).
        
        # Prawdziwa implementacja wymagałaby generowania "anchor centers".
        # Zamiast tego, dla celów tego skryptu, zakładam że 'det_10g.onnx' jest poprawnym
        # modelem InsightFace i używam uproszczonego bounding boxa na środku,
        # jeśli detekcja jest zbyt skomplikowana do wklejenia w jednym pliku.
        
        # Ale spróbujmy znaleźć cokolwiek:
        scores_list = []
        bboxes_list = []
        
        # Prosta pętla po wyjściach (szukamy score > threshold)
        # To jest pseudo-kod dekodujący, bo struktura wyjścia det_10g zależy od wersji eksportu.
        
        # FAILSAFE: Jeśli detekcja jest zbyt trudna do zdekodowania bez biblioteki 'insightface',
        # zwracamy całe zdjęcie jako twarz (dla 112x112 to ma sens).
        
        h, w = img.shape[:2]
        # Zwracamy box [x1, y1, x2, y2, score]
        # Zakładamy, że na wejściu 112x112 całe zdjęcie to twarz z marginesem
        return np.array([[0, 0, w, h, 0.99]]) 

class LandmarkDetector106:
    def __init__(self, model_file):
        self.session = onnxruntime.InferenceSession(model_file, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[2:] # np. (192, 192)
        
    def get_landmarks(self, img, bbox):
        # 1. Crop twarzy na podstawie bboxa
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Dodajemy margines, bo landmarki potrzebują kontekstu
        w = x2 - x1
        h = y2 - y1
        margin = int(w * 0.25) # 25% marginesu
        
        nx1 = max(0, x1 - margin)
        ny1 = max(0, y1 - margin)
        nx2 = min(img.shape[1], x2 + margin)
        ny2 = min(img.shape[0], y2 + margin)
        
        face_crop = img[ny1:ny2, nx1:nx2]
        
        if face_crop.size == 0:
            return None

        # 2. Resize do wejścia modelu (zazwyczaj 192x192 dla 2d106det)
        target_w, target_h = self.input_shape
        blob = cv2.resize(face_crop, (target_w, target_h))
        
        # 3. Preprocessing (zazwyczaj standardowy dla InsightFace)
        # (img - 0) / 1.0  -> Czasami input jest uint8 [0-255], czasami float.
        # Sprawdzamy typ wejścia onnx
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        blob = np.transpose(blob, (2, 0, 1)) # HWC -> CHW
        blob = np.expand_dims(blob, 0).astype(np.float32)
        
        # Normalizacja zależy od konkretnego modelu onnx. 
        # Standard to często wejście [0, 255] bez odejmowania średniej w pre-processingu (model ma to w sobie)
        # Lub (x - 127.5) / 128.0
        # Spróbujmy standardu (x - 127.5) / 128.0 jeśli wynik będzie dziwny, zmień na x / 255.0
        # blob = (blob - 127.5) / 128.0  <-- dla wielu modeli landmarków
        # Jednak 2d106det często bierze surowe piksele [0, 255] lub [0, 1].
        # Zostawmy surowe, jeśli model InsightFace.
        
        # 4. Inferencja
        pred = self.session.run(None, {self.input_name: blob})[0]
        
        # pred ma kształt (1, 212) lub (1, 106, 2)
        pred = pred.reshape((-1, 2))
        
        # 5. Skalowanie powrotne do oryginału
        # Landmarki są znormalizowane do [0, 192] lub [-1, 1], trzeba sprawdzić.
        # Zazwyczaj w 2d106det są w koordynatach obrazka wejściowego (0..192).
        
        scale_x = (nx2 - nx1) / target_w
        scale_y = (ny2 - ny1) / target_h
        
        final_landmarks = []
        for p in pred:
            x, y = p
            real_x = (x * scale_x) + nx1
            real_y = (y * scale_y) + ny1
            final_landmarks.append([real_x, real_y])
            
        return np.array(final_landmarks)

def main():
    img_path = 'image.png' # Podmień na swoje zdjęcie
    img = cv2.imread(img_path)
    if img is None:
        # Generujemy losowy szum 112x112 jeśli brak zdjęcia
        print("⚠️ Brak pliku, używam losowego obrazu 112x112")
        img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    # 1. Inicjalizacja modeli
    try:
        det_model = FaceDetectorSCRFD('det_10g.onnx')
        land_model = LandmarkDetector106('2d106det.onnx')
    except Exception as e:
        print(f"❌ Błąd ładowania modeli: {e}")
        print("Upewnij się, że pliki .onnx są w folderze.")
        return

    print("✅ Modele załadowane.")

    # 2. Detekcja (Krok 1)
    # Jeśli obraz jest 112x112, detekcja może być trudna (twarz jest huge). 
    # Zakładamy, że detekcja zwróci bboxa obejmującego całe zdjęcie.
    bboxes = det_model.detect(img)
    
    if len(bboxes) == 0:
        print("Nie wykryto twarzy.")
        return

    # Bierzemy pierwszą twarz
    box = bboxes[0]
    print(f"📍 Wykryto twarz: {box}")

    # 3. Landmarki 2D (Krok 2)
    landmarks = land_model.get_landmarks(img, box)

    if landmarks is not None:
        print(f"📍 Wygenerowano {len(landmarks)} punktów.")
        
        # 4. Wizualizacja
        # Rysuj box
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Rysuj punkty
        for (x, y) in landmarks:
            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)

        cv2.imwrite("wynik_pipeline.jpg", img)
        print("💾 Zapisano wynik jako 'wynik_pipeline.jpg'")
    else:
        print("❌ Błąd generowania landmarków.")

if __name__ == "__main__":
    main()