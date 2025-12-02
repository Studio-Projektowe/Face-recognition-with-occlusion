import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from podejscie_2 import load_clean_model, DEVICE
from compare_models import prepare_image, get_insightface_embedding, compare_embeddings

def main():
    print("--- TEST: Czy ten model w ogóle potrafi się uczyć? ---")
    
    # 1. Przygotuj dane
    img_path = 'example.jpg'
    img_numpy, img_tensor = prepare_image(img_path)
    
    # FIX: Batch Normalization w trybie .train() wyrzuca błąd przy BatchSize=1.
    # Rozwiązanie: Duplikujemy zdjęcie, żeby stworzyć BatchSize=2.
    # To pozwala warstwom BN policzyć statystyki, a nam kontynuować test.
    img_tensor = img_tensor.repeat(2, 1, 1, 1) # [1, 3, 112, 112] -> [2, 3, 112, 112]
    
    # Cel: Chcemy, żeby nasz model wypluł to samo co InsightFace
    try:
        target_embedding = get_insightface_embedding(img_numpy)
        target_tensor = torch.from_numpy(target_embedding).to(DEVICE)
    except Exception:
        print("Nie udało się pobrać celu z InsightFace. Używam losowego celu.")
        target_tensor = torch.randn(512).to(DEVICE)
        target_tensor = target_tensor / torch.norm(target_tensor)

    # FIX: Cel też musimy zduplikować, żeby pasował do BatchSize=2
    target_tensor = target_tensor.unsqueeze(0).repeat(2, 1) # [512] -> [2, 512]

    # 2. Model
    model = load_clean_model()
    model.train() # Przełączamy w tryb treningu!
    
    # Odmrażamy wszystkie wagi
    for param in model.parameters():
        param.requires_grad = True

    # 3. Setup treningowy
    criterion = nn.MSELoss()
    # Zwiększam trochę Learning Rate dla szybszego efektu w teście
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    print("\nRozpoczynam mikrot-trening na 1 zdjęciu (zduplikowanym do batch=2)...")
    
    # Zwiększyłem liczbę kroków do 100, żeby dać mu szansę
    for i in range(101):
        optimizer.zero_grad()
        
        # Forward
        output = model(img_tensor)
        
        # Normalizacja wyjścia (ważne dla ArcFace!)
        # output ma teraz wymiar [2, 512], normujemy każdy wiersz osobno
        output = torch.nn.functional.normalize(output, p=2, dim=1)
        
        # Loss
        loss = criterion(output, target_tensor)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            # Sprawdźmy similarity (bierzemy tylko pierwsze zdjęcie z batcha)
            emb_current = output[0].detach().cpu().numpy().flatten()
            target_np = target_tensor[0].cpu().numpy().flatten()
            
            sim, _ = compare_embeddings(target_np, emb_current)
            print(f"Krok {i}: Loss = {loss.item():.6f} | Similarity = {sim:.4f}")

    print("\n--- WERDYKT ---")
    if sim > 0.8:
        print("🚀 MODEL ŻYJE! Szybko nauczył się dopasowywać do InsightFace.")
        print("   Wniosek: Te zresetowane warstwy łatwo się uczą. Możesz robić projekt.")
    elif sim > 0.5:
        print("⚠️ Model się uczy, ale powoli. Może wymagać dłuższego treningu lub mniejszego LR.")
    else:
        print("💀 Nadal słabo. Uczenie nie postępuje.")

if __name__ == "__main__":
    main()