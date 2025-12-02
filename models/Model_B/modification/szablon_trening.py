import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# IMPORTUJEMY TWÓJ NAPRAWIONY LOADER
# Upewnij się, że pliki loader_check.py i iresnet.py są w tym samym folderze
from podejscie_2 import load_clean_model, DEVICE

# --- KONFIGURACJA ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001 # Mniejszy learning rate, bo to fine-tuning (nie chcemy zepsuć dobrych wag)
EPOCHS = 10
DATA_DIR = 'twoje_dane/train' # Struktura folderów: train/osoba1, train/osoba2...

def main():
    print("--- START PROJEKTU: TRENING WŁAŚCIWY ---")

    # 1. Ładowanie Twojego "Uleczonego" Modelu
    # Ta funkcja robi całą magię: ładuje wagi ONNX, naprawia nazwy, leczy NaN
    model = load_clean_model()
    
    # 2. Przygotowanie danych (Przykładowe transformacje)
    # InsightFace lubi obrazy 112x112 i normalizację do [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Zakładamy, że masz dane w folderach (ImageFolder)
    # Jeśli nie masz jeszcze danych, ten fragment wywali błąd - to tylko szablon!
    if os.path.exists(DATA_DIR):
        dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        num_classes = len(dataset.classes)
        print(f"✅ Znaleziono {len(dataset)} zdjęć w {num_classes} klasach.")
    else:
        print(f"⚠️ Nie znaleziono folderu {DATA_DIR}. Uruchamiam tryb DEMO (bez danych).")
        dataloader = None
        num_classes = 10 # Przykładowo

    # 3. Dostosowanie głowy modelu (Head)
    # Model buffalo_l wypluwa wektor 512 cech.
    # Żeby klasyfikować Twoje osoby, musimy dodać warstwę klasyfikującą na końcu.
    # UWAGA: W profesjonalnym FaceID używa się ArcFace Loss, ale na start zwykły Linear + CrossEntropy też zadziała.
    
    # Tworzymy nową klasę, która opakowuje backbone + klasyfikator
    class FaceNet(nn.Module):
        def __init__(self, backbone, num_classes):
            super(FaceNet, self).__init__()
            self.backbone = backbone
            self.classifier = nn.Linear(512, num_classes) # Z 512 cech na liczbę Twoich osób
        
        def forward(self, x):
            # Pobieramy cechy (embedding)
            features = self.backbone(x)
            # Flatten (w razie czego)
            features = features.view(features.size(0), -1)
            # Klasyfikacja
            logits = self.classifier(features)
            return logits

    full_model = FaceNet(model, num_classes)
    full_model.to(DEVICE)

    # 4. Optymalizator i Loss
    optimizer = optim.SGD(full_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 5. Pętla Treningowa
    if dataloader:
        full_model.train()
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for images, labels in dataloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = full_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(dataloader):.4f}")
            
            # Zapisywanie checkpointu
            torch.save(full_model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')

    print("✅ Gotowe! Możesz teraz pisać rozdział o implementacji.")

if __name__ == "__main__":
    main()