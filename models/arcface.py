import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from insightface.model_zoo import get_model

# -----------------------------
# 1️⃣ Autoencoder (U-Net style)
# -----------------------------
class FaceReconstructor(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, base_channels, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(base_channels, base_channels, 3, padding=1),
                                  nn.ReLU())
        self.enc2 = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
                                  nn.ReLU())
        self.enc3 = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
                                  nn.ReLU())
        # Decoder
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
                                  nn.ReLU())
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(base_channels, in_channels, 3, padding=1))
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        # Decoder
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        out = torch.tanh(self.dec1(d1))  # [-1,1] output
        return out

# -----------------------------------
# 2️⃣ Embedding network (ArcFace iResNet50 z insightface)
# -----------------------------------
class FaceEmbedder(nn.Module):
    def __init__(self, embedding_size=512, pretrained=True):
        super().__init__()
        # insightface model pre-trained on ArcFace loss
        self.model = get_model('arcface_r50_v1')  # iResNet50 + ArcFaceLoss pre-trained
        self.model.prepare(ctx_id=-1)  # CPU; jeśli masz GPU: ctx_id=0
    
    def forward(self, x):
        # Input: batch x 3 x H x W, normalized [-1,1]
        # Output: embeddings 512-D, znormalizowane
        emb = self.model.get_embedding(x)
        return emb

# -----------------------------
# 3️⃣ FR²Net łączy obie części
# -----------------------------
class FR2Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.reconstructor = FaceReconstructor()
        self.embedder = FaceEmbedder()
        
    def forward(self, x):
        reconstructed = self.reconstructor(x)
        embedding = self.embedder(reconstructed)
        return reconstructed, embedding

# -----------------------------
# 4️⃣ Loss
# -----------------------------
class FR2Loss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        # ArcFace loss w insightface jest wbudowany w model,
        # wystarczy mieć embeddingi i etykiety do training loop z insightface trainer
        # Jeśli chcesz manualnie:
        # insightface używa MarginSoftmax, więc można użyć ich implementacji ArcFace
        from insightface.loss import ArcFaceLoss
        self.arcface_loss = ArcFaceLoss(512, num_classes=10000, s=64.0, m=0.5)  # num_classes = ilość osób w zbiorze

    def forward(self, reconstructed, embedding, labels, clean_images):
        rec_loss = self.l1_loss(reconstructed, clean_images)
        arc_loss = self.arcface_loss(embedding, labels)
        return rec_loss + self.beta * arc_loss

# -----------------------------
# 5️⃣ Przykład użycia
# -----------------------------
if __name__ == "__main__":
    model = FR2Net()
    criterion = FR2Loss(beta=1.0)
    
    # Dummy data
    x_occluded = torch.randn(2,3,112,112)
    y_labels = torch.tensor([0,1])
    x_clean = torch.randn(2,3,112,112)
    
    reconstructed, embedding = model(x_occluded)
    loss = criterion(reconstructed, embedding, y_labels, clean_images=x_clean)
    
    print("Reconstructed shape:", reconstructed.shape)
    print("Embedding shape:", embedding.shape)
    print("Loss:", loss.item())
