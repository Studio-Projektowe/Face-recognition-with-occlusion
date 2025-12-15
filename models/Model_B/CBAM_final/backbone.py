import torch
import torch.nn as nn

# Ustawiamy Epsilon z Twojego screena
BN_EPSILON = 1e-5 

class InsightBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(InsightBlock, self).__init__()
        
        # --- GŁÓWNA ŚCIEŻKA (Main Path) ---
        # Ze schematu: BN -> Conv -> PReLU -> Conv -> BN
        
        # 1. BatchNormalization_2 (na wejściu bloku!)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=BN_EPSILON)
        
        # 2. Conv_3 (3x3)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=True)
        
        # 3. PRelu_4
        self.prelu1 = nn.PReLU(out_channels)
        
        # 4. Conv_5 (3x3) - Tutaj jest stride!
        # Na screenie Conv_5 ma strides=2,2 (jeśli to blok redukujący)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=True)
        
        # 5. BatchNormalization_8
        self.bn2 = nn.BatchNorm2d(out_channels, eps=BN_EPSILON)

        # --- ŚCIEŻKA SKRÓTU (Shortcut / Downsample) ---
        # Ze schematu: Conv_6 (1x1) idzie bezpośrednio od wejścia (od PReLU_1)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            # Conv_6 na schemacie
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                        stride=stride, bias=True)

    def forward(self, x):
        # x to wyjście z poprzedniego PReLU (479 na schemacie)
        
        # --- Main Path ---
        out = self.bn1(x)      # BatchNormalization_2
        out = self.conv1(out)  # Conv_3
        out = self.prelu1(out) # PRelu_4
        out = self.conv2(out)  # Conv_5
        out = self.bn2(out)    # BatchNormalization_8
        
        # --- Shortcut ---
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x) # Conv_6
            
        # --- Add ---
        out += identity        # Add_7
        
        return out

class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        
        # --- STEM (Początek sieci ze zdjęcia) ---
        # 1. Conv_0
        self.conv_stem = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        
        # 2. PRelu_1 (To wyjście idzie do bloków)
        self.prelu_stem = nn.PReLU(64)
        
        # --- BLOKI (Reszta sieci) ---
        # Tutaj musisz dodać kolejne warstwy na podstawie Netrona.
        # Ze schematu widać ciąg: Conv_9... Conv_11... itd.
        # Wygląda to na serię bloków InsightBlock.
        
        self.layers = nn.ModuleList()
        
        # PRZYKŁAD (dopasuj liczby kanałów i stride z Netrona!):
        # Layer 1
        self.layers.append(InsightBlock(64, 64, stride=2)) # To jest ten blok ze schematu (Conv_3..Conv_6)
        
        # Layer 2 (kolejny na schemacie Conv_9, PReLU_10...)
        # Na schemacie dalej widać Conv_9 (3x3), PReLU_10, Conv_11 (3x3), BN_13
        # To wygląda na kolejny blok typu "BN -> Conv -> PReLU -> Conv -> BN" (ale bez BN na wejściu?)
        # Czekaj, na schemacie Conv_9 bierze wyjście z Add_7.
        # I NIE MA BatchNormalizacji przed Conv_9!
        # Jest Conv_9 -> PReLU_10 -> Conv_11 -> BN_13 -> Add_12
        
        # To oznacza, że są DWA TYPY BLOKÓW:
        # Typ A (Startowy): BN -> Conv -> PReLU -> Conv -> BN (ten co napisałem wyżej)
        # Typ B (Kolejny):  Conv -> PReLU -> Conv -> BN
        
    def forward(self, x):
        # Stem
        x = self.conv_stem(x)
        x = self.prelu_stem(x)
        
        # Layers
        for layer in self.layers:
            x = layer(x)
            
        return x