#import torch
#import torch.nn as nn
#
## Ustawiamy Epsilon z Twojego screena
#BN_EPSILON = 1e-5 
#
#class InsightBlock(nn.Module):
#    def __init__(self, in_channels, out_channels, stride=1):
#        super(InsightBlock, self).__init__()
#        
#        # --- GŁÓWNA ŚCIEŻKA (Main Path) ---
#        # Ze schematu: BN -> Conv -> PReLU -> Conv -> BN
#        
#        # 1. BatchNormalization_2 (na wejściu bloku!)
#        self.bn1 = nn.BatchNorm2d(in_channels, eps=BN_EPSILON)
#        
#        # 2. Conv_3 (3x3)
#        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
#                               stride=1, padding=1, bias=True)
#        
#        # 3. PRelu_4
#        self.prelu1 = nn.PReLU(out_channels)
#        
#        # 4. Conv_5 (3x3) - Tutaj jest stride!
#        # Na screenie Conv_5 ma strides=2,2 (jeśli to blok redukujący)
#        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
#                               stride=stride, padding=1, bias=True)
#        
#        # 5. BatchNormalization_8
#        self.bn2 = nn.BatchNorm2d(out_channels, eps=BN_EPSILON)
#
#        # --- ŚCIEŻKA SKRÓTU (Shortcut / Downsample) ---
#        # Ze schematu: Conv_6 (1x1) idzie bezpośrednio od wejścia (od PReLU_1)
#        self.downsample = None
#        if stride != 1 or in_channels != out_channels:
#            # Conv_6 na schemacie
#            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
#                                        stride=stride, bias=True)
#
#    def forward(self, x):
#        # x to wyjście z poprzedniego PReLU (479 na schemacie)
#        
#        # --- Main Path ---
#        out = self.bn1(x)      # BatchNormalization_2
#        out = self.conv1(out)  # Conv_3
#        out = self.prelu1(out) # PRelu_4
#        out = self.conv2(out)  # Conv_5
#        out = self.bn2(out)    # BatchNormalization_8
#        
#        # --- Shortcut ---
#        identity = x
#        if self.downsample is not None:
#            identity = self.downsample(x) # Conv_6
#            
#        # --- Add ---
#        out += identity        # Add_7
#        
#        return out
#
#class CustomResNet(nn.Module):
#    def __init__(self):
#        super(CustomResNet, self).__init__()
#        
#        # --- STEM (Początek sieci ze zdjęcia) ---
#        # 1. Conv_0
#        self.conv_stem = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
#        
#        # 2. PRelu_1 (To wyjście idzie do bloków)
#        self.prelu_stem = nn.PReLU(64)
#        
#        # --- BLOKI (Reszta sieci) ---
#        # Tutaj musisz dodać kolejne warstwy na podstawie Netrona.
#        # Ze schematu widać ciąg: Conv_9... Conv_11... itd.
#        # Wygląda to na serię bloków InsightBlock.
#        
#        self.layers = nn.ModuleList()
#        
#        # PRZYKŁAD (dopasuj liczby kanałów i stride z Netrona!):
#        # Layer 1
#        self.layers.append(InsightBlock(64, 64, stride=2)) # To jest ten blok ze schematu (Conv_3..Conv_6)
#        
#        # Layer 2 (kolejny na schemacie Conv_9, PReLU_10...)
#        # Na schemacie dalej widać Conv_9 (3x3), PReLU_10, Conv_11 (3x3), BN_13
#        # To wygląda na kolejny blok typu "BN -> Conv -> PReLU -> Conv -> BN" (ale bez BN na wejściu?)
#        # Czekaj, na schemacie Conv_9 bierze wyjście z Add_7.
#        # I NIE MA BatchNormalizacji przed Conv_9!
#        # Jest Conv_9 -> PReLU_10 -> Conv_11 -> BN_13 -> Add_12
#        
#        # To oznacza, że są DWA TYPY BLOKÓW:
#        # Typ A (Startowy): BN -> Conv -> PReLU -> Conv -> BN (ten co napisałem wyżej)
#        # Typ B (Kolejny):  Conv -> PReLU -> Conv -> BN
#        
#    def forward(self, x):
#        # Stem
#        x = self.conv_stem(x)
#        x = self.prelu_stem(x)
#        
#        # Layers
#        for layer in self.layers:
#            x = layer(x)
#            
#        return x

from collections import OrderedDict
import re

__all__ = ['iresnet50']

# Ustawiamy Epsilon zgodnie ze screenem z Netron (image_577885.png)
BN_EPSILON = 0.000009999999747378752

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    # ZMIANA: bias=True, bo w Twoim modelu ONNX convy mają bias (np. Conv_3 ma wejście B)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True,
                     dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    # ZMIANA: bias=True
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=True)

class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        
        self.bn1 = nn.BatchNorm2d(inplanes, eps=BN_EPSILON)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=BN_EPSILON)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=BN_EPSILON)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        
        self.groups = groups
        self.base_width = width_per_group
        
        # ZMIANA: bias=True (zgodnie z Conv_0 w Netronie)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=BN_EPSILON)
        self.prelu = nn.PReLU(self.inplanes)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=BN_EPSILON)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=BN_EPSILON)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=BN_EPSILON),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, self.dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x

    def load_weights_from_pth(self, weights_path):
        """
        Ładuje wagi z Twojego pliku .pth, radząc sobie z mieszanką nazwanych 
        i nienazwanych (numerowanych) tensorów.
        """
        print(f"🔧 Ładowanie wag (tryb ONNX-match) z: {weights_path}")
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        except Exception as e:
            print(f"❌ Błąd odczytu pliku: {e}")
            return False

        new_state_dict = OrderedDict()
        
        # 1. Oddzielamy wagi nazwane (np. bn1, fc) od numerowanych (conv, prelu)
        named_source_keys = {}
        numbered_source_keys = []

        for key, val in state_dict.items():
            # Nazwane wagi (z pliku tekstowego widać, że BNy mają nazwy typu _initializer_layer...)
            if "layer" in key or "fc" in key or "bn" in key or "features" in key:
                # Normalizacja nazwy: _initializer_layer1_0_bn1_weight -> layer1.0.bn1.weight
                clean_key = key.replace("_initializer_", "")
                # Zamiana podkreślników na kropki, ale tylko tych strukturalnych
                # To jest trudne automatycznie, więc zrobimy mapowanie po dopasowaniu stringów
                named_source_keys[key] = val
            elif re.search(r'_initializer_\d+$', key):
                # Numerowane: _initializer_685
                num = int(key.split('_')[-1])
                numbered_source_keys.append((num, key, val))
        
        # Sortujemy numerowane klucze rosnąco - to odpowiada kolejności warstw w modelu
        numbered_source_keys.sort(key=lambda x: x[0])
        
        print(f"📊 Statystyka źródła: {len(named_source_keys)} nazwanych, {len(numbered_source_keys)} numerowanych.")

        # 2. Iterujemy po modelu PyTorch i dobieramy wagi
        model_keys = list(self.state_dict().keys())
        
        used_numbered_indices = []
        
        for m_key in model_keys:
            # Ignorujemy num_batches_tracked (nie ma ich w ONNX)
            if "num_batches_tracked" in m_key:
                new_state_dict[m_key] = torch.tensor(0, dtype=torch.long)
                continue

            # STRATEGIA A: Czy mamy pasujący klucz nazwany?
            # Przekształcamy m_key (np. layer1.0.bn1.weight) na format z pliku (_initializer_layer1_0_bn1_weight)
            candidate_name = "_initializer_" + m_key.replace(".", "_")
            
            if candidate_name in state_dict:
                new_state_dict[m_key] = state_dict[candidate_name]
            
            # STRATEGIA B: Jeśli nie ma nazwy, bierzemy kolejny numerowany klucz o pasującym kształcie
            else:
                target_shape = self.state_dict()[m_key].shape
                found = False
                
                # Przeszukujemy numerowane w kolejności
                for i, (num, s_key, s_val) in enumerate(numbered_source_keys):
                    if i in used_numbered_indices: continue
                    
                    # Sprawdzenie kształtu
                    if s_val.shape == target_shape:
                        new_state_dict[m_key] = s_val
                        used_numbered_indices.append(i)
                        found = True
                        break
                    
                    # FIX DLA PRELU: PyTorch [64], ONNX [64, 1, 1]
                    if len(target_shape) == 1 and len(s_val.shape) == 3:
                        if s_val.shape[0] == target_shape[0] and s_val.shape[1]==1 and s_val.shape[2]==1:
                            new_state_dict[m_key] = s_val.squeeze()
                            used_numbered_indices.append(i)
                            found = True
                            break
                            
                if not found:
                    print(f"⚠️ Nie znaleziono wagi dla: {m_key} (shape: {target_shape})")

        # 3. Ładowanie
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Załadowano wagi. Missing: {len(missing)} (powinno być 0 istotnych).")
        if len(missing) > 0:
            print(f"   Przykładowe braki: {missing[:5]}")

def iresnet50(weights_path=None, **kwargs):
    model = IResNet(IBasicBlock, [3, 4, 14, 3], **kwargs)
    if weights_path:
        model.load_weights_from_pth(weights_path)
    return model

