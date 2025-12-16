import torch
import torch.nn as nn
from collections import OrderedDict
import re

__all__ = ['iresnet50']

BN_EPSILON = 1e-5 

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True,
                     dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=True)

class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        
        # Architektura: BN -> Conv -> BN -> PReLU -> Conv -> BN (w Downsample)
        self.bn1 = nn.BatchNorm2d(inplanes, eps=BN_EPSILON)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=BN_EPSILON)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        # USUNIĘTO NADMIAROWY BN3 - trzeci BN siedzi w 'downsample'
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        # BN3 jest teraz częścią downsample, jeśli istnieje
        
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
        
        # Stem: Conv -> BN -> PReLU
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=BN_EPSILON)
        self.prelu = nn.PReLU(self.inplanes)
        
        # InsightFace r50 zaczyna od stride=2 w layer1 (redukcja wymiaru)
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
            
        # Tworzenie Downsample (Conv 1x1 + BN)
        # To "zjada" te dodatkowe wagi 1x1 z pliku .pth
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
        Inteligentny loader z 'Bucket Strategy' dopasowany do BasicBlock.
        """
        print(f"🔧 [IResNet] Zaawansowane ładowanie wag z: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        elif 'model' in state_dict: state_dict = state_dict['model']

        # 1. PRZYGOTOWANIE WIADEREK
        buckets = {
            'conv_weights': [], # ndim=4
            'conv_biases': [],  # ndim=1
            'bn_stats': [],     # BatchNormalization_XXX
            'prelu_weights': [] # ndim=1 lub 3
        }
        
        def get_id(key):
            match = re.search(r'(\d+)', key)
            return int(match.group(1)) if match else 999999

        for k, v in state_dict.items():
            if "layer" in k or "fc" in k or "features" in k or "bn" in k.lower():
                if not k.startswith("BatchNormalization"): continue

            if k.startswith("BatchNormalization"):
                bn_id = get_id(k)
                found = False
                for item in buckets['bn_stats']:
                    if item['id'] == bn_id:
                        item['keys'].append(k)
                        found = True
                        break
                if not found: buckets['bn_stats'].append({'id': bn_id, 'keys': [k]})
            
            elif k.startswith("_initializer") or k.startswith("Conv") or k.startswith("Gemm"):
                if v.ndim == 4: buckets['conv_weights'].append((get_id(k), v))
                elif v.ndim == 1: buckets['conv_biases'].append((get_id(k), v))
                elif v.ndim == 3: buckets['prelu_weights'].append((get_id(k), v))

        buckets['conv_weights'].sort(key=lambda x: x[0])
        buckets['conv_biases'].sort(key=lambda x: x[0])
        buckets['bn_stats'].sort(key=lambda x: x['id'])
        buckets['prelu_weights'].sort(key=lambda x: x[0])

        misc_1d_3d = buckets['conv_biases'] + buckets['prelu_weights']
        misc_1d_3d.sort(key=lambda x: x[0])

        new_state_dict = OrderedDict()
        idx_conv_w = 0
        idx_misc = 0
        idx_bn = 0
        last_seen_bn_prefix = ""

        # 2. ITERACJA
        for name, param in self.named_parameters():
            onnx_name = "_initializer_" + name.replace(".", "_")
            if onnx_name in state_dict:
                new_state_dict[name] = state_dict[onnx_name]
                continue
            
            # CONV WEIGHT
            if "conv" in name and "weight" in name:
                if idx_conv_w < len(buckets['conv_weights']):
                    src_id, src_val = buckets['conv_weights'][idx_conv_w]
                    if src_val.shape == param.shape:
                        new_state_dict[name] = src_val
                        idx_conv_w += 1
                    else:
                        print(f"⚠️ Mismatch shape conv: {name} {param.shape} vs {src_val.shape}")

            # CONV BIAS
            elif "conv" in name and "bias" in name:
                found = False
                temp_idx = idx_misc
                while temp_idx < len(misc_1d_3d):
                    src_id, src_val = misc_1d_3d[temp_idx]
                    if src_val.shape == param.shape:
                        new_state_dict[name] = src_val
                        misc_1d_3d.pop(temp_idx) 
                        found = True
                        break
                    temp_idx += 1

            # PRELU
            elif "prelu" in name and "weight" in name:
                found = False
                temp_idx = idx_misc
                while temp_idx < len(misc_1d_3d):
                    src_id, src_val = misc_1d_3d[temp_idx]
                    is_match = False
                    val_to_load = src_val
                    if src_val.shape == param.shape: is_match = True
                    elif src_val.ndim == 3 and src_val.shape[0] == param.shape[0]: 
                        is_match = True
                        val_to_load = src_val.squeeze()
                    if is_match:
                        new_state_dict[name] = val_to_load
                        misc_1d_3d.pop(temp_idx)
                        found = True
                        break
                    temp_idx += 1

            # BN (Weight/Bias)
            elif "bn" in name or "downsample.1" in name:
                prefix = ".".join(name.split(".")[:-1])
                if prefix != last_seen_bn_prefix:
                    if last_seen_bn_prefix != "": idx_bn += 1
                    last_seen_bn_prefix = prefix
                
                if idx_bn < len(buckets['bn_stats']):
                    group = buckets['bn_stats'][idx_bn]
                    suffix = name.split('.')[-1]
                    target_k = None
                    for k in group['keys']:
                        if suffix == "weight" and ("scale" in k or ".weight" in k): target_k = k
                        if suffix == "bias" and ("bias" in k and ".weight" not in k): target_k = k
                    if target_k: new_state_dict[name] = state_dict[target_k]

        # 3. BUFFERS
        all_model_keys = list(self.state_dict().keys())
        idx_bn = 0
        last_seen_bn_prefix = ""
        for m_key in all_model_keys:
            if "num_batches_tracked" in m_key:
                new_state_dict[m_key] = torch.tensor(0, dtype=torch.long)
                continue
            if m_key in new_state_dict: continue

            if "bn" in m_key or "downsample.1" in m_key:
                prefix = ".".join(m_key.split(".")[:-1])
                if prefix != last_seen_bn_prefix:
                    if last_seen_bn_prefix != "": idx_bn += 1
                    last_seen_bn_prefix = prefix
                if idx_bn < len(buckets['bn_stats']):
                    group = buckets['bn_stats'][idx_bn]
                    suffix = m_key.split('.')[-1]
                    target_k = None
                    for k in group['keys']:
                        if suffix == "running_mean" and "mean" in k: target_k = k
                        if suffix == "running_var" and "var" in k: target_k = k
                    if target_k: new_state_dict[m_key] = state_dict[target_k]

        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        real_missing = [k for k in missing if "num_batches_tracked" not in k]
        print(f"✅ [IResNet] Załadowano. Brakujące istotne: {len(real_missing)}")
        if len(real_missing) > 0: print(f"   Przykłady: {real_missing[:5]}")

def iresnet50(weights_path=None, **kwargs):
    # OSTATECZNA WERSJA: IBasicBlock (3x3) + Stride 2 w Layer1
    model = IResNet(IBasicBlock, [3, 4, 14, 3], **kwargs)
    if weights_path:
        model.load_weights_from_pth(weights_path)
    return model