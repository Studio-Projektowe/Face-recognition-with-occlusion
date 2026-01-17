import os
import sys
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# Dodaj katalog nadrzędny do importów
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from Aligned_Pretrained.load import load_clean_model
except ImportError:
    load_clean_model = None
    # Fallback: bezpośredni import z pliku
    try:
        import importlib.util

        load_path = os.path.join(MODELS_DIR, "Aligned_Pretrained", "load.py")
        spec = importlib.util.spec_from_file_location("aligned_pretrained_load", load_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        load_clean_model = getattr(module, "load_clean_model", None)
    except Exception:
        load_clean_model = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CONFIG ---
# MODEL_TYPE = "baseline"  # baseline | cbam | transformer
# WEIGHTS_PATH = "../Aligned_Pretrained/pytorch_model/baseline.pth"
# MODEL_TYPE = "transformer"  # baseline | cbam | transformer
# WEIGHTS_PATH = "../Aligned_Pretrained_Transformers/pytorch_model/Aligned_Pretrained_Transformers.pth"
MODEL_TYPE = "cbam"  # baseline | cbam | transformer
WEIGHTS_PATH = "../Aligned_Pretrained_CBAM_Block/pytorch_model/Aligned_Pretrained_CBAM_Block_v2.pth"
IMAGE_PATH = "./grad_cam/image5.jpg"
OUTPUT_PATH = "./grad_cam/cbam5.jpg"
REF_IMAGE_PATH = "ref_image4.jpg"  # obraz referencyjny do wektora podobieństwa (opcjonalny)
USE_SIM_TARGET = False  # jeśli True, Grad-CAM maksymalizuje podobieństwo do REF_IMAGE_PATH


def get_target_layer(model):
    if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        return model.backbone.layer4[-1]
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    raise ValueError("Nie znaleziono layer4 w modelu.")


def _filter_and_load(model, state_dict, min_match_ratio=0.2, weights_path=""):
    new_state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in new_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v

    if not filtered_state:
        raise ValueError(
            "Nie wczytano żadnych wag — sprawdź zgodność architektury i ścieżkę do pliku."
        )

    match_ratio = len(filtered_state) / max(1, len(model_state))
    if match_ratio < min_match_ratio:
        raise ValueError(
            f"Wczytano zbyt mało wag ({match_ratio:.1%}). "
            "Prawdopodobnie to inna architektura (np. CBAM/Transformer)."
        )

    model.load_state_dict(filtered_state, strict=False)
    print(f"Wczytano {len(filtered_state)}/{len(model_state)} warstw z: {weights_path}")


def load_model(weights_path=None, min_match_ratio=0.2, model_type="baseline"):
    if model_type == "baseline":
        if load_clean_model is not None:
            model = load_clean_model()
        else:
            # Fallback: zbuduj backbone bezpośrednio
            from Aligned_Pretrained.backbone_iresnet import iresnet50
            model = iresnet50(weights_path=None)
    elif model_type == "transformer":
        transformer_dir = os.path.join(MODELS_DIR, "Aligned_Pretrained_Transformers")
        if transformer_dir not in sys.path:
            sys.path.insert(0, transformer_dir)
        from Aligned_Pretrained_Transformers.load import load_clean_model as load_transformer
        model = load_transformer()
    elif model_type == "cbam":
        from Aligned_Pretrained_CBAM_Block.evaluation.run_evaluation import IResNetCBAM, CBAMBasicBlock
        model = IResNetCBAM(CBAMBasicBlock, [3, 4, 14, 3])
    else:
        raise ValueError("MODEL_TYPE musi być: baseline | cbam | transformer")

    if weights_path:
        state_dict = torch.load(weights_path, map_location=DEVICE)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        _filter_and_load(model, state_dict, min_match_ratio=min_match_ratio, weights_path=weights_path)

    model.to(DEVICE)
    model.eval()
    return model


def generate_gradcam(model, img_path, output_path, target_layer=None):
    if target_layer is None:
        target_layer = get_target_layer(model)

    rgb_img = cv2.imread(img_path, 1)
    if rgb_img is None:
        raise FileNotFoundError(f"Nie znaleziono obrazu: {img_path}")
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (112, 112))
    rgb_img = np.float32(rgb_img) / 255.0

    input_tensor = preprocess_image(
        rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    ).to(DEVICE)

    class CosineSimilarityTarget:
        def __init__(self, ref_emb):
            self.ref_emb = ref_emb

        def __call__(self, model_output):
            # model_output: (B, D) embedding
            if model_output.dim() > 2:
                model_output = torch.flatten(model_output, 1)
            return torch.nn.functional.cosine_similarity(model_output, self.ref_emb, dim=1).mean()

    class ScalarOutputTarget:
        def __call__(self, model_output):
            if model_output.dim() > 2:
                model_output = torch.flatten(model_output, 1)
            return model_output.mean()

    targets = None
    if USE_SIM_TARGET:
        if REF_IMAGE_PATH is None:
            raise ValueError("USE_SIM_TARGET=True, ale brak REF_IMAGE_PATH.")

        ref_img = cv2.imread(REF_IMAGE_PATH, 1)
        if ref_img is None:
            raise FileNotFoundError(f"Nie znaleziono obrazu referencyjnego: {REF_IMAGE_PATH}")
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_img = cv2.resize(ref_img, (112, 112))
        ref_img = np.float32(ref_img) / 255.0

        ref_tensor = preprocess_image(
            ref_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        ).to(DEVICE)

        with torch.no_grad():
            ref_emb = model(ref_tensor)
            if ref_emb.dim() > 2:
                ref_emb = torch.flatten(ref_emb, 1)
            ref_emb = torch.nn.functional.normalize(ref_emb, dim=1)

        targets = [CosineSimilarityTarget(ref_emb)]
    else:
        targets = [ScalarOutputTarget()]

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    model = load_model(weights_path=WEIGHTS_PATH, model_type=MODEL_TYPE)
    generate_gradcam(
        model=model,
        img_path=IMAGE_PATH,
        output_path=OUTPUT_PATH,
        target_layer=get_target_layer(model),
    )
    print(f"Grad-CAM zapisany jako {OUTPUT_PATH}")