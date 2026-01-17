# Aligned_Pretrained_Transformers

## Summary
The base model is `iResNet50` modified to return a spatial feature map `(B, 512, 7, 7)`. On top of that, the standard ArcFace head (Branch‑1) is used for training and inference, and a Transformer auxiliary head (Branch‑2) is used only during training.

## Training process
- **Data**: `../webface_112x112/train` and `../webface_112x112/test`.
- **Alignment**: 5‑point ArcFace landmarks, resize to `112×112`.
- **Occlusion**: random eye band, probability `0.7`, height `20` px.
- **Validation**: mean cosine similarity on `500` pairs, Branch‑1 only.
- **Checkpoint**: `best_model_transformer.pth` (best `Ver Sim`).
- **Early stopping**: `PATIENCE=5`.

## Base model
- **Architecture**: `iResNet50` (`[3,4,14,3]`) from `backbone_iresnet.py`.
- **Forward**: ends at `bn2` (no `flatten`/`fc`/`features`) — returns a spatial map.

## Loaded weights
- **Source**: `w600k_r50_from_onnx.pth`.
- **Mechanism**: hybrid name matching + shape matching; missing layers re‑initialized.
- **Loaded layer types**: all matching iResNet backbone layers (`conv*`, `bn*`, `prelu`, blocks `layer1–layer4`, `bn2`, and — if present in the checkpoint — `fc` and `features`).
- **Not loaded (random init)**: `ArcMarginProduct` and `TransformerHead` (new layers created in `train.py`).

## Frozen weights
- No explicit freezing in `train.py` (both backbone convolutional layers and new heads are trained). In the backbone, `features.weight` is typically frozen if that layer is used.

## Hyperparameters (`train.py`)
- `BATCH_SIZE=32`
- `EPOCHS=25`
- `LR_HEAD=0.01` (SGD, `momentum=0.9`, `weight_decay=5e-4`)
- `PATIENCE=5`
- `OCCLUSION_PROB=0.7`, `OCCLUSION_HEIGHT=20`
- `NUM_VERIFY_PAIRS=500`
- **Transformer**: `ALPHA=0.4`, `TRANSFORMER_LAYERS=6`, `TRANSFORMER_HEADS=8`
> **Actual batch size**: `32`.

## Results
- **Evaluation (`score.txt`)**: Rank-1 = `85.00%`, Rank-3 = `89.02%`.
- **Evaluation** in `evaluation/eval.py`: metrics saved to `metrics/evaluation_results.csv`.

## Training/Evaluation Issues That Could Affect Results
- In `evaluation/eval.py`, `get_embedding()` calls `model(img)` directly. With this backbone, that returns a **spatial map** `(B, 512, 7, 7)`, not the 512‑D embedding used in training. The evaluation thus uses flattened `512×7×7` features, which is inconsistent with the training embedding pipeline.
- The Transformer head is used only during training; evaluation ignores it, so any gains from Branch‑2 are not reflected at test time.
- No fixed random seed is set (augmentation + sampling), so results can vary between runs.

## Epochs (`logs.txt`)
- `logs.txt` is empty — no epochs recorded.
- Training configuration allows up to `25` epochs with early stopping.
