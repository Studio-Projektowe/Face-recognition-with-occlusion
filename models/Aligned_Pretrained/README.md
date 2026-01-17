# Aligned_Pretrained

## Summary
The base model is `iResNet50` (ArcFace) from `backbone_iresnet.py`. Training is classification with `ArcMarginProduct` on WebFace 112x112, with occlusion augmentation and cosine-similarity validation on pairs.

## Training process
- **Data**: `../webface_112x112/train` and `../webface_112x112/test`.
- **Alignment**: 5-point ArcFace landmarks, resize to `112×112`.
- **Occlusion**: random eye band, probability `0.7`, height `20` px.
- **Validation**: mean cosine similarity on `500` pairs (clean vs occluded).
- **Checkpoint**: best model saved as `baseline.pth` (full backbone + head `state_dict`).
- **Early stopping**: `PATIENCE=5` based on `Ver Sim`.

## Base model
- **Architecture**: `iResNet50` (`[3,4,14,3]`) with `BatchNorm` and `PReLU`.
- **Output**: `512D` embedding after `fc` and `features`.

## Loaded weights
- **Source**: `w600k_r50_from_onnx.pth`.
- **Mechanism**: hybrid name matching + shape matching; missing layers re-initialized.
- **Loaded layer types**: all backbone layers that matched by name or shape: `conv*`, `bn*`, `prelu`, blocks `layer1–layer4`, `fc`, and `features`.
- **From logs — missing (re-initialized) specific weights**:
	- layer1.2.bn3.{weight,bias,running_mean,running_var}
	- layer2.3.bn2.running_var
	- layer2.3.bn3.{weight,bias,running_mean,running_var}
	- layer2.3.prelu.weight
	- layer3.12.bn2.{running_mean,running_var}
	- layer3.12.bn3.{weight,bias,running_mean,running_var}
	- layer3.12.prelu.weight
	- layer3.13.bn2.{weight,bias,running_mean,running_var}
	- layer3.13.bn3.{weight,bias,running_mean,running_var}
	- layer3.13.prelu.weight

## Frozen weights
- No explicit freezing in `train.py`.
- In the backbone, only `features.weight` (final `BN1d`) is frozen; all other layers (including convolutional) are trainable.

## Hyperparameters (`train.py`)
- `BATCH_SIZE=32`
- `EPOCHS=25`
- `LR_HEAD=0.01` (SGD, `momentum=0.9`)
- `PATIENCE=5`
- `OCCLUSION_PROB=0.7`, `OCCLUSION_HEIGHT=20`
- `NUM_VERIFY_PAIRS=500`
> **Actual batch size**: `32`.

## Results
- **Evaluation (`score.txt`)**: Rank-1 = `87.11%`, Rank-3 = `90.56%`.
- **Evaluation (`logs_eval.txt`)**: Top-1 = `87.11%`, Top-3 = `90.56%` on `24,227` queries.

## Training/Evaluation Issues That Could Affect Results
- The training checkpoint stores the full model (`backbone` + `ArcMarginProduct` head), while evaluation loads only the backbone (head weights are ignored). This is fine for embeddings but can diverge from the exact training objective.
- No fixed random seed is set (augmentation + sampling), so results can vary between runs.

## Epochs (`logs.txt`)
- `logs.txt` contains epochs `1–12`.
- Best `Ver Sim` in the log: `0.4110` (epoch `8`).
- Last entry: epoch `12`, `Patience: 4/5` (no record of completing 25 epochs).

## File compatibility notes
- **Training** saves `baseline.pth`.
- **Evaluation** in `evaluation/eval.py` uses `baseline.pth`.
