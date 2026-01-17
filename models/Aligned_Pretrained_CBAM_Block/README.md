# Aligned_Pretrained_CBAM_Block

## Summary
This model is `iResNet50` with `CBAM` inside each block (`CBAMBasicBlock`). Training uses `ArcMarginProduct` on WebFace `112×112` with occlusion augmentation, and validation is mean cosine similarity on `500` pairs.

## Training process
- **Data**: `webface_112x112/train` and `webface_112x112/test`.
- **Alignment**: 5‑point ArcFace landmarks, resize to `112×112`.
- **Occlusion**: random eye band with probability `0.5` in the dataset.
- **Validation**: mean cosine similarity on `500` pairs.
- **Checkpoint**: `checkpoints_cbam_inside_v2/best_cbam_final.pth` (best `Avg Sim`).

## Base model
- **Architecture**: `iResNet50` (`[3,4,14,3]`) with `CBAM` inside blocks.
- **Output**: `512D` embedding after `fc` and `features`.

## Loaded weights
- **In code**: `BACKBONE_PATH = ./Aligned_Pretrained.pth`.
- **In logs**: `./Aligned_Pretrained_Aux_v2.pth` was loaded (`475` backbone layers).
- **Loading**: name/shape matching; missing layers ignored.
- **Loaded layer types**: standard backbone layers (`conv*`, `bn*`, `prelu`, blocks `layer1–layer4`, `fc`, `features`) that match by name/shape.
- **Not loaded (random init)**: `CBAM` layers (`cbam.ca.*`, `cbam.sa.*`) and `metric_fc` (`ArcMarginProduct`) — not present in backbone checkpoints.

## Frozen weights
- **Epoch 1**: all backbone layers frozen (including convolutional, `BN`, `PReLU`, `fc`, `features`); only `CBAM` and `metric_fc` trained.
- **From epoch 2**: full backbone unfrozen and trained jointly.

## Hyperparameters (`train.py`)
- `BATCH_SIZE=64`
- `EPOCHS=30`
- `LR_START=0.1` (SGD, `momentum=0.9`, `weight_decay=5e-4`)
- `StepLR(step_size=10, gamma=0.1)`
- Validation: `num_pairs=500`

## Results
- **Evaluation (`score.txt`)**: Rank-1 = `90.16%`, Rank-3 = `91.88%`.
- **Evaluation** in `evaluation/run_evaluation.py`: metrics saved to `metrics_final_correct/evaluation_results.csv`.

## Training/Evaluation Issues That Could Affect Results
- The code uses `BACKBONE_PATH = ./Aligned_Pretrained.pth`, but logs show `./Aligned_Pretrained_Aux_v2.pth` was actually loaded. This mismatch can make reproduction inconsistent.
- CBAM weights are **not** loaded from the backbone checkpoint (they are random at start), so early training depends heavily on random initialization.
- No fixed random seed is set (augmentation + sampling), so results can vary between runs.

## Epochs (`logs.txt`)
- `logs.txt` contains epochs `1–30`.
- Best `Avg Sim` in the log: `0.4381` (epoch `20`).
