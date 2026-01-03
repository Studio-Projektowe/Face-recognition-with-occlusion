# Aligned_Pretrained_CBAM_L1_v2

## Overview
Stage 2 continuation of CBAM Layer 1 training, fine-tuning pre-trained CBAM attention module with refined occlusion augmentation and pair-based validation.

## Base Model
- **Source**: Aligned_Pretrained_CBAM_L1_v1.pth (from v1 best checkpoint)
- **Architecture**: IResNet50 with CBAM attention at Layer 1
- **Input Size**: 112×112 RGB
- **Feature Dimension**: 512

## CBAM Configuration (Layer 1)
- **Channel Attention**: Ratio = 16
  - Adaptive pooling + Linear FC layers
- **Spatial Attention**: Kernel size = 7
- **Initialization**: Pre-trained from v1 (not random)
- **Focus**: Layer 1 early feature enhancement

## Training Configuration
- **Batch Size**: 64
- **Epochs**: 30
- **Training Data**: 442,710 images from WebFace
- **Validation Pairs**: 200 aligned image pairs
- **Starting Rank-1 Accuracy**: 18.50% (from v1 checkpoint)

## Learning Rates
- **CBAM LR**: 0.01
- **Head LR**: 0.01
- **Optimizer**: SGD with momentum
- **Fine-tuning Strategy**: Continued training of v1 weights

## Loss Function
- **Main Loss**: ArcMargin Loss (metric learning)
- **Validation Metric**: Rank-1 Accuracy on 200 verification pairs

## Data Augmentation
- **Occlusion Augmentation**: Refined from v1
  - Horizontal bar: height=10px
  - Position: center_y = 52 ± 5 (eyes region)
  - Color: Random RGB (0-255)
- **Face Alignment**: 5-point landmark-based normalization
  - Uses skimage SimilarityTransform
  - ArcFace-compatible alignment
  - Handles missing landmarks with resize fallback

## Verification Strategy
- **Pair Type**: Clean reference + occluded query
- **Occlusion Position**: Eyes region (y=42-62 for center=52±10)
- **Similarity Metric**: Cosine similarity between L2-normalized embeddings
- **Pair Count**: 200 validation pairs
- **Validation Frequency**: After each epoch

## Model Checkpointing
- **Best Model Saved**: `repaired_cbam_best.pth`
- **Output Directory**: `checkpoints_repair/`
- **Criterion**: Highest Rank-1 Accuracy

## Training Results (from logs)
- **Starting Point**: 18.50% Rank-1 (v1 checkpoint)
- **Epoch 1**: 20.50% Rank-1 Accuracy (improvement over starting point)
- **Epoch 2**: 22.00% Rank-1 Accuracy (best)
- **Epoch 4**: 23.00% Rank-1 Accuracy (achieved)
- **Plateau**: Around 21-22% after epoch 8

## Layer Freezing Strategy
- **Backbone (Layers 1-3)**:
  - Convolutions: Frozen (preserve v1 learning)
  - Batch normalization: Trainable
  - PReLU: Trainable
  - Conservative approach

- **Layer 4**: Fully trainable

- **CBAM (Layer 1)**:
  - Pre-trained from v1 (not random)
  - Fine-tuning mode: trainable at 0.01 LR
  - More refined than v1's from-scratch training

## Key Differences from v1
- **Checkpoint Loading**: Starts from v1's best weights (not random)
- **Transfer Learning**: Fine-tunes already-trained CBAM
- **Faster Convergence**: Reaches good accuracy faster due to warm-start
- **Same Augmentation**: Similar occlusion strategy maintained
- **Layer Freezing**: More conservative - only tunes BN/PReLU in layers 1-3

## Evaluation Results (Test Set with 20px Eye Occlusion)

### Test Configuration
- **Gallery**: Clean images from test set
- **Probe**: Synthetically occluded images (20px black bar at eye region)
- **Total Queries**: 24,227
- **Evaluation Method**: FAISS k-NN similarity search

### Performance Metrics

| Metric | Accuracy | Count (Correct/Total) | Delta (vs v1) | Delta (vs Aux_v2 Baseline) |
|--------|----------|----------------------|---------------|---------------------------|
| **Rank-1 Accuracy** | **43.76%** | 10,605 / 24,227 | +4.94% | **-34.76%** |
| **Rank-3 Accuracy** | **55.72%** | 13,505 / 24,227 | +4.44% | -28.88% |

### Observations
- **Plateau Effect**: Continued training from v1 achieved only marginal improvement (+4.94%)
- Saturation at ~44% indicates **structural limitation**, not temporal underfitting
- Frozen backbone with front-end attention remains fundamentally incompatible
- Extended training confirms: covariate shift is architectural problem, not a convergence problem
- Issue: Front-end CBAM introduces incompatible feature distribution for frozen layers

## Key Findings
- **Checkpoint Loading**: Starts from v1's best weights (not random)
- **Transfer Learning**: Fine-tunes already-trained CBAM
- **Plateau Phenomenon**: Shows law of diminishing returns with frozen backbone
- **Conclusion**: Attention mechanisms cannot work with frozen networks - requires full co-adaptation
