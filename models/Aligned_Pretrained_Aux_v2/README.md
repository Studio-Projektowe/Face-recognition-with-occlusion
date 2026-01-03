# Aligned_Pretrained_Aux_v2

## Overview
Stage 2 fine-tuning of Aligned_Pretrained_Aux_v1. Continues training with lower learning rates and stricter ArcMargin (easy_margin=False).

## Base Model
- **Source**: Aligned_Pretrained_Aux_v1.pth
- **Architecture**: IResNet50 with auxiliary occlusion head
- **Input Size**: 112×112 RGB
- **Feature Dimension**: 512

## Training Configuration
- **Batch Size**: 32 (with Gradient Accumulation × 2 = effective batch 64)
- **Epochs**: 15
- **Total Training Samples**: 442,710 images (same as v1)
- **Training/Validation Split**: 90% / 10%

## Learning Rates
- **Backbone LR**: 1e-5 (very small - fine-tuning)
- **Head (ArcMargin) LR**: 1e-3
- **Momentum**: 0.9
- **Weight Decay**: 5e-4

## Loss Functions
1. **ArcMargin Loss** (main classification):
   - ArcMargin Parameters: m=0.50, s=30.0
   - easy_margin=False (stricter margin learning)
   
2. **Auxiliary Loss** (occlusion prediction):
   - MSE Loss for 7×7 mask prediction
   - Weight: 0.1 (increased from v1's 0.01)

## Layer Freezing Strategy
- **All Layers Unfrozen**: No selective freezing in this stage
- **Training Strategy**: Conservative learning rates instead of freezing
  - Backbone: Fully trainable at 1e-5 (extremely low for fine-tuning)
  - Head layers: Trainable at 1e-3 (faster head adaptation)
- **Contrast to v1**: v1 used selective layer freezing; v2 uses rate-based control
- **Rationale**: Fine-tuning phase - all parameters can adapt with appropriate learning rates

## Data Augmentation
- **Occlusion Augmentation**: 70% probability
  - Random horizontal bar occlusion (height=20px, centered around y=52±5)
  - Random RGB color
- **Face Alignment**: Landmark-based normalization (5-point landmarks)

## Optimizer & Scheduler
- **Optimizer**: SGD with separate learning rates for backbone and heads
- **LR Scheduler**: ReduceLROnPlateau
  - Factor: 0.1
  - Patience: 2 epochs
- **Early Stopping**: Yes, patience=3 epochs

## Model Checkpointing
- **Best Model Saved**: `aligned_pretrained_v2.pth` (backbone state_dict)
- **Saved When**: Validation loss improves
- **Criterion**: Minimum validation loss

## Stage 2 Modifications vs v1
- Stricter ArcMargin enforcement (easy_margin=False) for better discriminative features
- Much lower backbone learning rate (1e-5 vs 5e-4) - conservative fine-tuning
- Increased auxiliary loss weight (0.1 vs 0.01) - stronger occlusion learning
- Reduced epochs (15 vs 25) - faster convergence expected

## Evaluation Results (Test Set with 20px Eye Occlusion)

### Test Configuration
- **Gallery**: Clean images from test set
- **Probe**: Synthetically occluded images (20px black bar at eye region)
- **Total Queries**: 24,227
- **Evaluation Method**: FAISS k-NN similarity search

### Performance Metrics

| Metric | Accuracy | Count (Correct/Total) | Delta (vs v1) |
|--------|----------|----------------------|---------------|
| **Rank-1 Accuracy** | **78.52%** | 19,023 / 24,227 | +0.71% |
| **Rank-3 Accuracy** | **84.60%** | 20,497 / 24,227 | +0.55% |

### Observations
- Unfreezing backbone allowed model to fine-tune low-level feature extractors
- Low-level features (early layers) adapted to be less sensitive to occlusion edges
- Full-network fine-tuning with low learning rate proved superior to partial freezing
- Marginal improvement validates Stage 2 fine-tuning approach

## Additional Notes
- Gradient accumulation and clipping applied (5.0 norm-based)
- NaN detection enabled
- Debug visualization: Training images saved to `training_photos_stage2/`
- This is a "Stage 2" fine-tuning approach, maintaining generalization while improving occlusion robustness
