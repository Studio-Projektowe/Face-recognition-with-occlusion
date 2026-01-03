# Aligned_Pretrained_CBAM_L3_v0

## Overview
IResNet50 with CBAM attention inserted after Layer 3, trained with comprehensive face alignment and rank-k evaluation on WebFace dataset.

## Base Model
- **Source**: Aligned_Pretrained_Aux_v2.pth (actually loads from w600k path in logs)
- **Architecture**: IResNet50 with single CBAM module (after Layer3, before Layer4)
- **Input Size**: 112×112 RGB
- **Feature Dimension**: 512

## CBAM Configuration (Layer 3 Output)
- **Channel Attention**: Ratio = 16
  - Linear FC layers with ReLU
  - Adaptive pooling on 256 channels
- **Spatial Attention**: Kernel size = 7
  - Conv2d(2, 1, kernel_size=7, padding=3)
- **Placement**: After Layer3 (256-dim feature maps)
- **Residual Connection**: CBAM output added to input (out = CBAM(x) + x)

## Training Configuration
- **Batch Size**: 32 (with Gradient Accumulation ×2 = effective batch 64)
- **Epochs**: 25
- **Total Training Samples**: 442,710 images in 9,514 identities
- **Validation Samples**: 500 pairs for Rank-1 evaluation
- **Training/Validation Split**: 90% / 10% of full dataset

## Learning Rates
- **Backbone LR**: 1e-3
- **Head (ArcMargin) LR**: 0.01
- **CBAM LR**: 5e-3
- **Optimizer**: SGD with momentum 0.9
- **Weight Decay**: 5e-4

## Loss Function
- **Main Loss**: ArcMargin Loss (metric learning)
  - m=0.50, s=30.0
  - easy_margin setting (boolean flag)
- **Auxiliary Loss**: MSE Loss for 7×7 occlusion mask (Weight: 0.01)
- **Total Loss**: ArcMargin + 0.01 × MSE

## Layer Freezing Strategy (Detailed)
- **Layer 1**: Fully frozen (conv, BN, PReLU) - preserves edge/color detection

- **Layer 2**: Fully frozen (conv, BN, PReLU) - preserves texture patterns

- **Layer 3**: Selectively trainable
  - Convolutions: Frozen
  - Batch Normalization: Trainable
  - PReLU: Trainable
  - Allows mid-level feature adaptation while preserving conv filters

- **CBAM Module (after Layer3)**: Fully trainable
  - New attention module trained from scratch
  - Channel attention LR: 5e-3
  - Spatial attention LR: 5e-3
  - Critical for occlusion robustness

- **Layer 4**: Fully trainable (LR: 1e-3)
  - All convolutions, BN, PReLU trainable
  - Final feature extraction

- **FC + ArcMargin Head**: Trainable
  - ArcMargin LR: 0.01
  - Identity classification

- **Statistics**: Frozen: 77 params | Trainable: 164 params
- **Strategy**: Conservative mid-layer freezing + full head/CBAM training

## Data Augmentation
- **Face Occlusion**: 70% probability
  - Random horizontal bar (height=20px, centered at y=52±5)
  - Random RGB color
- **Face Alignment**: Landmark-based with face_align module
  - 5-point landmarks from JSON
  - Handles missing landmarks with resize fallback

## Validation Metrics
- **Rank-1 Accuracy**: Percentage of queries matching to correct identity as top-1
- **Validation Pairs**: 500 image pairs
- **Frequency**: Every epoch

## Model Checkpointing
- **Path**: Based on epoch performance
- **Criterion**: Minimum validation loss
- **Metrics CSV**: `training_metrics_cbam_20251224_180801.csv`

## Training Results (from logs)
- **Epoch 1**: Train Loss: 15.8732 | Val Loss: 13.1350 | Rank-1: 6.20%
- **Epoch 5**: Train Loss: 9.3061 | Val Loss: 9.8954 | Rank-1: 7.20%
- **Epoch 9**: Train Loss: 8.4548 | Val Loss: 9.6487 | Rank-1: 7.80% (plateau)
- **Epoch 14**: Train Loss: 7.5100 | Val Loss: 9.4995 | Rank-1: 6.80%
- **Epoch 18**: Training stopped (likely early stopping triggered)

## Early Stopping
- **Patience**: 5 epochs
- **Metric**: Validation loss
- **Triggered at Epoch 18**: After 5 epochs without improvement

## Additional Notes
- **Hybrid Loading**: Greedy weight matching from ONNX/PTH files
- **79 Unused Layers**: Discarded from pre-training
- **475 Loaded Layers**: Successfully transferred
- **CSV Logging**: Detailed training metrics saved for analysis
- **CBAM Position**: After Layer3 (256-dim features) - good balance between early and deep features
- **Occlusion + Alignment**: Dual robustness targets (occlusion resistance + pose invariance)
- **Metrics Directory**: `metrics_cbam/`

## Evaluation Results (Test Set with 20px Eye Occlusion)

### Test Configuration
- **Gallery**: Clean images from test set
- **Probe**: Synthetically occluded images (20px black bar at eye region)
- **Total Queries**: 24,227
- **Evaluation Method**: FAISS k-NN similarity search

### Performance Metrics

| Metric | Accuracy | Count (Correct/Total) | Delta (vs Aux_v2 Baseline) | Delta (vs L1_v2) |
|--------|----------|----------------------|---------------------------|-------------------|
| **Rank-1 Accuracy** | **57.47%** | 13,935 / 24,227 | -21.05% | +13.71% |
| **Rank-3 Accuracy** | **66.58%** | 16,138 / 24,227 | -18.02% | +10.86% |

### Key Insight - Deep Placement Matters
- Substantial improvement over L1 models (+13.71%) shows **semantic features are more robust**
- Moving attention deeper (Layer3 vs Layer1) allows adaptation to abstract concepts
- Network better tolerates attention re-weighting at semantic level than pixel level
- Still underperforms baseline, indicating inserted layer creates bottleneck

### Conclusion
- Deep attention is more effective than front-end attention
- Partial fine-tuning not sufficient for full adaptation
- Need for full network co-adaptation with attention module
- Paves way for L3_v1 breakthrough with complete unfreezing

## Architecture Insight
- **Layer3 CBAM Focus**: 256-channel attention at mid-level features
- **Before Layer4**: Improves feature quality before final pooling
- **Residual Add**: Preserves original features while learning attention
