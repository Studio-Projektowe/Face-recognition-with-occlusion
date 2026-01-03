# Aligned_Pretrained_CBAM_Block_v1

## Overview
IResNet50 backbone with CBAM (Convolutional Block Attention Module) integrated into blocks, trained for occlusion-robust face recognition with 500-pair verification validation.

## Base Model
- **Source**: Aligned_Pretrained_Aux_v2.pth
- **Architecture**: IResNet50 with CBAM attention integrated in residual blocks
- **CBAM Integration**: Inside each CBAMBasicBlock (after convolutions, before residual add)
- **Input Size**: 112×112 RGB
- **Feature Dimension**: 512

## CBAM Configuration
- **Channel Attention**: Ratio = 16
- **Spatial Attention**: Kernel size = 7
- **Attention Placement**: Applied to each basic block
- **Fusion**: Multiplicative attention on input, then residual add

## Training Configuration
- **Batch Size**: 64
- **Epochs**: 30
- **Total Training Samples**: WebFace dataset (webface_112x112)
- **Validation**: 500 pairs of aligned images

## Learning Rates
- **Initial LR**: 0.1 (LR_START)
- **Optimizer**: SGD with momentum
- **Scheduler**: Not explicitly specified in provided code (likely step-based decay)

## Loss Function
- **Main Loss**: CosineSimilarity-based (inferred from verification validation)
- **Validation Metric**: Average cosine similarity on 500 test pairs

## Data Augmentation
- **Occlusion Augmentation**: 50% probability
  - Random horizontal bar occlusion (height=20px, centered at y=42-62)
  - Random RGB color
- **Face Alignment**: Landmark-based normalization (5-point landmarks from JSON)

## Model Checkpointing
- **Best Model Saved**: `best_model_cbam.pth`
- **Saved When**: Validation loss improves (similarity increases)
- **Verification Strategy**: 500 pairs with one clean and one occluded image

## Layer Freezing Strategy
- **Phase 1 (Epoch 1)**:
  - CBAM: Frozen
  - Backbone (Layers 1-3): Frozen
  - Backbone (Layer 4): Trainable
  - Purpose: Stabilize backbone before CBAM training
  
- **Phase 2 (Epochs 2+)**:
  - CBAM: Unfrozen and trained
  - Backbone: All layers trainable
  - All parameters participating in gradient updates

## Training Phases
1. **Phase 1**: CBAM frozen, backbone fine-tuned (first epoch)
2. **Phase 2**: Full unfreeze - all layers including CBAM trained

## Key Features
- **Layer Freezing**: Backbone initially frozen, then unfrozen for full training
- **Verification Validation**: Every epoch validates on 500 pairs (average cosine similarity)
- **Two-phase strategy**: Stabilize with frozen CBAM first, then full fine-tuning

## Training Results (from logs)
- Started: 32.02% similarity on 500 pairs
- Epoch 12: 43.12% similarity (best improvement)
- Consistent improvement in early epochs, stabilization after epoch 10

## Evaluation Results (Test Set with 20px Eye Occlusion)

### Test Configuration
- **Gallery**: Clean images from test set
- **Probe**: Synthetically occluded images (20px black bar at eye region)
- **Total Queries**: 24,227
- **Evaluation Method**: FAISS k-NN similarity search

### Performance Metrics

| Metric | Accuracy | Count (Correct/Total) | Delta (vs Aux_v2 Baseline) |
|--------|----------|----------------------|---------------------------|
| **Rank-1 Accuracy** | **87.58%** | 21,220 / 24,227 | +9.06% |
| **Rank-3 Accuracy** | **90.10%** | 21,850 / 24,227 | +5.50% |

### Key Achievement
- Integrated CBAM in every residual block (distributed attention)
- Significantly outperforms single-point attention insertion
- Integrated attention allows network to "clean" signal at every transformation step
- Model builds representation that is intrinsically robust to occlusion

## Additional Notes
- Occlusion-robust training: occluded images paired with clean references
- Model designed for handling facial occlusions (masks, hands, etc.)
- Small validation set (500 pairs) focuses on alignment quality
- Output directory: `checkpoints_cbam_inside_v2`
