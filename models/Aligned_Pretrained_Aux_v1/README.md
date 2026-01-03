# Aligned_Pretrained_Aux_v1

## Overview
IResNet50 with auxiliary occlusion prediction head (7×7 mask prediction), trained with face alignment on WebFace dataset.

## Base Model
- **Architecture**: IResNet50
- **Input Size**: 112×112 RGB
- **Feature Dimension**: 512

## Training Configuration
- **Batch Size**: 32 (with Gradient Accumulation × 2 = effective batch 64)
- **Epochs**: 25
- **Total Training Samples**: 442,710 images across 9,514 identities
- **Training/Validation Split**: 90% / 10%

## Learning Rates
- **Backbone LR**: 5e-4 (SGD)
- **Head (ArcMargin) LR**: 0.01
- **Momentum**: 0.9
- **Weight Decay**: 5e-4

## Loss Functions
1. **ArcMargin Loss** (main classification):
   - ArcMargin Parameters: m=0.50, s=30.0
   - easy_margin=True (during training)
   
2. **Auxiliary Loss** (occlusion prediction):
   - MSE Loss for 7×7 mask prediction
   - Weight: 0.01 (relative to ArcMargin loss)

## Layer Freezing Strategy
- **Initially Frozen**: Layer1, Layer2, and early backbone parts (conv1)
- **Selectively Trainable**:
  - SE (Squeeze-and-Excitation) modules: unfrozen
  - Layer3: unfrozen
  - Layer4: unfrozen
  - Batch Normalization layers: unfrozen
  - FC head layers: unfrozen
  - PReLU activations in layer3/layer4: unfrozen
- **Frozen**: 77 parameters | **Trainable**: 164 parameters
- **Rationale**: Preserve early learned features while allowing mid-to-late layers to adapt to occlusion task

## Data Augmentation
- **Occlusion Augmentation**: 70% probability
  - Random horizontal bar occlusion (height=20px, centered around y=52±5)
  - Random RGB color
- **Face Alignment**: Landmark-based alignment to normalize pose using landmarks (5-point)

## Optimizer & Scheduler
- **Optimizer**: SGD with separate learning rates for backbone and head
- **LR Scheduler**: ReduceLROnPlateau
  - Factor: 0.1
  - Patience: 2 epochs
- **Early Stopping**: Yes, patience=5 epochs

## Model Checkpointing
- **Best Model Saved**: `best_model_merged.pth` (backbone state_dict)
- **Saved When**: Validation loss improves
- **Criterion**: Minimum validation loss

## Training Logs Summary
- Pre-training weights loaded from: `w600k_r50_from_onnx.pth` (ONNX format converted)
- 79 layers from ONNX model were discarded (shape/naming incompatibility)
- 475 layers successfully loaded and frozen initially
- Layer freezing strategy: Selective unfreezing of SE modules, layer3, layer4, batch norm, FC layers

## Evaluation Results (Test Set with 20px Eye Occlusion)

### Test Configuration
- **Gallery**: Clean images from test set
- **Probe**: Synthetically occluded images (20px black bar at eye region)
- **Total Queries**: 24,227
- **Evaluation Method**: FAISS k-NN similarity search

### Performance Metrics

| Metric | Accuracy | Count (Correct/Total) |
|--------|----------|----------------------|
| **Rank-1 Accuracy** | **77.81%** | 18,851 / 24,227 |
| **Rank-3 Accuracy** | **84.05%** | 20,362 / 24,227 |

### Observations
- Strong baseline capabilities for handling severe eye occlusion
- Validates effectiveness of auxiliary loss strategy (mask prediction)
- Serves as foundation for downstream v2 fine-tuning and CBAM variants

## Additional Notes
- Gradient clipping: 5.0 (norm-based)
- NaN detection and step skipping enabled
- Debug visualization: Training images saved every epoch to `training_photos/`
- Designed for handling occluded faces with auxiliary mask prediction task 
