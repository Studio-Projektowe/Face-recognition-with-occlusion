# Aligned_Pretrained_CBAM_L1_v1

## Overview
Layer1-specific CBAM attention module trained on IResNet50 backbone for occlusion-robust face recognition with alignment and pair-based validation.

## Base Model
- **Source**: Aligned_Pretrained_Aux_v2.pth
- **Architecture**: IResNet50 with custom CBAM attention
- **Input Size**: 112×112 RGB
- **Feature Dimension**: 512

## CBAM Configuration (Layer 1 Focus)
- **Channel Attention**: Ratio = 16
  - FC: Linear layers with ReLU activation
  - Output: (B, C, 1, 1)
- **Spatial Attention**: Kernel size = 7
  - Conv2d with sigmoid activation
- **Forward**: Channel attention → Spatial attention → Multiplicative fusion
- **Placement**: Applied to Layer 1 of the backbone

## Training Configuration
- **Batch Size**: 64
- **Epochs**: 30
- **Training Data**: 442,710 images from WebFace (webface_112x112/train)
- **Validation Pairs**: 200 image pairs with alignment
- **Number of Identities**: 100 (for verification set)

## Learning Rates
- **CBAM LR**: 0.01
- **Optimizer**: SGD with momentum
- **Momentum**: 0.9 (standard)
- **Note**: All parameters (CBAM + backbone BN/PReLU + head) use single LR of 0.01

## Loss Function
- **Training Loss**: ArcMargin Loss (softmax-based metric learning)
- **Validation Metric**: Rank-1 Accuracy on aligned verification pairs

## Data Augmentation
- **Occlusion Augmentation**: Applied during training
  - Horizontal bar occlusion: height=20px (same as other models)
  - Position: center_y = 52 ± 5 (eyes region)
  - Color: Random RGB values
- **Face Alignment**: 5-point landmark normalization
  - Landmarks from JSON files
  - Uses skimage SimilarityTransform
  - Output: 112×112 aligned face

## Layer Freezing Strategy
- **Backbone (Layers 1-3)**:
  - Convolutions: Frozen
  - Batch normalization: Unfrozen (trainable)
  - PReLU activations: Unfrozen
  - Allows BN to adapt while preserving conv filters

- **Layer 4**: Fully trainable (all weights trainable)

- **CBAM (Layer 1 focus)**: Fully trainable
  - Channel attention FC layers: trainable (LR: 0.01)
  - Spatial attention Conv: trainable
  - New module trained from scratch

## Verification Strategy
- **Pair Generation**: Clean image + occluded version of same identity
- **Similarity Metric**: Cosine similarity between embeddings
- **Validation Set Size**: 200 pairs
- **Validation Frequency**: Every epoch

## Model Checkpointing
- **Best Model Saved**: `repaired_cbam_best.pth`
- **Output Directory**: `checkpoints_repair/`
- **Checkpoint Criterion**: Highest Rank-1 Accuracy

## Training Results (from logs)
- **Initial (Random CBAM)**: 1.50% Rank-1 Accuracy
- **Epoch 1**: 15.00% Rank-1 Accuracy
- **Epoch 5**: 22.00% Rank-1 Accuracy (best)
- **Final**: Convergence around 22% Rank-1 Accuracy on 200 validation pairs
- **Early Stopping**: Yes, patience monitored

## Architecture Notes
- **Repair Strategy**: CBAM module initialization and training from scratch
- **Alignment Focus**: Heavy emphasis on face alignment quality
- **Occluded Pair Validation**: Validates ability to match occluded to clean faces
- **Layer 1 Attention**: Early feature extraction enhancement

## Evaluation Results (Test Set with 20px Eye Occlusion)

### Test Configuration
- **Gallery**: Clean images from test set
- **Probe**: Synthetically occluded images (20px black bar at eye region)
- **Total Queries**: 24,227
- **Evaluation Method**: FAISS k-NN similarity search

### Performance Metrics

| Metric | Accuracy | Count (Correct/Total) | Delta (vs Aux_v2 Baseline) |
|--------|----------|----------------------|---------------------------|
| **Rank-1 Accuracy** | **38.82%** | 9,405 / 24,227 | **-39.70%** |
| **Rank-3 Accuracy** | **51.28%** | 12,429 / 24,227 | -33.32% |

### Observations - Significant Performance Degradation
- **Critical Finding**: Drastic drop demonstrates that front-end attention cannot be "plugged into" frozen networks
- Root cause: Randomly initialized CBAM disrupts input statistics that pre-trained backbone expects
- Input distribution mismatch creates "covariate shift" - frozen layers treat modified features as noise
- Training: From random initialization (1.5%) to 22% validation accuracy after 10 epochs
- Final test accuracy: 38.82% on occluded probe set, but far below baseline 78.52%
- **Conclusion**: Attention modules require **full co-adaptation** with the backbone

## Additional Notes
- Focus on **repairing/training CBAM attention** from scratch
- Alignment-based training (JSON landmarks required)
- Small validation set (200 pairs) for focused evaluation
- Two-stage approach: freeze/unfreeze backbone
- Designed to improve robustness to occluded regions in eyes area
