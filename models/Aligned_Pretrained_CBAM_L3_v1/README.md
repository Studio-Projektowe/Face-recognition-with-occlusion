# Aligned_Pretrained_CBAM_L3_v1

## Overview
Layer3-specific CBAM attention fine-tuning with checkpoint resume capability, trained on pre-trained CBAM weights with focus on repair/stabilization.

## Base Model
- **Source**: Aligned_Pretrained_L3_v0.pth (best checkpoint from v0)
- **Architecture**: IResNet50 with CBAM at Layer3 output
- **Input Size**: 112×112 RGB
- **Feature Dimension**: 512

## CBAM Configuration (Layer 3 Focus)
- **Channel Attention**: Ratio = 16
  - Linear FC layers with ReLU
  - Handles 256-dim input from Layer3
- **Spatial Attention**: Kernel size = 7
- **Fusion**: Multiplicative with residual connection
- **Initialization**: Loaded from v0 checkpoint (not random)

## Training Configuration
- **Batch Size**: 64
- **Epochs**: 30
- **Training Data**: WebFace dataset (webface_112x112/train)
- **Validation Data**: WebFace dataset (webface_112x112/test)
- **Starting Rank-1 Accuracy**: 65.00% (from v0 checkpoint)

## Learning Rates
- **CBAM LR**: 0.01
- **Optimizer**: SGD with momentum 0.9, weight_decay 5e-4
- **Scheduler**: StepLR with step_size=3, gamma=0.1
- **Fine-tuning Strategy**: Checkpoint resume with continued training

## Loss Function
- **Main Loss**: ArcMargin Loss (metric learning)
- **Validation Metric**: Rank-1 Accuracy on test set

## Layer Freezing Strategy
1. **Phase 1**: Freeze Layer1-3, Unfreeze Layer4 + CBAM
2. **Phase 2**: Full unfreeze - all layers trained

## Data Augmentation
- **Face Occlusion**: Applied during training 100% of the time
  - Horizontal bar: height=20px
  - Position: center_y = 52 ± 5 (eyes region)
  - Color: Random RGB (0-255)
- **Face Alignment**: 5-point landmark-based normalization
  - From JSON landmark files
  - ArcFace-compatible alignment

## Model Checkpointing
- **Best Model Saved**: `fixed_cbam_best.pth`
- **Output Directory**: `checkpoints_repair_layer3_fixed/`
- **Checkpoint Criterion**: Highest Rank-1 Accuracy

## Training Results (from logs)
- **Starting Point**: 65.00% Rank-1 (warm-start from v0)
- **Epoch 1**: Loss: 13.0932 | Acc: 57.00% (dip during unfreezing)
- **Epoch 2**: Loss: 10.2700 | Acc: 63.00%
- **Epoch 3**: Loss: 9.6701 | Acc: 66.00%
- **Epoch 4**: Loss: 8.0169 | Acc: 66.00%
- **Plateau Region**: Epochs 4-6 show stabilization around 65-66%

## Layer Freezing Strategy (Two-Phase)
**Phase 1 (Epoch 1): Selective Unfreeze**
- Layer 1-2: Frozen
- Layer 3: Frozen
- CBAM Module: Trainable (fine-tune pre-trained weights)
- Layer 4: Trainable
- Head: Trainable

**Phase 2 (Epochs 2+): Full Unfreeze**
- ALL layers trainable (Layer1, Layer2, Layer3, CBAM, Layer4, Head)
- Rationale: After CBAM stabilizes, unfreeze all layers
- Learning Rates: Conservative (0.01) to avoid catastrophic forgetting

## Two-Phase Training Strategy
1. **Phase 1** (Epoch 1): Layer4 + CBAM unfrozen, Layer1-3 frozen
   - Loss: 13.09, Accuracy: 57%
   - Allows CBAM and later layers to adapt
   
2. **Phase 2** (Epoch 6+): FULL UNFREEZE
   - All layers trainable
   - Loss: 7.11, Accuracy: 66%
   - Higher accuracy achieved

## Key Differences from v0
- **Checkpoint Resume**: Starts from v0's best weights (not random)
- **Transfer Learning**: Fine-tunes pre-trained CBAM (not training from scratch)
- **Faster Convergence**: Reaches high accuracy immediately (65% start vs 6% in v0)
- **Repair Focus**: Labeled as "repair" - stabilizing and improving v0 weights
- **Two-Phase Strategy**: Explicit freeze/unfreeze phases vs gradual v0

## Evaluation Results (Test Set with 20px Eye Occlusion)

### Test Configuration
- **Gallery**: Clean images from test set
- **Probe**: Synthetically occluded images (20px black bar at eye region)
- **Total Queries**: 24,227
- **Evaluation Method**: FAISS k-NN similarity search

### Performance Metrics

| Metric | Accuracy | Count (Correct/Total) | Delta (vs Aux_v2 Baseline) | Delta (vs L3_v0) |
|--------|----------|----------------------|---------------------------|-------------------|
| **Rank-1 Accuracy** | **81.77%** | 19,815 / 24,227 | **+3.25%** | +24.30% |
| **Rank-3 Accuracy** | **86.24%** | 20,900 / 24,227 | +1.64% | +19.66% |

### The Breakthrough - Full Unfreeze Works!
- **Combined Deep Attention + Full Fine-Tuning Strategy**
- Outperforms non-attention baseline for first time
- +24.30% improvement over L3_v0 shows power of co-adaptation
- Proves attention mechanisms need entire network to adapt together

### Key Success Factors
1. **Deep Placement**: CBAM after Layer3 (semantic features)
2. **Full Unfreeze**: All layers including early layers trainable
3. **Two-Phase Training**: Stabilize with frozen backbone first, then full unfreeze
4. **Conservative Learning**: Low rates prevent catastrophic forgetting

## Additional Notes
- **Warm-Start Advantage**: Begins at 65% accuracy vs 6.2% for v0
- **Checkpoint Detection**: Automatic 1:1 weight mapping for resume
- **Layer3 Refinement**: Deep feature attention at 256-dim level
- **Full Unfreeze**: Phase 2 achieves marginal improvements
- **Repair Strategy**: Named to indicate fixing/improving v0 model
- **Convergence Pattern**: Dip on unfreeze (57% phase 1) → recovery (66% phase 2)
