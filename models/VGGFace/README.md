# VGGFace

## Overview
VGGFace is a legacy face recognition model based on the VGG deep neural network architecture. It represents an earlier generation of deep learning approaches to face recognition (2015-2018 era). We include it as a historical comparison to demonstrate the significant advancements made by modern margin-based loss functions (such as ArcFace) over softmax-based approaches.

## Model Architecture
- **Backbone**: VGG-based deep convolutional network
- **Loss Function**: Softmax with standard cross-entropy loss (no margin)
- **Feature Dimension**: Variable (typically 512 or 2048 depending on variant)
- **Output**: Feature embeddings without explicit margin enforcement
- **Depth**: Deep VGG stack with multiple convolutional layers

## Training Dataset
- **Dataset**: VGGFace (original) or VGGFace2
- **Training Method**: Standard supervised learning with softmax loss
- **Pre-training Era**: 2015-2018 (pre-ArcFace era)
- **Feature Learning**: Standard cross-entropy without angular margin optimization

## Model Properties
- **Model Type**: Legacy / Historical baseline
- **Optimization Target**: Standard classification accuracy via softmax
- **Loss Function**: Cross-entropy (no margin enforcement)
- **Framework**: Originally TensorFlow/Keras, available in various implementations
- **Inference Speed**: Moderate (slower than modern efficient models)
- **Modern Status**: Largely superseded by ArcFace and modern metric learning approaches

## Evaluation Results (Test Set with 20px Eye Occlusion)

### Test Configuration
- **Gallery**: Clean images from WebFace test set
- **Probe**: Synthetically occluded images (20px black bar at eye region)
- **Total Queries**: 24,227
- **Evaluation Method**: FAISS k-NN similarity search (Cosine Similarity)

### Performance Metrics

| Metric | Accuracy | Count (Correct/Total) | Performance Tier |
|--------|----------|----------------------|-----------------|
| **Rank-1 Accuracy** | **39.64%** | 9,604 / 24,227 |  Severe Degradation |
| **Rank-3 Accuracy** | **57.01%** | 13,815 / 24,227 |  Poor Performance |

### Observations
- **Severe Occlusion Sensitivity**: Dramatic performance collapse under eye occlusion
- **Margin-Based Loss Advantage**: Demonstrates clear superiority of ArcFace/margin-based losses
- **Softmax Limitations**: Standard softmax loss produces suboptimal feature margins
- **Legacy Status**: Represents outdated approach to face recognition
- **Historical Baseline**: Shows 3-year advancement (VGGFace 2015 → ArcFace 2018) in loss function design
- **Generalization Gap**: Poor transfer to occluded scenarios - dramatic performance collapse

## Performance Degradation Comparison

| Model | Rank-1 Clean | Rank-1 Occluded | Degradation | Robustness |
|-------|--------------|-----------------|-------------|-----------|
| ArcFace Large | ~99%+ | 94.81% | ~4-5% |  Excellent |
| ArcFace Small | ~99%+ | 86.65% | ~12-13% |  Good |
| Aligned_Pretrained_CBAM_Block_v2 | ~93% | 90.16% | ~3% |  Very Good |
| **VGGFace** | ~73-75% | **39.64%** | **~34-35%** |  **Poor** |

### Key Insight: Margin-Based Loss Functions
The dramatic difference between VGGFace and modern models illustrates the critical importance of **margin-based softmax losses** (e.g., ArcFace) for learning robust features:

- **Softmax Loss** (VGGFace): Minimizes classification error but does not enforce feature margin
  - Result: Features cluster loosely; any perturbation (occlusion) causes massive degradation
  
- **ArcFace Loss** (Modern): Enforces angular margin between class features
  - Result: Features are tightly clustered with large inter-class separation
  - Benefit: Robust to noise and perturbations like occlusion

## Comparison to Modern Models

| Model | Rank-1 Accuracy | vs VGGFace | Performance Gain |
|-------|-----------------|-----------|------------------|
| ArcFace Large (buffalo_l) | 94.81% | +55.17% |  2.4× better |
| ArcFace Small (buffalo_s) | 86.65% | +47.01% |  2.2× better |
| **Aligned_Pretrained_CBAM_Block_v2** | 90.16% | +50.52% |  2.3× better |
| **Aligned_Pretrained_Aux_v2** | 78.52% | +38.88% | ✓ 2.0× better |
| VGGFace | 39.64% | — (Baseline) | — |

### Technology Gap
- **Year Developed**: VGGFace (2015), ArcFace (2018)
- **Time Gap**: 3 years of research advancement
- **Performance Gap**: 55.17 percentage points improvement (94.81% vs 39.64%)
- **Loss Function Evolution**:
  1. Standard Softmax (early 2010s): VGGFace era
  2. Margin-Based Softmax (2016+): CosFace, SphereFace
  3. Angular Margin (2018+): ArcFace era

## Architecture Details
- **Input Preprocessing**: 
  - Face detection and alignment
  - 224×224 or similar size (varies by variant)
  - ImageNet normalization
- **Backbone**: VGG deep stack
  - Multiple convolutional blocks
  - Max pooling layers
  - Dense layers at end
- **Feature Output**: Unbounded feature vectors (no L2 normalization)
- **Distance Metric**: Euclidean or Cosine Similarity

## Why VGGFace Fails on Occlusion

1. **No Angular Margin**: Features are not optimized for discriminative separation
2. **Softmax Saturation**: Once classification is correct, loss stops improving feature quality
3. **Low Feature Margin**: Classes can be separated with minimal margin; occlusion easily erases this margin
4. **Occlusion Brittleness**: Model has never learned robustness to occlusions during training

## Historical Context
- **Era**: Deep Learning 1.0 (pre-2015)
- **Loss Function**: Standard cross-entropy softmax
- **Benchmark Status**: Largely superseded; kept only for historical comparison
- **Lessons Learned**: Demonstrates critical importance of:
  - Angular/cosine margin enforcement
  - Metric learning principles
  - Contrastive feature learning

## Additional Notes
- **Source**: VGG Group, Oxford University (Parkhi et al., 2015)
- **Type**: Legacy research model
- **Current Status**: Superseded by ArcFace and modern approaches
- **Use Case**: Historical comparison only; NOT recommended for production
- **Research Value**: Demonstrates advances in loss function design
- **Modern Alternative**: Use ArcFace Large or Small instead
- **Key Takeaway**: Modern margin-based losses are 2-3× superior to older softmax approaches
- **Development Status**: No longer actively maintained; preserved for benchmarking

## Implications for Custom Models
The poor performance of VGGFace on occluded faces motivates our research into:
- ✓ Attention mechanisms (CBAM) for robust feature extraction
- ✓ Auxiliary loss functions for occlusion awareness
- ✓ Domain-specific fine-tuning on occluded faces
- ✓ Modern loss functions (ArcFace) combined with custom architectures

Our **CBAM_Block_v2** achieves 90.16%, which is:
- **2.3× better** than VGGFace (90.16% vs 39.64%)
- Demonstrating that careful architecture and training design can significantly improve robustness
