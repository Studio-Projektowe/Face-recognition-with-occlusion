# ArcFace Small (buffalo_s)

## Overview
ArcFace Small is a lightweight variant of ArcFace developed by the InsightFace team. It is optimized for speed and efficiency while maintaining competitive accuracy, making it suitable for mobile and edge deployment. This model serves as our primary industry-standard lightweight baseline for comparison.

## Model Architecture
- **Backbone**: MobileFaceNet (or equivalent lightweight architecture)
- **Loss Function**: ArcFace Loss (margin-based softmax)
  - Margin (m): 0.5
  - Scale (s): 64.0
- **Feature Dimension**: 512
- **Output**: 512-dimensional embedding vector
- **Depth**: Lightweight backbone (~11-15M parameters)

## Training Dataset
- **Dataset**: MS1MV2 (Microsoft Celeb 1M v2)
- **Total Images**: ~5.8 million images
- **Total Identities**: ~85,000 distinct identities
- **Training Paradigm**: Massive-scale supervised learning
- **Pre-training Duration**: Extensive training on massive cluster infrastructure

## Model Properties
- **Model Type**: Lightweight / Edge-optimized
- **Optimization Target**: Balance between accuracy and inference speed
- **Pre-training Method**: Supervised learning with margin-based softmax
- **Framework**: PyTorch (available via InsightFace library)
- **Inference Speed**: ~1-3 ms per embedding (on GPU)
- **Model Footprint**: ~20-40 MB (significantly smaller than Large variant)

## Evaluation Results (Test Set with 20px Eye Occlusion)

### Test Configuration
- **Gallery**: Clean images from WebFace test set
- **Probe**: Synthetically occluded images (20px black bar at eye region)
- **Total Queries**: 24,227
- **Evaluation Method**: FAISS k-NN similarity search (Cosine Similarity)

### Performance Metrics

| Metric | Accuracy | Count (Correct/Total) | Performance Tier |
|--------|----------|----------------------|-----------------|
| **Rank-1 Accuracy** | **86.65%** | 20,997 / 24,227 |  Industry Standard |
| **Rank-3 Accuracy** | **90.68%** | 21,975 / 24,227 |  Industry Standard |

### Observations
- **Industry-Standard Lightweight**: Represents the state-of-the-art in efficient face recognition
- **Occlusion Sensitivity**: Shows moderate degradation under eye occlusion compared to Large variant
- **Training Data Scale**: Benefits from 5.8M image pre-training dataset
- **Efficiency-Accuracy Trade-off**: Excellent balance for real-world deployment
- **Lightweight Champion**: Best-in-class accuracy for mobile/edge deployment scenarios

## Comparison to Our Custom Models

| Model | Rank-1 Accuracy | Delta (vs ArcFace Small) | Performance |
|-------|-----------------|--------------------------|-------------|
| **Aligned_Pretrained_CBAM_Block_v2** | 90.16% | **+3.51%** |  Surpasses |
| **Aligned_Pretrained_CBAM_Block_v1** | 87.58% | +0.93% | ✓ Competitive |
| **Aligned_Pretrained_Aux_v2** | 78.52% | -8.13% | — Below |
| ArcFace Small (buffalo_s) | 86.65% | — (Baseline) | — Baseline |

### Key Achievement
Our **CBAM_Block_v2** model achieves **90.16% accuracy**, effectively **surpassing ArcFace Small by +3.51%** while using:
- ✓ Only 500K WebFace images (vs 5.8M MS1MV2 images)
- ✓ Custom occlusion-specific fine-tuning
- ✓ Distributed CBAM attention mechanisms
- ✓ Domain-adapted training

This demonstrates that **domain-specific optimization can outperform generic SOTA models** on specialized tasks.

## Architecture Details
- **Lightweight Backbone**:
  - Depthwise separable convolutions
  - Low parameter count (~11-15M)
  - Minimal memory footprint
- **Input Preprocessing**: 
  - Face alignment via 5-point landmarks
  - 112×112 RGB image normalization
  - Standard ImageNet statistics
- **Feature Output**: 512-dimensional L2-normalized embeddings
- **Distance Metric**: Cosine Similarity

## Deployment Characteristics
- **Model Size**: ~20-40 MB
- **Computational Requirements**: Minimal (low-end GPU or CPU inference possible)
- **Latency**: Very fast (1-3 ms per image)
- **Memory Footprint**: Low (~100-200 MB including overhead)
- **Accuracy vs Speed Trade-off**: Optimized for speed without sacrificing too much accuracy
- **Ideal For**: Mobile apps, edge devices, real-time streaming, embedded systems

## Performance Degradation Under Occlusion
- **Clean Faces (No Occlusion)**: ~99%+ (typical benchmark)
- **With 20px Eye Occlusion**: 86.65% (this evaluation)
- **Degradation**: ~12-13% accuracy drop (expected for eye occlusion)

## Comparison with Larger Variant

| Aspect | ArcFace Large | ArcFace Small | Difference |
|--------|---------------|---------------|-----------|
| Backbone | IResNet50 | MobileFaceNet | - |
| Accuracy (Rank-1) | 94.81% | 86.65% | -8.16% |
| Model Size | ~200+ MB | ~20-40 MB | 5-10x smaller |
| Inference Time | ~5-10 ms | ~1-3 ms | 3-5x faster |
| Training Data | 5.8M (MS1MV2) | 5.8M (MS1MV2) | Same |
| Use Case | Production/Accuracy | Mobile/Edge | - |

## Additional Notes
- **Source**: InsightFace Project (InsightFace Community)
- **Availability**: Pre-trained weights available via InsightFace library
- **License**: Model weights available under InsightFace license
- **Reference Paper**: ArcFace: Additive Angular Margin Loss for Deep Face Recognition (Deng et al., 2019)
- **Industry Adoption**: Widely used in mobile face recognition applications
- **Use Case**: Best choice when balancing accuracy with speed and resource constraints
- **Benchmark Status**: Industry-standard baseline for efficient face recognition
- **Mobile-Friendly**: Optimized for deployment on smartphones and IoT devices
