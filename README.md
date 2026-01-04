# Face Recognition under Occlusion

## Quick Start - Interactive Demo

Want to see how our best model performs in practice? Open the **[`demo/`](demo/)** folder and check the **`demo_remote.ipynb`** notebook!

The notebook includes:
- Comparison of results on images **without occlusion** vs **with occlusion** (glasses, masks, shadows)
- Real-time visualization of face verification results
- Demonstration of our best model **CBAM_Block_v2**, which achieved **90.16% Rank-1** accuracy on occluded data
- Examples of both successful and failed identifications

** Note:** You can directly view the pre-computed results and outputs already stored in the notebook cells without running them - just open the `.ipynb` file to see the results! Of course, you're also welcome to run the cells yourself to reproduce the results.

### Run on Google Colab (No Setup Required!)

Want to try the demo without installing anything? Use our public Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Zld0rrVekVMX64UikFokj_OhhODpjsdL?usp=sharing)

Simply click the button above, make a copy to your Google Drive, and run the cells! You can instantly see how our best model performs on face verification with and without occlusion.

---

## 1. Project Overview

This project focuses on analyzing, implementing, and evaluating Deep Learning models for face recognition, with a specific emphasis on occlusion robustness.

The primary objectives are:

* Benchmark SOTA Models: Evaluate how current state-of-the-art models (ArcFace variants) handle occluded validation sets.

* Custom Architecture Design: Develop modified architectures based on IResNet50 utilizing CBAM (Convolutional Block Attention Module) and Auxiliary Loss functions to improve feature extraction in occluded scenarios.

* Retraining & Fine-tuning: Train models on the CASIA-WebFace dataset with specific augmentation and architectural changes.

### Analyzed Models

Baseline SOTA Models:

* ArcFace_Large (based on IResNet50, source: InsightFace)

* ArcFace_Small  (based on MobileFaceNet equivalent)

* VGGFace (Legacy comparison, 2018)

Our Custom Implementations:

* Aligned_Pretrained_Aux_v1 / v2 (Base IResNet with Auxiliary Loss)

* Aligned_Pretrained_CBAM_Block_v1 / v2 (Integration of Attention Modules)

* Aligned_Pretrained_CBAM_L1_v1 / v2 (Attention applied at Layer 1)

* Aligned_Pretrained_CBAM_L3_v0 / v1 / v2 (Attention applied at Layer 3)

## 2. Dataset Preparation

We utilize the CASIA-WebFace dataset, comprising approximately 500,000 images across 10,575 identities.

### Preprocessing Pipeline

All images undergo preprocessing using RetinaFace to detect 5-point facial landmarks (eyes, nose, mouth). This ensures consistent alignment before feeding data into the recognition network.

### Directory Structure

The dataset is organized into Train, Validation, and Test splits. Each image corresponds to a JSON file containing bounding box and landmark coordinates.

```
webface_112x112/
├── train/
│   ├── id_001/
│   │   ├── 001/
│   │   │   ├── 001.jpg
│   │   │   └── 001.json
│   │   └── 002/
│   │       ├── 002.jpg
│   │       └── 002.json
│   └── id_002/
│       └── ...
├── val/
│   └── id_010/
│       └── ...
└── test/
    └── id_020/
        └── ...
```

### Data Format:

* Images: .jpg (aligned/cropped to 112x112)

* Metadata: .json (contains landmarks and bbox)

* Naming Convention: img_name.jpg $\leftrightarrow$ img_name.json

## 3. Methodology & Baseline Setup

### Backbone Architecture

We selected IResNet50 as our baseline backbone due to its performance in the buffalo_l model (ArcFace Large) from the InsightFace project.

### Weight Initialization Strategy

To accelerate training, we attempted to transfer weights from the pre-trained buffalo_l ONNX model to our PyTorch implementation. Due to architectural differences between the ONNX export and the dynamic PyTorch graph, a Partial Weight Transfer strategy was employed.

### Initialization Protocol:

1. Matched Layers: Weights were successfully loaded for the majority of the ResNet blocks.

2. Unmatched Layers: 26 specific layers (primarily BatchNorm and PReLU instances inside specific blocks) could not be mapped 1:1. These layers were re-initialized using Kaiming Normal (He initialization).

<details>
<summary><strong>Click to expand: List of Re-initialized Layers</strong></summary>

The following layers were reset to random weights and trained from scratch:

Layer 1: layer1.2.bn3 (bias, running_mean, running_var, weight)

Layer 2: layer2.3.bn2 (running_var), layer2.3.bn3 (all params), layer2.3.prelu

Layer 3: layer3.12.bn2 (stats), layer3.12.bn3 (all params), layer3.12.prelu

Layer 3 (cont.): layer3.13.bn2 (all params), layer3.13.bn3 (all params), layer3.13.prelu

</details>

This hybrid initialization forces the model to re-learn batch statistics and activation thresholds during the first epoch, acting as a "warm-up" phase for the re-initialized blocks.

### 3.3. Evaluation Protocol (Identification Task)

To rigorously assess the performance of our models in a real-world identification scenario, we implemented a retrieval-based benchmark using FAISS (Facebook AI Similarity Search).

#### Test Set Construction:

* Gallery Set: Constructed from the first half of image sessions (folders) for each identity in the Test split. Embeddings from multiple images per identity are averaged to create a robust class prototype, then L2-normalized for cosine similarity matching.

* Probe Set: Constructed from the second half of image sessions for each identity. These images are synthetically occluded during evaluation to test robustness.

#### Evaluation Workflow:

1. Embedding Extraction: The model extracts 512-dimensional feature vectors for all gallery and probe images. Embeddings are L2-normalized to enable cosine similarity search.

2. Index Construction: Gallery embeddings are indexed using FAISS IndexFlatIP (Inner Product on L2-normalized vectors = Cosine Similarity).

3. Occlusion Simulation: A 20px height black bar is applied to each Probe image during evaluation. Bar placement is centered on eye region computed from 5-point landmarks (left_eye_y and right_eye_y center). If landmarks are unavailable, bar is placed at image_height/2 - 10px (fallback). Bar spans full image width and color is solid black (0,0,0).

4. Similarity Search: For each Probe image, we perform a k-Nearest Neighbors search (k=3) against the Gallery index using Cosine Similarity to retrieve top-3 candidates.

5. Metrics Computation:

* Rank-1 Accuracy: Percentage of queries where the highest-ranked match (1st result) corresponds to the correct identity.

* Rank-3 Accuracy: Percentage of queries where the correct identity appears anywhere in the top-3 results.

#### Output Details:

Results are logged to CSV with columns: query_identity, top1_identity, top1_similarity, top2_identity, top2_similarity, top3_identity, top3_similarity, found_in_top3_flag. Sample occluded images are saved periodically for visual inspection.

#### Important Note on Occlusion Consistency:

Training augmentation uses random-colored bars (RGB 0-255 random) at 70%-100% probability depending on model. Evaluation uses deterministic black bars (0,0,0) to create a standardized test condition. This ensures rigorous, reproducible assessment of occlusion robustness. Bar height is consistently 20px across training and evaluation.

## 4. SOTA Baselines Evaluation

Before developing our custom architectures, we evaluated existing State-of-the-Art (SOTA) solutions to establish a performance "Upper Bound". This benchmark helps contextualize our results against industry-standard models trained on massive datasets.

### 4.1. Evaluated Models

We tested the following pre-trained models using the InsightFace library:

ArcFace Large (buffalo_l): A heavy-weight model (ResNet50-based) optimized for high accuracy.

ArcFace Small (buffalo_s): A lightweight model optimized for speed.

VGGFace: A legacy model for historical comparison.

### 4.2. Benchmark Results (ArcFace Large)

We subjected the SOTA models to the exact same occlusion protocol as our custom models (Probe set with 20px eye occlusion).

Baseline Model | Rank-1 Accuracy | Rank-3 Accuracy |
---------------| --------------- | ----------------|
ArcFace Large (buffalo_l) | 94.81% | 95.51% |
ArcFace Small (buffalo_s) | 86.65% | 90.68% |
VGGFace (ResNet50) | 39.64% | 57.01% |

Observations:

ArcFace Large sets a very high "Upper Bound" (94.81%), likely due to its massive pre-training dataset (MS1MV2 with ~5.8M images) and deeper architecture.

ArcFace Small achieves 86.65%. Notably, our best custom model (Model I) achieves 90.16%, effectively surpassing this lightweight industry standard by a significant margin (+3.51%). This proves that domain-specific fine-tuning with attention mechanisms can outperform generic SOTA models on occlusion tasks.

VGGFace (2015/2018) shows significant degradation under occlusion (39.64%), highlighting the advancements made by modern margin-based loss functions (ArcFace) compared to older softmax-based approaches.

## 5. Proposed Architectures (Research Track)

### 5.1. Model A: Aligned_Pretrained_Aux_v1

This model represents our first approach to robust feature extraction using Multi-Task Learning (MTL).

Architecture Design:
We extended the standard IResNet50 backbone with a lightweight auxiliary head.

Primary Branch: Standard ArcFace projection (512-dim embedding) for identity classification.

Auxiliary Branch: A Multi-Layer Perceptron (MLP) composed of Linear(512, 256) -> ReLU -> Linear(256, 49) -> Sigmoid.

Objective: The aux head predicts a $7 \times 7$ grid mask representing the spatial location of the occlusion. This forces the shared feature extractor to be "aware" of which parts of the face are missing.

Training Configuration:

* Loss Function: $L_{total} = L_{ArcFace} + \lambda L_{MSE}$, where $\lambda=0.01$.

* Augmentation: Online synthetic occlusion with $P=0.7$, utilizing random colored bars (height: 20px).

* Freezing Strategy: To preserve the transferred low-level features, we froze layer1 and layer2. We fine-tuned the deeper semantic layers (layer3, layer4) along with all Batch Normalization and PReLU parameters.

* Optimization: SGD (momentum=0.9) with ReduceLROnPlateau. LR_Backbone=5e-4, LR_Head=0.01.

Results (25 Epochs):
The model was trained for 25 epochs. The learning curve shows consistent convergence without signs of overfitting, despite the high occlusion probability.

* Initial Loss (Val): ~20.22

* Final Loss (Val): ~10.72

Evaluation Protocol & Test Results:
We performed an identification test on the held-out Test split using FAISS for similarity search ($k$-NN).

Protocol: The test set was split into a Gallery (clean images) and a Probe set (synthetically occluded images).

Occlusion Type: A 20px height black bar applied to the eye region (based on landmarks).

Total Queries: 24,227

Metric | Accuracy | Count (Correct/Total) |
-------|----------|-----------------------|
Rank-1 Accuracy | 77.81% | 18,851 / 24,227 |
Rank-3 Accuracy | 84.05% | 20,362 / 24,227 |

Observation: The model demonstrates strong baseline capabilities in handling severe eye occlusion, validating the effectiveness of the auxiliary loss strategy.

### 5.2. Model B: Aligned_Pretrained_Aux_v2 (Sequential Fine-Tuning)

Building upon the features learned in v1, this model utilizes a Sequential Transfer Learning strategy to refine the feature space.

Architecture Design:
Identical to v1, but initialized with the weights from the best Aligned_Pretrained_Aux_v1 checkpoint.

Training Configuration:

Unfreezing: Unlike v1, we unfroze the entire backbone (including Layer 1 and Layer 2). This allows the lower-level features (edge/texture detectors) to adapt specifically to the occlusion patterns and the auxiliary task.

Hyperparameter Adjustment:

Low Learning Rate: LR_Backbone was reduced to 1e-5 (vs 5e-4 in v1) to prevent "catastrophic forgetting" of the stable features learned in the previous stage.

Increased Aux Weight: AUX_LOSS_WEIGHT was increased to 0.1 (vs 0.01 in v1). Since the backbone is already stable, we placed stronger emphasis on the mask prediction task to enforce spatial awareness.

Augmentation: Online synthetic occlusion with P=0.7, utilizing random colored bars (height: 20px) - same as v1.

Optimization: Trained for 15 epochs with Early Stopping (patience=3).

Results (15 Epochs):
The fine-tuning process resulted in a significant reduction in validation loss compared to v1, indicating improved generalization.

Initial Loss (Val): ~12.35 (Starting point from v1)

Final Loss (Val): ~8.58

Optimizer Configuration: SGD with momentum=0.9, weight_decay=5e-4. Scheduler: ReduceLROnPlateau (factor=0.1, patience=2).

Test Results (Occluded Probe):
Using the identical evaluation protocol (20px Eye Occlusion):

Metric | Accuracy | Count (Correct/Total) | Delta (vs v1) |
-------|----------|-----------------------|---------------|
Rank-1 Accuracy | 78.52% | 19,023 / 24,227 | + 0.71% |
Rank-3 Accuracy | 84.60% | 20,497 / 24,227 | + 0.55% |

Observation: Unfreezing the backbone allowed the model to fine-tune its low-level feature extractors (Gabor-like filters in early layers) to be less sensitive to the sharp edges of the occlusion masks. The improvement confirms that full-network fine-tuning with a low learning rate is superior to partial freezing.

### 5.3. Model C: Aligned_Pretrained_CBAM_L1_v1 (Modular Adaptation)

In this iteration, we hypothesized that adding a Channel and Spatial Attention Mechanism (CBAM) at the very beginning of the network (Front-End) could "clean" the occluded input signal before it reaches the backbone.

Architecture Design:

Base: Aligned_Pretrained_Aux_v2.

Modification: Inserted a CBAM block immediately after the initial convolution, before layer1.

Training Configuration:

Strategy: "Frozen Backbone Adaptation". We froze all layers of the pre-trained backbone and trained only the newly inserted CBAM module and Batch Normalization layers.

Rationale: We aimed to see if the attention module alone could adapt to occlusion without altering the deep feature extractors.

Optimization: High LR (0.01) for the CBAM module to encourage rapid adaptation. Epochs: 30. Augmentation: Online synthetic occlusion (height: 20px, 100% probability).

Test Results (Occluded Probe):
Evaluation revealed a significant performance degradation compared to the baseline v2 model.

Metric | Accuracy | Delta (vs v2) |
-------|----------|---------------|
Rank-1 Accuracy | 38.82% | -39.70% |
Rank-3 Accuracy | 51.28% | -33.32% |

Observation: The drastic drop in accuracy indicates that inserting a randomly initialized module at the front of a frozen, pre-trained network disrupts the input statistics (feature distribution) that the backbone expects. While the model improved from ~1.5% to ~38% during training, it could not recover the baseline performance without end-to-end fine-tuning. This confirms that attention modules cannot be simply "plugged in" to frozen networks; they require co-adaptation.

### 5.4. Model D: Aligned_Pretrained_CBAM_L1_v2 (Extended Training)

To verify if the poor performance of Model C was due to underfitting (insufficient convergence of the attention module), we extended the training process.

Architecture Design:
Identical to v1 (CBAM at Front-End + Frozen Backbone).

Training Configuration:

Initialization: Resumed from the best checkpoint of Aligned_Pretrained_CBAM_L1_v1 (starting accuracy ~38%).

Strategy: Continued training for 30 epochs with the backbone remaining frozen. Augmentation: Online synthetic occlusion (height: 20px, 100% probability). Optimizer: SGD with momentum=0.9, weight_decay=5e-4.

Test Results (Occluded Probe):
While performance improved slightly, it reached a saturation point far below the baseline.

Metric | Accuracy | Delta (vs v1) | Delta (vs Baseline v2) |
-------|----------|---------------|------------------------|
Rank-1 Accuracy | 43.76% | +4.94% |-34.76% |
Rank-3 Accuracy | 55.72% | +4.44% | -28.88% |

Observation: The plateau in accuracy confirms that the issue is structural, not temporal. A frozen backbone trained on "standard" faces cannot effectively process the modified feature map produced by a front-end attention module. The CBAM module essentially introduces a "covariate shift" that the subsequent frozen layers treat as noise.

### 5.5. Model E: Aligned_Pretrained_CBAM_L3_v0 (Deep Feature Attention)

Based on the failures of the Front-End attention models (C and D), we hypothesized that the attention mechanism might be more effective if applied to high-level semantic features rather than low-level input features.

Architecture Design:

Base: Initialized with weights from Aligned_Pretrained_Aux_v2.

Modification: A CBAM block was inserted deep in the network, specifically after layer3 and before layer4.

Training Configuration:

Selective Freezing: We employed a "Partial Unfreezing" strategy.

Frozen: layer1, layer2 (Low-level feature extractors).

Trainable: layer3, CBAM, layer4, and the classifier head.

Rationale: By keeping the early layers frozen, we preserve the basic edge/texture detection capabilities. By unfreezing the deeper layers, we allow the network to adapt its semantic understanding to utilize the new attention maps produced by the inserted CBAM block.

Augmentation: Online synthetic occlusion with P=0.7, height: 20px. Epochs: 25. Optimizer: SGD with momentum=0.9, weight_decay=5e-4. Scheduler: ReduceLROnPlateau (factor=0.1, patience=2).

Test Results (Occluded Probe):
Moving the attention module deeper yielded significantly better results than the Front-End approach, though still below the non-attention baseline.

Metric | Accuracy | Delta (vs L1_v2)| Delta (vs Baseline v2)|
-------|----------|---------------|------------------------|
Rank-1 Accuracy | 57.47% | +13.71% |-21.05% |
Rank-3 Accuracy | 66.58% | +10.86% | -18.02% |

Observation: The substantial improvement over the L1 models (+13.71%) suggests that semantic features are more robust to structural modification. The network can better tolerate (and potentially benefit from) attention re-weighting when it occurs at a stage where it processes abstract concepts (like "nose shape") rather than raw pixels. However, the drop compared to the baseline implies that the inserted layer still creates a bottleneck or misalignment that the partial fine-tuning has not fully resolved.

### 5.6. Model F: Aligned_Pretrained_CBAM_L3_v1 (The Breakthrough)

This experiment combined the Deep Attention architecture (Model E) with the Full Fine-Tuning strategy (Model B).

Architecture Design:

Base: Aligned_Pretrained_CBAM_L3_v0 (The "Mid-Level Attention" model).

Placement: CBAM between layer3 and layer4.

Training Configuration:

Phase 2 - Full Unfreeze: All layers of the backbone (L1-L4) and the CBAM module were unfrozen.

Initialization: Loaded weights from the best checkpoint of L3_v0 (starting accuracy ~65%).

Hypothesis: By allowing the entire network (including early layers) to adjust to the presence of the deep attention module, the network can align its low-level feature extraction to maximize the benefit of the semantic attention.

Optimization: Epochs: 30. LR_CBAM=0.01. Optimizer: SGD with momentum=0.9, weight_decay=5e-4. Scheduler: StepLR (step_size=3, gamma=0.1). Augmentation: Online synthetic occlusion (height: 20px, 100% probability).

Test Results (Occluded Probe):
This configuration yielded the best results of the entire project, outperforming the non-attention baseline.

Metric | Accuracy | Delta (vs Baseline v2)|
-------|----------|---------------|
Rank-1 Accuracy | 81.77% | +3.25% |
Rank-3 Accuracy |86.24% | +1.64% |

Conclusion: The attention mechanism (CBAM) is highly effective for occlusion robustness, provided two conditions are met:

Deep Placement: It must be placed at a high semantic level (after Layer 3).

Co-Adaptation: The entire network must be fine-tuned end-to-end to accommodate the modification. Merely inserting the module into a frozen or partially frozen network is insufficient.

### 5.7. Model G: Aligned_Pretrained_CBAM_L3_v2 (Differential Learning Rates)

This experiment aimed to squeeze maximum performance from the architecture by applying Differential Learning Rates (DLR) during the full fine-tuning phase.

Architecture Design:

Base: Aligned_Pretrained_CBAM_L3_v1.

Modification: None (Architecture remains IResNet50 + CBAM @ L3).

Training Configuration:

Initialization: Resumed from the best checkpoint of the previous model (v1).

Strategy: Full Network Unfreeze with DLR.

Backbone (Layers 1-3): LR = 1e-3 (Conservative updates to preserve extracted features).

High-Level Features (CBAM + Layer 4 + Head): LR = 1e-2 (Aggressive updates to adapt the attention mechanism and classifier).

Rationale: This allows the deep semantic layers to adapt quickly to the attention modulation while preventing catastrophic forgetting in the earlier layers.

Epochs: 25. Batch Size: 48. Optimizer: SGD with momentum=0.9, weight_decay=5e-4. Scheduler: StepLR (step_size=8, gamma=0.1). Augmentation: Online synthetic occlusion (height: 20px, 100% probability).

Test Results (Occluded Probe):

Metric | Accuracy | Delta (vs Baseline v2)| Delta (vs L3 v1)|
-------|----------|---------------|------------------------|
Rank-1 Accuracy | 86.55% | +8.03% | +4.78% |
Rank-3 Accuracy | 89.55% | +4.95% | +3.31% |

Conclusion: The combination of Deep Attention Placement and Differential Fine-Tuning proved to be an effective strategy for L3 bottleneck placement. The model not only recovered the performance lost by architectural modification but significantly surpassed the original non-attention baseline, demonstrating that the attention mechanism effectively learns to "look around" the occlusion.

### 5.8. Model H: Aligned_Pretrained_CBAM_Block_v1 (Integrated Attention)

In our final experiment, we tested whether "Distributed Attention" is superior to "Bottleneck Attention". Instead of a single CBAM module, we integrated the attention mechanism directly into every Residual Block of the IResNet architecture.

Architecture Design:

Mechanism: Inside each CBAMBasicBlock, the attention module refines the feature maps of the convolutional path before they are added to the residual identity connection.

Scale: This provides granular, hierarchical attention at every depth of the network, allowing the model to suppress noise (occlusion) locally at early stages and semantically at deeper stages.

Training Configuration:

Initialization: Weights transferred from Aligned_Pretrained_Aux_v2 (matched layers loaded, new intra-block CBAM layers initialized randomly).

Phased Training: 1.  Warm-up: Frozen backbone weights, training only the new CBAM modules (Epoch 1).
2.  Fine-tuning: Full network unfreeze to co-adapt convolutions with attention maps (Epochs 2-30).

Augmentation: Online synthetic occlusion with P=0.5, height: 20px. Optimizer: SGD with momentum=0.9, weight_decay=5e-4. Scheduler: StepLR (step_size=10, gamma=0.1). Initial LR: 0.1 (reinit on epoch 1 for full unfreeze).

Test Results (Occluded Probe):
This granular, distributed attention strategy yielded the absolute highest performance in our research.

Metric | Accuracy | Delta (vs L3_v2)| Delta (vs Baseline v2)|
-------|----------|---------------|------------------------|
Rank-1 Accuracy | 87.58% | +1.03% | +9.06% |
Rank-3 Accuracy | 90.10% | +0.55% | +5.50% |

Conclusion: Distributed attention outperforms single-point insertion. By allowing the network to "clean" the signal at every step of the transformation, the model builds a representation that is intrinsically robust to occlusion, rather than trying to fix the features at a single bottleneck.

### 5.9. Model I: Aligned_Pretrained_CBAM_Block_v2 (Aggressive Fine-Tuning)

Building on the architectural success of Model H, we explored the impact of hyperparameter optimization, specifically focusing on a High Learning Rate (Aggressive) Fine-Tuning strategy.

Architecture Design:
Identical to Model H (IResNet with Distributed CBAM in every block).

Training Configuration:

Initialization: Loaded weights from the baseline Aligned_Pretrained_Aux_v2.

Strategy: Rapid Adaptation.

Warm-up: Only 1 epoch with frozen backbone to initialize CBAM weights.

Full Unfreeze: Immediately unfroze the entire network at Epoch 2.

Aggressive LR: LR_START = 0.1 (compared to 0.01 or 0.001 in previous experiments).

Rationale: We hypothesized that the integration of attention modules into every block fundamentally changes the optimal weights for the convolutional layers. A high learning rate allows the network to escape local minima associated with the non-attention architecture and find a new, superior global minimum.

Augmentation: Online synthetic occlusion with P=0.5, height: 20px. Epochs: 30. Optimizer: SGD with momentum=0.9, weight_decay=5e-4. Scheduler: StepLR (step_size=10, gamma=0.1).

Test Results (Occluded Probe):
This strategy produced the Global State-of-the-Art (SOTA) for our project, breaking the 90% accuracy barrier.

Metric | Accuracy | Delta (vs  v1) | Delta (vs Baseline v2)|
-------|----------|---------------|------------------------|
Rank-1 Accuracy | 90.16% | +2.58% | +11.64% |
Rank-3 Accuracy | 91.88% | +1.78% | +7.28% |

Final Conclusion: The project demonstrates that standard face recognition models can be made significantly more robust to occlusion (+11.64% accuracy) by integrating distributed attention mechanisms (CBAM) and training them with an aggressive fine-tuning schedule that encourages deep co-adaptation between convolutional features and attention maps.

## 6. Comparative Summary & Conclusion

The table below summarizes the progression of our research, highlighting the impact of architectural choices and training strategies on occlusion robustness.

Model ID | Architecture | Strategy | Rank-1 Acc | Delta (vs Baseline) | 
---------|--------------|----------|------------|---------------------|
Aligned_Pretrained_Aux_v2 | Aux (Baseline) | Fine-Tuning | 78.52% | - |
Aligned_Pretrained_CBAM_L1_v2 | CBAM @ L1 | Front-End Adapt | 43.76% | -34.76% |
Aligned_Pretrained_CBAM_L3_v1 | CBAM @ L3 | Deep Adapt | 81.77% | +3.25% |
Aligned_Pretrained_CBAM_L3_v2 | CBAM @ L3 | Differential LR | 86.55% | +8.03% | 
Aligned_Pretrained_CBAM_Block_v1 | Dist. CBAM | Integrated | 87.58% | +9.06% |
Aligned_Pretrained_CBAM_Block_v2 | Dist. CBAM | High LR Tune | 90.16% | +11.64% | 

Final Conclusion 

Our proposed architecture (Aligned_Pretrained_CBAM_Block_v2) demonstrates that standard face recognition models can be made significantly more robust to occlusion by:

Integrating Distributed Attention: Modifying the IResNet blocks to include CBAM.

Aggressive Co-Adaptation: Using a high learning rate fine-tuning schedule to align convolutional features with attention maps.

Key Achievement: We successfully surpassed the industry-standard lightweight model (ArcFace Small: 86.65%) by +3.51%, narrowing the gap to the heavy-weight teacher model (ArcFace Large: 94.81%) using a significantly smaller training dataset.