# Face Recognition under Occlusion

## Overview
This project focuses on face recognition under occlusion (masks, glasses, shadows). We compared baseline models and our own modifications, and in this repository we kept **only the best results and models**. There were many trials and errors (this was our first such project), so some approaches were discarded.

## What’s inside and why
- **Model comparison**: we evaluate how classic approaches (`ArcFace`, `VGGFace`) handle occlusion and whether custom modifications improve results.
- **Custom architectures**: mainly `iResNet50` with additional mechanisms (`CBAM`, auxiliary heads, Transformer) to improve robustness to occlusion.
- **Only best variants**: the `models/` folder contains only the models and results that were most stable and promising.

## Key results (short)
- Best model: **`Aligned_Pretrained_CBAM_Block`**
- Occlusion benchmark: **Rank‑1 = `90.16%`, Rank‑3 = `91.88%`**

See details in:
- [models/Aligned_Pretrained/README.md](models/Aligned_Pretrained/README.md)
- [models/Aligned_Pretrained_CBAM_Block/README.md](models/Aligned_Pretrained_CBAM_Block/README.md)
- [models/Aligned_Pretrained_Transformers/README.md](models/Aligned_Pretrained_Transformers/README.md)

## How training data is created
Data preparation lives in `scripts/`:
- `scripts/download_and_preprocess_dataset/` — full WebFace download + preprocessing pipeline (RetinaFace + landmarks) per [scripts/download_and_preprocess_dataset/README.md](scripts/download_and_preprocess_dataset/README.md).
- `scripts/download_dataset_image/` — Kaggle → GCS transfer (Docker) per [scripts/download_dataset_image/README.md](scripts/download_dataset_image/README.md).
- `scripts/build_image_run_job/` — image build + Cloud Run job execution per [scripts/build_image_run_job/README.md](scripts/build_image_run_job/README.md).

## Why the results may look like this
- **Different occlusion strategies**: training uses colored bars, evaluation uses black bars (controlled test condition).
- **Mixed initialization**: some weights are loaded from `w600k_r50_from_onnx.pth`, while some layers are randomly initialized.
- **No fixed seeds**: lack of fixed random seeds increases result variance.
- **Architecture changes**: adding attention modules (e.g., `CBAM`) can hurt results if the backbone is frozen.

## Demo

Want to see how our best model performs in practice? Open the **[demo/](demo/)** folder and check **demo_local.ipynb**.

The notebook includes:
- Comparison of results on images **without occlusion** vs **with occlusion** (glasses, masks, shadows)
- Real-time visualization of face verification results
- Demonstration of our best model **CBAM_Block_v2**, which achieved **90.16% Rank-1** accuracy on occluded data
- Examples of both successful and failed identifications

**Note:** You can view the pre-computed outputs stored in the notebook cells without running them. You can also run the cells to reproduce the results.

### Run on Google Colab (No Setup Required!)

Want to try the demo without installing anything? Use our public Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Zld0rrVekVMX64UikFokj_OhhODpjsdL?usp=sharing)

Simply click the button above, make a copy to your Google Drive, and run the cells. You can instantly see how our best model performs on face verification with and without occlusion.

## Notes
This repository does not include all experiments — we kept only the ones we considered most useful for comparison and reporting.
