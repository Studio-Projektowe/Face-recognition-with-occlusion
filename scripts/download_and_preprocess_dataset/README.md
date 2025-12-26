# WebFace Dataset Preprocessing Pipeline

This repository contains a complete pipeline for downloading, preparing, restructuring, and processing the WebFace dataset (112x112). It utilizes RetinaFace for facial landmark detection and prepares the data for training deep learning models.

## Table of Contents

* Project Structure
* Prerequisites
* Configuration
* Pipeline Steps
* Docker Usage
* Known Issues & Notes

## Project Structure

The pipeline is divided into modular scripts executed via main.py:

| Script | Description |
| -------| ----------- |
| s_01_download.py | Downloads and unzips the dataset from Kaggle. |
| s_02_prepare.py | Splits the dataset into train, val, and test directories based on identity. |
| s_02b_restructure.py | Restructures file paths (creates subdirectories for each image). |
| s_03_process.py | Runs RetinaFace detection to generate metadata JSONs (bbox, landmarks). |
| s_04_upload.py | (Optional) Uploads the processed dataset to Google Cloud Storage. |
| main.py | Orchestrates the entire flow sequentially. |

## Prerequisites

* Docker (recommended for reproducibility)
* Kaggle Account & API Key (kaggle.json)
* Google Cloud SDK (only if using the upload step)

## Configuration

Global settings are defined in config.py. Key variables include:

* KAGGLE_DATASET: The ID of the dataset on Kaggle.
* BUCKET_NAME: GCS bucket name for upload.
* SPLIT_RATIOS: Dictionary defining train/val/test split (e.g., {"train": 0.8, ...}).
* DEVICE: cuda or cpu (auto-detected).
* NUM_WORKERS: Number of threads/processes for parallel processing.

## Pipeline Steps

You can run the full pipeline using python main.py, or execute individual steps as needed.

### Step 1: Download

Fetches the dataset using the Kaggle API and extracts it to the local directory.

### Step 2: Preparation & Split

Divides the dataset into subsets based on SPLIT_RATIOS.

> [!NOTE]
> See Known Issues regarding the split logic.

### Step 3: Processing (RetinaFace)

Iterates through images to detect faces. Generates a .json file for each image containing:

* Bounding Box (bbox)
* 5 Facial Landmarks
* Confidence Score

### Step 4: Upload (Optional)

Uploads the final dataset to the specified Google Cloud Storage bucket.

## Docker Usage

* Prepare Credentials: Ensure you have your kaggle.json file ready.
* Build the Image:

`docker build -t webface-pipeline .`

* Run the Container: You need to mount a volume if you want the data to persist on your host machine.

```
docker run -it --rm \
  -v $(pwd)/data:/app/../casia_dataset \
  -v $(pwd)/kaggle.json:/root/.kaggle/kaggle.json \
  webface-pipeline
```

And well done!

## Known Issues & Notes

* Train/Val Split Logic

Currently, s_02_prepare.py splits identities into separate train and val folders.

> [!WARNING]
> If your downstream training pipeline merges these folders back together, or if you intend to perform random splitting during data loading, you may want to disable this step or adjust `SPLIT_RATIOS` to assign 100% to `train`.

* RetinaFace Threshold

If detection misses faces, you can use the provided auxiliary script `extra_detect_lower_treshold.py` which attempts redetection on missing files with a lower confidence threshold (`0.5`).

* Upload Step

The `s_04_upload.py` script requires authenticated `gsutil`. Inside Docker, you may need to run `gcloud auth login` manually or mount Google credentials if not using a service account.