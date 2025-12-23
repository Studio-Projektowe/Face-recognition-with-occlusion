# Kaggle to GCS Transfer Tool

This directory contains a Dockerized utility designed to automate the transfer of datasets directly from Kaggle to Google Cloud Storage (GCS). It functions as a high-throughput bridge, downloading the dataset to a temporary container environment and immediately syncing it to a specified GCS bucket.

## Overview

The tool performs the following operations sequentially:

* Configures Kaggle authentication using environment variables.
* Downloads and unzips the specified dataset into a temporary directory.
* Uploads the extracted data to Google Cloud Storage using gsutil with multi-threading enabled (-m).
* Cleans up local temporary files and credentials.

## Prerequisites

* Docker installed on the host machine.
* Kaggle API Credentials: You need the content of your kaggle.json file.
* Google Cloud Credentials: The container requires authentication to write to the destination bucket.

## Configuration

The script behavior is controlled entirely via environment variables passed to the Docker container.

Variable | Description | Required | Example |
---------|-------------|----------|---------|
KAGGLE_DATASET | The dataset identifier on Kaggle (username/dataset-slug). | Yes | yakhyokhuja/webface-112x112 |
BUCKET_NAME | The name of the destination Google Cloud Storage bucket (without gs://). | Yes | my-data-bucket |
KAGGLE_JSON_CONTENT | The raw string content of your kaggle.json file. | Yes | {"username":"...","key":"..."} |

## Usage

1. Build the Docker Image

Build the image from the provided Dockerfile.

`docker build -t kaggle-gcs-transfer .`

2. Run the Transfer

To run the container, you must provide the environment variables and ensure GCP authentication.

### Option A: Running with local GCP credentials (mounted)

If you have gcloud configured locally, you can mount your configuration directory.

```
docker run --rm \
  -v ~/.config/gcloud:/root/.config/gcloud \
  -e KAGGLE_DATASET="yakhyokhuja/webface-112x112" \
  -e BUCKET_NAME="my-target-bucket" \
  -e KAGGLE_JSON_CONTENT='{"username":"your_user","key":"your_api_key"}' \
  kaggle-gcs-transfer
```

### Option B: Running with a Service Account Key

If using a service account JSON key:
```
docker run --rm \
  -v /path/to/service-account-key.json:/gcp/key.json \
  -e GOOGLE_APPLICATION_CREDENTIALS="/gcp/key.json" \
  -e KAGGLE_DATASET="yakhyokhuja/webface-112x112" \
  -e BUCKET_NAME="my-target-bucket" \
  -e KAGGLE_JSON_CONTENT='{"username":"your_user","key":"your_api_key"}' \
  kaggle-gcs-transfer
```

## Implementation Details

### Disk Space Warning

> [!IMPORTANT]
> 
> Ephemeral Storage Requirements
> 
> The script downloads and unzips the entire dataset locally inside the container before uploading. Ensure that the Docker daemon has enough disk space allocated to handle the uncompressed size of the dataset.
> 
> For example, if the dataset is 50GB compressed and 100GB uncompressed, the container needs at least ~150GB of available space during the process.

### Script Workflow (script.sh)

The entrypoint script executes as follows:

* Auth Setup: Writes KAGGLE_JSON_CONTENT to ~/.kaggle/kaggle.json and sets permissions.
* Download: Uses kaggle datasets download --unzip to fetch data into temp_data/.
* Upload: Executes gsutil -m cp -r to upload contents to gs://${BUCKET_NAME}/.
* Cleanup: Removes temp_data and credentials to prevent data leakage in cached layers (though the container should be run with --rm).