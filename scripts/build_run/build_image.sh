#!/bin/bash
set -e

export PROJECT_ID="face-recognition-476110"
export REGION="northamerica-northeast1"
export REPO_NAME="casia-downloader-repo"
export IMAGE_NAME="create-validation-dir"

export IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"

echo "Budowanie i wysyłanie obrazu: ${IMAGE_TAG}"

gcloud builds submit ../split_val/ --tag=${IMAGE_TAG} --project=${PROJECT_ID}

echo "Obraz zbudowany. Utwórz zadanie a następnie uruchom job_run.sh"
