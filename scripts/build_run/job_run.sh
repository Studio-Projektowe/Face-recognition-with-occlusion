#!/bin/bash
set -e

export PROJECT_ID="face-recognition-476110"
export REGION="northamerica-northeast1"
export REPO_NAME="casia-downloader-repo"
export JOB_NAME="testing-split-dir"

echo "-----------------------------------"
echo "URUCHAMIANIE ZADANIA: ${JOB_NAME}"
echo "-----------------------------------"

gcloud run jobs execute ${JOB_NAME} --region=${REGION} --wait --project=${PROJECT_ID}

echo "Zadanie zakończone!"