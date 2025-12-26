# GCP Cloud Run Job Deployment & Execution

This directory contains shell scripts for building the Docker image using Google Cloud Build and executing the associated Google Cloud Run Job. These scripts facilitate the deployment pipeline for the dataset processing task.

## Overview

The scripts automate two main phases of the deployment lifecycle:

* Build & Push: Submitting the source code to Cloud Build to create a Docker image and push it to the Artifact Registry.
* Execution: Triggering an existing Cloud Run Job to perform the processing task.

## Prerequisites

* Google Cloud CLI (gcloud) installed and authenticated.
* Permissions: The active user or service account must have permissions to:
    * Submit builds to Cloud Build.
    * Write to the Artifact Registry.
    * Execute Cloud Run Jobs.
* Directory Structure: The `build_image.sh` script assumes the source code for the image is located in a sibling directory named `../download_dataset_image/`.

## Configuration

Both scripts utilize hardcoded environment variables that define the project infrastructure.

> [!IMPORTANT]
> Configuration Required
> 
> Before running these scripts, you must open `build_image.sh` and `job_run.sh` and update the following variables to match your GCP environment:
> * PROJECT_ID: Your Google Cloud Project ID (e.g., face-recognition-476110).
> * REGION: The GCP region (e.g., northamerica-northeast1).
> * REPO_NAME: The name of your Artifact Registry repository.
> * JOB_NAME: The name of the Cloud Run Job (for execution).

## Scripts Description

1. Build Image (`build_image.sh`)This script submits a build request to Google Cloud Build.
* Source: It targets the `../download_dataset_image/ directory`.
* Destination: It builds the image and pushes it to the Google Artifact Registry using the tag format:
```
${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latestUsage:./build_image.sh
```

2. Job Execution (`job_run.sh`)

This script triggers the execution of a pre-configured Cloud Run Job.

* Command: Uses gcloud run jobs execute.
* Behavior: The script uses the --wait flag, meaning it will pause execution and stream logs/status until the Cloud Run Job completes or fails.

Usage:

`./job_run.sh`

## Workflow

To deploy an update and run the process:

* Ensure local changes in `../download_dataset_image/` are saved.
* Run `./build_image.sh` to update the image in the registry.
* Ensure the Cloud Run Job infrastructure is created (e.g., via Terraform) and is pointing to the latest image tag.
* Run `./job_run.sh` to start the processing task.