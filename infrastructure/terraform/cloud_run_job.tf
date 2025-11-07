resource "google_artifact_registry_repository" "casia_artifact_repo" {
  project       = var.PROJECT_ID
  location      = var.REGION
  repository_id = "casia-downloader-repo"
  format        = "DOCKER"
  depends_on = [google_project_service.apis]
}

locals {
  image_name = "kaggle-downloader"
  image_url  = "${var.REGION}-docker.pkg.dev/${var.PROJECT_ID}/${google_artifact_registry_repository.casia_artifact_repo.repository_id}/${local.image_name}:latest"
}

resource "google_cloud_run_v2_job" "casia_download" {
  client = "cloud-console"
  project  = var.PROJECT_ID
  name     = "kaggle-downloader-casia"
  location = var.REGION

  template {
    template {
      timeout = "86400s" // 24h
      max_retries = 0

      service_account = data.google_compute_default_service_account.default.email

      containers {
        name = "kaggle-downloader-1"
        image = local.image_url 

        resources {
          limits = {
            "cpu"    = "4"
            "memory" = "16Gi"
          }
        }

        env {
          name  = "PROJECT_ID"
          value = var.PROJECT_ID
        }
        env {
          name  = "BUCKET_NAME"
          value = google_storage_bucket.casia_dataset_original_bucket.name
        }
        env {
          name  = "KAGGLE_DATASET"
          value = "yakhyokhuja/webface-112x112"
        }
        env {
          name = "KAGGLE_JSON_CONTENT"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.kaggle_secret.secret_id
              version = "latest"
            }
          }
        }
      }
    }
  }

  depends_on = [
    google_storage_bucket_iam_member.gcs_access,
    google_secret_manager_secret_iam_member.secret_access
  ]
}