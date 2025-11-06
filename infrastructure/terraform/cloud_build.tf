
resource "google_cloudbuild_build" "casia_download" {
  steps {
    name       = "python:3.10"
    entrypoint = "bash"
    args = [
      "-c",
      "pip install kaggle gsutil && python script.py"
    ]
  }

  timeout = "1800s"

  source {
    storage_source {
      bucket = google_storage_bucket.casia-build-source-bucket.name
      object = google_storage_bucket_object.casia-build-source-zip.name
    }
  }

  substitutions = {
    _TARGET_BUCKET = google_storage_bucket.casia-dataset-original-bucket.name
  }

  secrets {
    kms_key_name = null
    secret_env = {
      KAGGLE_USERNAME = var.KAGGLE_USERNAME
      KAGGLE_KEY      = var.KAGGLE_KEY
      TARGET_BUCKET   = google_storage_bucket.casia-dataset-original-bucket.name
    }
  }
}