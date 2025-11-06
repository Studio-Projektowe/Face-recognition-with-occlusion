data "google_compute_default_service_account" "default" {
  project = var.PROJECT_ID
}

resource "google_storage_bucket_iam_member" "gcs_access" {
  bucket = google_storage_bucket.casia-dataset-original-bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${data.google_compute_default_service_account.default.email}"
  depends_on = [ google_storage_bucket.casia-dataset-original-bucket ]
}

resource "google_secret_manager_secret_iam_member" "secret_access" {
  project   = google_secret_manager_secret.kaggle_secret.project
  secret_id = google_secret_manager_secret.kaggle_secret.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${data.google_compute_default_service_account.default.email}"
}