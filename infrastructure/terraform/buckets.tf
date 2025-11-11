
resource "random_id" "suffix" {
  byte_length = 4
}

# Casia dataset but sorted for train:val:test and saved as tuples : (img, landmarks)

resource "google_storage_bucket" "casia_prepared_dataset_bucket" {
  name     = "casia_prepared_dataset-${random_id.suffix.hex}"
  location = "northamerica-northeast1"

  force_destroy            = false
  public_access_prevention = "enforced"
  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 120
    }
    action {
      type = "Delete"
    }
  }
}