
resource "random_id" "suffix" {
  byte_length = 4
}

resource "google_storage_bucket" "labeled-faces-in-the-wild-original" {
  name                     = "kaggle-labeled-faces-in-the-wild-original-${random_id.suffix.hex}"
  location                 = "US"

  force_destroy            = true
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
