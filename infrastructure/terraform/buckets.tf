
resource "random_id" "suffix" {
  byte_length = 4
}

# Labeled Faces in the Wild

resource "google_storage_bucket" "labeled-faces-in-the-wild-original" {
  name                     = "kaggle-labeled-faces-in-the-wild-original-${random_id.suffix.hex}"
  location                 = "US"

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

resource "google_storage_bucket" "labeled-faces-in-the-wild-occlusion" {
  name                     = "kaggle-labeled-faces-in-the-wild-occlusion-${random_id.suffix.hex}"
  location                 = "US"

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

# CelebA - CelebFaces

resource "google_storage_bucket" "celeb-faces-original" {
  name                     = "kaggle-celeb-faces-original-${random_id.suffix.hex}"
  location                 = "US"

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

resource "google_storage_bucket" "celeb-faces-occluded" {
  name                     = "kaggle-celeb-faces-occluded-${random_id.suffix.hex}"
  location                 = "US"

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

# CASIA-WebFace

resource "google_storage_bucket" "casia-dataset-original-bucket" {
  name     = "$kaggle-casia-dataset-original-${random_id.suffix.hex}"
  location = "US"

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

resource "google_storage_bucket" "casia-build-source-bucket" {
  name     = "$casia-build-source-${random_id.suffix.hex}"
  location = "US"

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

resource "google_storage_bucket_object" "casia-build-source-zip" {
  name   = "build-source.zip"
  bucket = google_storage_bucket.casia-build-source-bucket.name
  source = "${path.module}/../../scripts/download_dataset/build-source.zip"
}