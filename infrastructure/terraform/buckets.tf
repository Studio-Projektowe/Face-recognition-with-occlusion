
resource "random_id" "suffix" {
  byte_length = 4
}

# Labeled Faces in the Wild

resource "google_storage_bucket" "labeled_faces_in_the_wild_original" {
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

resource "google_storage_bucket" "labeled_faces_in_the_wild_occlusion" {
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

resource "google_storage_bucket" "celeb_faces_original" {
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

resource "google_storage_bucket" "celeb_faces_occluded" {
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

resource "google_storage_bucket" "casia_dataset_original_bucket" {
  name     = "kaggle-casia-dataset-original-${random_id.suffix.hex}"
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