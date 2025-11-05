terraform {
  backend "gcs" {
    bucket = "terraform-remote-backend-17b99faefb6860c1"
    prefix = "state"
  }
}

provider "google" {
  project = var.PROJECT_ID
  region  = var.REGION
}

resource "google_storage_bucket" "terraform_state_bucket" {
  name                     = "terraform-remote-backend-17b99faefb6860c1"
  location                 = "US"

  force_destroy            = false
  public_access_prevention = "enforced"
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
      num_newer_versions = 1
    }
    action {
      type = "Delete"
    }
  }
}

resource "google_project_service" "compute_api" {
  project                    = var.PROJECT_ID
  service                    = "compute.googleapis.com"
  disable_on_destroy         = false
  disable_dependent_services = false
}

resource "google_project_service" "container_registry_api" {
  project                    = var.PROJECT_ID
  service                    = "containerregistry.googleapis.com"
  disable_on_destroy         = false
  disable_dependent_services = false
}

resource "google_project_service" "aiplatform_api" {
  project                    = var.PROJECT_ID
  service                    = "aiplatform.googleapis.com" # Vertex AI API
  disable_on_destroy         = false
  disable_dependent_services = false
}