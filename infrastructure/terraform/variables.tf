variable "PROJECT_ID" {
  description = "The ID of Google Cloud Project"
  type        = string
}

variable "REGION" {
  description = "The default region for GCP resources."
  type        = string
  default     = "northamerica-northeast1"
}

variable "ZONE" {
  description = "The default zone for GCP resources."
  type        = string
  default     = "northamerica-northeast1-b"
}

variable "KAGGLE_USERNAME" {
  description = "Kaggle username"
  type        = string
  sensitive   = true
}

variable "KAGGLE_KEY" {
  description = "Kaggle API key"
  type        = string
  sensitive   = true
}