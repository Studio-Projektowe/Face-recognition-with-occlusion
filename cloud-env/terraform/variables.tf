
variable "PROJECT_ID" {
    description = "The ID of Google Cloud Project"
    type        = string
}

variable "STATE_BUCKET" {
    description = "The globally unique name for the GCS bucket to store Terraform state."
    type        = string
}

variable "REGION" {
  description = "The default region for GCP resources."
  type        = string
  default     = "us-central1"
}