variable "PROJECT_ID" {
  description = "The ID of Google Cloud Project"
  type        = string
  default     = "face-recognition-476110"
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