resource "google_workbench_instance" "notebook_instance" {
  name     = "vgg-model-${random_id.suffix.hex}"
  location = var.ZONE
  project  = var.PROJECT_ID

  gce_setup {
    machine_type = "g2-standard-4"

    accelerator_configs {
      type       = "NVIDIA_L4"
      core_count = 1
    }

    network_interfaces {
      network = "projects/${var.PROJECT_ID}/global/networks/default"
    }

    metadata = {
      idle-timeout-seconds = 1800
    }
  }

  // optional:
  labels = {
    "project" = "face-recognition"
    "owner"   = "eliza_jakub"
  }

  desired_state = "STOPPED"

  depends_on = [
    google_project_service.aiplatform_api
  ]
}
