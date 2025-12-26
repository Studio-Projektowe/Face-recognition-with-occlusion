locals {
  zone = "northamerica-northeast2-a"
  
  vm_name = "vgg-rescue-toronto"
  
  machine_type = "g2-standard-4"

  snapshot_boot_name = "face-rec-boot"
  snapshot_data_name = "face-rec-data-workspace"
}

resource "google_compute_disk" "boot_disk" {
  name  = "${local.vm_name}-boot"
  type  = "pd-ssd"
  zone  = local.zone
  snapshot = local.snapshot_boot_name
}

resource "google_compute_disk" "data_disk" {
  name  = "${local.vm_name}-data"
  type  = "pd-ssd"
  zone  = local.zone
  snapshot = local.snapshot_data_name
}

resource "google_compute_disk" "data_disk_copy_2_zone" {
  name  = "${local.vm_name}-copy-zone"
  type  = "pd-ssd"
  zone  = local.zone
  snapshot = "data-toronto-2"
}

resource "google_compute_instance" "rescue_vm" {
  name         = local.vm_name
  machine_type = local.machine_type
  zone         = local.zone

  boot_disk {
    source      = google_compute_disk.boot_disk.id
    auto_delete = false 
  }

  attached_disk {
    source      = google_compute_disk.data_disk.id
    device_name = "data_disk"
  }

  network_interface {
    network = "default"
    access_config {}
  }

  service_account {
    scopes = ["cloud-platform"]
  }

  metadata = {
    install-nvidia-driver = "True"
  }
  
  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
  }

  lifecycle {
    ignore_changes = [
      metadata["ssh-keys"], 
      desired_status 
    ]
  }
}

# resource "google_workbench_instance" "notebook_instance_copy" {
#   name     = "vgg-model-${random_id.suffix.hex}-copy"
#   location = "northamerica-northeast2-a"
#   project  = var.PROJECT_ID

#   gce_setup {
#     machine_type = "g2-standard-4"

#     accelerator_configs {
#       type       = "NVIDIA_L4"
#       core_count = 1
#     }

#     network_interfaces {
#       network = "projects/${var.PROJECT_ID}/global/networks/default"
#     }

#     metadata = {
#       idle-timeout-seconds = 1800
#     }

#     boot_disk {
#       disk_type = "PD_SSD"
#     }
#   }

#   // optional:
#   labels = {
#     "project" = "face-recognition"
#     "owner"   = "eliza_jakub"
#   }

#   desired_state = "STOPPED"

#   depends_on = [
#     google_project_service.aiplatform_api
#   ]
# }
