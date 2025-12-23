locals {
  zone_3 = "us-east1-c"

  vm_name_3 = "vgg-rescue-3-toronto"
}

resource "google_compute_disk" "boot_disk_3" {
  name  = "${local.vm_name_3}-boot"
  type  = "pd-ssd"
  zone  = local.zone_3
  snapshot = local.snapshot_boot_name_2
}

resource "google_compute_disk" "data_disk_3" {
  name  = "${local.vm_name_3}-data"
  type  = "pd-ssd"
  zone  = local.zone_3
  snapshot = local.snapshot_data_name_2
}

resource "google_compute_instance" "rescue_vm_3" {
  name         = local.vm_name_3
  machine_type = local.machine_type
  zone         = local.zone_3

  boot_disk {
    source      = google_compute_disk.boot_disk_3.id
    auto_delete = false 
  }

  attached_disk {
    source      = google_compute_disk.data_disk_3.id
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
#   location = "us-east1-c"
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
