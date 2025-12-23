locals {
  zone_2 = "us-central1-c"

  vm_name_2 = "vgg-rescue-2-toronto"

  snapshot_boot_name_2 = "zrzut-zrzutu-boot-toronto"
  snapshot_data_name_2 = "zrzut-zrzutu-data-toronto"
}

resource "google_compute_disk" "boot_disk_2" {
  name  = "${local.vm_name_2}-boot"
  type  = "pd-ssd"
  zone  = local.zone_2
  snapshot = local.snapshot_boot_name_2
}

resource "google_compute_disk" "data_disk_2" {
  name  = "${local.vm_name_2}-data"
  type  = "pd-ssd"
  zone  = local.zone_2
  snapshot = local.snapshot_data_name_2
}

resource "google_compute_instance" "rescue_vm_2" {
  name         = local.vm_name_2
  machine_type = local.machine_type
  zone         = local.zone_2

  boot_disk {
    source      = google_compute_disk.boot_disk_2.id
    auto_delete = false 
  }

  attached_disk {
    source      = google_compute_disk.data_disk_2.id
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
#   location = "us-central1-c"
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
