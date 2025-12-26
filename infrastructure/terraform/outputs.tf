
output "casia-bucket-prepared" {
    value = google_storage_bucket.casia_prepared_dataset_bucket.name
}

output "notebook_vertexai_name" {
    value = google_workbench_instance.notebook_instance.name
}

output "ssh_info" {
  value = "After creating open: gcloud compute ssh ${local.vm_name} --zone ${local.zone}"
}

output "ssh_info_2" {
  value = "After creating 2 open: gcloud compute ssh ${local.vm_name_2} --zone ${local.zone_2}"
}

output "ssh_info_3" {
  value = "After creating 3 open: gcloud compute ssh ${local.vm_name_3} --zone ${local.zone_3}"
}