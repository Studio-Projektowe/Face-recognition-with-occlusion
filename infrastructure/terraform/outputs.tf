
output "casia-bucket-prepared" {
    value = google_storage_bucket.casia_prepared_dataset_bucket.name
}

output "notebook_vertexai_name" {
    value = google_workbench_instance.notebook_instance.name
}