
output "lfitw-bucket-original" {
    value = google_storage_bucket.labeled-faces-in-the-wild-original.name
}

output "lfitw-bucket-occlusion" {
    value = google_storage_bucket.labeled-faces-in-the-wild-occlusion.name
}

output "celeba-bucket-original" {
    value = google_storage_bucket.celeb-faces-original.name
}

output "celeba-bucket-occlusion" {
    value = google_storage_bucket.celeb-faces-occluded.name
}

output "notebook_vertexai_name" {
    value = google_workbench_instance.notebook_instance.name
}