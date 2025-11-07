
output "lfitw_bucket_original" {
    value = google_storage_bucket.labeled_faces_in_the_wild_original.name
}

output "lfitw_bucket_occlusion" {
    value = google_storage_bucket.labeled_faces_in_the_wild_occlusion.name
}

output "celeba_bucket_original" {
    value = google_storage_bucket.celeb_faces_original.name
}

output "celeba_bucket_occlusion" {
    value = google_storage_bucket.celeb_faces_occluded.name
}

output "casia_bucket_occlusion" {
    value = google_storage_bucket.casia_dataset_original_bucket.name
}

output "notebook_vertexai_name" {
    value = google_workbench_instance.notebook_instance.name
}