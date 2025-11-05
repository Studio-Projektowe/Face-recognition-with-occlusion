
output "lfitw-bucket-original" {
    value = google_storage_bucket.labeled-faces-in-the-wild-original.name
}

output "lfitw-bucket-occlusion" {
    value = google_storage_bucket.labeled-faces-in-the-wild-occlusion.name
}