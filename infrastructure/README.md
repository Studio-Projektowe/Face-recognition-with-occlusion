# How to modify the environment?

## Initialize Terraform

This step configures the Terraform backend (to use the GCS bucket) and downloads necessary providers. Terraform will automatically detect the backend block in config.tf.

```bash
terraform init
```

Important: During the first terraform init, Terraform might ask if you want to migrate a local state or confirm backend configuration. Proceed by typing yes if prompted.

## Review and Apply Infrastructure Changes

Preview the changes Terraform will make and then apply them. This will create the GCS state bucket, enable required GCP APIs, and set up any other infrastructure defined in config.tf.

```bash
terraform plan
terraform apply (-auto-approve)
```

# How to create an environment in GCP?

#### 1. Install `gcloud` with [tutorial](https://docs.cloud.google.com/sdk/docs/install)

#### 2. Log in with your GCP account and create a project

```bash
gcloud auth login
gclodu projects list
gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID
```

#### 3. Create credentials for terraform

```bash
gcloud auth application-default login
```

#### 4. Create `terraform.tfvars` file in `terraform/` directory and write your _PROJECT_ID_ value there

```bash
touch terraform.tfvars
```

Use the following template for terraform.tfvars, replacing the placeholder values:

```
# terraform.tfvars
PROJECT_ID   = "YOUR_GCP_PROJECT_ID"                             # Your actual GCP Project ID
STATE_BUCKET = "your-new-bucket-name"                            # This MUST match the bucket name defined in config.tf
REGION       = "us-central1"                                     # Desired GCP region, e.g., "europe-west1"
ZONE         = "us-central1-a"                                   # Desired GCP zone, e.g., "europe-west1-b"
```

#### 5. Change backend name for your new unique bucket name

```
terraform {
  backend "gcs" {
    bucket = "your-new-bucket-name"
    prefix = "state"
  }
}
```

#### 6. Create a storage bucket for keeping a terraform state remote (or look [link](https://docs.cloud.google.com/docs/terraform/resource-management/store-state))

comment lines:

```
# terraform {
#   backend "gcs" {
#     bucket = "terraform-remote-backend-17b99faefb6860c1"
#     prefix = "state"
#   }
# }
```

than run script with provider, enabling services and this resource (`config.tf`):

```
resource "google_storage_bucket" "terraform_state_bucket" {
  name                     = var.STATE_BUCKET
  location                 = "US"

  force_destroy            = false
  public_access_prevention = "enforced"
  uniform_bucket_level_access = true
  # https://cloud.google.com/storage/docs/uniform-bucket-level-access

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30 # Delete older versions than 30 dni
      num_newer_versions = 1 # Always keep at least 1 latest version
    }
    action {
      type = "Delete"
    }
  }
}
```

after that uncomment line with `terrafrom { backend { "gcs" ...` and run:

```bash
terraform init
```

Type "yes" to migrate local state to remote backend:

```
Initializing the backend...
Acquiring state lock. This may take a few moments...
Do you want to copy existing state to the new backend?
  Pre-existing state was found while migrating the previous "local" backend to the
  newly configured "gcs" backend. No existing state was found in the newly
  configured "gcs" backend. Do you want to copy this state to the new "gcs"
  backend? Enter "yes" to copy and "no" to start with an empty state.

  Enter a value: yes
```

Now you are keeping terraform state remotely. To clear local env run:

```bash
rm -r .terraform
rm .terraform*
rm *.tfstate.backup
rm *.tfstate
```

#### 7. Initialize Terraform

This step configures the Terraform backend (to use the GCS bucket) and downloads necessary providers. Terraform will automatically detect the backend block in config.tf.

```bash
terraform init
```

Important: During the first terraform init, Terraform might ask if you want to migrate a local state or confirm backend configuration. Proceed by typing yes if prompted.

#### 8. Review and Apply Infrastructure Changes

Preview the changes Terraform will make and then apply them. This will create the GCS state bucket, enable required GCP APIs, and set up any other infrastructure defined in config.tf.

```bash
terraform plan
terraform apply
```

Note: The GCS bucket named your-new-bucket-name will be created (if it doesn't exist) and used to store Terraform's state file. This bucket's name is publicly visible in config.tf. Ensure its name is globally unique and do not store sensitive data directly in the bucket itself.
