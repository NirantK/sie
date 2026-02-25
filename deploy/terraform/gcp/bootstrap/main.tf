# SIE Deployer Service Account Bootstrap
#
# This module creates a service account that can deploy SIE infrastructure.
# Run this ONCE per project before using the main terraform modules.
#
# Prerequisites:
#   - User running this must have roles/iam.serviceAccountAdmin
#   - User must have roles/iam.serviceAccountKeyAdmin (if generating keys)
#   - Run: gcloud auth application-default login (for terraform to authenticate)
#
# Usage:
#   cd deploy/terraform/gcp/bootstrap
#   gcloud auth application-default login  # Required for terraform
#   export TF_VAR_project_id="your-project-id"
#   terraform init
#   terraform apply
#
# After apply:
#   - The service account email will be output
#   - Optionally download key: terraform output -raw key_file > sa-key.json

terraform {
  required_version = "~> 1.14.3"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.16.0"
    }
  }
}

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "service_account_id" {
  description = "ID for the deployer service account"
  type        = string
  default     = "sie-terraform"
}

variable "create_key" {
  description = "Whether to create and output a service account key"
  type        = bool
  default     = false
}

provider "google" {
  project = var.project_id
}

# Create the deployer service account
resource "google_service_account" "deployer" {
  project      = var.project_id
  account_id   = var.service_account_id
  display_name = "SIE Terraform Deployer"
  description  = "Service account for deploying SIE infrastructure via Terraform"
}

# Required roles for deploying SIE infrastructure
locals {
  deployer_roles = [
    # GKE cluster management
    "roles/container.admin",
    "roles/container.clusterAdmin",

    # Compute resources (networks, node pools)
    "roles/compute.admin",

    # Service accounts (create node SA, workload identity SA)
    "roles/iam.serviceAccountAdmin",
    "roles/iam.serviceAccountUser",

    # IAM bindings
    "roles/resourcemanager.projectIamAdmin",

    # Artifact Registry (push images)
    "roles/artifactregistry.admin",

    # Storage (for terraform state if using GCS backend)
    "roles/storage.admin",

    # Diagnostics (Cloud Logging reads)
    "roles/logging.viewer",
  ]
}

resource "google_project_iam_member" "deployer_roles" {
  for_each = toset(local.deployer_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

# Optionally create a key for the service account
resource "google_service_account_key" "deployer_key" {
  count = var.create_key ? 1 : 0

  service_account_id = google_service_account.deployer.name
}

# Outputs
output "service_account_email" {
  description = "Email of the deployer service account"
  value       = google_service_account.deployer.email
}

output "service_account_id" {
  description = "Fully qualified ID of the deployer service account"
  value       = google_service_account.deployer.id
}

output "key_file" {
  description = "Base64-encoded service account key (if create_key=true)"
  value       = var.create_key ? base64decode(google_service_account_key.deployer_key[0].private_key) : ""
  sensitive   = true
}

output "usage_instructions" {
  description = "Instructions for using this service account"
  value       = <<-EOT
    Deployer service account created: ${google_service_account.deployer.email}

    To use with Terraform:
      export GOOGLE_APPLICATION_CREDENTIALS="path/to/key.json"
      export TF_VAR_project_id="${var.project_id}"
      export TF_VAR_deployer_service_account="${google_service_account.deployer.email}"

    To download the key (if create_key=true):
      terraform output -raw key_file > ~/.secrets/${var.project_id}-deployer.json
      chmod 600 ~/.secrets/${var.project_id}-deployer.json
  EOT
}
