# SIE GKE Cluster - Production Example
#
# Multi-zone HA cluster with multiple GPU types for production workloads.
# Includes A100 for large models and L4 for cost-effective inference.
#
# Prerequisites:
#   1. GCP project with billing enabled
#   2. GPU quota for A100 and L4 in target zones
#   3. GCS bucket for model cache (optional)
#   4. APIs enabled: container.googleapis.com, compute.googleapis.com
#
# Usage:
#   cp terraform.tfvars.example terraform.tfvars
#   # Edit terraform.tfvars with your values
#   terraform init
#   terraform plan
#   terraform apply

terraform {
  required_version = "~> 1.14.3"

  # Use GCS backend for state (required for production)
  backend "gcs" {
    # Configure via backend config or environment variables:
    #   bucket = "your-terraform-state-bucket"
    #   prefix = "sie/gke-production"
  }
}

# =============================================================================
# Variables
# =============================================================================

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "gcs_bucket_name" {
  description = "GCS bucket for model cache"
  type        = string
  default     = ""
}

variable "authorized_networks" {
  description = "CIDR blocks authorized to access cluster master"
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  default = []
}

# Git-sync for config hot reload (optional)
variable "sie_git_sync_enabled" {
  description = "Enable git-sync for config hot reload"
  type        = bool
  default     = false
}

variable "sie_git_sync_repo" {
  description = "Git repository URL for config sync"
  type        = string
  default     = ""
}

variable "sie_git_sync_ssh_secret" {
  description = "K8s secret with SSH key for private repos"
  type        = string
  default     = ""
}

variable "sie_platform" {
  description = "Docker image platform for SIE server (cuda12, cuda11, cpu). Must match cluster GPU hardware."
  type        = string
  default     = "cuda12" # Production uses A100/L4 GPUs which require CUDA 12
}

# =============================================================================
# SIE GKE Module
# =============================================================================

module "sie_gke" {
  source = "../../"

  project_id   = var.project_id
  region       = var.region
  cluster_name = "sie-production"

  # Network
  create_network = true
  network        = "sie-prod-network"
  subnetwork     = "sie-prod-subnet"
  subnet_cidr    = "10.0.0.0/20"
  pods_cidr      = "10.1.0.0/16"
  services_cidr  = "10.2.0.0/20"

  # Private cluster with authorized networks
  enable_private_nodes = true
  authorized_networks  = var.authorized_networks

  # Release channel - STABLE for production
  release_channel = "STABLE"

  # Node Auto-Provisioning - higher limits for production
  enable_node_auto_provisioning = true
  nap_max_cpu                   = 500
  nap_max_memory_gb             = 2000

  # CPU node pool - HA across zones
  cpu_node_pool = {
    machine_type   = "e2-standard-8"
    min_node_count = 2 # HA minimum
    max_node_count = 10
    spot           = false # On-demand for reliability
  }

  # GPU node pools - multiple tiers
  gpu_node_pools = [
    # L4 pool - cost-effective for most models
    {
      name           = "l4-ondemand"
      machine_type   = "g2-standard-8"
      gpu_type       = "nvidia-l4"
      gpu_count      = 1
      min_node_count = 1 # Always-on for fast response
      max_node_count = 20
      spot           = false
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "present"
        effect = "NO_SCHEDULE"
      }]
      labels = {
        "sie.superlinked.com/gpu-type" = "l4"
        "sie.superlinked.com/tier"     = "standard"
      }
    },
    # L4 spot pool - burst capacity
    {
      name           = "l4-spot"
      machine_type   = "g2-standard-8"
      gpu_type       = "nvidia-l4"
      gpu_count      = 1
      min_node_count = 0
      max_node_count = 50
      spot           = true
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "present"
        effect = "NO_SCHEDULE"
      }]
      labels = {
        "sie.superlinked.com/gpu-type" = "l4"
        "sie.superlinked.com/tier"     = "spot"
      }
    },
    # A100 pool - large models (7B+)
    {
      name           = "a100-ondemand"
      machine_type   = "a2-highgpu-1g"
      gpu_type       = "nvidia-tesla-a100"
      gpu_count      = 1
      min_node_count = 0 # Scale to zero - expensive!
      max_node_count = 10
      spot           = false
      disk_size_gb   = 200 # Larger disk for big models
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "present"
        effect = "NO_SCHEDULE"
      }]
      labels = {
        "sie.superlinked.com/gpu-type" = "a100"
        "sie.superlinked.com/tier"     = "premium"
      }
    }
  ]

  # Workload Identity for GCS access
  enable_workload_identity = true
  sie_namespace            = "sie"
  sie_service_account_name = "sie-server"
  gcs_bucket_name          = var.gcs_bucket_name

  # KEDA for autoscaling
  install_keda = true
  keda_version = "2.14.0"

  # Observability - full stack
  enable_managed_prometheus = true
  enable_cloud_logging      = true

  # Artifact Registry
  create_artifact_registry = true

  # Git-sync for config hot reload
  sie_git_sync_enabled    = var.sie_git_sync_enabled
  sie_git_sync_repo       = var.sie_git_sync_repo
  sie_git_sync_ssh_secret = var.sie_git_sync_ssh_secret

  labels = {
    "environment" = "production"
    "managed-by"  = "terraform"
    "team"        = "platform"
  }
}

# =============================================================================
# Outputs
# =============================================================================

output "cluster_name" {
  value = module.sie_gke.cluster_name
}

output "cluster_endpoint" {
  value     = module.sie_gke.cluster_endpoint
  sensitive = true
}

output "kubectl_command" {
  value = module.sie_gke.kubectl_config_command
}

output "artifact_registry_url" {
  value = module.sie_gke.artifact_registry_url
}

output "workload_identity_annotation" {
  value = module.sie_gke.workload_identity_annotation
}

output "gpu_node_pools" {
  value = module.sie_gke.gpu_node_pool_names
}
