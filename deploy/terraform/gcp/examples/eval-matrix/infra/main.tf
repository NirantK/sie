# SIE GKE Cluster - Matrix Evaluation (Infrastructure Layer)
#
# GCP infrastructure for running matrix evaluations across GPU types.
# This layer creates the GKE cluster, node pools, service accounts, and registry.
#
# Usage (via mise task):
#   mise run cluster create --name eval-matrix
#
# Manual usage:
#   export TF_VAR_project_id="your-project-id"
#   terraform init
#   terraform apply -auto-approve

terraform {
  required_version = "~> 1.14.3"
  # Local backend - ephemeral cluster, no need for remote state
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
  default     = "us-west4"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "sie-dev-eval"
}

variable "deployer_service_account" {
  description = "Email of the service account running Terraform (optional, for CI/CD)"
  type        = string
  default     = ""
}

variable "create_artifact_registry" {
  description = "Create Artifact Registry repository for SIE images"
  type        = bool
  default     = true
}

# =============================================================================
# Infrastructure Module
# =============================================================================

module "infra" {
  source = "../../../infra"

  project_id               = var.project_id
  region                   = var.region
  cluster_name             = var.cluster_name
  deployer_service_account = var.deployer_service_account
  deletion_protection      = false # Ephemeral - must allow easy cleanup

  # Network (module uses {cluster_name}-network/subnet naming)
  create_network = true

  # Private cluster with NAT
  enable_private_nodes = true

  # Node Auto-Provisioning
  enable_node_auto_provisioning = true
  nap_max_cpu                   = 200
  nap_max_memory_gb             = 800

  # CPU node pool - minimal for system workloads
  cpu_node_pool = {
    machine_type   = "e2-standard-4"
    min_node_count = 1
    max_node_count = 2
  }

  # GPU node pools - L4 pool for standard matrix evaluation
  gpu_node_pools = [
    # L4 spot pool - up to 16 nodes total (across zones) for parallel model evaluation
    {
      name           = "l4-spot"
      machine_type   = "g2-standard-8" # 8 vCPU, 32GB RAM, 1x L4
      gpu_type       = "nvidia-l4"
      gpu_count      = 1
      min_node_count = 0
      max_node_count = 16 # Full L4 quota for parallel eval
      spot           = true
      zones          = ["us-west4-a", "us-west4-c"]
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "present"
        effect = "NO_SCHEDULE"
      }]
      labels = {
        "sie.superlinked.com/gpu-type" = "l4"
      }
    }
    # A100 pool disabled - XL models will OOM on L4 (expected)
  ]

  # Workload Identity for GCS access
  enable_workload_identity = true
  sie_namespace            = "sie"
  sie_service_account_name = "sie-server"

  # Artifact Registry for SIE images
  create_artifact_registry = var.create_artifact_registry

  # GKE native logging (useful for debugging)
  enable_cloud_logging = true

  labels = {
    "environment" = "eval"
    "managed-by"  = "terraform"
    "purpose"     = "matrix-evaluation"
  }
}

# =============================================================================
# Outputs (for k8s layer)
# =============================================================================

output "cluster_endpoint" {
  description = "GKE cluster API endpoint"
  value       = module.infra.cluster_endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate (base64 encoded)"
  value       = module.infra.cluster_ca_certificate
  sensitive   = true
}

output "cluster_name" {
  description = "GKE cluster name"
  value       = module.infra.cluster_name
}

output "project_id" {
  description = "GCP project ID"
  value       = module.infra.project_id
}

output "region" {
  description = "GCP region"
  value       = module.infra.region
}

output "sie_workload_service_account" {
  description = "Service account email for SIE workloads"
  value       = module.infra.sie_workload_service_account
}

output "gpu_node_pools" {
  description = "GPU node pool configurations"
  value       = module.infra.gpu_node_pools
}

output "artifact_registry_url" {
  description = "Artifact Registry URL for SIE images"
  value       = module.infra.artifact_registry_url
}

output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value       = module.infra.kubectl_config_command
}
