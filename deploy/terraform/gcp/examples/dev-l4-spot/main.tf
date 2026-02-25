# SIE GKE Cluster - Development Example (L4 Spot)
#
# Single-command deploy: Creates a fully operational SIE cluster with
# GPU nodes, autoscaling, observability, and the SIE application.
#
# Features:
#   - 1x L4 GPU spot pool (scale 0-5)
#   - NAP enabled for automatic node provisioning
#   - KEDA for scale-to-zero autoscaling
#   - kube-prometheus-stack + DCGM for metrics (required for KEDA)
#   - Loki + Alloy for log aggregation
#   - Grafana with pre-built SIE dashboards
#   - SIE application (router + workers) pre-installed
#   - Workload Identity for GCS access
#
# Prerequisites:
#   1. GCP project with billing enabled
#   2. GPU quota (check with: mise run gcp quota)
#   3. APIs enabled: container.googleapis.com, compute.googleapis.com
#   4. SIE Docker images pushed to Artifact Registry
#
# Usage:
#   export TF_VAR_project_id="your-project-id"
#   # Optional: if using a service account for Terraform (CI/CD):
#   # export TF_VAR_deployer_service_account="terraform-sa@your-project.iam.gserviceaccount.com"
#   terraform init
#   terraform plan
#   terraform apply
#
# After apply:
#   # Get kubectl credentials
#   $(terraform output -raw kubectl_command)
#
#   # Access Grafana (admin / admin)
#   kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
#
#   # Check SIE status
#   kubectl get pods -n sie
#
# Cleanup (IMPORTANT - avoid ongoing costs):
#   terraform destroy

terraform {
  required_version = "~> 1.14.3"

  # Uncomment to use GCS backend for state
  # backend "gcs" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "sie/gke"
  # }
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

variable "deployer_service_account" {
  description = "Email of the service account running Terraform (optional, for CI/CD)"
  type        = string
  default     = ""
}

variable "sie_router_image" {
  description = "SIE router Docker image (defaults to ghcr.io/superlinked/sie-router)"
  type        = string
  default     = ""
}

variable "sie_server_image" {
  description = "SIE server Docker image (defaults to ghcr.io/superlinked/sie-server)"
  type        = string
  default     = ""
}

variable "sie_image_tag" {
  description = "Docker image tag for SIE server"
  type        = string
  default     = ""
}

variable "sie_router_image_tag" {
  description = "Docker image tag for SIE router (defaults to 'latest')"
  type        = string
  default     = "latest"
}

variable "sie_platform" {
  description = "Docker image platform for SIE server (cuda12, cuda11, cpu). Must match cluster GPU hardware."
  type        = string
  default     = "cuda12" # This cluster has L4 GPUs which require CUDA 12
}

variable "sie_autoscaling_cooldown" {
  description = "KEDA cooldown period in seconds (default 600, use 120 for testing)"
  type        = number
  default     = 600
}

# =============================================================================
# SIE GKE Module
# =============================================================================

module "sie_gke" {
  source = "../../"

  project_id               = var.project_id
  region                   = var.region
  cluster_name             = "sie-dev"
  deployer_service_account = var.deployer_service_account
  deletion_protection      = false # Dev cluster - allow easy cleanup

  # Network
  create_network = true
  network        = "sie-network"
  subnetwork     = "sie-subnet"

  # Private cluster with NAT
  enable_private_nodes = true

  # Node Auto-Provisioning (NAP) - let GKE create nodes as needed
  enable_node_auto_provisioning = true
  nap_max_cpu                   = 100
  nap_max_memory_gb             = 400

  # CPU node pool for system workloads
  cpu_node_pool = {
    machine_type   = "e2-standard-4"
    min_node_count = 1
    max_node_count = 3
  }

  # GPU node pool - L4 for inference
  gpu_node_pools = [
    {
      name            = "l4-spot"
      machine_type    = "g2-standard-8" # 8 vCPU, 32GB RAM, 1x L4
      gpu_type        = "nvidia-l4"
      gpu_count       = 1
      min_node_count  = 0 # Scale to zero when idle
      max_node_count  = 5
      spot            = true                                                # Use spot instances for ~60% savings
      local_ssd_count = 1                                                   # 375GB local SSD for model cache
      zones           = ["us-central1-a", "us-central1-b", "us-central1-c"] # L4 availability
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "present"
        effect = "NO_SCHEDULE"
      }]
      labels = {
        "sie.superlinked.com/gpu-type" = "l4"
      }
    }
  ]

  # Workload Identity for GCS access
  enable_workload_identity = true
  sie_namespace            = "sie"
  sie_service_account_name = "sie-server"

  # KEDA for autoscaling
  install_keda = true

  # ==========================================================================
  # Observability (Tier 1 - Core, always installed)
  # ==========================================================================
  # kube-prometheus-stack provides Prometheus that KEDA needs for metrics.
  # DCGM Exporter provides GPU metrics for capacity planning.
  # To use external Prometheus: external_prometheus_url = "http://your-prometheus:9090"

  # Tier 2 - Logging (optional, default: true)
  install_loki = true

  # Tier 3 - Tracing (opt-in, default: false)
  install_tempo = false

  # GKE native logging (in addition to Loki)
  enable_cloud_logging = true

  # ==========================================================================
  # SIE Application
  # ==========================================================================
  install_sie = true

  # SIE images - pass through from variables (allows override for testing)
  sie_router_image     = var.sie_router_image != "" ? var.sie_router_image : "ghcr.io/superlinked/sie-router"
  sie_server_image     = var.sie_server_image != "" ? var.sie_server_image : "ghcr.io/superlinked/sie-server"
  sie_image_tag        = var.sie_image_tag != "" ? var.sie_image_tag : ""
  sie_router_image_tag = var.sie_router_image_tag

  # Autoscaling - shorter cooldown for testing
  sie_autoscaling_cooldown = var.sie_autoscaling_cooldown

  # Artifact Registry for SIE images
  create_artifact_registry = true

  labels = {
    "environment" = "dev"
    "managed-by"  = "terraform"
  }
}

# =============================================================================
# Outputs
# =============================================================================

output "cluster_name" {
  description = "GKE cluster name"
  value       = module.sie_gke.cluster_name
}

output "kubectl_command" {
  description = "Command to configure kubectl"
  value       = module.sie_gke.kubectl_config_command
}

output "artifact_registry_url" {
  description = "Artifact Registry URL for pushing images"
  value       = module.sie_gke.artifact_registry_url
}

output "workload_identity_annotation" {
  description = "Annotation for Kubernetes service accounts"
  value       = module.sie_gke.workload_identity_annotation
}

output "prometheus_url" {
  description = "Prometheus URL for KEDA and internal queries"
  value       = module.sie_gke.prometheus_url
}

output "grafana_url" {
  description = "Grafana URL (port-forward to access)"
  value       = module.sie_gke.grafana_url
}

output "sie_router_service" {
  description = "SIE router service for internal access"
  value       = module.sie_gke.sie_router_service
}
