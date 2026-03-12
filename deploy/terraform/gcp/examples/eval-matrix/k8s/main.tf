# SIE GKE Cluster - Matrix Evaluation (Kubernetes Layer)
#
# K8s/Helm resources for running matrix evaluations.
# This layer installs KEDA, Prometheus, and SIE application.
#
# Prerequisites:
#   - infra layer must be applied first
#   - Cluster must be accessible (kubectl configured)
#
# Usage (via mise task):
#   mise run cluster create --name eval-matrix
#
# Manual usage:
#   # Get outputs from infra layer
#   export TF_VAR_cluster_endpoint=$(terraform -chdir=../infra output -raw cluster_endpoint)
#   export TF_VAR_cluster_ca_certificate=$(terraform -chdir=../infra output -raw cluster_ca_certificate)
#   # ... other TF_VARs ...
#   terraform init
#   terraform apply -auto-approve

terraform {
  required_version = "~> 1.14.3"
  # Local backend - ephemeral cluster, no need for remote state
}

# =============================================================================
# Variables (from infra layer outputs)
# =============================================================================

variable "cluster_endpoint" {
  description = "GKE cluster API endpoint (from infra layer)"
  type        = string
}

variable "cluster_ca_certificate" {
  description = "GKE cluster CA certificate (from infra layer)"
  type        = string
  sensitive   = true
}

variable "cluster_name" {
  description = "GKE cluster name (from infra layer)"
  type        = string
}

variable "project_id" {
  description = "GCP project ID (from infra layer)"
  type        = string
}

variable "region" {
  description = "GCP region (from infra layer)"
  type        = string
}

variable "sie_workload_service_account" {
  description = "Service account email for SIE workloads (from infra layer)"
  type        = string
  default     = null
}

variable "gpu_node_pools" {
  description = "GPU node pool configurations (from infra layer)"
  type = list(object({
    name           = string
    machine_type   = string
    gpu_type       = string
    gpu_count      = number
    min_node_count = number
    max_node_count = number
    disk_size_gb   = optional(number, 100)
    disk_type      = optional(string, "pd-ssd")
    spot           = optional(bool, false)
    zones          = optional(list(string), [])
    taints = optional(list(object({
      key    = string
      value  = string
      effect = string
    })), [])
    labels = optional(map(string), {})
  }))
  default = []
}

# =============================================================================
# K8s Configuration Variables
# =============================================================================

variable "install_sie" {
  description = "Install SIE application (set false for first apply to skip SIE until images are ready)"
  type        = bool
  default     = true
}

variable "sie_router_image" {
  description = "SIE router Docker image"
  type        = string
  default     = ""
}

variable "sie_server_image" {
  description = "SIE server Docker image"
  type        = string
  default     = ""
}

variable "sie_image_tag" {
  description = "Docker image tag for SIE server"
  type        = string
  default     = ""
}

variable "sie_router_image_tag" {
  description = "Docker image tag for SIE router (defaults to sie_image_tag if not set)"
  type        = string
  default     = ""
}

variable "artifact_registry_url" {
  description = "Artifact Registry URL (from infra layer, used for default image URLs)"
  type        = string
  default     = ""
}

variable "sie_hf_token" {
  description = "HuggingFace token for gated model access (creates K8s secret automatically)"
  type        = string
  default     = ""
  sensitive   = true
}

# =============================================================================
# Kubernetes Module
# =============================================================================

module "k8s" {
  source = "../../../k8s"

  # Connection from infra layer
  cluster_endpoint       = var.cluster_endpoint
  cluster_ca_certificate = var.cluster_ca_certificate
  cluster_name           = var.cluster_name
  project_id             = var.project_id
  region                 = var.region

  # Service account for Workload Identity
  sie_workload_service_account = var.sie_workload_service_account

  # GPU pools for worker autoscaling
  gpu_node_pools = var.gpu_node_pools

  # KEDA for autoscaling (required for scale-to-zero)
  install_keda = true

  # Deploy all bundles for matrix evaluation
  # Creates worker pools: l4-spot-default
  sie_bundles = ["default", "sglang"]

  # Minimal observability - just what KEDA needs
  # kube-prometheus-stack is always installed (KEDA dependency)
  install_loki = false # Skip logging for ephemeral cluster

  # SIE Application (controlled by TF_VAR_install_sie for two-phase deploy)
  install_sie = var.install_sie

  # Router access via ingress
  install_ingress_nginx   = true
  sie_ingress_enabled     = true
  sie_router_service_type = "ClusterIP"

  # SIE images - use Artifact Registry if available, otherwise fall back to defaults
  sie_router_image = var.sie_router_image != "" ? var.sie_router_image : (
    var.artifact_registry_url != "" ? "${var.artifact_registry_url}/sie-router" : "ghcr.io/superlinked/sie-router"
  )
  sie_server_image = var.sie_server_image != "" ? var.sie_server_image : (
    var.artifact_registry_url != "" ? "${var.artifact_registry_url}/sie-server" : "ghcr.io/superlinked/sie-server"
  )
  sie_image_tag        = var.sie_image_tag != "" ? var.sie_image_tag : "latest"
  sie_router_image_tag = var.sie_router_image_tag != "" ? var.sie_router_image_tag : var.sie_image_tag

  # HuggingFace token for gated model access
  sie_hf_token = var.sie_hf_token

  # Fast autoscaling for benchmarks
  # Keep workers warm during matrix eval to avoid mid-run scale-downs
  sie_autoscaling_cooldown = 600
}

# =============================================================================
# Outputs
# =============================================================================

output "router_url" {
  description = "SIE router URL for benchmarks"
  value       = module.k8s.router_url
}

output "sie_namespace" {
  description = "SIE namespace"
  value       = module.k8s.sie_namespace
}

output "prometheus_url" {
  description = "Prometheus server URL"
  value       = module.k8s.prometheus_url
}

output "grafana_url" {
  description = "Grafana URL"
  value       = module.k8s.grafana_url
}

output "health_tests_status" {
  description = "Status of health test jobs"
  value       = module.k8s.health_tests_status
}
