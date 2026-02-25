# SIE GKE Cluster - Infrastructure Outputs
#
# Outputs needed by the k8s module and external consumers.

# =============================================================================
# Cluster Connection (for k8s module providers)
# =============================================================================

output "cluster_endpoint" {
  description = "GKE cluster API endpoint"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate (base64 encoded)"
  value       = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}

output "cluster_id" {
  description = "GKE cluster ID"
  value       = google_container_cluster.primary.id
}

output "cluster_location" {
  description = "GKE cluster location (region)"
  value       = google_container_cluster.primary.location
}

# =============================================================================
# Project and Region (pass-through for k8s module)
# =============================================================================

output "project_id" {
  description = "GCP project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP region"
  value       = var.region
}

# =============================================================================
# Service Accounts (for k8s module Workload Identity)
# =============================================================================

output "gke_nodes_service_account" {
  description = "Service account email for GKE nodes"
  value       = google_service_account.gke_nodes.email
}

output "sie_workload_service_account" {
  description = "Service account email for SIE workloads (Workload Identity)"
  value       = var.enable_workload_identity ? google_service_account.sie_workload[0].email : null
}

# =============================================================================
# Network Information
# =============================================================================

output "network_name" {
  description = "VPC network name"
  value       = var.create_network ? local.names.network : var.network
}

output "subnetwork_name" {
  description = "Subnetwork name"
  value       = var.create_network ? local.names.subnet : var.subnetwork
}

# =============================================================================
# Artifact Registry
# =============================================================================

output "artifact_registry_url" {
  description = "Artifact Registry URL for SIE images"
  value       = var.create_artifact_registry ? "${var.artifact_registry_location != "" ? var.artifact_registry_location : var.region}-docker.pkg.dev/${var.project_id}/${local.names.registry}" : null
}

# =============================================================================
# Node Pools
# =============================================================================

output "cpu_node_pool_name" {
  description = "CPU node pool name"
  value       = google_container_node_pool.cpu.name
}

output "gpu_node_pool_names" {
  description = "GPU node pool names"
  value       = [for pool in google_container_node_pool.gpu : pool.name]
}

# =============================================================================
# GPU Pool Configuration (for k8s module worker pools)
# =============================================================================

output "gpu_node_pools" {
  description = "GPU node pool configurations (passed to k8s module)"
  value       = var.gpu_node_pools
}

# =============================================================================
# kubectl Configuration
# =============================================================================

output "kubectl_config_command" {
  description = "Command to configure kubectl for this cluster"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --region ${var.region} --project ${var.project_id}"
}

# =============================================================================
# Workload Identity Annotation
# =============================================================================

output "workload_identity_annotation" {
  description = "Annotation to add to K8s service account for Workload Identity"
  value       = var.enable_workload_identity ? "iam.gke.io/gcp-service-account=${google_service_account.sie_workload[0].email}" : null
}

# =============================================================================
# Kubernetes Connection Details (for external providers)
# =============================================================================

output "kubernetes_host" {
  description = "Kubernetes API server host"
  value       = "https://${google_container_cluster.primary.endpoint}"
  sensitive   = true
}

output "kubernetes_token" {
  description = "Kubernetes auth token (from gcloud)"
  value       = data.google_client_config.current.access_token
  sensitive   = true
}
