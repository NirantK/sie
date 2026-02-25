# SIE GKE Cluster - Outputs

# =============================================================================
# Cluster Information
# =============================================================================

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

output "cluster_endpoint" {
  description = "GKE cluster endpoint (API server URL)"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate (base64 encoded)"
  value       = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

# =============================================================================
# kubectl Configuration
# =============================================================================

output "kubectl_config_command" {
  description = "Command to configure kubectl for this cluster"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --region ${var.region} --project ${var.project_id}"
}

# =============================================================================
# Network Information
# =============================================================================

output "network_name" {
  description = "VPC network name"
  value       = var.create_network ? google_compute_network.vpc[0].name : var.network
}

output "subnetwork_name" {
  description = "Subnetwork name"
  value       = var.create_network ? google_compute_subnetwork.subnet[0].name : var.subnetwork
}

# =============================================================================
# Service Accounts
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
# Artifact Registry
# =============================================================================

output "artifact_registry_url" {
  description = "Artifact Registry URL for SIE images"
  value       = var.create_artifact_registry ? "${var.artifact_registry_location != "" ? var.artifact_registry_location : var.region}-docker.pkg.dev/${var.project_id}/sie" : null
}

# =============================================================================
# Router Access
# =============================================================================

output "router_url" {
  description = "Router URL (ingress host or LoadBalancer IP/hostname)"
  value = (
    var.sie_ingress_enabled && var.sie_ingress_host != "" ?
    format("%s://%s", var.sie_ingress_tls_enabled ? "https" : "http", var.sie_ingress_host) :
    (
      var.sie_ingress_enabled && var.install_ingress_nginx ?
      (
        try(data.kubernetes_service_v1.ingress_nginx[0].status[0].load_balancer[0].ingress[0].ip, "") != "" ?
        format("http://%s", data.kubernetes_service_v1.ingress_nginx[0].status[0].load_balancer[0].ingress[0].ip) :
        (
          try(data.kubernetes_service_v1.ingress_nginx[0].status[0].load_balancer[0].ingress[0].hostname, "") != "" ?
          format("http://%s", data.kubernetes_service_v1.ingress_nginx[0].status[0].load_balancer[0].ingress[0].hostname) :
          null
        )
      ) :
      (
        var.sie_router_service_type == "LoadBalancer" ?
        (
          try(data.kubernetes_service_v1.router[0].status[0].load_balancer[0].ingress[0].ip, "") != "" ?
          format("http://%s", data.kubernetes_service_v1.router[0].status[0].load_balancer[0].ingress[0].ip) :
          (
            try(data.kubernetes_service_v1.router[0].status[0].load_balancer[0].ingress[0].hostname, "") != "" ?
            format("http://%s", data.kubernetes_service_v1.router[0].status[0].load_balancer[0].ingress[0].hostname) :
            null
          )
        ) :
        null
      )
    )
  )
}

output "router_service_name" {
  description = "Router service name (for diagnostics)"
  value       = local.router_service_name
}

output "sie_namespace" {
  description = "SIE namespace"
  value       = var.sie_namespace
}

output "ingress_controller_service_name" {
  description = "Ingress controller service name (if installed)"
  value       = local.ingress_service_name
}

# =============================================================================
# Workload Identity Annotation
# =============================================================================

output "workload_identity_annotation" {
  description = "Annotation to add to K8s service account for Workload Identity"
  value       = var.enable_workload_identity ? "iam.gke.io/gcp-service-account=${google_service_account.sie_workload[0].email}" : null
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
# Connection Details (for Helm/kubectl providers)
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
