# SIE GKE Cluster - Kubernetes Module Variables
#
# Variables for K8s/Helm resources. Includes inputs from infra module.

# =============================================================================
# Required Inputs (from infra module)
# =============================================================================

variable "cluster_endpoint" {
  description = "GKE cluster API endpoint (from infra module)"
  type        = string
}

variable "cluster_ca_certificate" {
  description = "GKE cluster CA certificate base64 (from infra module)"
  type        = string
  sensitive   = true
}

variable "cluster_name" {
  description = "GKE cluster name (from infra module)"
  type        = string
}

variable "project_id" {
  description = "GCP project ID (from infra module)"
  type        = string
}

variable "region" {
  description = "GCP region (from infra module)"
  type        = string
}

variable "sie_workload_service_account" {
  description = "SIE workload GCP service account email (from infra module)"
  type        = string
  default     = null
}

variable "gpu_node_pools" {
  description = "GPU node pool configurations (from infra module)"
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
# Workload Identity
# =============================================================================

variable "enable_workload_identity" {
  description = "Enable Workload Identity for GCS/S3 access"
  type        = bool
  default     = true
}

variable "sie_service_account_name" {
  description = "Name of the K8s service account for SIE workloads"
  type        = string
  default     = "sie-server"
}

variable "sie_namespace" {
  description = "Kubernetes namespace for SIE workloads"
  type        = string
  default     = "sie"
}

variable "gcs_bucket_name" {
  description = "GCS bucket for model cache"
  type        = string
  default     = ""
}

# =============================================================================
# KEDA (Autoscaling)
# =============================================================================

variable "install_keda" {
  description = "Install KEDA for Kubernetes event-driven autoscaling"
  type        = bool
  default     = true
}

variable "keda_version" {
  description = "KEDA Helm chart version"
  type        = string
  default     = "2.14.0"
}

# =============================================================================
# Observability - Prometheus
# =============================================================================

variable "external_prometheus_url" {
  description = "External Prometheus URL. If set, skips kube-prometheus-stack installation."
  type        = string
  default     = ""
}

variable "external_grafana_url" {
  description = "External Grafana URL"
  type        = string
  default     = ""
}

variable "prometheus_stack_version" {
  description = "kube-prometheus-stack Helm chart version"
  type        = string
  default     = "67.4.0"
}

variable "dcgm_exporter_version" {
  description = "NVIDIA DCGM Exporter Helm chart version"
  type        = string
  default     = "3.6.0"
}

variable "prometheus_retention" {
  description = "Prometheus data retention period"
  type        = string
  default     = "15d"
}

variable "prometheus_retention_size" {
  description = "Prometheus data retention size limit"
  type        = string
  default     = "50GB"
}

variable "prometheus_storage_size" {
  description = "Prometheus persistent volume size"
  type        = string
  default     = "100Gi"
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  default     = "admin"
  sensitive   = true
}

# =============================================================================
# Observability - Logging
# =============================================================================

variable "install_loki" {
  description = "Install Loki + Alloy + Event Exporter for log aggregation"
  type        = bool
  default     = true
}

variable "loki_version" {
  description = "Loki Helm chart version"
  type        = string
  default     = "6.23.0"
}

variable "alloy_version" {
  description = "Grafana Alloy Helm chart version"
  type        = string
  default     = "0.10.0"
}

variable "event_exporter_version" {
  description = "Kubernetes Event Exporter Helm chart version"
  type        = string
  default     = "3.6.3"
}

# =============================================================================
# SIE Application
# =============================================================================

variable "install_sie" {
  description = "Install SIE application (router + workers) via Helm"
  type        = bool
  default     = true
}

variable "install_ingress_nginx" {
  description = "Install ingress-nginx controller"
  type        = bool
  default     = false
}

variable "ingress_nginx_class" {
  description = "Ingress class name for ingress-nginx"
  type        = string
  default     = "nginx"
}

variable "ingress_nginx_service_annotations" {
  description = "Annotations for ingress-nginx service"
  type        = map(string)
  default     = {}
}

variable "ingress_nginx_service_type" {
  description = "Ingress-nginx service type"
  type        = string
  default     = "LoadBalancer"
}

variable "install_dex" {
  description = "Install Dex (OIDC issuer) via Helm"
  type        = bool
  default     = false
}

variable "dex_chart_version" {
  description = "Dex Helm chart version"
  type        = string
  default     = "0.8.2"
}

variable "dex_namespace" {
  description = "Dex namespace"
  type        = string
  default     = "dex"
}

variable "dex_values_yaml" {
  description = "Dex Helm values (YAML string)"
  type        = string
  default     = ""
}

variable "sie_router_image" {
  description = "SIE router Docker image"
  type        = string
  default     = "ghcr.io/superlinked/sie-router"
}

variable "sie_server_image" {
  description = "SIE server Docker image"
  type        = string
  default     = "ghcr.io/superlinked/sie-server"
}

variable "sie_image_tag" {
  description = "Docker image tag for SIE server"
  type        = string
  default     = ""
}

variable "sie_router_image_tag" {
  description = "Docker image tag for SIE router"
  type        = string
  default     = "latest"
}

variable "sie_bundle" {
  description = "SIE server bundle (default, florence2, sglang)"
  type        = string
  default     = "default"
}

variable "sie_bundles" {
  description = "List of SIE server bundles to deploy"
  type        = list(string)
  default     = []
}

variable "sie_router_replicas" {
  description = "Number of router replicas"
  type        = number
  default     = 2
}

variable "sie_router_resources" {
  description = "Resource requests/limits for router"
  type = object({
    requests = object({
      cpu    = string
      memory = string
    })
    limits = object({
      cpu    = string
      memory = string
    })
  })
  default = {
    requests = {
      cpu    = "500m"
      memory = "512Mi"
    }
    limits = {
      cpu    = "2"
      memory = "2Gi"
    }
  }
}

variable "sie_router_auth_mode" {
  description = "Router auth mode (none | static)"
  type        = string
  default     = "none"
}

variable "sie_router_auth_secret_name" {
  description = "K8s secret name for router auth token"
  type        = string
  default     = ""
}

variable "sie_router_auth_secret_key" {
  description = "Secret key name for router auth token"
  type        = string
  default     = "SIE_AUTH_TOKEN"
}

variable "sie_router_service_type" {
  description = "Router service type (ClusterIP, LoadBalancer)"
  type        = string
  default     = "ClusterIP"
}

variable "sie_router_service_annotations" {
  description = "Annotations for router service"
  type        = map(string)
  default     = {}
}

variable "sie_cache_volume_size" {
  description = "Size of persistent volume for model cache"
  type        = string
  default     = "50Gi"
}

variable "sie_hf_token" {
  description = "HuggingFace token for gated model access"
  type        = string
  default     = ""
  sensitive   = true
}

variable "sie_image_pull_secrets" {
  description = "Image pull secrets for SIE images"
  type        = list(string)
  default     = []
}

variable "sie_autoscaling_cooldown" {
  description = "KEDA cooldown period in seconds"
  type        = number
  default     = 600
}

# =============================================================================
# SIE Ingress + Auth
# =============================================================================

variable "sie_ingress_enabled" {
  description = "Enable ingress for router access"
  type        = bool
  default     = false
}

variable "sie_ingress_class" {
  description = "Ingress class name"
  type        = string
  default     = "nginx"
}

variable "sie_ingress_host" {
  description = "Ingress hostname"
  type        = string
  default     = ""
}

variable "sie_ingress_tls_enabled" {
  description = "Enable ingress TLS"
  type        = bool
  default     = false
}

variable "sie_ingress_tls_secret_name" {
  description = "TLS secret name for ingress"
  type        = string
  default     = ""
}

variable "sie_ingress_annotations" {
  description = "Ingress annotations"
  type        = map(string)
  default     = {}
}

variable "sie_auth_enabled" {
  description = "Enable oauth2-proxy for ingress auth"
  type        = bool
  default     = false
}

variable "sie_auth_oidc_issuer_url" {
  description = "OIDC issuer URL for oauth2-proxy"
  type        = string
  default     = ""
}

variable "sie_auth_redirect_url" {
  description = "OAuth2 redirect URL"
  type        = string
  default     = ""
}

variable "sie_auth_email_domain" {
  description = "Email domain filter for oauth2-proxy"
  type        = string
  default     = "*"
}

variable "sie_auth_secret_name" {
  description = "K8s secret name for oauth2-proxy"
  type        = string
  default     = "oauth2-proxy"
}

variable "sie_auth_client_id_key" {
  description = "Secret key name for client ID"
  type        = string
  default     = "OAUTH2_PROXY_CLIENT_ID"
}

variable "sie_auth_client_secret_key" {
  description = "Secret key name for client secret"
  type        = string
  default     = "OAUTH2_PROXY_CLIENT_SECRET"
}

variable "sie_auth_cookie_secret_key" {
  description = "Secret key name for cookie secret"
  type        = string
  default     = "OAUTH2_PROXY_COOKIE_SECRET"
}

variable "sie_auth_oauth2_proxy_image" {
  description = "oauth2-proxy Docker image"
  type        = string
  default     = "ghcr.io/oauth2-proxy/oauth2-proxy"
}

variable "sie_auth_oauth2_proxy_image_tag" {
  description = "oauth2-proxy Docker image tag"
  type        = string
  default     = "v7.9.0"
}

variable "sie_auth_extra_jwt_issuers" {
  description = "Additional JWT issuers"
  type        = list(string)
  default     = []
}

# =============================================================================
# SIE Config Hot Reload (git-sync)
# =============================================================================

variable "sie_git_sync_enabled" {
  description = "Enable git-sync sidecar for config hot reload"
  type        = bool
  default     = false
}

variable "sie_git_sync_repo" {
  description = "Git repository URL for config sync"
  type        = string
  default     = ""
}

variable "sie_git_sync_branch" {
  description = "Git branch to sync"
  type        = string
  default     = "main"
}

variable "sie_git_sync_period" {
  description = "Sync interval"
  type        = string
  default     = "60s"
}

variable "sie_git_sync_bundles_path" {
  description = "Path within repo to bundles directory"
  type        = string
  default     = "packages/sie_server/bundles"
}

variable "sie_git_sync_models_path" {
  description = "Path within repo to models directory"
  type        = string
  default     = "packages/sie_server/models"
}

variable "sie_git_sync_ssh_secret" {
  description = "Name of K8s secret containing SSH key"
  type        = string
  default     = ""
}
