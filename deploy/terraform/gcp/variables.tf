# SIE GKE Cluster - Variables
# See examples/ for usage

# =============================================================================
# Required Variables
# =============================================================================

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "deployer_service_account" {
  description = "Email of the service account running Terraform (for granting iam.serviceAccountUser on node SA). If empty, uses project editors."
  type        = string
  default     = ""
}

variable "region" {
  description = "GCP region for the cluster (e.g., us-central1, europe-west4)"
  type        = string
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "sie-cluster"
}

variable "deletion_protection" {
  description = "Enable deletion protection for the cluster (set to false for dev/test)"
  type        = bool
  default     = true
}

# =============================================================================
# Network Configuration
# =============================================================================

variable "network" {
  description = "VPC network name (created if create_network=true)"
  type        = string
  default     = "sie-network"
}

variable "subnetwork" {
  description = "Subnetwork name (created if create_network=true)"
  type        = string
  default     = "sie-subnet"
}

variable "create_network" {
  description = "Create VPC network and subnetwork (set false to use existing)"
  type        = bool
  default     = true
}

variable "subnet_cidr" {
  description = "CIDR range for the subnetwork"
  type        = string
  default     = "10.0.0.0/20"
}

variable "pods_cidr" {
  description = "Secondary CIDR range for pods"
  type        = string
  default     = "10.1.0.0/16"
}

variable "services_cidr" {
  description = "Secondary CIDR range for services"
  type        = string
  default     = "10.2.0.0/20"
}

# =============================================================================
# Cluster Configuration
# =============================================================================

variable "kubernetes_version" {
  description = "Kubernetes version (null = latest available)"
  type        = string
  default     = null
}

variable "release_channel" {
  description = "GKE release channel: RAPID, REGULAR, STABLE, or UNSPECIFIED"
  type        = string
  default     = "REGULAR"
}

variable "enable_private_nodes" {
  description = "Enable private nodes (no public IPs on nodes)"
  type        = bool
  default     = true
}

variable "master_ipv4_cidr_block" {
  description = "CIDR block for the master network (private cluster)"
  type        = string
  default     = "172.16.0.0/28"
}

variable "authorized_networks" {
  description = "CIDR blocks authorized to access the cluster master"
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  default = []
}

# =============================================================================
# Node Pool Auto-Provisioning (NAP)
# =============================================================================

variable "enable_node_auto_provisioning" {
  description = "Enable cluster autoscaler node auto-provisioning"
  type        = bool
  default     = true
}

variable "nap_min_cpu" {
  description = "Minimum total CPU cores for NAP"
  type        = number
  default     = 0
}

variable "nap_max_cpu" {
  description = "Maximum total CPU cores for NAP"
  type        = number
  default     = 1000
}

variable "nap_min_memory_gb" {
  description = "Minimum total memory (GB) for NAP"
  type        = number
  default     = 0
}

variable "nap_max_memory_gb" {
  description = "Maximum total memory (GB) for NAP"
  type        = number
  default     = 4000
}

# =============================================================================
# GPU Node Pools
# =============================================================================

variable "gpu_node_pools" {
  description = "GPU node pool configurations"
  type = list(object({
    name            = string
    machine_type    = string
    gpu_type        = string # nvidia-l4, nvidia-tesla-a100, nvidia-tesla-t4, etc.
    gpu_count       = number
    min_node_count  = number
    max_node_count  = number
    disk_size_gb    = optional(number, 100)
    disk_type       = optional(string, "pd-ssd")
    local_ssd_count = optional(number, 0)
    spot            = optional(bool, false)
    zones           = optional(list(string), []) # Empty = all zones in region
    taints = optional(list(object({
      key    = string
      value  = string
      effect = string
    })), [])
    labels = optional(map(string), {})
  }))
  default = [
    {
      name           = "l4-pool"
      machine_type   = "g2-standard-8" # 8 vCPU, 32GB RAM
      gpu_type       = "nvidia-l4"
      gpu_count      = 1
      min_node_count = 0
      max_node_count = 10
      spot           = true
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
}

# =============================================================================
# CPU Node Pool (for system workloads)
# =============================================================================

variable "cpu_node_pool" {
  description = "CPU node pool for system workloads (kube-system, monitoring, etc.)"
  type = object({
    machine_type    = string
    min_node_count  = number
    max_node_count  = number
    disk_size_gb    = optional(number, 50)
    disk_type       = optional(string, "pd-standard")
    local_ssd_count = optional(number, 0)
    spot            = optional(bool, false)
  })
  default = {
    machine_type   = "e2-standard-4"
    min_node_count = 1
    max_node_count = 5
  }
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
  description = "GCS bucket for model cache (optional, creates bucket if set)"
  type        = string
  default     = ""
}

# =============================================================================
# Artifact Registry
# =============================================================================

variable "artifact_registry_location" {
  description = "Location for Artifact Registry (defaults to region)"
  type        = string
  default     = ""
}

variable "create_artifact_registry" {
  description = "Create Artifact Registry repository for SIE images"
  type        = bool
  default     = true
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
# Observability - Tier 1 (Core - Required for KEDA)
# =============================================================================

variable "external_prometheus_url" {
  description = "BYOI: External Prometheus URL. If set, skips kube-prometheus-stack installation and uses this for KEDA."
  type        = string
  default     = ""
}

variable "external_grafana_url" {
  description = "BYOI: External Grafana URL. Used for documentation/links when using external Prometheus."
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
  description = "Grafana admin password (change in production!)"
  type        = string
  default     = "admin"
  sensitive   = true
}

# =============================================================================
# Observability - Tier 2 (Logging - Optional)
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
# Observability - Tier 3 (Tracing - Opt-in)
# =============================================================================

variable "install_tempo" {
  description = "Install Tempo for distributed tracing"
  type        = bool
  default     = false
}

variable "tempo_version" {
  description = "Tempo Helm chart version"
  type        = string
  default     = "1.10.0"
}

# =============================================================================
# Observability - GKE Native (informational, not used by KEDA)
# =============================================================================

variable "enable_managed_prometheus" {
  description = "Enable GKE Managed Prometheus (for GCP Console metrics, NOT used by KEDA)"
  type        = bool
  default     = false # Changed default: standalone Prometheus is primary
}

variable "enable_cloud_logging" {
  description = "Enable Cloud Logging for cluster"
  type        = bool
  default     = true
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
  description = "Annotations for ingress-nginx service (e.g., internal LB)"
  type        = map(string)
  default     = {}
}

variable "ingress_nginx_service_type" {
  description = "Ingress-nginx service type (LoadBalancer recommended)"
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
  description = "Dex Helm values (YAML string, must include config)"
  type        = string
  default     = ""
  validation {
    condition     = !var.install_dex || var.dex_values_yaml != ""
    error_message = "dex_values_yaml must be set when install_dex is true."
  }
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
  description = "Docker image tag for SIE server (defaults to Chart appVersion)"
  type        = string
  default     = ""
}

variable "sie_router_image_tag" {
  description = "Docker image tag for SIE router (defaults to 'latest')"
  type        = string
  default     = "latest"
}

variable "sie_bundle" {
  description = "SIE server bundle (default, florence2, sglang). Deprecated: use sie_bundles instead."
  type        = string
  default     = "default"
}

variable "sie_bundles" {
  description = "List of SIE server bundles to deploy. Creates worker pools for each (gpu_pool × bundle) combination. If empty, falls back to sie_bundle."
  type        = list(string)
  default     = []
  # Example: ["default", "sglang", "florence2"]
  # With gpu_node_pools = [{name = "l4-spot", ...}], this creates:
  #   - l4-spot-default (workers with bundle=default)
  #   - l4-spot-sglang (workers with bundle=sglang)
  #   - l4-spot-florence2 (workers with bundle=florence2)
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
  validation {
    condition     = contains(["none", "static"], var.sie_router_auth_mode)
    error_message = "sie_router_auth_mode must be one of: none, static."
  }
}

variable "sie_router_auth_secret_name" {
  description = "K8s secret name for router auth token"
  type        = string
  default     = ""
  validation {
    condition     = var.sie_router_auth_mode != "static" || var.sie_router_auth_secret_name != ""
    error_message = "sie_router_auth_secret_name must be set when sie_router_auth_mode is static."
  }
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
  description = "Annotations for router service (e.g., internal LB)"
  type        = map(string)
  default     = {}
}

variable "sie_cache_volume_size" {
  description = "Size of persistent volume for model cache"
  type        = string
  default     = "50Gi"
}

variable "sie_hf_token" {
  description = "HuggingFace token for gated model access (creates secret automatically)"
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
  description = "KEDA cooldown period in seconds before scaling down (default 600s = 10 min)"
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
  description = "Ingress class name (e.g., nginx)"
  type        = string
  default     = "nginx"
}

variable "sie_ingress_host" {
  description = "Ingress hostname (optional; empty means catch-all)"
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
  validation {
    condition     = !var.sie_ingress_tls_enabled || var.sie_ingress_tls_secret_name != ""
    error_message = "sie_ingress_tls_secret_name must be set when sie_ingress_tls_enabled is true."
  }
}

variable "sie_ingress_annotations" {
  description = "Ingress annotations (auth, cert-manager, etc.)"
  type        = map(string)
  default     = {}
}

variable "sie_auth_enabled" {
  description = "Enable oauth2-proxy for ingress auth"
  type        = bool
  default     = false
  validation {
    condition     = !var.sie_auth_enabled || var.sie_ingress_enabled
    error_message = "sie_auth_enabled requires sie_ingress_enabled=true."
  }
}

variable "sie_auth_oidc_issuer_url" {
  description = "OIDC issuer URL for oauth2-proxy"
  type        = string
  default     = ""
  validation {
    condition     = !var.sie_auth_enabled || var.sie_auth_oidc_issuer_url != ""
    error_message = "sie_auth_oidc_issuer_url must be set when sie_auth_enabled is true."
  }
}

variable "sie_auth_redirect_url" {
  description = "OAuth2 redirect URL for oauth2-proxy (optional, required for browser login)"
  type        = string
  default     = ""
}

variable "sie_auth_email_domain" {
  description = "Email domain filter for oauth2-proxy (use * to allow all)"
  type        = string
  default     = "*"
}

variable "sie_auth_secret_name" {
  description = "K8s secret name for oauth2-proxy client and cookie secrets"
  type        = string
  default     = "oauth2-proxy"
}

variable "sie_auth_client_id_key" {
  description = "Secret key name for oauth2-proxy client ID"
  type        = string
  default     = "OAUTH2_PROXY_CLIENT_ID"
}

variable "sie_auth_client_secret_key" {
  description = "Secret key name for oauth2-proxy client secret"
  type        = string
  default     = "OAUTH2_PROXY_CLIENT_SECRET"
}

variable "sie_auth_cookie_secret_key" {
  description = "Secret key name for oauth2-proxy cookie secret"
  type        = string
  default     = "OAUTH2_PROXY_COOKIE_SECRET"
}

variable "sie_auth_oauth2_proxy_image" {
  description = "oauth2-proxy Docker image repository"
  type        = string
  default     = "ghcr.io/oauth2-proxy/oauth2-proxy"
}

variable "sie_auth_oauth2_proxy_image_tag" {
  description = "oauth2-proxy Docker image tag"
  type        = string
  default     = "v7.9.0"
}

variable "sie_auth_extra_jwt_issuers" {
  description = "Additional JWT issuers for oauth2-proxy"
  type        = list(string)
  default     = []
}

# =============================================================================
# SIE Config Hot Reload (git-sync)
# =============================================================================

variable "sie_git_sync_enabled" {
  description = "Enable git-sync sidecar for config hot reload (bundles and models)"
  type        = bool
  default     = false
}

variable "sie_git_sync_repo" {
  description = "Git repository URL for config sync (e.g., https://github.com/org/repo or git@github.com:org/repo)"
  type        = string
  default     = ""
}

variable "sie_git_sync_branch" {
  description = "Git branch to sync"
  type        = string
  default     = "main"
}

variable "sie_git_sync_period" {
  description = "Sync interval (e.g., 60s, 5m)"
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
  description = "Name of K8s secret containing SSH key for private repos (key: ssh)"
  type        = string
  default     = ""
}

# =============================================================================
# Labels & Tags
# =============================================================================

variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default = {
    "managed-by" = "terraform"
    "app"        = "sie"
  }
}
