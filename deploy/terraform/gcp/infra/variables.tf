# SIE GKE Cluster - Infrastructure Variables
#
# Variables for GCP-only resources (no K8s/Helm configuration).

# =============================================================================
# Required Variables
# =============================================================================

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for the cluster (e.g., us-central1, europe-west4)"
  type        = string
}

variable "deployer_service_account" {
  description = "Email of the service account running Terraform (for granting iam.serviceAccountUser on node SA). If empty, uses project editors."
  type        = string
  default     = ""
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
# Observability (GKE Native)
# =============================================================================

variable "enable_managed_prometheus" {
  description = "Enable GKE Managed Prometheus (for GCP Console metrics)"
  type        = bool
  default     = false
}

variable "enable_cloud_logging" {
  description = "Enable Cloud Logging for cluster"
  type        = bool
  default     = true
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
