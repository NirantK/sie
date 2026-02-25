# SIE GKE Cluster - Main Configuration
#
# Creates a GKE cluster optimized for GPU inference workloads with:
# - Private nodes with NAT for egress
# - Node Auto-Provisioning (NAP) for dynamic GPU scaling
# - Workload Identity for secure GCS access
# - KEDA for event-driven autoscaling
#
# Usage:
#   cd deploy/terraform/gcp/examples/dev-l4-spot
#   terraform init
#   terraform apply

terraform {
  required_version = "~> 1.14.3"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.16.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 7.16.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 3.1.1"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 3.0.1"
    }
  }
}

# =============================================================================
# Data Sources
# =============================================================================

data "google_client_config" "current" {}

# Get current identity - used to auto-detect service account for CI/CD
data "google_client_openid_userinfo" "current" {}

# =============================================================================
# Local Variables
# =============================================================================

locals {
  # Auto-detect deployer service account if not explicitly provided
  # This handles CI/CD scenarios where Terraform runs as a service account
  # Note: google_client_openid_userinfo.email may be null for compute service accounts
  current_email              = data.google_client_openid_userinfo.current.email != null ? data.google_client_openid_userinfo.current.email : ""
  current_is_service_account = local.current_email != "" && endswith(local.current_email, ".iam.gserviceaccount.com")
  deployer_service_account   = var.deployer_service_account != "" ? var.deployer_service_account : (local.current_is_service_account ? local.current_email : "")
}

# =============================================================================
# VPC Network
# =============================================================================

resource "google_compute_network" "vpc" {
  count = var.create_network ? 1 : 0

  project                 = var.project_id
  name                    = var.network
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"
}

resource "google_compute_subnetwork" "subnet" {
  count = var.create_network ? 1 : 0

  project       = var.project_id
  name          = var.subnetwork
  region        = var.region
  network       = google_compute_network.vpc[0].id
  ip_cidr_range = var.subnet_cidr

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pods_cidr
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.services_cidr
  }

  private_ip_google_access = true
}

# Cloud NAT for private nodes to access internet (pull images, etc.)
resource "google_compute_router" "router" {
  count = var.create_network && var.enable_private_nodes ? 1 : 0

  project = var.project_id
  name    = "${var.cluster_name}-router"
  region  = var.region
  network = google_compute_network.vpc[0].id
}

resource "google_compute_router_nat" "nat" {
  count = var.create_network && var.enable_private_nodes ? 1 : 0

  project                            = var.project_id
  name                               = "${var.cluster_name}-nat"
  router                             = google_compute_router.router[0].name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# =============================================================================
# GKE Cluster
# =============================================================================

resource "google_container_cluster" "primary" {
  provider = google-beta
  project  = var.project_id
  name     = var.cluster_name
  location = var.region

  # Use regional cluster for HA (nodes spread across zones)
  # For single-zone, set location to specific zone (e.g., us-central1-a)

  network    = var.create_network ? google_compute_network.vpc[0].name : var.network
  subnetwork = var.create_network ? google_compute_subnetwork.subnet[0].name : var.subnetwork

  # Deletion protection (set to false for dev/test environments)
  deletion_protection = var.deletion_protection

  # Remove default node pool immediately after cluster creation
  remove_default_node_pool = true
  initial_node_count       = 1

  # Release channel for automatic upgrades
  release_channel {
    channel = var.release_channel
  }

  # Kubernetes version (null = use release channel default)
  min_master_version = var.kubernetes_version

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = var.enable_private_nodes
    enable_private_endpoint = false # Allow public access to master
    master_ipv4_cidr_block  = var.enable_private_nodes ? var.master_ipv4_cidr_block : null
  }

  # Master authorized networks
  dynamic "master_authorized_networks_config" {
    for_each = length(var.authorized_networks) > 0 ? [1] : []
    content {
      dynamic "cidr_blocks" {
        for_each = var.authorized_networks
        content {
          cidr_block   = cidr_blocks.value.cidr_block
          display_name = cidr_blocks.value.display_name
        }
      }
    }
  }

  # IP allocation policy (required for VPC-native cluster)
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Workload Identity
  dynamic "workload_identity_config" {
    for_each = var.enable_workload_identity ? [1] : []
    content {
      workload_pool = "${var.project_id}.svc.id.goog"
    }
  }

  # Node Auto-Provisioning (NAP)
  dynamic "cluster_autoscaling" {
    for_each = var.enable_node_auto_provisioning ? [1] : []
    content {
      enabled = true

      resource_limits {
        resource_type = "cpu"
        minimum       = var.nap_min_cpu
        maximum       = var.nap_max_cpu
      }

      resource_limits {
        resource_type = "memory"
        minimum       = var.nap_min_memory_gb
        maximum       = var.nap_max_memory_gb
      }

      # Allow NAP to provision GPU nodes
      resource_limits {
        resource_type = "nvidia-l4"
        minimum       = 0
        maximum       = 100
      }

      resource_limits {
        resource_type = "nvidia-tesla-a100"
        minimum       = 0
        maximum       = 50
      }

      resource_limits {
        resource_type = "nvidia-tesla-t4"
        minimum       = 0
        maximum       = 100
      }

      auto_provisioning_defaults {
        oauth_scopes = [
          "https://www.googleapis.com/auth/cloud-platform"
        ]

        service_account = google_service_account.gke_nodes.email

        management {
          auto_upgrade = true
          auto_repair  = true
        }

        disk_type = "pd-ssd"
        disk_size = 100
      }
    }
  }

  # Managed Prometheus
  dynamic "monitoring_config" {
    for_each = var.enable_managed_prometheus ? [1] : []
    content {
      enable_components = ["SYSTEM_COMPONENTS"]
      managed_prometheus {
        enabled = true
      }
    }
  }

  # Cloud Logging
  logging_config {
    enable_components = var.enable_cloud_logging ? ["SYSTEM_COMPONENTS", "WORKLOADS"] : []
  }

  # Addons
  addons_config {
    http_load_balancing {
      disabled = false
    }

    horizontal_pod_autoscaling {
      disabled = false
    }

    gce_persistent_disk_csi_driver_config {
      enabled = true
    }

    gcp_filestore_csi_driver_config {
      enabled = true
    }
  }

  # Resource labels
  resource_labels = var.labels

  # Ignore changes to node pool (we manage separately)
  lifecycle {
    ignore_changes = [
      node_pool,
      initial_node_count
    ]
  }
}

# =============================================================================
# Service Accounts
# =============================================================================

# Service account for GKE nodes
resource "google_service_account" "gke_nodes" {
  project      = var.project_id
  account_id   = "${var.cluster_name}-nodes"
  display_name = "GKE Nodes Service Account for ${var.cluster_name}"
}

# Minimal permissions for GKE nodes
resource "google_project_iam_member" "gke_nodes_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_project_iam_member" "gke_nodes_metric_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_project_iam_member" "gke_nodes_monitoring_viewer" {
  project = var.project_id
  role    = "roles/monitoring.viewer"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# Artifact Registry reader (for pulling SIE images) - project level
resource "google_project_iam_member" "gke_nodes_artifact_registry" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}


# Allow deployer SA to use node SA (required for creating node pools with a service account)
# Auto-detects when running as a service account (CI/CD) and grants necessary permission
# For interactive use with `gcloud auth login`, the user typically has sufficient permissions
resource "google_service_account_iam_member" "deployer_can_use_node_sa" {
  count = local.deployer_service_account != "" ? 1 : 0

  service_account_id = google_service_account.gke_nodes.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${local.deployer_service_account}"
}

# =============================================================================
# Workload Identity for SIE
# =============================================================================

# GCP service account for SIE workloads (GCS access)
resource "google_service_account" "sie_workload" {
  count = var.enable_workload_identity ? 1 : 0

  project      = var.project_id
  account_id   = "sie-workload"
  display_name = "SIE Workload Identity Service Account"
}

# Allow K8s service account to impersonate GCP service account
resource "google_service_account_iam_member" "sie_workload_identity" {
  count = var.enable_workload_identity ? 1 : 0

  service_account_id = google_service_account.sie_workload[0].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.sie_namespace}/${var.sie_service_account_name}]"
}

# GCS bucket access for model cache
resource "google_storage_bucket_iam_member" "sie_gcs_access" {
  count = var.enable_workload_identity && var.gcs_bucket_name != "" ? 1 : 0

  bucket = var.gcs_bucket_name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.sie_workload[0].email}"
}

# =============================================================================
# Artifact Registry
# =============================================================================

resource "google_artifact_registry_repository" "sie" {
  count = var.create_artifact_registry ? 1 : 0

  project       = var.project_id
  location      = var.artifact_registry_location != "" ? var.artifact_registry_location : var.region
  repository_id = "sie"
  description   = "SIE Docker images"
  format        = "DOCKER"

  labels = var.labels

  # Cleanup policies to manage storage costs
  # See: https://cloud.google.com/artifact-registry/docs/repositories/cleanup-policy
  cleanup_policy_dry_run = false

  # Keep production releases indefinitely
  cleanup_policies {
    id     = "keep-production-releases"
    action = "KEEP"
    condition {
      tag_state    = "TAGGED"
      tag_prefixes = ["v", "prod-", "release-"]
    }
  }

  # Keep last 10 tagged images for any tag prefix
  cleanup_policies {
    id     = "keep-recent-tagged"
    action = "KEEP"
    most_recent_versions {
      keep_count = 10
    }
  }

  # Delete untagged images older than 30 days
  cleanup_policies {
    id     = "delete-old-untagged"
    action = "DELETE"
    condition {
      tag_state  = "UNTAGGED"
      older_than = "2592000s" # 30 days
    }
  }

  # Delete dev/test tags older than 14 days
  cleanup_policies {
    id     = "delete-old-dev-tags"
    action = "DELETE"
    condition {
      tag_state    = "TAGGED"
      tag_prefixes = ["dev-", "test-", "pr-", "sha-"]
      older_than   = "1209600s" # 14 days
    }
  }
}
