# SIE GKE Cluster - Infrastructure Module
#
# GCP-only resources: VPC, GKE cluster, node pools, service accounts, IAM, Artifact Registry.
# Does NOT include kubernetes or helm providers - those are in the k8s module.
#
# This module can be applied independently without a running cluster.

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
  # ===========================================================================
  # CENTRALIZED NAMING CONVENTION
  # ===========================================================================
  # Suffixes are defined in ../naming.tf (shared with Python cleanup scripts).
  # Pattern: {cluster_name}{suffix}
  # Label: sie-cluster={cluster_name} (for resources that support labels)
  #
  # To change naming: edit ../naming.tf (single source of truth)
  # ===========================================================================
  names = {
    for key, suffix in local.name_suffixes :
    key => "${var.cluster_name}${suffix}"
  }

  # Auto-detect deployer service account if not explicitly provided
  # This handles CI/CD scenarios where Terraform runs as a service account
  # Note: google_client_openid_userinfo.email may be null for compute service accounts
  current_email              = data.google_client_openid_userinfo.current.email != null ? data.google_client_openid_userinfo.current.email : ""
  current_is_service_account = local.current_email != "" && endswith(local.current_email, ".iam.gserviceaccount.com")
  deployer_service_account   = var.deployer_service_account != "" ? var.deployer_service_account : (local.current_is_service_account ? local.current_email : "")

  # Standard labels for all resources - includes cluster name for cleanup/filtering
  resource_labels = merge(var.labels, {
    "sie-cluster" = var.cluster_name
  })
}

# =============================================================================
# VPC Network
# =============================================================================

resource "google_compute_network" "vpc" {
  count = var.create_network ? 1 : 0

  project                 = var.project_id
  name                    = local.names.network # Use centralized naming
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"

  # Note: google_compute_network doesn't support labels
  # Cleanup relies on naming convention: {cluster_name}-network
}

resource "google_compute_subnetwork" "subnet" {
  count = var.create_network ? 1 : 0

  project       = var.project_id
  name          = local.names.subnet # Use centralized naming
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
  name    = local.names.router
  region  = var.region
  network = google_compute_network.vpc[0].id
}

resource "google_compute_router_nat" "nat" {
  count = var.create_network && var.enable_private_nodes ? 1 : 0

  project                            = var.project_id
  name                               = local.names.nat
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

  # Resource labels (includes sie-cluster for filtering/cleanup)
  resource_labels = local.resource_labels

  # Ignore changes to node pool (we manage separately)
  lifecycle {
    ignore_changes = [
      node_pool,
      initial_node_count
    ]
  }
}
