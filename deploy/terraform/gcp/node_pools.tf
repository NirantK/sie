# SIE GKE Cluster - Node Pools
#
# Manages GPU and CPU node pools for the cluster.
# GPU pools use spot instances by default for cost savings.

# =============================================================================
# CPU Node Pool (System Workloads)
# =============================================================================

resource "google_container_node_pool" "cpu" {
  provider = google-beta
  project  = var.project_id
  name     = "cpu-pool"
  location = var.region
  cluster  = google_container_cluster.primary.name

  # Autoscaling
  autoscaling {
    min_node_count = var.cpu_node_pool.min_node_count
    max_node_count = var.cpu_node_pool.max_node_count
  }

  # Management
  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type    = var.cpu_node_pool.machine_type
    disk_size_gb    = var.cpu_node_pool.disk_size_gb
    disk_type       = var.cpu_node_pool.disk_type
    local_ssd_count = var.cpu_node_pool.local_ssd_count

    # Spot instances (preemptible)
    spot = var.cpu_node_pool.spot

    # Service account
    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    # Workload Identity
    dynamic "workload_metadata_config" {
      for_each = var.enable_workload_identity ? [1] : []
      content {
        mode = "GKE_METADATA"
      }
    }

    labels = merge(var.labels, {
      "sie.superlinked.com/node-type" = "cpu"
    })

    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }

    # Shielded instance
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }

  lifecycle {
    ignore_changes = [
      initial_node_count
    ]
  }

  # Ensure IAM binding for deployer SA is complete before creating node pool
  # This prevents "The user does not have access to service account" errors
  depends_on = [
    google_service_account_iam_member.deployer_can_use_node_sa
  ]
}

# =============================================================================
# GPU Node Pools
# =============================================================================

resource "google_container_node_pool" "gpu" {
  provider = google-beta
  for_each = { for pool in var.gpu_node_pools : pool.name => pool }

  project  = var.project_id
  name     = each.value.name
  location = var.region
  cluster  = google_container_cluster.primary.name

  # Node locations (zones) - empty means all zones in region
  node_locations = length(each.value.zones) > 0 ? each.value.zones : null

  # Autoscaling - use total limits for cross-zone control
  # location_policy = "ANY" prioritizes spot availability over zone balance
  autoscaling {
    location_policy      = "ANY"
    total_min_node_count = each.value.min_node_count
    total_max_node_count = each.value.max_node_count
  }

  # Management
  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type    = each.value.machine_type
    disk_size_gb    = each.value.disk_size_gb
    disk_type       = each.value.disk_type
    local_ssd_count = each.value.local_ssd_count

    # Spot instances for cost savings
    spot = each.value.spot

    # GPU configuration
    guest_accelerator {
      type  = each.value.gpu_type
      count = each.value.gpu_count

      # GPU driver installation
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }

    # Service account
    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    # Workload Identity
    dynamic "workload_metadata_config" {
      for_each = var.enable_workload_identity ? [1] : []
      content {
        mode = "GKE_METADATA"
      }
    }

    # Labels
    labels = merge(var.labels, each.value.labels, {
      "sie.superlinked.com/node-type" = "gpu"
      "sie.superlinked.com/gpu-type"  = each.value.gpu_type
    })

    # Taints for GPU isolation
    dynamic "taint" {
      for_each = each.value.taints
      content {
        key    = taint.value.key
        value  = taint.value.value
        effect = taint.value.effect
      }
    }

    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }

    # Shielded instance
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }

  lifecycle {
    ignore_changes = [
      initial_node_count
    ]
  }

  # Ensure IAM binding for deployer SA is complete before creating node pool
  # This prevents "The user does not have access to service account" errors
  depends_on = [
    google_service_account_iam_member.deployer_can_use_node_sa
  ]
}

# =============================================================================
# GPU Node Pool Presets
# =============================================================================
#
# Common GPU configurations for reference:
#
# NVIDIA L4 (24GB VRAM) - Best price/performance for inference
#   machine_type: g2-standard-8 (1x L4), g2-standard-24 (2x L4), g2-standard-48 (4x L4)
#   gpu_type: nvidia-l4
#
# NVIDIA A100 40GB - High-end inference
#   machine_type: a2-highgpu-1g (1x A100 40GB)
#   gpu_type: nvidia-tesla-a100
#
# NVIDIA A100 80GB - Large models
#   machine_type: a2-ultragpu-1g (1x A100 80GB)
#   gpu_type: nvidia-a100-80gb
#
# NVIDIA T4 (16GB VRAM) - Budget option
#   machine_type: n1-standard-8 + T4
#   gpu_type: nvidia-tesla-t4
#
# NVIDIA H100 80GB - Latest generation
#   machine_type: a3-highgpu-1g (1x H100)
#   gpu_type: nvidia-h100-80gb
#
# Zone availability varies - use scripts/list-gpu-zones.sh to check
