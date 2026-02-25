# SIE GKE Terraform - Validation Tests
#
# Run with: terraform test
# Requires Terraform >= 1.7.0

# =============================================================================
# Variable Validation Tests (plan-only, no infrastructure)
# =============================================================================

run "validate_cluster_name" {
  command = plan

  variables {
    project_id   = "test-project"
    cluster_name = "sie-test"
    region       = "us-central1"
  }

  # Cluster name should be set correctly
  assert {
    condition     = google_container_cluster.primary.name == "sie-test"
    error_message = "Cluster name should match input variable"
  }
}

run "validate_gpu_node_pool_config" {
  command = plan

  variables {
    project_id   = "test-project"
    cluster_name = "sie-test"
    region       = "us-central1"
    gpu_node_pools = [
      {
        name           = "l4-pool"
        machine_type   = "g2-standard-24"
        gpu_type       = "nvidia-l4"
        gpu_count      = 2
        min_node_count = 0
        max_node_count = 10
        disk_size_gb   = 100
        disk_type      = "pd-ssd"
        spot           = true
        zones          = []
        labels         = {}
        taints         = []
      }
    ]
  }

  # GPU node pool should have LATEST driver
  assert {
    condition     = google_container_node_pool.gpu["l4-pool"].node_config[0].guest_accelerator[0].gpu_driver_installation_config[0].gpu_driver_version == "LATEST"
    error_message = "GPU driver version should be LATEST"
  }

  # GPU type should match
  assert {
    condition     = google_container_node_pool.gpu["l4-pool"].node_config[0].guest_accelerator[0].type == "nvidia-l4"
    error_message = "GPU type should be nvidia-l4"
  }

  # Spot should be enabled
  assert {
    condition     = google_container_node_pool.gpu["l4-pool"].node_config[0].spot == true
    error_message = "Spot VMs should be enabled for cost savings"
  }
}

run "validate_workload_identity_enabled" {
  command = plan

  variables {
    project_id               = "test-project"
    cluster_name             = "sie-test"
    region                   = "us-central1"
    enable_workload_identity = true
  }

  # Workload Identity pool should be configured
  assert {
    condition     = google_container_cluster.primary.workload_identity_config[0].workload_pool == "test-project.svc.id.goog"
    error_message = "Workload Identity pool should be configured"
  }
}

run "validate_nap_gpu_limits" {
  command = plan

  variables {
    project_id                    = "test-project"
    cluster_name                  = "sie-test"
    region                        = "us-central1"
    enable_node_auto_provisioning = true
  }

  # NAP should have GPU resource limits defined
  assert {
    condition     = length([for r in google_container_cluster.primary.cluster_autoscaling[0].resource_limits : r if r.resource_type == "nvidia-l4"]) > 0
    error_message = "NAP should have nvidia-l4 resource limits"
  }
}

run "validate_private_cluster_config" {
  command = plan

  variables {
    project_id           = "test-project"
    cluster_name         = "sie-test"
    region               = "us-central1"
    create_network       = true
    enable_private_nodes = true
  }

  # Private nodes should be enabled
  assert {
    condition     = google_container_cluster.primary.private_cluster_config[0].enable_private_nodes == true
    error_message = "Private nodes should be enabled"
  }

  # Cloud NAT should be created for private nodes
  assert {
    condition     = length(google_compute_router_nat.nat) > 0
    error_message = "Cloud NAT should be created for private nodes"
  }
}

# =============================================================================
# Security Validation Tests
# =============================================================================

run "validate_shielded_nodes" {
  command = plan

  variables {
    project_id   = "test-project"
    cluster_name = "sie-test"
    region       = "us-central1"
  }

  # CPU pool should have shielded instance config
  assert {
    condition     = google_container_node_pool.cpu.node_config[0].shielded_instance_config[0].enable_secure_boot == true
    error_message = "Secure boot should be enabled on CPU nodes"
  }

  assert {
    condition     = google_container_node_pool.cpu.node_config[0].shielded_instance_config[0].enable_integrity_monitoring == true
    error_message = "Integrity monitoring should be enabled on CPU nodes"
  }
}

run "validate_service_account_permissions" {
  command = plan

  variables {
    project_id   = "test-project"
    cluster_name = "sie-test"
    region       = "us-central1"
  }

  # Node service account should have artifact registry reader
  assert {
    condition     = google_project_iam_member.gke_nodes_artifact_registry.role == "roles/artifactregistry.reader"
    error_message = "Node SA should have Artifact Registry reader role"
  }
}
