# OPA Policy: GPU Node Pool Validation
#
# Validates GPU configurations in Terraform plans.
# Run with: conftest test plan.json -p policies/
#
# Generate plan.json with:
#   terraform plan -out=tfplan
#   terraform show -json tfplan > plan.json

package terraform.gpu_validation

import future.keywords.in

# Allowed GPU types for SIE inference workloads
allowed_gpu_types := {
  "nvidia-l4",
  "nvidia-tesla-t4",
  "nvidia-tesla-a100",
  "nvidia-a100-80gb",
  "nvidia-h100-80gb",
}

# Recommended machine types for each GPU
recommended_machine_types := {
  "nvidia-l4": ["g2-standard-8", "g2-standard-24", "g2-standard-48"],
  "nvidia-tesla-t4": ["n1-standard-4", "n1-standard-8", "n1-standard-16"],
  "nvidia-tesla-a100": ["a2-highgpu-1g", "a2-highgpu-2g", "a2-highgpu-4g"],
  "nvidia-a100-80gb": ["a2-ultragpu-1g", "a2-ultragpu-2g", "a2-ultragpu-4g"],
  "nvidia-h100-80gb": ["a3-highgpu-1g", "a3-highgpu-2g", "a3-highgpu-4g"],
}

# =============================================================================
# Deny Rules (Block Invalid Configurations)
# =============================================================================

# Deny unsupported GPU types
deny[msg] {
  resource := input.resource_changes[_]
  resource.type == "google_container_node_pool"
  resource.change.actions[_] in ["create", "update"]

  accelerator := resource.change.after.node_config[0].guest_accelerator[0]
  gpu_type := accelerator.type

  not gpu_type in allowed_gpu_types

  msg := sprintf(
    "GPU node pool '%s' uses unsupported GPU type '%s'. Allowed types: %v",
    [resource.address, gpu_type, allowed_gpu_types]
  )
}

# Deny GPU nodes without LATEST driver
deny[msg] {
  resource := input.resource_changes[_]
  resource.type == "google_container_node_pool"
  resource.change.actions[_] in ["create", "update"]

  accelerator := resource.change.after.node_config[0].guest_accelerator[0]
  accelerator.type != null

  driver_config := accelerator.gpu_driver_installation_config[0]
  driver_config.gpu_driver_version != "LATEST"

  msg := sprintf(
    "GPU node pool '%s' should use gpu_driver_version = 'LATEST' for automatic updates",
    [resource.address]
  )
}

# Deny GPU nodes without shielded instance config
deny[msg] {
  resource := input.resource_changes[_]
  resource.type == "google_container_node_pool"
  resource.change.actions[_] in ["create", "update"]

  node_config := resource.change.after.node_config[0]
  node_config.guest_accelerator[0].type != null

  shielded := node_config.shielded_instance_config[0]
  shielded.enable_secure_boot != true

  msg := sprintf(
    "GPU node pool '%s' should have enable_secure_boot = true",
    [resource.address]
  )
}

# =============================================================================
# Warn Rules (Recommendations)
# =============================================================================

# Warn if Spot VMs not enabled for GPU nodes (cost optimization)
warn[msg] {
  resource := input.resource_changes[_]
  resource.type == "google_container_node_pool"
  resource.change.actions[_] in ["create", "update"]

  node_config := resource.change.after.node_config[0]
  node_config.guest_accelerator[0].type != null
  node_config.spot != true

  msg := sprintf(
    "GPU node pool '%s' does not use Spot VMs. Consider enabling for up to 91%% cost savings.",
    [resource.address]
  )
}

# Warn if max_node_count is too high without NAP
warn[msg] {
  resource := input.resource_changes[_]
  resource.type == "google_container_node_pool"
  resource.change.actions[_] in ["create", "update"]

  resource.change.after.autoscaling[0].max_node_count > 50

  msg := sprintf(
    "GPU node pool '%s' has max_node_count > 50. Ensure sufficient GPU quota.",
    [resource.address]
  )
}

# Warn if disk is not SSD for GPU nodes
warn[msg] {
  resource := input.resource_changes[_]
  resource.type == "google_container_node_pool"
  resource.change.actions[_] in ["create", "update"]

  node_config := resource.change.after.node_config[0]
  node_config.guest_accelerator[0].type != null
  node_config.disk_type != "pd-ssd"

  msg := sprintf(
    "GPU node pool '%s' uses disk_type '%s'. Consider pd-ssd for better performance.",
    [resource.address, node_config.disk_type]
  )
}

# =============================================================================
# Cost Validation (Optional - for use with Infracost)
# =============================================================================

# Maximum monthly cost threshold (adjust as needed)
max_monthly_cost := 50000

deny[msg] {
  # This rule requires Infracost JSON output
  input.totalMonthlyCost != null
  cost := to_number(input.totalMonthlyCost)
  cost > max_monthly_cost

  msg := sprintf(
    "Estimated monthly cost $%.2f exceeds limit of $%.2f",
    [cost, max_monthly_cost]
  )
}
