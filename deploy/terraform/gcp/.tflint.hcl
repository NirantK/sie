# TFLint Configuration for SIE GKE Terraform
#
# Run with: tflint --init && tflint
# Install: brew install tflint (macOS) or see https://github.com/terraform-linters/tflint

config {
  # Enable all available rules by default
  module = true
  force = false
}

# =============================================================================
# Google Cloud Provider Plugin
# =============================================================================

plugin "google" {
  enabled = true
  version = "0.28.0"
  source  = "github.com/terraform-linters/tflint-ruleset-google"
}

# =============================================================================
# Terraform Language Rules
# =============================================================================

plugin "terraform" {
  enabled = true
  preset  = "recommended"
}

# Enforce consistent naming conventions
rule "terraform_naming_convention" {
  enabled = true

  # Resource naming (snake_case)
  resource {
    format = "snake_case"
  }

  # Variable naming (snake_case)
  variable {
    format = "snake_case"
  }

  # Output naming (snake_case)
  output {
    format = "snake_case"
  }
}

# Ensure all variables have descriptions
rule "terraform_documented_variables" {
  enabled = true
}

# Ensure all outputs have descriptions
rule "terraform_documented_outputs" {
  enabled = true
}

# Enforce standard module structure
rule "terraform_standard_module_structure" {
  enabled = true
}

# Warn on deprecated syntax
rule "terraform_deprecated_index" {
  enabled = true
}

# Warn on deprecated interpolation
rule "terraform_deprecated_interpolation" {
  enabled = true
}

# Ensure required providers are versioned
rule "terraform_required_providers" {
  enabled = true
}

# Ensure terraform version constraint
rule "terraform_required_version" {
  enabled = true
}

# =============================================================================
# Google Cloud Specific Rules
# =============================================================================

# Ensure GKE clusters use VPC-native networking
rule "google_container_cluster_node_pool" {
  enabled = true
}

# Ensure compute instances use appropriate machine types
rule "google_compute_instance_invalid_machine_type" {
  enabled = true
}
