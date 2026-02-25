# SIE GKE Cluster - Kubernetes Module Providers
#
# Kubernetes and Helm providers configured using VARIABLES passed from the infra module.
# This is the key difference from the original module - providers no longer reference
# the cluster resource directly, eliminating the chicken-and-egg problem.
#
# The infra module must be applied first, and its outputs passed to this module
# as variables (via TF_VAR_* environment variables from mise tasks).

terraform {
  required_version = "~> 1.14.3"

  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 3.0.1"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 3.1.1"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 7.16.0"
    }
  }
}

# =============================================================================
# Data Sources
# =============================================================================

# Get current GCP client config for auth token
data "google_client_config" "current" {}

# =============================================================================
# Provider Configuration
# =============================================================================

provider "kubernetes" {
  host                   = "https://${var.cluster_endpoint}"
  token                  = data.google_client_config.current.access_token
  cluster_ca_certificate = base64decode(var.cluster_ca_certificate)
}

provider "helm" {
  kubernetes = {
    host                   = "https://${var.cluster_endpoint}"
    token                  = data.google_client_config.current.access_token
    cluster_ca_certificate = base64decode(var.cluster_ca_certificate)
  }
}
