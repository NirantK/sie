# SIE GKE Cluster - Artifact Registry
#
# Docker image repository for SIE images with cleanup policies.

# =============================================================================
# Artifact Registry
# =============================================================================

resource "google_artifact_registry_repository" "sie" {
  count = var.create_artifact_registry ? 1 : 0

  project       = var.project_id
  location      = var.artifact_registry_location != "" ? var.artifact_registry_location : var.region
  repository_id = local.names.registry
  description   = "SIE Docker images for ${var.cluster_name}"
  format        = "DOCKER"

  labels = local.resource_labels

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
