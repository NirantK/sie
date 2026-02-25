# SIE GKE Cluster - IAM Configuration
#
# Service accounts and IAM bindings for GKE nodes and SIE workloads.

# =============================================================================
# Service Accounts
# =============================================================================

# Service account for GKE nodes
resource "google_service_account" "gke_nodes" {
  project      = var.project_id
  account_id   = local.names.nodes_sa
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
  account_id   = local.names.workload_sa
  display_name = "SIE Workload Identity for ${var.cluster_name}"
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
