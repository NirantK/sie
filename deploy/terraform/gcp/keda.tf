# SIE GKE Cluster - KEDA Installation
#
# Installs KEDA (Kubernetes Event-Driven Autoscaling) for
# scaling SIE deployments based on queue depth or custom metrics.

# =============================================================================
# Helm/Kubernetes Providers Configuration
# =============================================================================

provider "kubernetes" {
  host                   = "https://${google_container_cluster.primary.endpoint}"
  token                  = data.google_client_config.current.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
}

provider "helm" {
  # Helm provider 3.x uses object syntax for kubernetes configuration
  kubernetes = {
    host                   = "https://${google_container_cluster.primary.endpoint}"
    token                  = data.google_client_config.current.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
  }
}

# =============================================================================
# KEDA Installation
# =============================================================================

resource "helm_release" "keda" {
  count = var.install_keda ? 1 : 0

  name             = "keda"
  repository       = "https://kedacore.github.io/charts"
  chart            = "keda"
  version          = var.keda_version
  namespace        = "keda"
  create_namespace = true

  # Wait for KEDA to be ready
  wait            = true
  timeout         = 300 # 5 minutes
  atomic          = true
  cleanup_on_fail = true

  # KEDA configuration (Helm provider 3.x syntax)
  set = [
    {
      name  = "resources.operator.requests.cpu"
      value = "100m"
    },
    {
      name  = "resources.operator.requests.memory"
      value = "128Mi"
    },
    {
      name  = "resources.metricServer.requests.cpu"
      value = "100m"
    },
    {
      name  = "resources.metricServer.requests.memory"
      value = "128Mi"
    },
    {
      name  = "prometheus.metricServer.enabled"
      value = "true"
    },
    {
      name  = "prometheus.operator.enabled"
      value = "true"
    }
  ]

  depends_on = [
    google_container_cluster.primary,
    google_container_node_pool.cpu
  ]
}

# =============================================================================
# SIE Namespace
# =============================================================================

resource "kubernetes_namespace_v1" "sie" {
  count = var.enable_workload_identity ? 1 : 0

  metadata {
    name = var.sie_namespace

    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/name"       = "sie"
    }
  }

  depends_on = [
    google_container_cluster.primary,
    google_container_node_pool.cpu
  ]
}

# =============================================================================
# SIE Service Account (with Workload Identity)
# =============================================================================

resource "kubernetes_service_account_v1" "sie" {
  count = var.enable_workload_identity ? 1 : 0

  metadata {
    name      = var.sie_service_account_name
    namespace = kubernetes_namespace_v1.sie[0].metadata[0].name

    annotations = {
      "iam.gke.io/gcp-service-account" = google_service_account.sie_workload[0].email
    }

    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/name"       = "sie"
    }
  }

  depends_on = [
    kubernetes_namespace_v1.sie,
    google_service_account_iam_member.sie_workload_identity
  ]
}

# =============================================================================
# HuggingFace Token Secret (for gated model access)
# =============================================================================

resource "kubernetes_secret_v1" "hf_token" {
  count = var.sie_hf_token != "" ? 1 : 0

  metadata {
    name      = "hf-token"
    namespace = kubernetes_namespace_v1.sie[0].metadata[0].name

    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/name"       = "sie"
    }
  }

  data = {
    token = var.sie_hf_token
  }

  depends_on = [
    kubernetes_namespace_v1.sie
  ]
}
