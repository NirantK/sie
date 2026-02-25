# SIE GKE Cluster - Kubernetes Module Main
#
# KEDA installation, SIE namespace, and service account configuration.
# This module requires the infra module to be applied first.

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
      "iam.gke.io/gcp-service-account" = var.sie_workload_service_account
    }

    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/name"       = "sie"
    }
  }

  depends_on = [
    kubernetes_namespace_v1.sie
  ]
}

# =============================================================================
# HuggingFace Token Secret (for gated model access)
# =============================================================================

resource "kubernetes_secret_v1" "hf_token" {
  count = var.sie_hf_token != "" && var.enable_workload_identity ? 1 : 0

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
