# KEDA (Kubernetes Event-driven Autoscaling)
#
# Required for scale-to-zero and demand-based autoscaling.
# KEDA queries Prometheus for the sie_router_pending_demand metric.

variable "keda_version" {
  description = "Version of KEDA Helm chart"
  type        = string
  default     = "2.14.0"
}

variable "sie_hf_token" {
  description = "HuggingFace token for gated model access (creates K8s secret automatically)"
  type        = string
  default     = ""
  sensitive   = true
}

# SIE Application Namespace
resource "kubernetes_namespace_v1" "sie" {
  metadata {
    name = var.sie_namespace

    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/name"       = "sie"
    }
  }

  depends_on = [module.eks]
}

# SIE Service Account with IRSA annotation
resource "kubernetes_service_account_v1" "sie" {
  metadata {
    name      = var.sie_service_account_name
    namespace = kubernetes_namespace_v1.sie.metadata[0].name

    annotations = {
      "eks.amazonaws.com/role-arn" = module.sie_irsa_role.iam_role_arn
    }

    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/name"       = "sie"
    }
  }

  depends_on = [
    kubernetes_namespace_v1.sie,
    module.sie_irsa_role
  ]
}

# KEDA Namespace
resource "kubernetes_namespace_v1" "keda" {
  metadata {
    name = "keda"

    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/name"       = "keda"
    }
  }

  depends_on = [module.eks]
}

resource "helm_release" "keda" {
  name             = "keda"
  repository       = "https://kedacore.github.io/charts"
  chart            = "keda"
  version          = var.keda_version
  namespace        = kubernetes_namespace_v1.keda.metadata[0].name
  create_namespace = false

  # Don't wait for pods - let them come up in background
  wait    = false
  timeout = 300

  set = [
    # Resource limits for KEDA operator
    {
      name  = "resources.operator.requests.cpu"
      value = "100m"
    },
    {
      name  = "resources.operator.requests.memory"
      value = "128Mi"
    },
    {
      name  = "resources.operator.limits.cpu"
      value = "500m"
    },
    {
      name  = "resources.operator.limits.memory"
      value = "256Mi"
    },
    # Resource limits for KEDA metrics server
    {
      name  = "resources.metricServer.requests.cpu"
      value = "100m"
    },
    {
      name  = "resources.metricServer.requests.memory"
      value = "128Mi"
    },
    {
      name  = "resources.metricServer.limits.cpu"
      value = "500m"
    },
    {
      name  = "resources.metricServer.limits.memory"
      value = "256Mi"
    },
    # Prometheus integration
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
    kubernetes_namespace_v1.keda,
    helm_release.prometheus
  ]
}

# =============================================================================
# HuggingFace Token Secret (for gated model access)
# =============================================================================

resource "kubernetes_secret_v1" "hf_token" {
  count = var.sie_hf_token != "" ? 1 : 0

  metadata {
    name      = "hf-token"
    namespace = kubernetes_namespace_v1.sie.metadata[0].name

    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/name"       = "hf-token"
    }
  }

  data = {
    token = var.sie_hf_token
  }

  depends_on = [kubernetes_namespace_v1.sie]
}
