# Prometheus & GPU Metrics (Tier 1 Observability)
#
# Required for KEDA autoscaling - KEDA queries Prometheus for scaling decisions.

variable "prometheus_stack_version" {
  description = "Version of kube-prometheus-stack Helm chart"
  type        = string
  default     = "67.4.0"
}

variable "dcgm_exporter_version" {
  description = "Version of DCGM exporter Helm chart"
  type        = string
  default     = "3.6.0"
}

variable "prometheus_retention" {
  description = "Prometheus data retention period"
  type        = string
  default     = "15d"
}

variable "prometheus_storage_size" {
  description = "Prometheus storage size"
  type        = string
  default     = "50Gi"
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  default     = "admin"
  sensitive   = true
}

# Monitoring Namespace (only if using internal Prometheus)
resource "kubernetes_namespace_v1" "monitoring" {
  count = var.external_prometheus_url == "" ? 1 : 0

  metadata {
    name = "monitoring"

    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/name"       = "monitoring"
    }
  }

  depends_on = [module.eks]
}

# Priority Class for Monitoring (only if using internal Prometheus)
resource "kubernetes_priority_class_v1" "monitoring" {
  count = var.external_prometheus_url == "" ? 1 : 0

  metadata {
    name = "monitoring-critical"
  }

  value             = 1000000000
  global_default    = false
  preemption_policy = "PreemptLowerPriority"
  description       = "High priority for monitoring infrastructure"
}

# kube-prometheus-stack (Prometheus + Grafana + Alertmanager)
resource "helm_release" "prometheus" {
  count = var.external_prometheus_url == "" ? 1 : 0

  name             = "prometheus"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  version          = var.prometheus_stack_version
  namespace        = kubernetes_namespace_v1.monitoring[0].metadata[0].name
  create_namespace = false

  # Don't wait for pods - let them come up in background
  # This avoids timeout issues during long terraform applies
  wait    = false
  timeout = 600

  values = [
    yamlencode({
      prometheus = {
        prometheusSpec = {
          serviceMonitorSelectorNilUsesHelmValues = false
          podMonitorSelectorNilUsesHelmValues     = false
          ruleSelectorNilUsesHelmValues           = false
          priorityClassName                       = kubernetes_priority_class_v1.monitoring[0].metadata[0].name
          retention                               = var.prometheus_retention
          retentionSize                           = var.prometheus_retention_size
          # Reduced for t3.medium nodes - increase for production
          resources = {
            requests = { cpu = "200m", memory = "512Mi" }
            limits   = { cpu = "1", memory = "2Gi" }
          }
          # Disable persistence to speed up initial deployment
          # Enable storageSpec for production to persist metrics across restarts
          # storageSpec = {
          #   volumeClaimTemplate = {
          #     spec = {
          #       accessModes = ["ReadWriteOnce"]
          #       resources   = { requests = { storage = var.prometheus_storage_size } }
          #     }
          #   }
          # }
        }
      }
      grafana = {
        adminPassword = var.grafana_admin_password
        resources = {
          requests = { cpu = "50m", memory = "128Mi" }
          limits   = { cpu = "200m", memory = "256Mi" }
        }
        # Disable persistence to speed up initial deployment
        persistence = { enabled = false }
      }
      alertmanager = {
        enabled = true
        alertmanagerSpec = {
          priorityClassName = kubernetes_priority_class_v1.monitoring[0].metadata[0].name
          resources = {
            requests = { cpu = "10m", memory = "64Mi" }
            limits   = { cpu = "100m", memory = "128Mi" }
          }
        }
      }
      # Reduce node-exporter and kube-state-metrics resources
      prometheus-node-exporter = {
        resources = {
          requests = { cpu = "10m", memory = "32Mi" }
          limits   = { cpu = "100m", memory = "64Mi" }
        }
      }
      kube-state-metrics = {
        resources = {
          requests = { cpu = "10m", memory = "64Mi" }
          limits   = { cpu = "100m", memory = "128Mi" }
        }
      }
      # Disable EKS-managed control plane monitoring
      kubeApiServer         = { enabled = false }
      kubeControllerManager = { enabled = false }
      kubeScheduler         = { enabled = false }
      kubeEtcd              = { enabled = false }
    })
  ]

  depends_on = [
    kubernetes_namespace_v1.monitoring,
    kubernetes_priority_class_v1.monitoring
  ]
}

# DCGM Exporter for GPU Metrics
resource "helm_release" "dcgm_exporter" {
  count = var.external_prometheus_url == "" ? 1 : 0

  name             = "dcgm-exporter"
  repository       = "https://nvidia.github.io/dcgm-exporter/helm-charts"
  chart            = "dcgm-exporter"
  version          = var.dcgm_exporter_version
  namespace        = kubernetes_namespace_v1.monitoring[0].metadata[0].name
  create_namespace = false

  # Don't wait - DCGM only runs on GPU nodes which may not exist yet
  wait    = false
  timeout = 300

  values = [
    yamlencode({
      serviceMonitor = {
        enabled  = true
        interval = "15s"
      }
      tolerations = [
        { key = "nvidia.com/gpu", operator = "Exists", effect = "NoSchedule" }
      ]
    })
  ]

  depends_on = [
    kubernetes_namespace_v1.monitoring,
    helm_release.prometheus
  ]
}

output "prometheus_url" {
  description = "Prometheus server URL for KEDA"
  value       = var.external_prometheus_url != "" ? var.external_prometheus_url : "http://prometheus-kube-prometheus-prometheus.monitoring.svc:9090"
}
