# SIE GKE Cluster - Prometheus & GPU Metrics (Tier 1 Observability)
#
# Installs kube-prometheus-stack (Prometheus + Grafana + Alertmanager) and
# DCGM Exporter for GPU metrics. These are REQUIRED for KEDA autoscaling
# to function - KEDA queries Prometheus for scaling decisions.

# =============================================================================
# Monitoring Namespace
# =============================================================================

resource "kubernetes_namespace_v1" "monitoring" {
  count = var.external_prometheus_url == "" ? 1 : 0

  metadata {
    name = "monitoring"

    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/name"       = "monitoring"
    }
  }
}

# =============================================================================
# Priority Class for Monitoring
# =============================================================================

resource "kubernetes_priority_class_v1" "monitoring" {
  count = var.external_prometheus_url == "" ? 1 : 0

  metadata {
    name = "monitoring-critical"
  }

  value             = 1000000000
  global_default    = false
  preemption_policy = "PreemptLowerPriority"
  description       = "High priority for monitoring infrastructure (Prometheus, KEDA dependency)"
}

# =============================================================================
# kube-prometheus-stack (Prometheus + Grafana + Alertmanager)
# =============================================================================

resource "helm_release" "prometheus" {
  count = var.external_prometheus_url == "" ? 1 : 0

  name             = "prometheus"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  version          = var.prometheus_stack_version
  namespace        = kubernetes_namespace_v1.monitoring[0].metadata[0].name
  create_namespace = false

  wait            = true
  timeout         = 300 # 5 minutes
  atomic          = true
  cleanup_on_fail = true

  values = [
    yamlencode({
      prometheus = {
        prometheusSpec = {
          # Discover all ServiceMonitors (not just helm-labeled ones)
          serviceMonitorSelectorNilUsesHelmValues = false
          podMonitorSelectorNilUsesHelmValues     = false
          ruleSelectorNilUsesHelmValues           = false

          priorityClassName = kubernetes_priority_class_v1.monitoring[0].metadata[0].name

          retention     = var.prometheus_retention
          retentionSize = var.prometheus_retention_size

          resources = {
            requests = {
              cpu    = "500m"
              memory = "2Gi"
            }
            limits = {
              cpu    = "2"
              memory = "8Gi"
            }
          }

          storageSpec = {
            volumeClaimTemplate = {
              spec = {
                accessModes = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = var.prometheus_storage_size
                  }
                }
              }
            }
          }
        }
      }

      grafana = {
        adminPassword = var.grafana_admin_password

        sidecar = {
          dashboards = {
            enabled          = true
            label            = "grafana_dashboard"
            labelValue       = "1"
            folderAnnotation = "grafana_folder"
            provider = {
              foldersFromFilesStructure = true
            }
          }
          datasources = {
            enabled = true
          }
        }

        # Add Loki datasource if Loki is installed
        additionalDataSources = var.install_loki ? [
          {
            name      = "Loki"
            type      = "loki"
            url       = "http://loki.monitoring:3100"
            access    = "proxy"
            isDefault = false
          }
        ] : []

        resources = {
          requests = {
            cpu    = "100m"
            memory = "256Mi"
          }
          limits = {
            cpu    = "500m"
            memory = "512Mi"
          }
        }

        persistence = {
          enabled = true
          size    = "10Gi"
        }
      }

      alertmanager = {
        enabled = true
        alertmanagerSpec = {
          priorityClassName = kubernetes_priority_class_v1.monitoring[0].metadata[0].name
          resources = {
            requests = {
              cpu    = "50m"
              memory = "64Mi"
            }
            limits = {
              cpu    = "100m"
              memory = "128Mi"
            }
          }
        }
      }

      # Enable node metrics
      nodeExporter = {
        enabled = true
      }

      kubeStateMetrics = {
        enabled = true
      }

      # Disable control plane monitoring (managed by GKE)
      kubeApiServer = {
        enabled = false
      }
      kubeControllerManager = {
        enabled = false
      }
      kubeScheduler = {
        enabled = false
      }
      kubeEtcd = {
        enabled = false
      }
    })
  ]

  depends_on = [
    kubernetes_namespace_v1.monitoring,
    kubernetes_priority_class_v1.monitoring,
    helm_release.keda
  ]
}

# =============================================================================
# DCGM Exporter (GPU Metrics)
# =============================================================================

resource "helm_release" "dcgm_exporter" {
  count = var.external_prometheus_url == "" ? 1 : 0

  name             = "dcgm-exporter"
  repository       = "https://nvidia.github.io/dcgm-exporter/helm-charts"
  chart            = "dcgm-exporter"
  version          = var.dcgm_exporter_version
  namespace        = kubernetes_namespace_v1.monitoring[0].metadata[0].name
  create_namespace = false

  wait            = true
  timeout         = 300
  atomic          = true
  cleanup_on_fail = true

  values = [
    yamlencode({
      serviceMonitor = {
        enabled  = true
        interval = "15s"
        additionalLabels = {
          "app.kubernetes.io/part-of" = "sie"
        }
      }

      # Run on GPU nodes only
      nodeSelector = {
        "cloud.google.com/gke-accelerator-count" = "1"
      }

      tolerations = [
        {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }
      ]

      resources = {
        requests = {
          cpu    = "50m"
          memory = "64Mi"
        }
        limits = {
          cpu    = "200m"
          memory = "128Mi"
        }
      }

      arguments = ["-f", "/etc/dcgm-exporter/dcp-metrics-included.csv"]
    })
  ]

  depends_on = [
    kubernetes_namespace_v1.monitoring,
    helm_release.prometheus
  ]
}
