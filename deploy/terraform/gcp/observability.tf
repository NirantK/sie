# SIE GKE Cluster - Logging Stack (Tier 2 Observability)
#
# Installs Loki (log aggregation), Alloy (log collection), and
# Event Exporter (K8s events to Loki). Optional - controlled by install_loki variable.
#
# Prerequisites: prometheus.tf must be applied first (creates monitoring namespace)

# =============================================================================
# Loki (Log Aggregation)
# =============================================================================

resource "helm_release" "loki" {
  count = var.install_loki && var.external_prometheus_url == "" ? 1 : 0

  name             = "loki"
  repository       = "https://grafana.github.io/helm-charts"
  chart            = "loki"
  version          = var.loki_version
  namespace        = "monitoring"
  create_namespace = false

  wait            = true
  timeout         = 300
  atomic          = true
  cleanup_on_fail = true

  # Values from deploy/helm/observability/loki-values.yaml
  values = [
    yamlencode({
      # Single binary mode for simplicity
      deploymentMode = "SingleBinary"

      loki = {
        # Authentication disabled for internal use
        auth_enabled = false

        # Storage configuration
        storage = {
          type = "filesystem"
        }

        # Retention
        limits_config = {
          retention_period = "168h" # 7 days
        }

        # Schema configuration
        schemaConfig = {
          configs = [
            {
              from         = "2024-01-01"
              store        = "tsdb"
              object_store = "filesystem"
              schema       = "v13"
              index = {
                prefix = "loki_index_"
                period = "24h"
              }
            }
          ]
        }

        # Common config for single binary
        commonConfig = {
          replication_factor = 1
        }
      }

      # Single binary deployment
      singleBinary = {
        replicas = 1
        resources = {
          requests = {
            cpu    = "100m"
            memory = "256Mi"
          }
          limits = {
            cpu    = "500m"
            memory = "1Gi"
          }
        }
        persistence = {
          enabled = true
          size    = "50Gi"
        }
      }

      # Disable distributed components
      read = {
        replicas = 0
      }
      write = {
        replicas = 0
      }
      backend = {
        replicas = 0
      }

      # Disable test pod
      test = {
        enabled = false
      }

      # Disable gateway (not needed for internal use)
      gateway = {
        enabled = false
      }
    })
  ]

  depends_on = [
    helm_release.prometheus
  ]
}

# =============================================================================
# Grafana Alloy (Log Collection)
# =============================================================================

resource "helm_release" "alloy" {
  count = var.install_loki && var.external_prometheus_url == "" ? 1 : 0

  name             = "alloy"
  repository       = "https://grafana.github.io/helm-charts"
  chart            = "alloy"
  version          = var.alloy_version
  namespace        = "monitoring"
  create_namespace = false

  wait            = true
  timeout         = 300
  atomic          = true
  cleanup_on_fail = true

  # Values from deploy/helm/observability/alloy-values.yaml
  values = [
    yamlencode({
      alloy = {
        configMap = {
          content = <<-EOT
            // Kubernetes log discovery
            discovery.kubernetes "pods" {
              role = "pod"
            }

            // Relabel to filter SIE pods and extract metadata
            discovery.relabel "sie_pods" {
              targets = discovery.kubernetes.pods.targets

              // Only collect logs from SIE containers
              rule {
                source_labels = ["__meta_kubernetes_pod_label_app_kubernetes_io_part_of"]
                regex         = "sie"
                action        = "keep"
              }

              // Extract namespace
              rule {
                source_labels = ["__meta_kubernetes_namespace"]
                target_label  = "namespace"
              }

              // Extract pod name
              rule {
                source_labels = ["__meta_kubernetes_pod_name"]
                target_label  = "pod"
              }

              // Extract container name
              rule {
                source_labels = ["__meta_kubernetes_pod_container_name"]
                target_label  = "container"
              }

              // Extract component (router/worker)
              rule {
                source_labels = ["__meta_kubernetes_pod_label_app_kubernetes_io_component"]
                target_label  = "component"
              }

              // Set log file path
              rule {
                source_labels = ["__meta_kubernetes_pod_uid", "__meta_kubernetes_pod_container_name"]
                target_label  = "__path__"
                separator     = "/"
                replacement   = "/var/log/pods/*$1/$2/*.log"
              }
            }

            // Collect logs from files
            loki.source.file "sie_logs" {
              targets    = discovery.relabel.sie_pods.output
              forward_to = [loki.process.sie_json.receiver]
            }

            // Parse SIE JSON logs
            loki.process "sie_json" {
              forward_to = [loki.write.default.receiver]

              stage.json {
                expressions = {
                  level      = "level",
                  logger     = "logger",
                  message    = "message",
                  model      = "model",
                  request_id = "request_id",
                  trace_id   = "trace_id",
                }
              }

              stage.labels {
                values = {
                  level  = "",
                  logger = "",
                }
              }
            }

            // Write to Loki
            loki.write "default" {
              endpoint {
                url = "http://loki:3100/loki/api/v1/push"
              }
            }
          EOT
        }
      }

      # DaemonSet to collect logs from all nodes
      controller = {
        type = "daemonset"
      }

      # Mount host log directories
      mounts = {
        varlog           = true
        dockercontainers = true
      }

      # Resources
      resources = {
        requests = {
          cpu    = "50m"
          memory = "64Mi"
        }
        limits = {
          cpu    = "200m"
          memory = "256Mi"
        }
      }

      # Run on all nodes including GPU nodes
      tolerations = [
        {
          operator = "Exists"
        }
      ]
    })
  ]

  depends_on = [
    helm_release.loki
  ]
}

# =============================================================================
# Kubernetes Event Exporter
# =============================================================================

resource "helm_release" "event_exporter" {
  count = var.install_loki && var.external_prometheus_url == "" ? 1 : 0

  name             = "event-exporter"
  repository       = "https://charts.bitnami.com/bitnami"
  chart            = "kubernetes-event-exporter"
  version          = var.event_exporter_version
  namespace        = "monitoring"
  create_namespace = false

  wait    = false # Don't block on event-exporter - it's non-critical for SIE operation
  timeout = 300

  # Values from deploy/helm/observability/event-exporter-values.yaml
  values = [
    yamlencode({
      config = {
        logLevel  = "info"
        logFormat = "json"

        route = {
          routes = [
            {
              match = [
                {
                  receiver = "loki"
                }
              ]
            }
          ]
        }

        receivers = [
          {
            name = "loki"
            webhook = {
              endpoint = "http://loki:3100/loki/api/v1/push"
              headers = {
                "Content-Type" = "application/json"
              }
              layout = {
                streams = [
                  {
                    stream = {
                      app       = "kubernetes-events"
                      namespace = "{{ .InvolvedObject.Namespace }}"
                      kind      = "{{ .InvolvedObject.Kind }}"
                      name      = "{{ .InvolvedObject.Name }}"
                      reason    = "{{ .Reason }}"
                      type      = "{{ .Type }}"
                    }
                    values = [
                      ["{{ .LastTimestamp.UnixNano }}", "{{ .Message }}"]
                    ]
                  }
                ]
              }
            }
          }
        ]
      }

      resources = {
        requests = {
          cpu    = "10m"
          memory = "32Mi"
        }
        limits = {
          cpu    = "100m"
          memory = "128Mi"
        }
      }
    })
  ]

  depends_on = [
    helm_release.loki
  ]
}

# =============================================================================
# Outputs
# =============================================================================

output "loki_url" {
  description = "Loki URL for log queries"
  value       = var.install_loki && var.external_prometheus_url == "" ? "http://loki.monitoring.svc:3100" : "N/A (Loki not installed)"
}
