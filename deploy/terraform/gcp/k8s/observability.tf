# SIE GKE Cluster - Logging Stack (Tier 2 Observability)
#
# Installs Loki (log aggregation), Alloy (log collection), and
# Event Exporter (K8s events to Loki). Optional - controlled by install_loki variable.

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

  values = [
    yamlencode({
      deploymentMode = "SingleBinary"

      loki = {
        auth_enabled = false

        storage = {
          type = "filesystem"
        }

        limits_config = {
          retention_period = "168h" # 7 days
        }

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

        commonConfig = {
          replication_factor = 1
        }
      }

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

      read = {
        replicas = 0
      }
      write = {
        replicas = 0
      }
      backend = {
        replicas = 0
      }

      test = {
        enabled = false
      }

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

              rule {
                source_labels = ["__meta_kubernetes_pod_label_app_kubernetes_io_part_of"]
                regex         = "sie"
                action        = "keep"
              }

              rule {
                source_labels = ["__meta_kubernetes_namespace"]
                target_label  = "namespace"
              }

              rule {
                source_labels = ["__meta_kubernetes_pod_name"]
                target_label  = "pod"
              }

              rule {
                source_labels = ["__meta_kubernetes_pod_container_name"]
                target_label  = "container"
              }

              rule {
                source_labels = ["__meta_kubernetes_pod_label_app_kubernetes_io_component"]
                target_label  = "component"
              }

              rule {
                source_labels = ["__meta_kubernetes_pod_uid", "__meta_kubernetes_pod_container_name"]
                target_label  = "__path__"
                separator     = "/"
                replacement   = "/var/log/pods/*$1/$2/*.log"
              }
            }

            loki.source.file "sie_logs" {
              targets    = discovery.relabel.sie_pods.output
              forward_to = [loki.process.sie_json.receiver]
            }

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

            loki.write "default" {
              endpoint {
                url = "http://loki:3100/loki/api/v1/push"
              }
            }
          EOT
        }
      }

      controller = {
        type = "daemonset"
      }

      mounts = {
        varlog           = true
        dockercontainers = true
      }

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

  wait    = false
  timeout = 300

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
