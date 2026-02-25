# SIE GKE Cluster - Health Gate Jobs
#
# Kubernetes Jobs that verify cluster components are ready before Terraform completes.

# =============================================================================
# Prometheus Ready Test
# =============================================================================

resource "kubernetes_service_account_v1" "health_prometheus" {
  count = var.install_sie && var.install_keda ? 1 : 0

  metadata {
    name      = "sie-health-prometheus"
    namespace = "monitoring"
  }

  depends_on = [helm_release.prometheus]
}

resource "kubernetes_job_v1" "prometheus_ready_test" {
  count = var.install_sie && var.install_keda ? 1 : 0

  metadata {
    name      = "sie-prometheus-health-${try(helm_release.prometheus[0].metadata.revision, "0")}"
    namespace = "monitoring"
    labels = {
      "app.kubernetes.io/name"      = "sie-health-test"
      "app.kubernetes.io/component" = "prometheus"
    }
  }

  spec {
    backoff_limit              = 0
    active_deadline_seconds    = 660
    ttl_seconds_after_finished = 300

    template {
      metadata {
        labels = {
          "app.kubernetes.io/name"      = "sie-health-test"
          "app.kubernetes.io/component" = "prometheus"
        }
      }

      spec {
        service_account_name = kubernetes_service_account_v1.health_prometheus[0].metadata[0].name
        restart_policy       = "Never"

        container {
          name  = "test"
          image = "alpine:3.19"

          command = ["/bin/sh", "-c"]
          args = [<<-EOT
            set -e
            apk add --no-cache jq >/dev/null 2>&1
            PROMETHEUS_URL="http://prometheus-kube-prometheus-prometheus.monitoring.svc:9090"
            TIMEOUT=600
            INTERVAL=5
            echo "=== Prometheus Readiness Test ==="
            elapsed=0
            while [ $elapsed -lt $TIMEOUT ]; do
              response=$(wget -qO- --timeout=5 "$${PROMETHEUS_URL}/api/v1/query?query=up" 2>/dev/null || echo "{}")
              if echo "$response" | jq -e '.status == "success"' >/dev/null 2>&1; then
                echo "SUCCESS: Prometheus query API is ready"
                exit 0
              fi
              echo "Waiting... ($${elapsed}s/$${TIMEOUT}s)"
              sleep $INTERVAL
              elapsed=$((elapsed + INTERVAL))
            done
            echo "FAILED: Prometheus did not become ready within $${TIMEOUT}s"
            exit 1
          EOT
          ]

          resources {
            requests = {
              cpu    = "10m"
              memory = "32Mi"
            }
            limits = {
              cpu    = "100m"
              memory = "64Mi"
            }
          }
        }

        toleration {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }
      }
    }
  }

  wait_for_completion = true

  timeouts {
    create = "12m"
  }

  depends_on = [
    helm_release.prometheus,
    helm_release.keda,
    kubernetes_service_account_v1.health_prometheus
  ]
}

# =============================================================================
# ScaledObject Health Test
# =============================================================================

resource "kubernetes_service_account_v1" "health_scaledobject" {
  count = var.install_sie && var.install_keda ? 1 : 0

  metadata {
    name      = "sie-health-scaledobject"
    namespace = var.sie_namespace
  }

  depends_on = [kubernetes_namespace_v1.sie]
}

resource "kubernetes_cluster_role_v1" "health_scaledobject" {
  count = var.install_sie && var.install_keda ? 1 : 0

  metadata {
    name = "sie-health-scaledobject-reader-${var.cluster_name}"
    labels = {
      "app.kubernetes.io/name"      = "sie-health-test"
      "app.kubernetes.io/component" = "scaledobject"
    }
  }

  rule {
    api_groups = ["keda.sh"]
    resources  = ["scaledobjects"]
    verbs      = ["get", "list"]
  }
}

resource "kubernetes_cluster_role_binding_v1" "health_scaledobject" {
  count = var.install_sie && var.install_keda ? 1 : 0

  metadata {
    name = "sie-health-scaledobject-reader-${var.cluster_name}"
    labels = {
      "app.kubernetes.io/name"      = "sie-health-test"
      "app.kubernetes.io/component" = "scaledobject"
    }
  }

  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "ClusterRole"
    name      = kubernetes_cluster_role_v1.health_scaledobject[0].metadata[0].name
  }

  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account_v1.health_scaledobject[0].metadata[0].name
    namespace = var.sie_namespace
  }
}

resource "kubernetes_job_v1" "scaledobject_health_test" {
  count = var.install_sie && var.install_keda ? 1 : 0

  metadata {
    name      = "sie-scaledobject-health-${try(helm_release.sie[0].metadata.revision, "0")}"
    namespace = var.sie_namespace
    labels = {
      "app.kubernetes.io/name"      = "sie-health-test"
      "app.kubernetes.io/component" = "scaledobject"
    }
  }

  spec {
    backoff_limit              = 0
    active_deadline_seconds    = 240
    ttl_seconds_after_finished = 300

    template {
      metadata {
        labels = {
          "app.kubernetes.io/name"      = "sie-health-test"
          "app.kubernetes.io/component" = "scaledobject"
        }
      }

      spec {
        service_account_name = kubernetes_service_account_v1.health_scaledobject[0].metadata[0].name
        restart_policy       = "Never"

        container {
          name  = "test"
          image = "alpine:3.19"

          command = ["/bin/sh", "-c"]
          args = [<<-EOT
            set -e
            apk add --no-cache curl jq >/dev/null 2>&1
            NAMESPACE="${var.sie_namespace}"
            TIMEOUT=180
            INTERVAL=5
            TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
            CA_CERT=/var/run/secrets/kubernetes.io/serviceaccount/ca.crt
            API_SERVER="https://kubernetes.default.svc"
            API_PATH="/apis/keda.sh/v1alpha1/namespaces/$${NAMESPACE}/scaledobjects"
            echo "=== ScaledObject Health Test ==="
            k8s_get() {
              curl -sS --cacert "$${CA_CERT}" -H "Authorization: Bearer $${TOKEN}" "$${API_SERVER}$${API_PATH}" 2>/dev/null || echo "{}"
            }
            elapsed=0
            while [ $elapsed -lt $TIMEOUT ]; do
              response=$(k8s_get)
              count=$(echo "$response" | jq -r '.items | length // 0')
              if [ "$count" -eq "0" ]; then
                echo "Waiting for ScaledObjects... ($${elapsed}s/$${TIMEOUT}s)"
                sleep $INTERVAL
                elapsed=$((elapsed + INTERVAL))
                continue
              fi
              fallback_count=$(echo "$response" | jq -r '[.items[].status.conditions[]? | select(.type=="Fallback" and .status=="True")] | length')
              if [ "$fallback_count" -gt "0" ]; then
                echo "FAILED: ScaledObjects in Fallback mode"
                exit 1
              fi
              ready_count=$(echo "$response" | jq -r '[.items[].status.conditions[]? | select(.type=="Ready" and .status=="True")] | length')
              not_ready_count=$(echo "$response" | jq -r '[.items[].status.conditions[]? | select(.type=="Ready" and .status!="True")] | length')
              if [ "$ready_count" -gt "0" ] && [ "$not_ready_count" -eq "0" ]; then
                echo "SUCCESS: All ScaledObjects are Ready"
                exit 0
              fi
              echo "Waiting... ($${elapsed}s/$${TIMEOUT}s)"
              sleep $INTERVAL
              elapsed=$((elapsed + INTERVAL))
            done
            echo "FAILED: ScaledObjects did not become Ready"
            exit 1
          EOT
          ]

          resources {
            requests = {
              cpu    = "10m"
              memory = "32Mi"
            }
            limits = {
              cpu    = "100m"
              memory = "64Mi"
            }
          }
        }

        toleration {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }
      }
    }
  }

  wait_for_completion = true

  timeouts {
    create = "5m"
  }

  depends_on = [
    helm_release.sie,
    kubernetes_cluster_role_binding_v1.health_scaledobject
  ]
}

# =============================================================================
# Router Ready Test
# =============================================================================

resource "kubernetes_service_account_v1" "health_router" {
  count = var.install_sie && var.install_keda ? 1 : 0

  metadata {
    name      = "sie-health-router"
    namespace = var.sie_namespace
  }

  depends_on = [kubernetes_namespace_v1.sie]
}

resource "kubernetes_job_v1" "router_ready_test" {
  count = var.install_sie && var.install_keda ? 1 : 0

  metadata {
    name      = "sie-router-health-${try(helm_release.sie[0].metadata.revision, "0")}"
    namespace = var.sie_namespace
    labels = {
      "app.kubernetes.io/name"      = "sie-health-test"
      "app.kubernetes.io/component" = "router"
    }
  }

  spec {
    backoff_limit              = 0
    active_deadline_seconds    = 360
    ttl_seconds_after_finished = 300

    template {
      metadata {
        labels = {
          "app.kubernetes.io/name"      = "sie-health-test"
          "app.kubernetes.io/component" = "router"
        }
      }

      spec {
        service_account_name = kubernetes_service_account_v1.health_router[0].metadata[0].name
        restart_policy       = "Never"

        container {
          name  = "test"
          image = "alpine:3.19"

          command = ["/bin/sh", "-c"]
          args = [<<-EOT
            set -e
            apk add --no-cache jq >/dev/null 2>&1
            ROUTER_SERVICE="sie-sie-cluster-router.${var.sie_namespace}.svc:8080"
            TIMEOUT=300
            INTERVAL=5
            echo "=== Router Readiness Test ==="
            elapsed=0
            while [ $elapsed -lt $TIMEOUT ]; do
              response=$(wget -qO- --timeout=5 "http://$${ROUTER_SERVICE}/healthz" 2>/dev/null || echo "{}")
              if echo "$response" | jq -e '.status == "ok"' >/dev/null 2>&1; then
                echo "SUCCESS: Router is healthy"
                exit 0
              fi
              echo "Waiting... ($${elapsed}s/$${TIMEOUT}s)"
              sleep $INTERVAL
              elapsed=$((elapsed + INTERVAL))
            done
            echo "FAILED: Router did not become ready"
            exit 1
          EOT
          ]

          resources {
            requests = {
              cpu    = "10m"
              memory = "32Mi"
            }
            limits = {
              cpu    = "100m"
              memory = "64Mi"
            }
          }
        }

        toleration {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }
      }
    }
  }

  wait_for_completion = true

  timeouts {
    create = "7m"
  }

  depends_on = [
    helm_release.sie,
    kubernetes_service_account_v1.health_router
  ]
}
