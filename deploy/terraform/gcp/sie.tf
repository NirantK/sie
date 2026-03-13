# SIE GKE Cluster - SIE Application Installation
#
# Installs the SIE Helm chart (router + worker pools) with configuration
# for autoscaling via KEDA and Prometheus metrics.
#
# Prerequisites:
#   - prometheus.tf must be applied first (KEDA needs Prometheus for scaling)
#   - Docker images must be pushed to registry
#
# =============================================================================
# IMPORTANT: Machine Profile Naming Convention
# =============================================================================
#
# The `machineProfile` value is used consistently across all components:
#
#   1. KEDA ScaledObjects query: machine_profile="<machineProfile>"
#   2. Worker pod labels: sie.superlinked.com/machine-profile: "<machineProfile>"
#   3. Router demand metrics: sie_router_pending_demand{machine_profile="<machineProfile>"}
#   4. Matrix eval configs: gpus: [<machineProfile>]
#   5. SDK pool specs: {"gpus": {"<machineProfile>": count}}
#
# machineProfile = pool.name (e.g., "l4-spot", "a100-40gb")
# gpuType = hardware type without nvidia prefix (e.g., "l4", "a100")
#
# These MUST match for KEDA autoscaling to work. If they don't match,
# workers won't scale because KEDA queries won't find the demand metrics.

# =============================================================================
# SIE Application (Router + Workers)
# =============================================================================
#
# Note: Prometheus readiness is now verified by kubernetes_job_v1.prometheus_ready_test
# in health_gates.tf, which runs inside the cluster without external auth dependencies.

resource "helm_release" "sie" {
  count = var.install_sie ? 1 : 0

  name             = "sie"
  chart            = abspath("${path.module}/../../helm/sie-cluster")
  namespace        = var.sie_namespace
  create_namespace = false # Created by keda.tf

  wait    = false # Don't block - images may not be pushed yet, matrix-eval handles readiness
  timeout = 600   # 10 min - just for helm install, not pod readiness

  # Dynamic values based on Terraform configuration
  values = [
    yamlencode({
      global = {
        namespace        = var.sie_namespace
        imagePullSecrets = var.sie_image_pull_secrets
      }

      router = {
        replicas = var.sie_router_replicas
        image = {
          repository = var.sie_router_image
          tag        = var.sie_router_image_tag != "" ? var.sie_router_image_tag : var.sie_image_tag
          pullPolicy = "IfNotPresent"
        }
        resources = var.sie_router_resources
        service = {
          type        = var.sie_router_service_type
          port        = 8080
          annotations = var.sie_router_service_annotations
        }
        auth = {
          mode            = var.sie_router_auth_mode
          tokenSecretName = var.sie_router_auth_secret_name
          tokenSecretKey  = var.sie_router_auth_secret_key
        }

        # Git-sync for config hot reload
        gitSync = {
          enabled     = var.sie_git_sync_enabled
          repo        = var.sie_git_sync_repo
          branch      = var.sie_git_sync_branch
          period      = var.sie_git_sync_period
          bundlesPath = var.sie_git_sync_bundles_path
          modelsPath  = var.sie_git_sync_models_path
          sshSecret   = var.sie_git_sync_ssh_secret
        }
      }

      workers = {
        common = {
          image = {
            repository = var.sie_server_image
            tag        = var.sie_image_tag
            pullPolicy = "IfNotPresent"
          }
          bundle          = var.sie_bundle
          cacheVolumeSize = var.sie_cache_volume_size
          clusterCache = {
            enabled = var.gcs_bucket_name != ""
            url     = var.gcs_bucket_name != "" ? "gs://${var.gcs_bucket_name}/models" : ""
          }
          hfCache = {
            home        = "/models/huggingface"
            tokenSecret = var.sie_hf_token != "" ? "hf-token" : ""
          }
        }

        pools = {
          # Generate pool configs from expanded_worker_pools (gpu_pools × bundles)
          # Each pool gets a unique name like "l4-spot-default"
          for pool_key, pool in local.expanded_worker_pools : pool.expanded_name => {
            enabled     = true
            minReplicas = pool.min_node_count
            maxReplicas = pool.max_node_count
            gpuType     = replace(pool.gpu_type, "nvidia-", "")
            # machineProfile uses ORIGINAL pool name (e.g., "l4-spot") for routing
            # Router emits demand metrics with machine_profile="l4-spot", bundle="default"
            # KEDA queries on both labels to match the right pool
            machineProfile = pool.pool_name
            # Bundle for this worker pool (used by KEDA and worker --bundle flag)
            bundle = pool.bundle
            nodeSelector = {
              "cloud.google.com/gke-accelerator" = pool.gpu_type
            }
            gpu = {
              count   = pool.gpu_count
              product = lookup(local.gpu_product_names, pool.gpu_type, "")
            }
            resources = lookup(local.worker_resources, pool.gpu_type, local.default_worker_resources)
            tolerations = [
              {
                key      = "nvidia.com/gpu"
                operator = "Exists"
                effect   = "NoSchedule"
              }
            ]
          }
        }
      }

      autoscaling = {
        enabled                = var.install_keda
        prometheusAddress      = local.prometheus_url
        cooldownPeriod         = var.sie_autoscaling_cooldown
        scaleDownStabilization = min(var.sie_autoscaling_cooldown / 2, 300)
        queueDepthThreshold    = 10
        queueDepthActivation   = 2
      }

      ingress = {
        enabled     = var.sie_ingress_enabled
        className   = var.sie_ingress_class
        annotations = var.sie_ingress_annotations
        host        = var.sie_ingress_host
        tls = {
          enabled    = var.sie_ingress_tls_enabled
          secretName = var.sie_ingress_tls_secret_name
        }
      }

      auth = {
        enabled = var.sie_auth_enabled
        oauth2Proxy = {
          image = {
            repository = var.sie_auth_oauth2_proxy_image
            tag        = var.sie_auth_oauth2_proxy_image_tag
            pullPolicy = "IfNotPresent"
          }
          secret = {
            name            = var.sie_auth_secret_name
            clientIDKey     = var.sie_auth_client_id_key
            clientSecretKey = var.sie_auth_client_secret_key
            cookieSecretKey = var.sie_auth_cookie_secret_key
          }
          oidcIssuerUrl   = var.sie_auth_oidc_issuer_url
          redirectUrl     = var.sie_auth_redirect_url
          emailDomain     = var.sie_auth_email_domain
          extraJwtIssuers = var.sie_auth_extra_jwt_issuers
        }
      }

      serviceAccount = {
        create = false # Already created by keda.tf with Workload Identity
        name   = var.sie_service_account_name
        annotations = var.enable_workload_identity ? {
          "iam.gke.io/gcp-service-account" = google_service_account.sie_workload[0].email
        } : {}
      }

      rbac = {
        create = true
      }

      serviceMonitor = {
        enabled = var.external_prometheus_url == ""
        labels = {
          "app.kubernetes.io/part-of" = "sie"
        }
      }

      dashboards = {
        enabled = var.external_prometheus_url == ""
        labels = {
          grafana_dashboard = "1"
        }
        folder = "SIE"
      }

      logging = {
        level = "INFO"
        json  = true
      }
    })
  ]

  depends_on = [
    kubernetes_namespace_v1.sie,
    kubernetes_service_account_v1.sie,
    helm_release.prometheus,
    helm_release.keda,
    kubernetes_job_v1.prometheus_ready_test, # Ensures Prometheus is serving queries (from health_gates.tf)
    kubernetes_secret_v1.hf_token,           # HF_TOKEN secret must exist before workers start
  ]
}

# =============================================================================
# Router Service Lookup (LoadBalancer)
# =============================================================================

data "kubernetes_service_v1" "router" {
  count = var.install_sie && var.sie_router_service_type == "LoadBalancer" ? 1 : 0

  metadata {
    name      = local.router_service_name
    namespace = var.sie_namespace
  }

  depends_on = [
    helm_release.sie
  ]
}

data "kubernetes_service_v1" "ingress_nginx" {
  count = var.install_ingress_nginx ? 1 : 0

  metadata {
    name      = local.ingress_service_name
    namespace = "ingress-nginx"
  }

  depends_on = [
    helm_release.ingress_nginx
  ]
}

# =============================================================================
# Local Values
# =============================================================================

locals {
  # Prometheus URL (internal or external)
  prometheus_url = var.external_prometheus_url != "" ? var.external_prometheus_url : "http://prometheus-kube-prometheus-prometheus.monitoring.svc:9090"

  router_service_name  = try("${helm_release.sie[0].name}-sie-cluster-router", "")
  ingress_service_name = try("${helm_release.ingress_nginx[0].name}-controller", "ingress-nginx-controller")

  # GPU product names for node affinity
  gpu_product_names = {
    "nvidia-l4"         = "NVIDIA-L4"
    "nvidia-tesla-t4"   = "NVIDIA-T4"
    "nvidia-tesla-a100" = "NVIDIA-A100-SXM4-40GB"
    "nvidia-a100-80gb"  = "NVIDIA-A100-SXM4-80GB"
    "nvidia-h100-80gb"  = "NVIDIA-H100-80GB-HBM3"
  }

  # Default worker resources
  default_worker_resources = {
    requests = {
      cpu    = "4"
      memory = "16Gi"
    }
    limits = {
      cpu    = "8"
      memory = "32Gi"
    }
  }

  # Worker resources by GPU type
  worker_resources = {
    "nvidia-l4" = {
      requests = {
        cpu    = "4"
        memory = "16Gi"
      }
      limits = {
        cpu    = "8"
        memory = "32Gi"
      }
    }
    "nvidia-tesla-t4" = {
      requests = {
        cpu    = "4"
        memory = "12Gi"
      }
      limits = {
        cpu    = "8"
        memory = "24Gi"
      }
    }
    "nvidia-tesla-a100" = {
      requests = {
        cpu    = "8"
        memory = "64Gi"
      }
      limits = {
        cpu    = "16"
        memory = "128Gi"
      }
    }
    "nvidia-a100-80gb" = {
      requests = {
        cpu    = "8"
        memory = "128Gi"
      }
      limits = {
        cpu    = "16"
        memory = "256Gi"
      }
    }
  }

  # Effective bundles list: use sie_bundles if set, else fall back to [sie_bundle]
  effective_bundles = length(var.sie_bundles) > 0 ? var.sie_bundles : [var.sie_bundle]

  # Expand GPU pools × bundles into worker pool configs
  # Each (gpu_pool, bundle) pair becomes a separate worker StatefulSet
  # Example: l4-spot × [default] → l4-spot-default
  expanded_worker_pools = merge([
    for pool in var.gpu_node_pools : {
      for bundle in local.effective_bundles :
      "${pool.name}-${bundle}" => {
        # Original pool properties
        pool_name      = pool.name
        machine_type   = pool.machine_type
        gpu_type       = pool.gpu_type
        gpu_count      = pool.gpu_count
        min_node_count = pool.min_node_count
        max_node_count = pool.max_node_count
        disk_size_gb   = pool.disk_size_gb
        disk_type      = pool.disk_type
        spot           = pool.spot
        zones          = pool.zones
        taints         = pool.taints
        labels         = pool.labels
        # Bundle-specific properties
        bundle        = bundle
        expanded_name = "${pool.name}-${bundle}"
      }
    }
  ]...)
}

# =============================================================================
# Outputs
# =============================================================================

output "sie_router_service" {
  description = "SIE router service name for internal access"
  value       = var.install_sie ? "sie-router.${var.sie_namespace}.svc:8080" : "N/A (SIE not installed)"
}

output "sie_cluster_status" {
  description = "Command to check SIE cluster status"
  value       = var.install_sie ? "kubectl get pods -n ${var.sie_namespace}" : "N/A (SIE not installed)"
}

# =============================================================================
# Health Gates
# =============================================================================
#
# Cluster health verification is now handled by kubernetes_job_v1 resources
# in health_gates.tf. These run inside the cluster without external auth
# dependencies (no gke-gcloud-auth-plugin required).
#
# See health_gates.tf for:
# - prometheus_ready_test: Verifies Prometheus query API
# - scaledobject_health_test: Verifies KEDA ScaledObjects not in Fallback
# - router_ready_test: Verifies router pods are healthy
