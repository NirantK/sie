# SIE GKE Cluster - Kubernetes Module Outputs

# =============================================================================
# Router Access
# =============================================================================

output "router_url" {
  description = "Router URL (ingress host or LoadBalancer IP/hostname)"
  value = (
    var.sie_ingress_enabled && var.sie_ingress_host != "" ?
    format("%s://%s", var.sie_ingress_tls_enabled ? "https" : "http", var.sie_ingress_host) :
    (
      var.sie_ingress_enabled && var.install_ingress_nginx ?
      (
        try(data.kubernetes_service_v1.ingress_nginx[0].status[0].load_balancer[0].ingress[0].ip, "") != "" ?
        format("http://%s", data.kubernetes_service_v1.ingress_nginx[0].status[0].load_balancer[0].ingress[0].ip) :
        (
          try(data.kubernetes_service_v1.ingress_nginx[0].status[0].load_balancer[0].ingress[0].hostname, "") != "" ?
          format("http://%s", data.kubernetes_service_v1.ingress_nginx[0].status[0].load_balancer[0].ingress[0].hostname) :
          null
        )
      ) :
      (
        var.sie_router_service_type == "LoadBalancer" ?
        (
          try(data.kubernetes_service_v1.router[0].status[0].load_balancer[0].ingress[0].ip, "") != "" ?
          format("http://%s", data.kubernetes_service_v1.router[0].status[0].load_balancer[0].ingress[0].ip) :
          (
            try(data.kubernetes_service_v1.router[0].status[0].load_balancer[0].ingress[0].hostname, "") != "" ?
            format("http://%s", data.kubernetes_service_v1.router[0].status[0].load_balancer[0].ingress[0].hostname) :
            null
          )
        ) :
        null
      )
    )
  )
}

output "router_service_name" {
  description = "Router service name"
  value       = local.router_service_name
}

output "sie_namespace" {
  description = "SIE namespace"
  value       = var.sie_namespace
}

output "ingress_controller_service_name" {
  description = "Ingress controller service name (if installed)"
  value       = local.ingress_service_name
}

# =============================================================================
# Observability URLs
# =============================================================================

output "prometheus_url" {
  description = "Prometheus server URL for KEDA and internal queries"
  value       = var.external_prometheus_url != "" ? var.external_prometheus_url : "http://prometheus-kube-prometheus-prometheus.monitoring.svc:9090"
}

output "grafana_url" {
  description = "Grafana URL (port-forward to access)"
  value       = var.external_prometheus_url != "" ? var.external_grafana_url : "http://prometheus-grafana.monitoring.svc:80"
}

output "loki_url" {
  description = "Loki URL for log queries"
  value       = var.install_loki && var.external_prometheus_url == "" ? "http://loki.monitoring.svc:3100" : "N/A (Loki not installed)"
}

# =============================================================================
# Status
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
# Health Tests
# =============================================================================

output "health_tests_status" {
  description = "Status of health test jobs"
  value = var.install_sie && var.install_keda ? {
    prometheus_test   = try(kubernetes_job_v1.prometheus_ready_test[0].metadata[0].name, "N/A")
    scaledobject_test = try(kubernetes_job_v1.scaledobject_health_test[0].metadata[0].name, "N/A")
    router_test       = try(kubernetes_job_v1.router_ready_test[0].metadata[0].name, "N/A")
  } : null
}
