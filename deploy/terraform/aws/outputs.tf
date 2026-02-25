output "cluster_name" {
  description = "The name of the EKS cluster."
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "The endpoint for the EKS cluster's Kubernetes API."
  value       = module.eks.cluster_endpoint
}

output "cluster_ca_certificate" {
  description = "The base64 encoded CA certificate for the EKS cluster."
  value       = module.eks.cluster_certificate_authority_data
}

# =============================================================================
# Router Access
# =============================================================================

output "router_url" {
  description = "Router URL (ingress-nginx LoadBalancer hostname)"
  value = (
    var.install_ingress_nginx ?
    (
      try(data.kubernetes_service_v1.ingress_nginx[0].status[0].load_balancer[0].ingress[0].hostname, "") != "" ?
      format("http://%s", data.kubernetes_service_v1.ingress_nginx[0].status[0].load_balancer[0].ingress[0].hostname) :
      (
        try(data.kubernetes_service_v1.ingress_nginx[0].status[0].load_balancer[0].ingress[0].ip, "") != "" ?
        format("http://%s", data.kubernetes_service_v1.ingress_nginx[0].status[0].load_balancer[0].ingress[0].ip) :
        null
      )
    ) :
    null
  )
}

data "kubernetes_service_v1" "ingress_nginx" {
  count = var.install_ingress_nginx ? 1 : 0

  metadata {
    name      = "ingress-nginx-controller"
    namespace = "ingress-nginx"
  }

  depends_on = [helm_release.ingress_nginx]
}
