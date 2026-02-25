# Ingress-NGINX Installation
#
# Provides a turnkey ingress controller for external access to the SIE router.
# Matches GCP's ingress.tf for feature parity.

resource "helm_release" "ingress_nginx" {
  count = var.install_ingress_nginx ? 1 : 0

  name             = "ingress-nginx"
  repository       = "https://kubernetes.github.io/ingress-nginx"
  chart            = "ingress-nginx"
  version          = "4.11.3"
  namespace        = "ingress-nginx"
  create_namespace = true

  wait            = true
  timeout         = 300
  atomic          = true
  cleanup_on_fail = true

  values = [
    yamlencode({
      controller = {
        ingressClassResource = {
          name = var.ingress_nginx_class
        }
        ingressClass = var.ingress_nginx_class
        service = {
          type        = var.ingress_nginx_service_type
          annotations = var.ingress_nginx_service_annotations
        }
        publishService = {
          enabled = true
        }
      }
    })
  ]

  depends_on = [
    module.eks
  ]
}
