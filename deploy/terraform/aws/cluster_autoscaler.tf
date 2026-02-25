# Cluster Autoscaler for EKS
#
# GKE has cluster autoscaling built-in; EKS requires an explicit deployment.
# The autoscaler watches for unschedulable pods and scales node groups up/down.
# It discovers node groups via the k8s.io/cluster-autoscaler tags on eks.tf.

# IRSA role for cluster autoscaler
module "cluster_autoscaler_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.55.0"

  role_name = "${var.project_name}-cluster-autoscaler"

  attach_cluster_autoscaler_policy = true
  cluster_autoscaler_cluster_names = [module.eks.cluster_name]

  oidc_providers = {
    eks = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:cluster-autoscaler"]
    }
  }
}

resource "helm_release" "cluster_autoscaler" {
  name       = "cluster-autoscaler"
  repository = "https://kubernetes.github.io/autoscaler"
  chart      = "cluster-autoscaler"
  version    = "9.43.2"
  namespace  = "kube-system"

  values = [
    yamlencode({
      autoDiscovery = {
        clusterName = module.eks.cluster_name
      }
      awsRegion = var.aws_region
      rbac = {
        serviceAccount = {
          annotations = {
            "eks.amazonaws.com/role-arn" = module.cluster_autoscaler_irsa.iam_role_arn
          }
        }
      }
      extraArgs = {
        scale-down-unneeded-time    = "10m"
        scale-down-delay-after-add  = "10m"
      }
    })
  ]

  depends_on = [module.eks]
}
