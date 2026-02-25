# Kubernetes and Helm Providers for EKS
#
# Uses exec-based authentication to get fresh tokens on each API call.
# This avoids token expiration during long terraform applies (EKS cluster
# creation takes 15-20 min, but tokens expire after 15 min).
#
# try() wraps module.eks outputs so providers can configure on first apply
# before the cluster exists. The dummy values are never used in practice
# because all kubernetes/helm resources depend_on the EKS cluster.

provider "kubernetes" {
  host                   = try(module.eks.cluster_endpoint, "https://localhost")
  cluster_ca_certificate = try(base64decode(module.eks.cluster_certificate_authority_data), "")

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", var.project_name, "--region", var.aws_region]
  }
}

provider "helm" {
  kubernetes = {
    host                   = try(module.eks.cluster_endpoint, "https://localhost")
    cluster_ca_certificate = try(base64decode(module.eks.cluster_certificate_authority_data), "")

    exec = {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", var.project_name, "--region", var.aws_region]
    }
  }
}
