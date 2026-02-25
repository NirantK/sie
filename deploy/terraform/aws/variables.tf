variable "aws_region" {
  description = "The AWS region to deploy resources in."
  type        = string
  default     = "eu-central-1"
}

variable "project_name" {
  description = "The general project name for resource naming."
  type        = string
  default     = "sie"
}

variable "server_ecr_repository_name" {
  description = "The name of the ECR repository for the sie-server."
  type        = string
  default     = "sie-server"
}

variable "router_ecr_repository_name" {
  description = "The name of the ECR repository for the sie-router."
  type        = string
  default     = "sie-router"
}

# =============================================================================
# SIE Application Configuration
# =============================================================================

variable "sie_namespace" {
  description = "Kubernetes namespace where SIE workloads run"
  type        = string
  default     = "sie"
}

variable "sie_service_account_name" {
  description = "Kubernetes ServiceAccount name for SIE workloads"
  type        = string
  default     = "sie-server"
}

variable "external_prometheus_url" {
  description = "External Prometheus URL (if set, skip internal Prometheus installation)"
  type        = string
  default     = ""
}

variable "prometheus_retention_size" {
  description = "Prometheus data retention size limit"
  type        = string
  default     = "45GB"
}

# =============================================================================
# Ingress-NGINX
# =============================================================================

variable "install_ingress_nginx" {
  description = "Install ingress-nginx controller"
  type        = bool
  default     = true
}

variable "ingress_nginx_class" {
  description = "Ingress class name for ingress-nginx"
  type        = string
  default     = "nginx"
}

variable "ingress_nginx_service_type" {
  description = "Ingress-nginx service type (LoadBalancer recommended)"
  type        = string
  default     = "LoadBalancer"
}

variable "ingress_nginx_service_annotations" {
  description = "Annotations for ingress-nginx service (e.g., internal LB)"
  type        = map(string)
  default     = {}
}
