# AWS EKS Deployment Gap Analysis

Based on analysis of the AWS setup and the GCP deployment architecture.

---

## Implementation Progress

| Task | Status | File(s) |
|------|--------|---------|
| EKS cluster hardening (KMS, logging) | ✅ Done | `eks.tf` |
| GPU node pool (taints, scale-to-zero) | ✅ Done | `eks.tf` |
| NVIDIA device plugin | ✅ Done | `nvidia.tf` |
| IRSA configuration | ✅ Done | `irsa.tf` |
| KEDA installation | ✅ Done | `keda.tf` |
| Prometheus + Grafana + DCGM | ✅ Done | `prometheus.tf` |
| `mise run aws-deploy` task | ✅ Done | `tools/mise_tasks/aws_deploy.py` |
| `mise run aws-docker` task | ✅ Done | `tools/mise_tasks/aws_docker.py` |
| SIE Helm deployment | ✅ Done | via `mise run aws-deploy` |
| Health gate verification | N/A | Not needed (deployment via mise, not Terraform) |

---

## 1. Production Hardening Assessment

| Area | Status | Notes |
|------|--------|-------|
| **API Endpoint** | ⚠️ Open | Public, open to 0.0.0.0/0 (TODO: restrict when VPN/bastion available) |
| **Secrets Encryption** | ✅ Done | KMS encryption enabled for etcd secrets |
| **Cluster Logging** | ✅ Done | audit, api, authenticator, controllerManager, scheduler |
| **IRSA** | ✅ Done | ECR pull permissions configured |
| **GPU Node Pool** | ✅ Done | min_size=0, taints configured |
| **NVIDIA Plugin** | ✅ Done | Helm chart installed |

---

## 2. Deployment Gap Analysis (AWS vs GCP)

| Component | GCP Status | AWS Status |
|-----------|-----------|------------|
| **KEDA** | Installed via Terraform | ✅ Installed |
| **Prometheus + Grafana** | kube-prometheus-stack | ✅ Installed |
| **DCGM (GPU metrics)** | Installed | ✅ Installed |
| **SIE Helm Chart** | Auto-deployed | ✅ via `mise run aws-deploy` |
| **IRSA/Workload Identity** | Configured | ✅ Configured |
| **HuggingFace Secret** | Created | ✅ via `sie_hf_token` variable |
| **Health Gates** | Verification jobs | N/A (Terraform-managed in GCP only) |
| **Scale-to-zero** | min=0 + KEDA | ✅ Configured |

---

## 3. Deployment Commands

```bash
# Build and push images to ECR
mise run aws-docker --tag v1.0.0

# Deploy to EKS
mise run aws-deploy --tag v1.0.0

# Preview deployment (dry-run)
mise run aws-deploy --tag v1.0.0 --dry-run
```

### Optional enhancements:
- Create `sie.tf` for Terraform-managed SIE deployment (like GCP)
- Add health gate verification jobs to Terraform
- Restrict API endpoint when VPN/bastion is available
