# SIE GKE Terraform Module

Creates a GKE cluster with SIE (Search Inference Engine) pre-installed, including:

- GPU node pools with KEDA autoscaling (scale-to-zero)
- Prometheus/Grafana observability stack
- Workload Identity for GCS access
- Optional git-sync for config hot reload

## Quick Start

```bash
# Dev/test cluster (L4 spot, ~$0.50/hr when active)
cd examples/dev-l4-spot
export TF_VAR_project_id="your-project"
terraform init && terraform apply

# Production cluster (multi-GPU, HA)
cd examples/production
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars
terraform init && terraform apply
```

## Examples

| Example | Description |
|---------|-------------|
| `dev-l4-spot` | Single L4 spot pool, scale-to-zero, minimal cost |
| `production` | Multi-GPU pools (L4 + A100), HA, git-sync ready |
| `eval-matrix` | Ephemeral cluster for matrix evaluation runs |

## Key Variables

### Required

| Variable | Description |
|----------|-------------|
| `project_id` | GCP project ID |
| `region` | GCP region (e.g., `us-central1`) |

### GPU Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `gpu_node_pools` | L4 spot pool | List of GPU node pool configs |
| `cpu_node_pool` | e2-standard-4 | CPU pool for system workloads |

### SIE Application

| Variable | Default | Description |
|----------|---------|-------------|
| `install_sie` | `true` | Install SIE via Helm |
| `sie_bundle` | `default` | Server bundle (default, sglang, florence2) |
| `sie_router_replicas` | `2` | Router replicas (HA) |
| `sie_autoscaling_cooldown` | `600` | KEDA cooldown (seconds) |
| `sie_router_service_type` | `ClusterIP` | Router service type (`ClusterIP` or `LoadBalancer`) |
| `sie_router_service_annotations` | `{}` | Router service annotations (e.g., internal LB) |

### Ingress Controller

| Variable | Default | Description |
|----------|---------|-------------|
| `install_ingress_nginx` | `false` | Install ingress-nginx controller |
| `ingress_nginx_class` | `nginx` | Ingress class name |
| `ingress_nginx_service_type` | `LoadBalancer` | Ingress controller service type |
| `ingress_nginx_service_annotations` | `{}` | Service annotations (internal LB, etc.) |

### Dex (Optional OIDC Issuer)

| Variable | Default | Description |
|----------|---------|-------------|
| `install_dex` | `false` | Install Dex OIDC issuer |
| `dex_chart_version` | `0.8.2` | Dex Helm chart version |
| `dex_namespace` | `dex` | Dex namespace |
| `dex_values_yaml` | `""` | Helm values YAML (must include Dex config) |

**Dex + oauth2-proxy example (programmatic auth):**

```bash
export TF_VAR_install_dex=true
export TF_VAR_dex_values_yaml="$(cat dex-values.yaml)"
export TF_VAR_sie_auth_enabled=true
export TF_VAR_sie_auth_oidc_issuer_url="http://dex.dex.svc.cluster.local:5556/dex"
```

### Ingress + Auth (Programmatic)

| Variable | Default | Description |
|----------|---------|-------------|
| `sie_ingress_enabled` | `false` | Enable ingress for router |
| `sie_ingress_class` | `nginx` | Ingress class |
| `sie_ingress_host` | `""` | Hostname (optional; empty = catch-all) |
| `sie_ingress_tls_enabled` | `false` | Enable TLS for ingress |
| `sie_ingress_tls_secret_name` | `""` | TLS secret name |
| `sie_ingress_annotations` | `{}` | Ingress annotations |
| `sie_auth_enabled` | `false` | Enable oauth2-proxy auth |
| `sie_auth_oidc_issuer_url` | `""` | OIDC issuer URL |
| `sie_auth_secret_name` | `oauth2-proxy` | Secret name with OAuth client + cookie secrets |
| `sie_auth_redirect_url` | `""` | Redirect URL (only needed for browser login) |
| `sie_auth_email_domain` | `"*"` | Email domain filter |
| `sie_auth_extra_jwt_issuers` | `[]` | Additional JWT issuer strings |
| `sie_router_auth_mode` | `none` | Router auth mode (`none` or `static`) |
| `sie_router_auth_secret_name` | `""` | Secret name for static token |
| `sie_router_auth_secret_key` | `SIE_AUTH_TOKEN` | Secret key for static token |

Minimal oauth2-proxy secret example:

```bash
kubectl create secret generic oauth2-proxy -n sie \
  --from-literal=OAUTH2_PROXY_CLIENT_ID=... \
  --from-literal=OAUTH2_PROXY_CLIENT_SECRET=... \
  --from-literal=OAUTH2_PROXY_COOKIE_SECRET=...
```

Programmatic clients should set `SIE_API_KEY` to pass a Bearer token to the SDK.

### Config Hot Reload (git-sync)

Enable git-sync to automatically update model/bundle configs without redeploying:

| Variable | Default | Description |
|----------|---------|-------------|
| `sie_git_sync_enabled` | `false` | Enable git-sync sidecar |
| `sie_git_sync_repo` | `""` | Git repo URL |
| `sie_git_sync_branch` | `main` | Branch to sync |
| `sie_git_sync_period` | `60s` | Sync interval |
| `sie_git_sync_bundles_path` | `packages/sie_server/bundles` | Path to bundles in repo |
| `sie_git_sync_models_path` | `packages/sie_server/models` | Path to models in repo |
| `sie_git_sync_ssh_secret` | `""` | K8s secret for SSH key (private repos) |

#### Public Repository Example

```hcl
module "sie_gke" {
  source = "../../"

  # ... other config ...

  sie_git_sync_enabled = true
  sie_git_sync_repo    = "https://github.com/your-org/sie-configs"
  sie_git_sync_branch  = "main"
}
```

#### Private Repository with SSH

```bash
# 1. Create SSH key for git access
ssh-keygen -t ed25519 -f ~/.ssh/sie-git-sync -N ""

# 2. Add public key to repo (GitHub: Settings > Deploy keys)

# 3. Create K8s secret
kubectl create secret generic git-ssh \
  --namespace=sie \
  --from-file=ssh=~/.ssh/sie-git-sync
```

```hcl
module "sie_gke" {
  source = "../../"

  # ... other config ...

  sie_git_sync_enabled    = true
  sie_git_sync_repo       = "git@github.com:your-org/sie-configs.git"
  sie_git_sync_ssh_secret = "git-ssh"
}
```

### Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `external_prometheus_url` | `""` | BYOI: Skip Prometheus install, use external |
| `install_loki` | `true` | Install Loki for log aggregation |
| `install_tempo` | `false` | Install Tempo for tracing |
| `enable_cloud_logging` | `true` | Enable GKE Cloud Logging |

## Outputs

| Output | Description |
|--------|-------------|
| `cluster_name` | GKE cluster name |
| `kubectl_config_command` | Command to configure kubectl |
| `artifact_registry_url` | URL for pushing Docker images |
| `prometheus_url` | Prometheus URL for queries |
| `grafana_url` | Grafana URL (port-forward to access) |
| `router_url` | Router base URL (ingress host or LB IP/hostname) |
| `router_service_name` | Router service name |
| `ingress_controller_service_name` | Ingress controller service name |
| `sie_namespace` | SIE namespace |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ GKE Cluster                                                             │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ sie namespace                                                    │   │
│  │                                                                  │   │
│  │  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │   │
│  │  │   Router     │     │   Worker     │     │   Worker     │    │   │
│  │  │  (git-sync)  │────▶│  (L4 GPU)    │     │  (A100 GPU)  │    │   │
│  │  └──────────────┘     └──────────────┘     └──────────────┘    │   │
│  │         │                    │                    │             │   │
│  │         │         ┌─────────────────────┐         │             │   │
│  │         └────────▶│  KEDA ScaledObject  │◀────────┘             │   │
│  │                   └─────────────────────┘                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ monitoring namespace                                             │   │
│  │                                                                  │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐│   │
│  │  │ Prometheus │  │  Grafana   │  │    Loki    │  │   DCGM     ││   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘│   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌────────────────┐  ┌────────────────┐                                │
│  │  CPU Node Pool │  │  GPU Node Pool │                                │
│  │  (system pods) │  │  (L4/A100)     │                                │
│  └────────────────┘  └────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Cleanup

```bash
terraform destroy
```

**Important**: GPU nodes can be expensive. Always destroy clusters when not in use.
