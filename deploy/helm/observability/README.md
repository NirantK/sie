# SIE Observability Stack

Complete observability for SIE GPU inference clusters: metrics, logs, events, and alerts.

## Components

| Component | Purpose | Values File |
|-----------|---------|-------------|
| kube-prometheus-stack | Metrics + Grafana + Alertmanager | `prometheus-values.yaml` |
| Loki | Log aggregation | `loki-values.yaml` |
| Alloy | Log collection (ships to Loki) | `alloy-values.yaml` |
| DCGM Exporter | GPU metrics (utilization, temp, power) | `dcgm-exporter-values.yaml` |
| Event Exporter | K8s events to Loki (scheduling failures) | `event-exporter-values.yaml` |
| Alert Rules | Pre-configured SIE alerts | `alerting-rules.yaml` |

## Quick Start

### 1. Add Helm repositories

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
```

### 2. Install observability stack

```bash
# Create namespace
kubectl create namespace monitoring

# Core: Prometheus + Grafana + Alertmanager
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring -f prometheus-values.yaml

# Logs: Loki
helm install loki grafana/loki \
  -n monitoring -f loki-values.yaml

# Log collector: Alloy (runs on all nodes)
helm install alloy grafana/alloy \
  -n monitoring -f alloy-values.yaml

# GPU metrics: DCGM Exporter (runs on GPU nodes)
helm install dcgm-exporter gpu-helm-charts/dcgm-exporter \
  -n monitoring -f dcgm-exporter-values.yaml

# K8s events: Event Exporter (for debugging scheduling failures)
helm install event-exporter bitnami/kubernetes-event-exporter \
  -n monitoring -f event-exporter-values.yaml

# SIE-specific alerts
kubectl apply -f alerting-rules.yaml -n monitoring
```

### 3. Enable SIE dashboards

In your SIE Helm values:

```yaml
dashboards:
  enabled: true
  labels:
    grafana_dashboard: "1"
  folder: SIE

serviceMonitor:
  enabled: true
```

## What You Can See

### Metrics (Prometheus/Grafana)

- Request latency (p50, p95, p99)
- Throughput (requests/sec, tokens/sec)
- Queue depth per worker
- GPU utilization, memory, temperature
- Model load times

### Logs (Loki)

```
# SIE application logs
{namespace="sie", component="worker"} |= "error"

# Find slow requests
{namespace="sie"} | json | latency_ms > 1000
```

### Events (Loki)

```
# GPU scheduling failures
{app="kubernetes-events", reason="FailedScheduling"} |= "nvidia"

# Pod scaling events
{app="kubernetes-events", namespace="sie", kind="Pod"}
```

### Alerts

| Alert | Severity | Description |
|-------|----------|-------------|
| SIENoHealthyWorkers | critical | All workers down |
| SIEWorkerHighQueueDepth | warning | Queue > 50 for 5min |
| SIEGPUMemoryHigh | warning | GPU memory > 90% |
| SIEGPUTemperatureHigh | warning | GPU temp > 80°C |
| SIEProvisioningStuck | warning | Pod pending > 10min |
| SIEHighErrorRate | warning | Error rate > 5% |

## Accessing Grafana

```bash
# Port-forward
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Get password
kubectl get secret -n monitoring prometheus-grafana \
  -o jsonpath="{.data.admin-password}" | base64 -d && echo

# Login: admin / <password>
```

## Debugging GPU Provisioning Failures

When a request times out waiting for GPU:

1. **Check K8s events in Grafana**:
   ```
   {app="kubernetes-events", reason="FailedScheduling"}
   ```

2. **Common causes**:
   - `Insufficient nvidia.com/gpu` → No GPU capacity in zone
   - `node(s) didn't match Pod's node affinity` → Wrong GPU type requested
   - `PodToleratesNodeTaints` → Missing toleration for GPU nodes

3. **Check KEDA scaling**:
   ```bash
   kubectl get scaledobject -n sie
   kubectl describe scaledobject <name> -n sie
   ```
