# NVIDIA Device Plugin for Kubernetes
#
# Required for GPU nodes to expose nvidia.com/gpu resource to Kubernetes.
# Without this, pods cannot request GPU resources.

# Node Feature Discovery (NFD) - automatically detects and labels GPU nodes
# Required for NVIDIA device plugin to schedule on GPU nodes
resource "helm_release" "nfd" {
  name             = "node-feature-discovery"
  repository       = "https://kubernetes-sigs.github.io/node-feature-discovery/charts"
  chart            = "node-feature-discovery"
  namespace        = "node-feature-discovery"
  create_namespace = true

  wait    = false
  timeout = 300

  # NFD worker must tolerate GPU taints to label GPU nodes.
  # Without this, the NVIDIA device plugin never schedules (it requires NFD labels),
  # so nvidia.com/gpu is never advertised and GPU pods stay Pending.
  values = [
    yamlencode({
      worker = {
        tolerations = [
          {
            key      = "nvidia.com/gpu"
            operator = "Exists"
            effect   = "NoSchedule"
          }
        ]
      }
    })
  ]

  depends_on = [
    module.eks
  ]
}

resource "helm_release" "nvidia_device_plugin" {
  name             = "nvidia-device-plugin"
  repository       = "https://nvidia.github.io/k8s-device-plugin"
  chart            = "nvidia-device-plugin"
  namespace        = "kube-system"
  create_namespace = false

  # Don't wait for pods - let them come up in background
  wait    = false
  timeout = 300

  # Configure via values for complex nested structures
  values = [
    yamlencode({
      # Tolerate GPU taints so plugin can run on GPU nodes
      tolerations = [
        {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }
      ]
      # Node affinity to match NFD labels
      # NFD creates pci-0302_10de (PCI class 0302 = display controller + vendor 10de = NVIDIA)
      # This replaces the default affinity that expects pci-10de (vendor only)
      affinity = {
        nodeAffinity = {
          requiredDuringSchedulingIgnoredDuringExecution = {
            nodeSelectorTerms = [
              {
                matchExpressions = [
                  {
                    key      = "feature.node.kubernetes.io/pci-0302_10de.present"
                    operator = "In"
                    values   = ["true"]
                  }
                ]
              }
            ]
          }
        }
      }
    })
  ]

  depends_on = [
    module.eks,
    helm_release.nfd
  ]
}
