# Dex Installation (optional OIDC issuer)

resource "helm_release" "dex" {
  count = var.install_dex ? 1 : 0

  name             = "dex"
  repository       = "https://charts.dexidp.io"
  chart            = "dex"
  version          = var.dex_chart_version
  namespace        = var.dex_namespace
  create_namespace = true

  wait            = true
  timeout         = 300
  atomic          = true
  cleanup_on_fail = true

  values = [
    var.dex_values_yaml
  ]
}
