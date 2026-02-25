# Centralized naming convention for SIE GCP resources.
# This file is the single source of truth - read by both Terraform and Python.
#
# Terraform: uses local.name_suffixes in infra/main.tf
# Python: parses this file with python-hcl2 in tools/mise_tasks/cluster.py

locals {
  name_suffixes = {
    cluster     = ""
    network     = "-network"
    subnet      = "-subnet"
    router      = "-router"
    nat         = "-nat"
    nodes_sa    = "-nodes"
    workload_sa = "-workload"
    registry    = "-images"
  }
}
