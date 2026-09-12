#!/bin/bash
CLUSTER="${MLX_CLUSTER_DIR:-$HOME/mlx-cluster}"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
exec "$CLUSTER/libexec/mlxctl" --config "$CLUSTER/mlxctl.toml" controller
