#!/bin/bash
# Run from the rank-0 controller (or any node that can SSH to the mesh).
set -euo pipefail
PYTHON="${MLX_PYTHON:-$HOME/mlx-env/bin/python}"
LAUNCH="${MLX_LAUNCH:-$HOME/mlx-env/bin/mlx.launch}"
HOSTFILE="${MLX_HOSTFILE:-$HOME/mlx-cluster/jaccl-mesh.json}"
CWD="${MLX_CLUSTER_DIR:-$HOME/mlx-cluster}"

exec "$LAUNCH" --verbose --hostfile "$HOSTFILE" --cwd "$CWD" -- \
  "$PYTHON" "$@"
