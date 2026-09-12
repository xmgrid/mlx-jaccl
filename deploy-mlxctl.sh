#!/bin/bash
# Build mlxctl and install agent/controller + launchd on every node in mlxctl.toml.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN_SRC="$ROOT/mlxctl/target/release/mlxctl"
SUDO_PW="${MLXCTL_SUDO_PASSWORD:?set MLXCTL_SUDO_PASSWORD}"
SSH_USER="${MLXCTL_SSH_USER:-${USER:?}}"
CONFIG="${MLXCTL_CONFIG:-$ROOT/mlxctl.toml}"
REMOTE="${MLXCTL_REMOTE:-$HOME/mlx-cluster}"
PYTHON_REMOTE="${MLX_PYTHON:-$HOME/mlx-env/bin/python}"
UV_REMOTE="${MLX_UV:-$HOME/.local/bin/uv}"

if [[ ! -x "$BIN_SRC" ]]; then
  echo "missing $BIN_SRC — cargo build --release first" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "missing $CONFIG — copy mlxctl.toml.example to mlxctl.toml" >&2
  exit 1
fi

mapfile -t NODES < <(/usr/bin/python3 - "$CONFIG" <<'PY'
import sys, tomllib
from pathlib import Path
cfg = tomllib.loads(Path(sys.argv[1]).read_text())
for n in cfg["nodes"]:
    print(n["ssh"])
PY
)
CONTROLLER="$(/usr/bin/python3 - "$CONFIG" <<'PY'
import sys, tomllib
from pathlib import Path
cfg = tomllib.loads(Path(sys.argv[1]).read_text())
nodes = cfg["nodes"]
rank0 = next((n for n in nodes if n.get("rank") == 0), nodes[0])
print(rank0["ssh"])
PY
)"

TOKEN="$(python3 - <<'PY'
import secrets
print(secrets.token_hex(16))
PY
)"

if ssh -o BatchMode=yes "${SSH_USER}@${CONTROLLER}" "test -f $REMOTE/mlxctl.toml"; then
  OLD="$(ssh -o BatchMode=yes "${SSH_USER}@${CONTROLLER}" "awk -F'\"' '/^token /{print \$2}' $REMOTE/mlxctl.toml" || true)"
  if [[ -n "${OLD:-}" && "$OLD" != "CHANGE_ME" ]]; then
    TOKEN="$OLD"
  fi
fi

TMP="$(mktemp)"
sed "s/token = \"CHANGE_ME\"/token = \"$TOKEN\"/" "$CONFIG" > "$TMP"

for ip in "${NODES[@]}"; do
  echo "==== deploy $ip ===="
  ssh -o BatchMode=yes "${SSH_USER}@${ip}" "mkdir -p $REMOTE/libexec $REMOTE/bin $REMOTE/ui $REMOTE/logs"
  scp -o BatchMode=yes "$BIN_SRC" "${SSH_USER}@${ip}":$REMOTE/libexec/mlxctl
  ssh -o BatchMode=yes "${SSH_USER}@${ip}" "xattr -cr $REMOTE/libexec/mlxctl && ln -sfn $REMOTE/libexec/mlxctl $REMOTE/bin/mlxctl && chmod +x $REMOTE/libexec/mlxctl $REMOTE/setup-tb.sh"
  scp -o BatchMode=yes "$TMP" "${SSH_USER}@${ip}":$REMOTE/mlxctl.toml
  scp -o BatchMode=yes "$ROOT/mlxctl/python/serve_entry.py" \
    "$ROOT/setup-tb.sh" "$ROOT/jaccl-mesh.json" "${SSH_USER}@${ip}":$REMOTE/
  scp -o BatchMode=yes "$ROOT/mlxctl/web/index.html" "$ROOT/mlxctl/web/styles.css" \
    "$ROOT/mlxctl/web/app.js" "${SSH_USER}@${ip}":$REMOTE/ui/
  ssh -o BatchMode=yes "${SSH_USER}@${ip}" "chmod +x $REMOTE/libexec/mlxctl $REMOTE/setup-tb.sh $REMOTE/serve_entry.py"
  if [[ "${MLXCTL_SKIP_MLX_LM:-}" != "1" ]]; then
    echo "==== mlx-lm $ip ===="
    ssh -o BatchMode=yes "${SSH_USER}@${ip}" \
      "$UV_REMOTE pip install mlx-lm --python $PYTHON_REMOTE"
  fi
  ROLE_FLAG=""
  if [[ "$ip" == "$CONTROLLER" ]]; then
    ROLE_FLAG="--controller"
  fi
  ssh -o BatchMode=yes "${SSH_USER}@${ip}" \
    "$REMOTE/bin/mlxctl --config $REMOTE/mlxctl.toml install $ROLE_FLAG --sudo-password $(printf %q "$SUDO_PW")"
done

rm -f "$TMP"
echo
echo "UI:  http://${CONTROLLER}:9090"
echo "API: http://${CONTROLLER}:8088/v1  (after a model is loaded)"
