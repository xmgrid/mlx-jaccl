#!/bin/bash
# Re-apply Thunderbolt point-to-point IPs from mlxctl.toml for this host.
set -euo pipefail

CONFIG="${MLXCTL_CONFIG:-${1:-$HOME/mlx-cluster/mlxctl.toml}}"
if [[ ! -f "$CONFIG" ]]; then
  echo "missing config: $CONFIG" >&2
  exit 1
fi

ifconfig bridge0 down 2>/dev/null || true

/usr/bin/python3 - "$CONFIG" <<'PY'
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    raise SystemExit("python 3.11+ with tomllib is required")

cfg = tomllib.loads(Path(sys.argv[1]).read_text())
try:
    en0 = subprocess.check_output(
        ["/usr/sbin/ipconfig", "getifaddr", "en0"], text=True
    ).strip()
except subprocess.CalledProcessError as e:
    raise SystemExit(f"could not read en0 address: {e}") from e

node = next((n for n in cfg.get("nodes") or [] if n.get("ssh") == en0), None)
if not node:
    raise SystemExit(f"en0 {en0} is not listed in mlxctl.toml nodes")

for link in node.get("links") or []:
    iface, ip, peer = link["iface"], link["ip"], link["peer"]
    subprocess.check_call(
        ["/sbin/ifconfig", iface, "inet", ip, "netmask", "255.255.255.252"]
    )
    changed = subprocess.run(
        ["/sbin/route", "change", peer, "-interface", iface]
    )
    if changed.returncode != 0:
        subprocess.check_call(
            ["/sbin/route", "add", peer, "-interface", iface]
        )

print(f"Thunderbolt mesh IPs applied on {en0}")
PY
