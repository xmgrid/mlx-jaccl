# MLX-JACCL

Control plane and serving stack for a Thunderbolt-mesh Apple Silicon cluster running [MLX](https://github.com/ml-explore/mlx) with the JACCL distributed backend.

`mlxctl` is a Rust agent/controller: load models, sync weights between nodes, proxy OpenAI- and Anthropic-compatible APIs, and host a small web console.

## What is not in this repo

Local cluster files are gitignored so IPs, hostnames, tokens, and SSH identities stay off GitHub:

- `mlxctl.toml`
- `jaccl-mesh.json` / `jaccl-ring.json`
- SSH keys (`~/.ssh/mlxctl_sync` and similar)
- sudo passwords and API keys

Copy the examples and fill in **your** machines:

```bash
cp mlxctl.toml.example mlxctl.toml
cp jaccl-mesh.json.example jaccl-mesh.json
cp jaccl-ring.json.example jaccl-ring.json
```

Then set `token` to a random hex string and replace placeholder paths / node addresses.

## Build

Requires a recent Rust toolchain.

```bash
cd mlxctl
cargo build --release
```

Default config path is `$HOME/mlx-cluster/mlxctl.toml`. Override with `--config`.

## Deploy

```bash
export MLXCTL_SUDO_PASSWORD='...'   # required; no default
# optional: MLXCTL_SSH_USER, MLXCTL_REMOTE, MLX_PYTHON, MLX_UV
./deploy-mlxctl.sh
```

`deploy-mlxctl.sh` reads node LAN addresses from local `mlxctl.toml`. Thunderbolt interface IPs are applied by `setup-tb.sh` from the same file.

## Layout

| Path | Role |
| --- | --- |
| `mlxctl/` | Controller, agent, inference proxy, console |
| `mlxctl/python/serve_entry.py` | `mlx.launch` entry (includes tensor-parallel patches) |
| `mlxctl/web/` | Console UI |
| `setup-tb.sh` | Apply Thunderbolt P2P addresses from config |
| `launch-cluster.sh` | Manual `mlx.launch` wrapper |

Runtime copies typically live in `$HOME/mlx-cluster` on each node. Model weights are not part of this repository.
