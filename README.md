# MLX-JACCL

[中文](#mlx-jaccl) · [English](#english)

面向 Apple Silicon 集群的 [MLX](https://github.com/ml-explore/mlx) 控制面与推理栈，分布式后端为 **JACCL**（Thunderbolt 点对点网状互联）。

`mlxctl` 是 Rust 写的 agent / controller：加载模型、在节点间同步权重、提供 OpenAI 与 Anthropic 兼容 API，并带一个简易 Web 控制台。

## 能做什么

- 在多台 Mac 上拉起 JACCL 网格，用 `mlx.launch` 做张量并行推理
- 控制台加载 / 停止模型，查看节点在线、网格和副本分布
- 把完整模型目录同步到其余节点（SSH + rsync，优先走 Thunderbolt 对端 IP）
- 对外推理代理：OpenAI Chat Completions / Responses，以及 Anthropic Messages
- 为部分架构提供 mlx-vlm 张量并行补丁（见 `mlxctl/python/serve_entry.py`）

## 本仓库不包含什么

本地集群身份信息已加入 `.gitignore`，请勿提交：

- `mlxctl.toml`（节点名、局域网 IP、Thunderbolt 地址、控制面 token）
- `jaccl-mesh.json` / `jaccl-ring.json`
- SSH 私钥（如 `~/.ssh/mlxctl_sync`）
- sudo 密码、API key、模型权重

克隆后先复制示例再填自己的机器：

```bash
cp mlxctl.toml.example mlxctl.toml
cp jaccl-mesh.json.example jaccl-mesh.json
cp jaccl-ring.json.example jaccl-ring.json
```

把 `token` 换成随机十六进制字符串，并替换路径、节点名和地址。

## 环境要求

- macOS，Apple Silicon
- 节点之间 Thunderbolt 直连（JACCL 全网格）以及可 SSH 的局域网
- Rust 工具链（编译 `mlxctl`）
- 各节点 Python 环境，内含 `mlx`、`mlx-lm`；视觉模型另需 `mlx-vlm`
- 运行副本一般放在每台机器的 `$HOME/mlx-cluster`

## 配置要点

`mlxctl.toml` 描述整张网格：

| 字段 | 含义 |
| --- | --- |
| `token` | agent / controller 互访的 Bearer token |
| `python` / `mlx_launch` | 各节点上的 Python 与 `mlx.launch` |
| `hostfile` | JACCL hostfile（可由 mesh 示例改写） |
| `nodes[].name` | 逻辑名（不要用真实主机名也行） |
| `nodes[].ssh` | 局域网 IP，供 SSH 与 agent HTTP |
| `nodes[].rank` | 0 为 controller |
| `nodes[].links` | Thunderbolt 网卡、本端 IP、对端 IP |
| `endpoint` | 对外推理代理（默认 8088） |
| `controller_bind` | 控制台（默认 9090） |

`setup-tb.sh` 会读这份 toml：按本机 `en0` 地址匹配节点，再给 Thunderbolt 网卡配点对点 IP。

## 编译

```bash
cd mlxctl
cargo build --release
```

默认配置路径是 `$HOME/mlx-cluster/mlxctl.toml`，可用 `--config` 覆盖。

## 部署

```bash
export MLXCTL_SUDO_PASSWORD='...'   # 必填，脚本没有默认密码
# 可选：MLXCTL_SSH_USER、MLXCTL_REMOTE、MLX_PYTHON、MLX_UV
./deploy-mlxctl.sh
```

脚本从本地 `mlxctl.toml` 读取节点地址，把二进制、配置、UI 和 `serve_entry.py` 拷到各节点，并安装 launchd。rank 0 同时装 controller。

模型同步使用专用 SSH 密钥 `$HOME/.ssh/mlxctl_sync`（需事先配到各节点 `authorized_keys`）。

## 使用

1. 浏览器打开 `http://<rank0局域网IP>:9090`
2. 确认四台（或你配置的全部）节点可达且网格就绪
3. 在控制台加载已同步完成的本地模型目录
4. 推理走 `http://<rank0>:8088`：OpenAI 兼容接口带 `/v1`，Anthropic Messages 的 base URL **不要**加 `/v1`

对话请求里的 `model` 必须是本机模型目录路径，不是 Hub 上的仓库 id。

## 目录

| 路径 | 作用 |
| --- | --- |
| `mlxctl/` | 控制面、agent、推理代理、控制台 |
| `mlxctl/python/serve_entry.py` | `mlx.launch` 入口（含张量并行补丁） |
| `mlxctl/web/` | 控制台 UI |
| `setup-tb.sh` | 按配置写入 Thunderbolt 点对点地址 |
| `launch-cluster.sh` | 手动 `mlx.launch` 包装 |
| `deploy-mlxctl.sh` | 编译产物安装到各节点 |

---

## English

Control plane and serving stack for a Thunderbolt-mesh Apple Silicon cluster running [MLX](https://github.com/ml-explore/mlx) with the JACCL distributed backend.

`mlxctl` is a Rust agent/controller: load models, sync weights between nodes, proxy OpenAI- and Anthropic-compatible APIs, and host a small web console.

### What is not in this repo

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

### Build

Requires a recent Rust toolchain.

```bash
cd mlxctl
cargo build --release
```

Default config path is `$HOME/mlx-cluster/mlxctl.toml`. Override with `--config`.

### Deploy

```bash
export MLXCTL_SUDO_PASSWORD='...'   # required; no default
# optional: MLXCTL_SSH_USER, MLXCTL_REMOTE, MLX_PYTHON, MLX_UV
./deploy-mlxctl.sh
```

`deploy-mlxctl.sh` reads node LAN addresses from local `mlxctl.toml`. Thunderbolt interface IPs are applied by `setup-tb.sh` from the same file.

### Layout

| Path | Role |
| --- | --- |
| `mlxctl/` | Controller, agent, inference proxy, console |
| `mlxctl/python/serve_entry.py` | `mlx.launch` entry (includes tensor-parallel patches) |
| `mlxctl/web/` | Console UI |
| `setup-tb.sh` | Apply Thunderbolt P2P addresses from config |
| `launch-cluster.sh` | Manual `mlx.launch` wrapper |

Runtime copies typically live in `$HOME/mlx-cluster` on each node. Model weights are not part of this repository.
