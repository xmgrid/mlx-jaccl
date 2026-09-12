use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub cluster_id: String,
    pub token: String,
    pub python: PathBuf,
    pub mlx_launch: PathBuf,
    pub hostfile: PathBuf,
    pub setup_tb: PathBuf,
    pub serve_entry: PathBuf,
    pub cluster_dir: PathBuf,
    pub log_dir: PathBuf,
    pub state_path: PathBuf,
    #[serde(default)]
    pub ui_dir: Option<PathBuf>,
    pub model_roots: Vec<PathBuf>,
    pub controller_bind: String,
    pub agent_bind: String,
    pub internal_infer_port: u16,
    pub endpoint: EndpointConfig,
    pub nodes: Vec<NodeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EndpointConfig {
    pub advertise_host: String,
    pub bind: String,
    pub port: u16,
    #[serde(default)]
    pub api_key: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub name: String,
    pub ssh: String,
    pub rank: u32,
    pub memory_gb: u32,
    #[serde(default)]
    pub rdma: Vec<String>,
    #[serde(default)]
    pub links: Vec<TbLink>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TbLink {
    pub iface: String,
    pub ip: String,
    pub peer: String,
    pub peer_name: String,
}

pub fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

pub fn username() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "mlx".into())
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("read config {}", path.display()))?;
        let cfg: Self = toml::from_str(&raw).context("parse mlxctl.toml")?;
        Ok(cfg)
    }

    pub fn this_node(&self) -> Option<&NodeConfig> {
        let host = local_ipv4s();
        self.nodes
            .iter()
            .find(|n| host.iter().any(|ip| ip == &n.ssh) || hostname_matches(&n.name))
    }

    pub fn controller_node(&self) -> &NodeConfig {
        self.nodes
            .iter()
            .find(|n| n.rank == 0)
            .unwrap_or(&self.nodes[0])
    }

    pub fn agent_url(&self, node: &NodeConfig) -> String {
        let addr: SocketAddr = self
            .agent_bind
            .parse()
            .unwrap_or_else(|_| "0.0.0.0:9091".parse().unwrap());
        format!("http://{}:{}", node.ssh, addr.port())
    }

    pub fn ui_dir(&self) -> PathBuf {
        self.ui_dir
            .clone()
            .unwrap_or_else(|| self.cluster_dir.join("ui"))
    }

    /// Prefer the Thunderbolt peer IP from `from` toward `to`; fall back to LAN ssh.
    pub fn hop_ip(&self, from: &str, to: &str) -> Option<String> {
        let src = self.nodes.iter().find(|n| n.name == from)?;
        let dst = self.nodes.iter().find(|n| n.name == to)?;
        if let Some(link) = src.links.iter().find(|l| l.peer_name == to) {
            if !link.peer.is_empty() {
                return Some(link.peer.clone());
            }
        }
        Some(dst.ssh.clone())
    }

    pub fn binary_path() -> PathBuf {
        std::env::current_exe().unwrap_or_else(|_| PathBuf::from("mlxctl"))
    }
}

fn hostname_matches(name: &str) -> bool {
    hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .is_some_and(|h| h.to_lowercase().contains(&name.to_lowercase()))
}

fn local_ipv4s() -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(ip) = std::process::Command::new("/usr/sbin/ipconfig")
        .args(["getifaddr", "en0"])
        .output()
    {
        if ip.status.success() {
            if let Ok(s) = String::from_utf8(ip.stdout) {
                let s = s.trim();
                if !s.is_empty() {
                    out.push(s.to_string());
                }
            }
        }
    }
    out
}
