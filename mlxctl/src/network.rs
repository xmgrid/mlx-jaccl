use crate::config::{Config, NodeConfig};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkStatus {
    pub iface: String,
    pub ip: String,
    pub peer: String,
    pub peer_name: String,
    pub assigned: bool,
    pub peer_up: bool,
    pub ping_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatus {
    pub hostname: String,
    pub en0: Option<String>,
    pub rdma_enabled: bool,
    pub links: Vec<LinkStatus>,
    pub mesh_ready: bool,
}

pub fn apply_this_node(cfg: &Config) -> Result<NetworkStatus> {
    let node = cfg
        .this_node()
        .context("this machine is not in mlxctl.toml nodes")?;
    let _ = Command::new("/sbin/ifconfig").args(["bridge0", "down"]).status();
    let mut last_err = None;
    for attempt in 0..12 {
        match apply_links(node) {
            Ok(()) => return Ok(collect(node)),
            Err(e) => {
                last_err = Some(e);
                if attempt + 1 < 12 {
                    thread::sleep(Duration::from_secs(5));
                }
            }
        }
    }
    bail!("thunderbolt net-up failed: {:?}", last_err)
}

fn apply_links(node: &NodeConfig) -> Result<()> {
    for link in &node.links {
        let st = Command::new("/sbin/ifconfig")
            .args([&link.iface, "inet", &link.ip, "netmask", "255.255.255.252"])
            .status()
            .with_context(|| format!("ifconfig {}", link.iface))?;
        if !st.success() {
            bail!("ifconfig {} {} failed", link.iface, link.ip);
        }
        let change = Command::new("/sbin/route")
            .args(["change", &link.peer, "-interface", &link.iface])
            .status()?;
        if !change.success() {
            let add = Command::new("/sbin/route")
                .args(["add", &link.peer, "-interface", &link.iface])
                .status()?;
            if !add.success() {
                bail!("route {} via {} failed", link.peer, link.iface);
            }
        }
    }
    Ok(())
}

pub fn collect(node: &NodeConfig) -> NetworkStatus {
    let hostname = hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_default();
    let en0 = ifconfig_inet("en0");
    let rdma_enabled = rdma_enabled();
    let links: Vec<LinkStatus> = node
        .links
        .iter()
        .map(|l| {
            let assigned = ifconfig_inet(&l.iface).as_deref() == Some(l.ip.as_str());
            let (peer_up, ping_ms) = ping_ms(&l.peer);
            LinkStatus {
                iface: l.iface.clone(),
                ip: l.ip.clone(),
                peer: l.peer.clone(),
                peer_name: l.peer_name.clone(),
                assigned,
                peer_up,
                ping_ms,
            }
        })
        .collect();
    let mesh_ready = !links.is_empty() && links.iter().all(|l| l.assigned && l.peer_up);
    NetworkStatus {
        hostname,
        en0,
        rdma_enabled,
        links,
        mesh_ready,
    }
}

fn ifconfig_inet(iface: &str) -> Option<String> {
    let out = Command::new("/sbin/ifconfig").arg(iface).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("inet ") {
            return rest.split_whitespace().next().map(|s| s.to_string());
        }
    }
    None
}

fn rdma_enabled() -> bool {
    let out = Command::new("/usr/bin/rdma_ctl")
        .arg("status")
        .output()
        .ok()
        .or_else(|| Command::new("rdma_ctl").arg("status").output().ok());
    let Some(out) = out else {
        return false;
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
    .to_lowercase();
    text.contains("enabled") && !text.contains("disabled")
}

fn ping_ms(ip: &str) -> (bool, Option<f64>) {
    let out = Command::new("/sbin/ping")
        .args(["-c", "1", "-W", "1000", ip])
        .output();
    let Ok(out) = out else {
        return (false, None);
    };
    let text = String::from_utf8_lossy(&out.stdout);
    if !out.status.success() {
        return (false, None);
    }
    for part in text.split_whitespace() {
        if let Some(v) = part.strip_prefix("time=") {
            if let Ok(ms) = v.parse::<f64>() {
                return (true, Some(ms));
            }
        }
    }
    (true, None)
}

pub fn sudo_net_up(cfg: &Config) -> Result<()> {
    let wrapper = cfg.cluster_dir.join("bin/mlx-net-up");
    let status = Command::new("/usr/bin/sudo")
        .args(["-n", wrapper.to_str().unwrap_or("mlx-net-up")])
        .status();
    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(_) | Err(_) => apply_this_node(cfg).map(|_| ()),
    }
}
