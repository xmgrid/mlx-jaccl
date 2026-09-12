use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

const ALLOWED: &[&str] = &["mlx-vlm", "mlx-lm", "mlx", "huggingface_hub"];

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Packages {
    pub mlx: Option<String>,
    pub mlx_lm: Option<String>,
    pub mlx_vlm: Option<String>,
    pub huggingface_hub: Option<String>,
}

#[derive(Clone)]
pub struct Cache {
    inner: Packages,
    at: Instant,
}

impl Cache {
    pub fn new() -> Self {
        Self {
            inner: Packages::default(),
            at: Instant::now() - Duration::from_secs(60),
        }
    }

    pub fn get(&self) -> Packages {
        self.inner.clone()
    }

    pub fn stale(&self) -> bool {
        self.at.elapsed() > Duration::from_secs(45)
    }

    pub fn set(&mut self, pkgs: Packages) {
        self.inner = pkgs;
        self.at = Instant::now();
    }
}

pub fn probe(python: &Path) -> Packages {
    let code = r#"
import importlib.metadata as m
def v(n):
    try:
        print(n, m.version(n))
    except Exception:
        print(n, "")
v("mlx")
v("mlx-lm")
v("mlx-vlm")
v("huggingface_hub")
"#;
    let out = Command::new(python).args(["-c", code]).output();
    let mut pkgs = Packages::default();
    let Ok(out) = out else {
        return pkgs;
    };
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let mut parts = line.split_whitespace();
        let Some(name) = parts.next() else { continue };
        let ver = parts.next().unwrap_or("");
        let ver = if ver.is_empty() {
            None
        } else {
            Some(ver.to_string())
        };
        match name {
            "mlx" => pkgs.mlx = ver,
            "mlx-lm" => pkgs.mlx_lm = ver,
            "mlx-vlm" => pkgs.mlx_vlm = ver,
            "huggingface_hub" => pkgs.huggingface_hub = ver,
            _ => {}
        }
    }
    pkgs
}

pub fn install(python: &Path, package: &str) -> Result<String> {
    if !ALLOWED.contains(&package) {
        bail!("package {package} is not allowed");
    }
    let local_uv = crate::config::home_dir().join(".local/bin/uv");
    let uv: PathBuf = if local_uv.is_file() {
        local_uv
    } else {
        PathBuf::from("uv")
    };
    let out = Command::new(&uv)
        .args(["pip", "install", package, "--python"])
        .arg(python)
        .output()
        .context("run uv pip install")?;
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if !out.status.success() {
        bail!("install {package} failed:\n{log}");
    }
    Ok(log)
}
