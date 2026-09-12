use crate::config::EndpointConfig;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PersistentState {
    #[serde(default)]
    pub desired_serving: bool,
    #[serde(default)]
    pub autostart: bool,
    pub model_id: Option<String>,
    pub model_path: Option<String>,
    #[serde(default = "default_runtime")]
    pub runtime: String,
    pub endpoint: Option<EndpointConfig>,
    pub launch_pid: Option<u32>,
    #[serde(default)]
    pub hub: HubSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HubSettings {
    #[serde(default)]
    pub token: String,
    #[serde(default)]
    pub endpoint: String,
    #[serde(default)]
    pub python: String,
    #[serde(default)]
    pub dest_dir: String,
    #[serde(default)]
    pub node: String,
}

fn default_runtime() -> String {
    "mlx_lm".to_string()
}

impl PersistentState {
    pub fn load(path: &Path) -> Self {
        fs::read_to_string(path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).ok();
        }
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, serde_json::to_vec_pretty(self)?)
            .with_context(|| format!("write {}", tmp.display()))?;
        fs::rename(tmp, path).context("rename state")?;
        Ok(())
    }
}
