use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServeProfile {
    pub family: String,
    pub title: String,
    pub default_runtime: String,
    pub allow_vlm: bool,
    pub allow_lm: bool,
    pub vision: bool,
    pub load_vlm_label: String,
    pub load_lm_label: String,
    pub hint: String,
    pub chat_safe_max_tokens: u32,
    pub chat_long_max_tokens: u32,
    pub chat_long_default_tokens: u32,
    pub chat_long_hint: String,
    pub thinking: bool,
    pub temperature: f32,
}

impl Default for ServeProfile {
    fn default() -> Self {
        generic_profile(true)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModel {
    pub id: String,
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub complete: bool,
    pub kind: String,
    pub architecture: Option<String>,
    #[serde(default)]
    pub profile: ServeProfile,
    #[serde(default)]
    pub replicas: Vec<ModelReplica>,
    #[serde(default)]
    pub cluster_complete: bool,
    #[serde(default)]
    pub source_node: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelReplica {
    pub node: String,
    pub complete: bool,
    pub size_bytes: u64,
    #[serde(default)]
    pub path: String,
}

pub fn scan_models(roots: &[PathBuf]) -> Vec<LocalModel> {
    let mut models = Vec::new();
    for root in roots {
        if !root.is_dir() {
            continue;
        }
        let Ok(entries) = fs::read_dir(root) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if name == "caches" || name.starts_with('.') {
                continue;
            }
            if let Some(m) = inspect_model(&path, &name) {
                models.push(m);
            }
        }
        // incomplete HF cache snapshots under caches/
        let caches = root.join("caches");
        if caches.is_dir() {
            if let Ok(entries) = fs::read_dir(&caches) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !path.is_dir() {
                        continue;
                    }
                    let name = entry.file_name().to_string_lossy().to_string();
                    if let Some(mut m) = inspect_model(&path, &name) {
                        if !m.complete {
                            m.kind = format!("{}-cache", m.kind);
                            models.push(m);
                        }
                    }
                }
            }
        }
    }
    models.sort_by(|a, b| b.complete.cmp(&a.complete).then(a.name.cmp(&b.name)));
    models
}

pub fn inspect_path(path: &Path) -> Option<LocalModel> {
    let name = path.file_name()?.to_string_lossy().to_string();
    inspect_model(path, &name)
}

pub fn attach_profile(model: &mut LocalModel) {
    let vision = model.kind.starts_with("vlm");
    model.profile = serve_profile(
        &model.kind,
        model.architecture.as_deref(),
        &model.id,
        &model.name,
        vision,
    );
}

fn inspect_model(path: &Path, folder: &str) -> Option<LocalModel> {
    let config_path = path.join("config.json");
    let has_config = config_path.is_file();
    let has_weights = has_weight_files(path);
    if !has_config && !has_weights {
        return None;
    }
    let size_bytes = dir_size(path);
    let complete = has_config && has_weights && size_bytes > 10 * 1024 * 1024;
    let (kind, architecture) = if has_config {
        classify(&config_path)
    } else {
        ("unknown".into(), None)
    };
    let id = folder.to_string();
    let name = folder.replace("--", "/");
    let vision = kind.starts_with("vlm");
    let profile = serve_profile(&kind, architecture.as_deref(), &id, &name, vision);
    Some(LocalModel {
        id,
        name,
        path: path.display().to_string(),
        size_bytes,
        complete,
        kind,
        architecture,
        profile,
        replicas: Vec::new(),
        cluster_complete: false,
        source_node: None,
    })
}

fn has_weight_files(path: &Path) -> bool {
    let Ok(entries) = fs::read_dir(path) else {
        return false;
    };
    entries.flatten().any(|e| {
        let n = e.file_name().to_string_lossy().to_string();
        n.ends_with(".safetensors") || n.ends_with(".npz") || n == "model.safetensors.index.json"
    })
}

fn classify(config_path: &Path) -> (String, Option<String>) {
    let Ok(raw) = fs::read_to_string(config_path) else {
        return ("unknown".into(), None);
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return ("unknown".into(), None);
    };
    let model_type = v
        .get("model_type")
        .and_then(|x| x.as_str())
        .or_else(|| {
            v.get("text_config")
                .and_then(|t| t.get("model_type"))
                .and_then(|x| x.as_str())
        });
    let arch = v
        .get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
        .or_else(|| model_type.map(|s| s.to_string()));
    let family = family_of(arch.as_deref(), "", "");
    let vision = family != "deepseek_v4"
        && (v.get("vision_config").is_some()
            || v.get("video_preprocessor_config").is_some()
            || v.get("video_processor_config").is_some()
            || path_exists_near(config_path, "video_preprocessor_config.json")
            || path_exists_near(config_path, "preprocessor_config.json"));
    let kind = if vision { "vlm" } else { "llm" };
    (kind.into(), arch)
}

fn family_of(architecture: Option<&str>, id: &str, name: &str) -> &'static str {
    let blob = format!(
        "{} {} {}",
        architecture.unwrap_or(""),
        id,
        name
    )
    .to_lowercase();
    if blob.contains("qwen4_exp")
        || blob.contains("qwen4exp")
        || blob.contains("flash-next")
        || blob.contains("flash_next")
    {
        return "qwen4_exp";
    }
    if blob.contains("deepseek_v4")
        || blob.contains("deepseek-v4")
        || blob.contains("deepseekv4")
    {
        return "deepseek_v4";
    }
    if blob.contains("qwen3_5") || blob.contains("qwen3.5") {
        return "qwen35";
    }
    "generic"
}

pub fn serve_profile(
    kind: &str,
    architecture: Option<&str>,
    id: &str,
    name: &str,
    vision: bool,
) -> ServeProfile {
    match family_of(architecture, id, name) {
        "qwen4_exp" => ServeProfile {
            family: "qwen4_exp".into(),
            title: "Qwen3.8-Flash-Next".into(),
            default_runtime: "mlx_vlm".into(),
            allow_vlm: true,
            allow_lm: false,
            vision: true,
            load_vlm_label: "加载视觉".into(),
            load_lm_label: "加载文本".into(),
            hint: "必须走 mlx-vlm。语言侧 TP 切分，PLE n-gram 各机 mmap/复制。不要点「加载文本」。".into(),
            chat_safe_max_tokens: 2048,
            chat_long_max_tokens: 8192,
            chat_long_default_tokens: 4096,
            chat_long_hint: "Flash-Next 超长补全上限 8192；PLE 热路径会读盘".into(),
            thinking: false,
            temperature: 0.7,
        },
        "deepseek_v4" => ServeProfile {
            family: "deepseek_v4".into(),
            title: "DeepSeek-V4-Flash".into(),
            default_runtime: "mlx_vlm".into(),
            allow_vlm: true,
            allow_lm: false,
            vision: false,
            load_vlm_label: "加载 V4".into(),
            load_lm_label: "加载文本".into(),
            hint: "mlx-lm 0.31 没有 deepseek_v4。用 mlx-vlm 文本 TP（按钮叫「加载 V4」）。无图。".into(),
            chat_safe_max_tokens: 2048,
            chat_long_max_tokens: 8192,
            chat_long_default_tokens: 4096,
            chat_long_hint: "V4 长解码可能撑爆 Metal 对象；先控在 8k 以内".into(),
            thinking: true,
            temperature: 0.6,
        },
        "qwen35" => ServeProfile {
            family: "qwen35".into(),
            title: "Qwen3.5".into(),
            default_runtime: if vision {
                "mlx_vlm".into()
            } else {
                "mlx_lm".into()
            },
            allow_vlm: vision,
            allow_lm: true,
            vision,
            load_vlm_label: "加载视觉".into(),
            load_lm_label: if vision {
                "加载文本".into()
            } else {
                "加载".into()
            },
            hint: if vision {
                "视觉走 mlx-vlm TP；纯文本可改走 mlx-lm。".into()
            } else {
                "按文本模型加载。".into()
            },
            chat_safe_max_tokens: 2048,
            chat_long_max_tokens: 12288,
            chat_long_default_tokens: 8192,
            chat_long_hint: "Qwen3.5 单次超过约 1.4 万 token 可能崩溃".into(),
            thinking: false,
            temperature: 0.7,
        },
        _ => generic_profile(vision || kind.starts_with("vlm")),
    }
}

fn generic_profile(vision: bool) -> ServeProfile {
    ServeProfile {
        family: "generic".into(),
        title: "通用".into(),
        default_runtime: if vision {
            "mlx_vlm".into()
        } else {
            "mlx_lm".into()
        },
        allow_vlm: vision,
        allow_lm: true,
        vision,
        load_vlm_label: "加载视觉".into(),
        load_lm_label: if vision {
            "加载文本".into()
        } else {
            "加载".into()
        },
        hint: "按扫描到的类型加载。".into(),
        chat_safe_max_tokens: 2048,
        chat_long_max_tokens: 4096,
        chat_long_default_tokens: 2048,
        chat_long_hint: "未识别架构，补全上限保持保守".into(),
        thinking: false,
        temperature: 0.7,
    }
}

fn path_exists_near(config_path: &Path, name: &str) -> bool {
    config_path
        .parent()
        .map(|p| p.join(name).is_file())
        .unwrap_or(false)
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                let name = e.file_name().to_string_lossy().to_string();
                if name == ".cache" {
                    continue;
                }
                stack.push(p);
            } else if let Ok(meta) = e.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

pub fn pull_model(python: &Path, repo: &str, dest: &Path) -> Result<String> {
    fs::create_dir_all(dest)?;
    let code = r#"
import os, sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, local_dir=dest)
print(dest)
"#;
    let out = std::process::Command::new(python)
        .env("HF_HUB_DISABLE_XET", "1")
        .env("HF_HUB_DISABLE_TELEMETRY", "1")
        .args(["-c", code, repo, &dest.display().to_string()])
        .output()?;
    if !out.status.success() {
        anyhow::bail!(
            "pull failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub fn ssh_user() -> String {
    crate::config::username()
}

fn sh_single_quote(path: &Path) -> String {
    format!("'{}'", path.display().to_string().replace('\'', "'\\''"))
}

/// Copy a local model directory to another Studio over SSH (Thunderbolt IP preferred).
pub fn rsync_push(src_dir: &Path, dest_host: &str, dest_dir: &Path) -> Result<String> {
    if !src_dir.is_dir() {
        anyhow::bail!("source missing: {}", src_dir.display());
    }
    let user = ssh_user();
    let identity = crate::config::home_dir()
        .join(".ssh/mlxctl_sync")
        .display()
        .to_string();
    let remote = format!("{user}@{dest_host}");
    let ssh_opts = [
        "-i",
        identity.as_str(),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
    ];
    let mkdir = std::process::Command::new("/usr/bin/ssh")
        .args(ssh_opts)
        .arg(&remote)
        .arg(format!("mkdir -p {}", sh_single_quote(dest_dir)))
        .output()?;
    if !mkdir.status.success() {
        anyhow::bail!(
            "ssh mkdir {remote}: {}",
            String::from_utf8_lossy(&mkdir.stderr)
        );
    }
    let src = format!("{}/", src_dir.display());
    let dest = format!("{remote}:{}/", dest_dir.display());
    let rsh = format!(
        "ssh -i {identity} -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
    );
    let out = std::process::Command::new("/usr/bin/rsync")
        .args(["-aH", "--partial", "--delete", "--stats", "-e", &rsh, &src, &dest])
        .output()?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        anyhow::bail!(
            "rsync {src} -> {dest}: {stderr}{stdout}"
        );
    }
    Ok(format!("{stdout}{stderr}"))
}
