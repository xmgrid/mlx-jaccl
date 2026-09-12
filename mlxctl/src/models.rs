use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

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
    #[serde(default)]
    pub downloading: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelReplica {
    pub node: String,
    pub complete: bool,
    pub size_bytes: u64,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub downloading: bool,
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
        model.size_bytes,
    );
}

fn inspect_model(path: &Path, folder: &str) -> Option<LocalModel> {
    let config_path = path.join("config.json");
    let has_config = config_path.is_file();
    let has_weights = has_weight_files(path);
    if !has_config && !has_weights {
        return None;
    }
    let downloading = download_in_progress(path);
    let size_bytes = dir_size(path, downloading);
    let complete = has_config
        && has_weights
        && size_bytes > 10 * 1024 * 1024
        && snapshot_complete(path);
    let (kind, architecture) = if has_config {
        classify(&config_path)
    } else {
        ("unknown".into(), None)
    };
    let id = folder.to_string();
    let name = folder.replace("--", "/");
    let vision = kind.starts_with("vlm");
    let profile = serve_profile(
        &kind,
        architecture.as_deref(),
        &id,
        &name,
        vision,
        size_bytes,
    );
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
        downloading,
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

fn snapshot_complete(path: &Path) -> bool {
    if let Some(ok) = numbered_shards_complete(path) {
        return ok;
    }
    if let Some(ok) = index_shards_complete(path) {
        return ok;
    }
    !hf_cache_has_incomplete(path)
}

pub fn download_in_progress(path: &Path) -> bool {
    if numbered_shards_complete(path) == Some(true) {
        return false;
    }
    if index_shards_complete(path) == Some(true) {
        return false;
    }
    if hf_cache_has_incomplete(path) {
        return true;
    }
    let Ok(entries) = fs::read_dir(path) else {
        return false;
    };
    entries.flatten().any(|e| {
        let n = e.file_name().to_string_lossy().into_owned();
        n.starts_with('.') && n.contains(".safetensors")
    })
}

fn hf_cache_has_incomplete(path: &Path) -> bool {
    let download = path.join(".cache").join("huggingface").join("download");
    let Ok(entries) = fs::read_dir(download) else {
        return false;
    };
    entries.flatten().any(|e| {
        e.file_name()
            .to_string_lossy()
            .ends_with(".incomplete")
    })
}

fn parse_numbered_shard(name: &str) -> Option<(u32, u32)> {
    let rest = name.strip_prefix("model-")?.strip_suffix(".safetensors")?;
    let (idx, total) = rest.split_once("-of-")?;
    Some((idx.parse().ok()?, total.parse().ok()?))
}

fn numbered_shards_complete(path: &Path) -> Option<bool> {
    let entries = fs::read_dir(path).ok()?;
    let mut total = None;
    let mut seen = std::collections::BTreeSet::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some((idx, n)) = parse_numbered_shard(&name) else {
            continue;
        };
        match total {
            None => total = Some(n),
            Some(prev) if prev != n => return Some(false),
            Some(_) => {}
        }
        seen.insert(idx);
    }
    let total = total?;
    Some(total > 0 && seen.len() as u32 == total && (1..=total).all(|i| seen.contains(&i)))
}

fn index_shards_complete(path: &Path) -> Option<bool> {
    let index = path.join("model.safetensors.index.json");
    if !index.is_file() {
        return None;
    }
    let raw = fs::read_to_string(&index).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let map = v.get("weight_map")?.as_object()?;
    if map.is_empty() {
        return Some(false);
    }
    let files: std::collections::BTreeSet<&str> = map
        .values()
        .filter_map(|x| x.as_str())
        .collect();
    Some(files.iter().all(|f| path.join(f).is_file()))
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

fn looks_4bit(id: &str, name: &str) -> bool {
    let blob = format!("{id} {name}").to_lowercase();
    blob.contains("4bit")
        || blob.contains("4-bit")
        || blob.contains("q4_")
        || blob.contains("-q4")
}

pub fn serve_profile(
    kind: &str,
    architecture: Option<&str>,
    id: &str,
    name: &str,
    vision: bool,
    _size_bytes: u64,
) -> ServeProfile {
    match family_of(architecture, id, name) {
        "qwen4_exp" => {
            let fourbit = looks_4bit(id, name);
            let (title, hint) = if fourbit {
                (
                    "Qwen3.8-Flash-Next 4bit".into(),
                    "必须走 mlx-vlm（点「加载视觉」）。4bit 约 104GB，四卡 TP 后 64GB worker 放得下；PLE n-gram 磁盘 mmap。日志应持续出现 JACCL / sharded_load；若停在 Loading model 超过 30 秒，是通信卡住，点「中止加载」再重载。不要点「加载文本」。".into(),
                )
            } else {
                (
                    "Qwen3.8-Flash-Next bf16".into(),
                    "必须走 mlx-vlm。这是 bf16 全量约 336GB，PLE n-gram 约 99GB 不能切分，64GB worker 放不下。请改加载同系列 4bit，或 Qwen3.8-27B。不要点「加载文本」。".into(),
                )
            };
            ServeProfile {
                family: "qwen4_exp".into(),
                title,
                default_runtime: "mlx_vlm".into(),
                allow_vlm: true,
                allow_lm: false,
                vision: true,
                load_vlm_label: "加载视觉".into(),
                load_lm_label: "加载文本".into(),
                hint,
                chat_safe_max_tokens: 2048,
                chat_long_max_tokens: 8192,
                chat_long_default_tokens: 4096,
                chat_long_hint: "Flash-Next 超长补全上限 8192；PLE 热路径会读盘".into(),
                thinking: false,
                temperature: 0.7,
            }
        }
        "deepseek_v4" => ServeProfile {
            family: "deepseek_v4".into(),
            title: "DeepSeek-V4-Flash".into(),
            default_runtime: "mlx_vlm".into(),
            allow_vlm: true,
            allow_lm: false,
            vision: false,
            load_vlm_label: "加载 V4".into(),
            load_lm_label: "加载文本".into(),
            hint: "mlx-lm 0.31 没有 deepseek_v4。用 mlx-vlm 文本 TP（按钮叫「加载 V4」）。无图。中文请用 ByteLevel 解码；若仍乱码先卸下再加载。".into(),
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

fn dir_size(path: &Path, include_cache: bool) -> u64 {
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
                if name == ".cache" && !include_cache {
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

pub struct PullOpts<'a> {
    pub python: &'a Path,
    pub repo: &'a str,
    pub dest: &'a Path,
    pub token: Option<&'a str>,
    pub endpoint: Option<&'a str>,
}

pub fn folder_for_repo(repo: &str) -> String {
    repo.trim().trim_matches('/').replace('/', "--")
}

pub fn path_under_roots(roots: &[PathBuf], dest: &Path) -> bool {
    if dest.as_os_str().is_empty() {
        return false;
    }
    let dest = if dest.is_absolute() {
        dest.to_path_buf()
    } else {
        return false;
    };
    roots.iter().any(|root| dest == *root || dest.starts_with(root))
}

pub fn pull_model(opts: PullOpts<'_>, mut on_line: impl FnMut(&str)) -> Result<String> {
    fs::create_dir_all(opts.dest)?;
    let code = r#"
import os, sys, threading, time
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
print(f"start {repo} -> {dest}", flush=True)
stop = threading.Event()

def watch():
    while not stop.wait(4):
        total = 0
        files = 0
        incomplete = 0
        for root, _, names in os.walk(dest):
            for name in names:
                path = os.path.join(root, name)
                try:
                    total += os.path.getsize(path)
                except OSError:
                    continue
                files += 1
                if name.endswith(".incomplete"):
                    incomplete += 1
        gb = total / (1024 ** 3)
        print(f"progress files={files} incomplete={incomplete} {gb:.2f}GB", flush=True)

t = threading.Thread(target=watch, daemon=True)
t.start()
try:
    path = snapshot_download(repo_id=repo, local_dir=dest)
    print(f"done {path}", flush=True)
except Exception as exc:
    print(f"error {exc}", flush=True)
    raise
finally:
    stop.set()
"#;
    let mut cmd = Command::new(opts.python);
    cmd.arg("-u")
        .env("HF_HUB_DISABLE_XET", "1")
        .env("HF_HUB_DISABLE_TELEMETRY", "1")
        .env("HF_HUB_ENABLE_HF_TRANSFER", "0")
        .env("PYTHONUNBUFFERED", "1");
    if let Some(token) = opts.token.filter(|t| !t.is_empty()) {
        cmd.env("HF_TOKEN", token)
            .env("HUGGING_FACE_HUB_TOKEN", token);
    }
    if let Some(endpoint) = opts.endpoint.filter(|e| !e.is_empty()) {
        cmd.env("HF_ENDPOINT", endpoint);
    }
    let dest = opts.dest.display().to_string();
    let mut child = cmd
        .args(["-c", code, opts.repo, &dest])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()?;
    let stdout = child.stdout.take().expect("stdout");
    let stderr = child.stderr.take().expect("stderr");
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let tx2 = tx.clone();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let _ = tx.send(line);
        }
    });
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            let _ = tx2.send(line);
        }
    });
    let mut last = String::new();
    while let Ok(line) = rx.recv() {
        last = line.clone();
        on_line(&line);
    }
    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!(
            "pull failed: {}",
            if last.is_empty() {
                format!("exit {status}")
            } else {
                last
            }
        );
    }
    Ok(dest)
}

pub fn delete_model(roots: &[PathBuf], folder: &str) -> Result<String> {
    let folder = folder.trim();
    if folder.is_empty()
        || folder.contains("..")
        || folder.contains('/')
        || folder.contains('\\')
        || Path::new(folder).is_absolute()
    {
        anyhow::bail!("bad model id");
    }
    let mut removed = Vec::new();
    for root in roots {
        let dest = root.join(folder);
        if !dest.exists() {
            continue;
        }
        let Ok(root_c) = root.canonicalize() else {
            continue;
        };
        let Ok(dest_c) = dest.canonicalize() else {
            continue;
        };
        if dest_c == root_c || !dest_c.starts_with(&root_c) {
            anyhow::bail!("refusing to delete {}", dest_c.display());
        }
        fs::remove_dir_all(&dest_c)?;
        removed.push(dest_c.display().to_string());
    }
    if removed.is_empty() {
        anyhow::bail!("not found");
    }
    Ok(removed.join("\n"))
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
