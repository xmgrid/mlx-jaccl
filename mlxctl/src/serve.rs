use crate::config::Config;
use anyhow::{bail, Context, Result};
use nix::sys::signal::{kill, killpg, Signal};
use nix::unistd::Pid;
use std::fs::OpenOptions;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::time::sleep;

pub fn pid_alive(pid: u32) -> bool {
    let r = unsafe { libc::kill(pid as i32, 0) };
    r == 0
}

pub fn stop_pid(pid: u32) {
    let p = Pid::from_raw(pid as i32);
    let _ = killpg(p, Signal::SIGTERM);
    let _ = kill(p, Signal::SIGTERM);
    std::thread::sleep(Duration::from_millis(1500));
    if pid_alive(pid) {
        let _ = killpg(p, Signal::SIGKILL);
        let _ = kill(p, Signal::SIGKILL);
    }
}

pub fn cleanup_serve_processes(serve_entry: Option<&Path>) {
    if let Some(path) = serve_entry.and_then(|p| p.to_str()) {
        let _ = Command::new("/usr/bin/pkill").args(["-f", path]).status();
    }
    let _ = Command::new("/usr/bin/pkill")
        .args(["-f", "serve_entry.py"])
        .status();
    let _ = Command::new("/usr/bin/pkill")
        .args(["-f", "mlx_lm.server"])
        .status();
    let _ = Command::new("/usr/bin/pkill")
        .args(["-f", "mlx_lm server"])
        .status();
    let _ = Command::new("/usr/bin/pkill")
        .args(["-f", "mlx_vlm.server"])
        .status();
    let _ = Command::new("/usr/bin/pkill")
        .args(["-f", "mlx_vlm.server.cli"])
        .status();
    let _ = Command::new("/usr/bin/pkill")
        .args(["-f", "mlx.launch"])
        .status();
}

pub fn spawn_launch(
    cfg: &Config,
    model_path: &str,
    runtime: &str,
) -> Result<u32> {
    std::fs::create_dir_all(&cfg.log_dir).ok();
    let log_path = cfg.log_dir.join("serve.log");
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("open {}", log_path.display()))?;
    let log2 = log.try_clone()?;
    let marker = format!(
        "\n==== mlxctl serve {} {} ====\n",
        chrono::Local::now().to_rfc3339(),
        model_path
    );
    let _ = std::fs::OpenOptions::new()
        .append(true)
        .open(&log_path)
        .and_then(|mut f| {
            use std::io::Write;
            f.write_all(marker.as_bytes())
        });

    let port = cfg.internal_infer_port.to_string();
    let mut cmd = Command::new(&cfg.mlx_launch);
    cmd.arg("--verbose")
        .arg("--backend")
        .arg("jaccl")
        .arg("--hostfile")
        .arg(&cfg.hostfile)
        .arg("--cwd")
        .arg(&cfg.cluster_dir)
        .arg("--env")
        .arg("MLX_METAL_FAST_SYNCH=1")
        .arg("--env")
        .arg(format!("MLXCTL_RUNTIME={runtime}"))
        .arg("--env")
        .arg("MLX_DISTRIBUTED_BACKEND=jaccl")
        .arg("--env")
        .arg("PYTHONUNBUFFERED=1")
        .arg("--")
        .arg(&cfg.python)
        .arg("-u")
        .arg(&cfg.serve_entry)
        .arg("--model")
        .arg(model_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(&port)
        .arg("--log-level")
        .arg("INFO");
    if runtime == "mlx_vlm" {
        cmd.arg("--trust-remote-code");
    }
    cmd.current_dir(&cfg.cluster_dir)
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log2))
        .process_group(0);

    let mut child = cmd.spawn().context("spawn mlx.launch")?;
    let pid = child.id();
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(pid)
}

pub async fn wait_port_free(port: u16, timeout: Duration) -> Result<()> {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let start = tokio::time::Instant::now();
    loop {
        match std::net::TcpListener::bind(addr) {
            Ok(listener) => {
                drop(listener);
                return Ok(());
            }
            Err(_) => {
                if start.elapsed() > timeout {
                    bail!("port {port} still in use after unloading the previous model");
                }
                sleep(Duration::from_millis(250)).await;
            }
        }
    }
}

pub async fn wait_ready(port: u16, timeout: Duration, pid: Option<u32>) -> Result<()> {
    let url = format!("http://127.0.0.1:{port}/v1/models");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()?;
    let start = tokio::time::Instant::now();
    loop {
        if let Some(p) = pid {
            if !pid_alive(p) {
                bail!(
                    "model server process {p} exited before becoming ready; see logs/serve.log"
                );
            }
        }
        if start.elapsed() > timeout {
            bail!("model server did not become ready on {url}");
        }
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                return Ok(());
            }
        }
        sleep(Duration::from_secs(2)).await;
    }
}

pub fn tail_log(path: &std::path::Path, max_bytes: usize) -> String {
    let Ok(data) = std::fs::read(path) else {
        return String::new();
    };
    if data.len() <= max_bytes {
        return String::from_utf8_lossy(&data).to_string();
    }
    String::from_utf8_lossy(&data[data.len() - max_bytes..]).to_string()
}

pub fn clear_log(path: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    let marker = format!(
        "==== mlxctl logs cleared {} ====\n",
        chrono::Local::now().to_rfc3339()
    );
    file.write_all(marker.as_bytes())?;
    file.flush()?;
    Ok(())
}
