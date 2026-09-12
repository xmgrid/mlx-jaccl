use crate::config::Config;
use crate::models::{self, LocalModel};
use crate::network;
use crate::serve;
use crate::stack;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use sysinfo::System;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tracing::info;

#[derive(Clone)]
struct AgentState {
    cfg: Arc<Config>,
    token: String,
    jobs: Arc<Mutex<HashMap<String, Job>>>,
    stack: Arc<Mutex<stack::Cache>>,
}

#[derive(Clone, Serialize)]
struct Job {
    id: String,
    kind: String,
    status: String,
    log: String,
}

#[derive(Serialize)]
struct Health {
    ok: bool,
    name: String,
    ssh: String,
    rank: u32,
}

#[derive(Serialize)]
struct AgentInfo {
    health: Health,
    hostname: String,
    memory_total_bytes: u64,
    memory_used_bytes: u64,
    python_ok: bool,
    mlx_ok: bool,
    mlx_lm: Option<String>,
    mlx_vlm: Option<String>,
    network: network::NetworkStatus,
}

pub async fn run(cfg: Config) -> anyhow::Result<()> {
    let this = cfg
        .this_node()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("this host is not listed in mlxctl.toml"))?;
    let st = AgentState {
        token: cfg.token.clone(),
        cfg: Arc::new(cfg.clone()),
        jobs: Arc::new(Mutex::new(HashMap::new())),
        stack: Arc::new(Mutex::new(stack::Cache::new())),
    };
    let app = Router::new()
        .route("/v1/health", get(health))
        .route("/v1/info", get(info))
        .route("/v1/models", get(list_models))
        .route("/v1/models/pull", post(pull))
        .route("/v1/models/push", post(push))
        .route("/v1/jobs/{id}", get(job))
        .route("/v1/network", get(net_status))
        .route("/v1/network/up", post(net_up))
        .route("/v1/cleanup", post(cleanup))
        .route("/v1/stack/install", post(stack_install))
        .with_state(st);
    let addr = cfg.agent_bind.clone();
    info!("mlxctl agent {} rank {} on {addr}", this.name, this.rank);
    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn check_token(st: &AgentState, headers: &HeaderMap) -> Result<(), StatusCode> {
    let got = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if got == format!("Bearer {}", st.token) {
        Ok(())
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

async fn health(State(st): State<AgentState>) -> Json<Health> {
    Json(health_inner(&st))
}

fn health_inner(st: &AgentState) -> Health {
    let n = st.cfg.this_node();
    Health {
        ok: true,
        name: n.map(|x| x.name.clone()).unwrap_or_default(),
        ssh: n.map(|x| x.ssh.clone()).unwrap_or_default(),
        rank: n.map(|x| x.rank).unwrap_or(0),
    }
}

async fn info(
    State(st): State<AgentState>,
    headers: HeaderMap,
) -> Result<Json<AgentInfo>, StatusCode> {
    check_token(&st, &headers)?;
    let mut sys = System::new();
    sys.refresh_memory();
    let python_ok = st.cfg.python.exists();
    let pkgs = {
        let cache = st.stack.lock().await;
        if cache.stale() {
            drop(cache);
            let py = st.cfg.python.clone();
            let probed = tokio::task::spawn_blocking(move || stack::probe(&py))
                .await
                .unwrap_or_default();
            let mut cache = st.stack.lock().await;
            cache.set(probed.clone());
            probed
        } else {
            cache.get()
        }
    };
    let mlx_ok = pkgs.mlx.is_some();
    let node = st.cfg.this_node().cloned();
    let network = {
        let n = node.clone();
        tokio::task::spawn_blocking(move || {
            n.as_ref()
                .map(network::collect)
                .unwrap_or(network::NetworkStatus {
                    hostname: String::new(),
                    en0: None,
                    rdma_enabled: false,
                    links: vec![],
                    mesh_ready: false,
                })
        })
        .await
        .unwrap_or(network::NetworkStatus {
            hostname: String::new(),
            en0: None,
            rdma_enabled: false,
            links: vec![],
            mesh_ready: false,
        })
    };
    Ok(Json(AgentInfo {
        health: health_inner(&st),
        hostname: network.hostname.clone(),
        memory_total_bytes: sys.total_memory(),
        memory_used_bytes: sys.used_memory(),
        python_ok,
        mlx_ok,
        mlx_lm: pkgs.mlx_lm,
        mlx_vlm: pkgs.mlx_vlm,
        network,
    }))
}

async fn list_models(
    State(st): State<AgentState>,
    headers: HeaderMap,
) -> Result<Json<Vec<LocalModel>>, StatusCode> {
    check_token(&st, &headers)?;
    Ok(Json(models::scan_models(&st.cfg.model_roots)))
}

async fn net_status(
    State(st): State<AgentState>,
    headers: HeaderMap,
) -> Result<Json<network::NetworkStatus>, StatusCode> {
    check_token(&st, &headers)?;
    let node = st.cfg.this_node().ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(network::collect(node)))
}

async fn net_up(
    State(st): State<AgentState>,
    headers: HeaderMap,
) -> Result<Json<network::NetworkStatus>, StatusCode> {
    check_token(&st, &headers)?;
    let cfg = st.cfg.clone();
    tokio::task::spawn_blocking(move || network::sudo_net_up(&cfg))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let node = st.cfg.this_node().ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(network::collect(node)))
}

async fn cleanup(
    State(st): State<AgentState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, StatusCode> {
    check_token(&st, &headers)?;
    serve::cleanup_serve_processes(Some(&st.cfg.serve_entry));
    Ok(Json(serde_json::json!({"ok": true})))
}

#[derive(Deserialize)]
struct PullReq {
    repo: String,
}

async fn pull(
    State(st): State<AgentState>,
    headers: HeaderMap,
    Json(req): Json<PullReq>,
) -> Result<Json<Job>, StatusCode> {
    check_token(&st, &headers)?;
    let id = uuid::Uuid::new_v4().to_string();
    let job = Job {
        id: id.clone(),
        kind: "pull".into(),
        status: "running".into(),
        log: format!("pull {}\n", req.repo),
    };
    st.jobs.lock().await.insert(id.clone(), job.clone());
    let cfg = st.cfg.clone();
    let jobs = st.jobs.clone();
    let repo = req.repo.clone();
    tokio::task::spawn_blocking(move || {
        let folder = repo.replace('/', "--");
        let dest = cfg.model_roots[0].join(&folder);
        let result = models::pull_model(&cfg.python, &repo, &dest);
        let handle = tokio::runtime::Handle::current();
        handle.block_on(async {
            let mut g = jobs.lock().await;
            if let Some(j) = g.get_mut(&id) {
                match result {
                    Ok(p) => {
                        j.status = "ok".into();
                        j.log.push_str(&format!("saved {p}\n"));
                    }
                    Err(e) => {
                        j.status = "error".into();
                        j.log.push_str(&format!("{e:#}\n"));
                    }
                }
            }
        });
    });
    Ok(Json(job))
}

#[derive(Deserialize)]
struct PushDest {
    name: String,
    host: String,
    path: String,
}

#[derive(Deserialize)]
struct PushReq {
    src_path: String,
    dests: Vec<PushDest>,
}

async fn push(
    State(st): State<AgentState>,
    headers: HeaderMap,
    Json(req): Json<PushReq>,
) -> Result<Json<Job>, StatusCode> {
    check_token(&st, &headers)?;
    if req.dests.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }
    let src = std::path::PathBuf::from(&req.src_path);
    if !src.is_dir() {
        return Err(StatusCode::NOT_FOUND);
    }
    let id = uuid::Uuid::new_v4().to_string();
    let names: Vec<String> = req.dests.iter().map(|d| d.name.clone()).collect();
    let job = Job {
        id: id.clone(),
        kind: "sync".into(),
        status: "running".into(),
        log: format!(
            "rsync {} → {}\n",
            req.src_path,
            names.join(", ")
        ),
    };
    st.jobs.lock().await.insert(id.clone(), job.clone());
    let jobs = st.jobs.clone();
    tokio::spawn(async move {
        let mut handles = Vec::new();
        for dest in req.dests {
            let src = src.clone();
            handles.push(tokio::task::spawn_blocking(move || {
                let result = models::rsync_push(&src, &dest.host, std::path::Path::new(&dest.path));
                (dest.name, dest.host, result)
            }));
        }
        let mut failed = false;
        for handle in handles {
            let (name, host, result) = match handle.await {
                Ok(v) => v,
                Err(e) => {
                    failed = true;
                    let mut g = jobs.lock().await;
                    if let Some(j) = g.get_mut(&id) {
                        j.log.push_str(&format!("\n== join error ==\n{e}\n"));
                    }
                    continue;
                }
            };
            let mut g = jobs.lock().await;
            if let Some(j) = g.get_mut(&id) {
                match result {
                    Ok(log) => {
                        j.log.push_str(&format!("\n== {name} ({host}) ok ==\n{log}\n"));
                    }
                    Err(e) => {
                        failed = true;
                        j.log.push_str(&format!("\n== {name} ({host}) error ==\n{e:#}\n"));
                    }
                }
            }
        }
        let mut g = jobs.lock().await;
        if let Some(j) = g.get_mut(&id) {
            j.status = if failed { "error" } else { "ok" }.into();
        }
    });
    Ok(Json(job))
}

#[derive(Deserialize)]
struct StackInstall {
    package: String,
}

async fn stack_install(
    State(st): State<AgentState>,
    headers: HeaderMap,
    Json(req): Json<StackInstall>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    check_token(&st, &headers)?;
    let py = st.cfg.python.clone();
    let pkg = req.package.clone();
    let result = tokio::task::spawn_blocking(move || stack::install(&py, &pkg))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    match result {
        Ok(log) => {
            let probed = stack::probe(&st.cfg.python);
            st.stack.lock().await.set(probed.clone());
            Ok(Json(serde_json::json!({"ok": true, "log": log, "stack": probed})))
        }
        Err(e) => Ok(Json(serde_json::json!({"ok": false, "error": format!("{e:#}")}))),
    }
}

async fn job(
    State(st): State<AgentState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<Job>, StatusCode> {
    check_token(&st, &headers)?;
    st.jobs
        .lock()
        .await
        .get(&id)
        .cloned()
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}
