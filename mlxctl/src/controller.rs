use crate::config::{Config, EndpointConfig, NodeConfig};
use crate::models::{LocalModel, ServeProfile};
use crate::proxy;
use crate::serve;
use crate::state::PersistentState;
use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Request, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::{watch, Mutex};
use tokio::time::sleep;
use tower_http::services::{ServeDir, ServeFile};
use tracing::info;

#[derive(Clone)]
struct App {
    cfg: Arc<Config>,
    persist: Arc<Mutex<PersistentState>>,
    serving: Arc<Mutex<Serving>>,
    nodes: Arc<Mutex<Vec<NodeView>>>,
    endpoint_tx: watch::Sender<EndpointConfig>,
    sync: Arc<Mutex<Option<SyncView>>>,
    pull: Arc<Mutex<Option<PullView>>>,
}

#[derive(Default)]
struct Serving {
    status: String,
    error: Option<String>,
    pid: Option<u32>,
    started_at: Option<String>,
    model_id: Option<String>,
    model_path: Option<String>,
    runtime: String,
    profile: Option<ServeProfile>,
}

#[derive(Clone, Serialize, Default)]
struct NodeView {
    name: String,
    ssh: String,
    rank: u32,
    memory_gb: u32,
    reachable: bool,
    hostname: Option<String>,
    memory_total_bytes: Option<u64>,
    memory_used_bytes: Option<u64>,
    python_ok: Option<bool>,
    mlx_ok: Option<bool>,
    mlx_lm: Option<String>,
    mlx_vlm: Option<String>,
    huggingface_hub: Option<String>,
    python: Option<String>,
    model_roots: Vec<String>,
    rdma_enabled: Option<bool>,
    mesh_ready: Option<bool>,
    links: Vec<crate::network::LinkStatus>,
    error: Option<String>,
}

#[derive(Serialize)]
struct StatusView {
    cluster_id: String,
    controller: String,
    serving: ServingView,
    endpoint: EndpointView,
    autostart: bool,
    nodes: Vec<NodeView>,
    mesh_ready: bool,
    stack: StackView,
    hub: HubView,
    sync: Option<SyncView>,
    pull: Option<PullView>,
}

#[derive(Clone, Serialize, Deserialize, Default)]
struct SyncView {
    id: String,
    status: String,
    source: String,
    model_id: String,
    dests: Vec<String>,
    log: String,
}

#[derive(Clone, Serialize, Deserialize, Default)]
struct PullView {
    id: String,
    status: String,
    node: String,
    repo: String,
    dest: String,
    log: String,
}

#[derive(Serialize)]
struct ServingView {
    status: String,
    error: Option<String>,
    pid: Option<u32>,
    started_at: Option<String>,
    model_id: Option<String>,
    model_path: Option<String>,
    runtime: String,
    profile: Option<ServeProfile>,
}

#[derive(Serialize)]
struct StackPkg {
    installed: usize,
    total: usize,
    version: Option<String>,
}

#[derive(Serialize)]
struct StackView {
    mlx_lm: StackPkg,
    mlx_vlm: StackPkg,
}

#[derive(Serialize)]
struct HubView {
    token_set: bool,
    endpoint: String,
    python: String,
    dest_dir: String,
    node: String,
}

#[derive(Serialize)]
struct EndpointView {
    advertise_host: String,
    bind: String,
    port: u16,
    api_key: String,
    enabled: bool,
    public_url: String,
    claude_url: String,
    internal_url: String,
}

pub async fn run(cfg: Config) -> anyhow::Result<()> {
    std::fs::create_dir_all(&cfg.log_dir).ok();
    let mut persist = PersistentState::load(&cfg.state_path);
    if persist.endpoint.is_none() {
        persist.endpoint = Some(cfg.endpoint.clone());
    }
    let endpoint = persist.endpoint.clone().unwrap_or_else(|| cfg.endpoint.clone());
    let (endpoint_tx, endpoint_rx) = watch::channel(endpoint.clone());
    proxy::spawn(cfg.internal_infer_port, endpoint_rx);

    let mut serving = Serving {
        status: "stopped".into(),
        runtime: persist.runtime.clone(),
        model_id: persist.model_id.clone(),
        model_path: persist.model_path.clone(),
        ..Default::default()
    };
    if persist.desired_serving {
        if let Some(pid) = persist.launch_pid {
            if serve::pid_alive(pid) {
                serving.status = "ready".into();
                serving.pid = Some(pid);
                serving.started_at = Some(chrono::Local::now().to_rfc3339());
                if let Some(path) = serving.model_path.as_deref() {
                    serving.profile = crate::models::inspect_path(std::path::Path::new(path))
                        .map(|m| m.profile);
                }
            }
        }
    }

    let app_state = App {
        persist: Arc::new(Mutex::new(persist)),
        serving: Arc::new(Mutex::new(serving)),
        nodes: Arc::new(Mutex::new(
            cfg.nodes
                .iter()
                .map(|n| NodeView {
                    name: n.name.clone(),
                    ssh: n.ssh.clone(),
                    rank: n.rank,
                    memory_gb: n.memory_gb,
                    ..Default::default()
                })
                .collect(),
        )),
        endpoint_tx,
        cfg: Arc::new(cfg.clone()),
        sync: Arc::new(Mutex::new(None)),
        pull: Arc::new(Mutex::new(None)),
    };

    let poller = app_state.clone();
    tokio::spawn(async move { poll_loop(poller).await });
    let recon = app_state.clone();
    tokio::spawn(async move { reconcile_loop(recon).await });

    let ui_dir = {
        let configured = cfg.ui_dir();
        let bundled = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("web");
        if configured.join("index.html").exists() {
            configured
        } else {
            bundled
        }
    };
    let index = ui_dir.join("index.html");
    let mut api = Router::new()
        .route("/api/status", get(api_status))
        .route("/api/cluster/start", post(api_cluster_start))
        .route("/api/cluster/stop", post(api_cluster_stop))
        .route("/api/serve/start", post(api_serve_start))
        .route("/api/serve/stop", post(api_serve_stop))
        .route("/api/models", get(api_models))
        .route("/api/models/pull", post(api_pull))
        .route("/api/models/sync", post(api_sync))
        .route("/api/models/delete", post(api_delete))
        .route("/api/hub", get(api_hub).put(api_hub_put))
        .route("/api/endpoints", get(api_endpoints).put(api_endpoints_put))
        .route("/api/logs", get(api_logs))
        .route("/api/logs/clear", post(api_logs_clear))
        .route("/api/infer/chat", post(api_infer_chat))
        .route("/api/stack/install", post(api_stack_install))
        .with_state(app_state);

    if index.exists() {
        api = api.fallback_service(
            ServeDir::new(&ui_dir).not_found_service(ServeFile::new(&index)),
        );
    } else {
        api = api.fallback(get(fallback_ui));
    }

    info!("mlxctl controller on {}", cfg.controller_bind);
    let listener = TcpListener::bind(&cfg.controller_bind).await?;
    axum::serve(listener, api).await?;
    Ok(())
}

async fn poll_loop(app: App) {
    loop {
        refresh_nodes(&app).await;
        refresh_sync(&app).await;
        refresh_pull(&app).await;
        {
            let mut s = app.serving.lock().await;
            if let Some(pid) = s.pid {
                if !serve::pid_alive(pid) && s.status == "ready" {
                    s.status = "error".into();
                    s.error = Some("mlx.launch exited".into());
                    s.pid = None;
                }
            }
        }
        sleep(Duration::from_secs(3)).await;
    }
}

async fn reconcile_loop(app: App) {
    sleep(Duration::from_secs(8)).await;
    loop {
        let persist = app.persist.lock().await.clone();
        let status = app.serving.lock().await.status.clone();
        if persist.autostart && persist.desired_serving && persist.model_path.is_some() {
            if status == "stopped" {
                info!("autostart: bringing cluster and model back");
                let _ = cluster_up(&app).await;
                if let (Some(id), Some(path)) = (persist.model_id.clone(), persist.model_path.clone()) {
                    let runtime = persist.runtime.clone();
                    let profile = crate::models::inspect_path(std::path::Path::new(&path))
                        .map(|m| m.profile);
                    let _ = start_serve(&app, id, path, runtime, profile).await;
                }
            }
        }
        sleep(Duration::from_secs(10)).await;
    }
}

async fn refresh_nodes(app: &App) {
    let mut views = Vec::new();
    for node in &app.cfg.nodes {
        views.push(fetch_node(app, node).await);
    }
    *app.nodes.lock().await = views;
}

async fn agent_call(token: &str, url: &str, method: &str, body: Option<&str>, timeout: u64) -> Result<(u16, String), String> {
    let mut cmd = tokio::process::Command::new("/usr/bin/curl");
    cmd.arg("-sS")
        .arg("-m")
        .arg(timeout.to_string())
        .arg("-X")
        .arg(method)
        .arg("-H")
        .arg(format!("Authorization: Bearer {token}"))
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("-w")
        .arg("\n__STATUS__%{http_code}");
    if let Some(b) = body {
        cmd.arg("--data-binary").arg(b);
    }
    cmd.arg(url);
    let out = cmd.output().await.map_err(|e| e.to_string())?;
    if !out.status.success() && out.stdout.is_empty() {
        return Err(format!(
            "curl {}: {}",
            url,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    if let Some((body, status)) = text.rsplit_once("\n__STATUS__") {
        let code = status.trim().parse().unwrap_or(0);
        Ok((code, body.to_string()))
    } else {
        Err(format!("bad curl output for {url}"))
    }
}

async fn fetch_node(app: &App, node: &NodeConfig) -> NodeView {
    let mut v = NodeView {
        name: node.name.clone(),
        ssh: node.ssh.clone(),
        rank: node.rank,
        memory_gb: node.memory_gb,
        ..Default::default()
    };
    let url = format!("{}/v1/info", app.cfg.agent_url(node));
    match agent_call(&app.cfg.token, &url, "GET", None, 12).await {
        Ok((200, body)) => {
            if let Ok(body) = serde_json::from_str::<serde_json::Value>(&body) {
                v.reachable = true;
                v.hostname = body
                    .get("hostname")
                    .and_then(|x| x.as_str())
                    .map(|s| s.to_string());
                v.memory_total_bytes = body.get("memory_total_bytes").and_then(|x| x.as_u64());
                v.memory_used_bytes = body.get("memory_used_bytes").and_then(|x| x.as_u64());
                v.python_ok = body.get("python_ok").and_then(|x| x.as_bool());
                v.mlx_ok = body.get("mlx_ok").and_then(|x| x.as_bool());
                v.mlx_lm = body
                    .get("mlx_lm")
                    .and_then(|x| x.as_str())
                    .map(|s| s.to_string());
                v.mlx_vlm = body
                    .get("mlx_vlm")
                    .and_then(|x| x.as_str())
                    .map(|s| s.to_string());
                v.huggingface_hub = body
                    .get("huggingface_hub")
                    .and_then(|x| x.as_str())
                    .map(|s| s.to_string());
                v.python = body
                    .get("python")
                    .and_then(|x| x.as_str())
                    .map(|s| s.to_string());
                if let Some(roots) = body.get("model_roots").and_then(|x| x.as_array()) {
                    v.model_roots = roots
                        .iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect();
                }
                if let Some(net) = body.get("network") {
                    v.rdma_enabled = net.get("rdma_enabled").and_then(|x| x.as_bool());
                    v.mesh_ready = net.get("mesh_ready").and_then(|x| x.as_bool());
                    if let Some(links) = net.get("links") {
                        v.links = serde_json::from_value(links.clone()).unwrap_or_default();
                    }
                }
            }
        }
        Ok((code, body)) => {
            v.error = Some(format!("agent HTTP {code} {body}"));
        }
        Err(e) => {
            v.error = Some(e);
        }
    }
    v
}

fn endpoint_view(cfg: &Config, ep: &EndpointConfig) -> EndpointView {
    EndpointView {
        advertise_host: ep.advertise_host.clone(),
        bind: ep.bind.clone(),
        port: ep.port,
        api_key: ep.api_key.clone(),
        enabled: ep.enabled,
        public_url: proxy::public_url(ep),
        claude_url: proxy::claude_base_url(ep),
        internal_url: format!("http://127.0.0.1:{}/v1", cfg.internal_infer_port),
    }
}

async fn snapshot(app: &App) -> StatusView {
    let persist = app.persist.lock().await.clone();
    let serving = app.serving.lock().await;
    let nodes = app.nodes.lock().await.clone();
    let ep = persist
        .endpoint
        .clone()
        .unwrap_or_else(|| app.cfg.endpoint.clone());
    let mesh_ready = nodes.iter().all(|n| n.reachable && n.mesh_ready.unwrap_or(false));
    StatusView {
        cluster_id: app.cfg.cluster_id.clone(),
        controller: app.cfg.controller_node().ssh.clone(),
        serving: ServingView {
            status: serving.status.clone(),
            error: serving.error.clone(),
            pid: serving.pid,
            started_at: serving.started_at.clone(),
            model_id: serving.model_id.clone(),
            model_path: serving.model_path.clone(),
            runtime: serving.runtime.clone(),
            profile: serving.profile.clone(),
        },
        endpoint: endpoint_view(&app.cfg, &ep),
        autostart: persist.autostart,
        nodes: nodes.clone(),
        mesh_ready,
        stack: summarize_stack(&nodes),
        hub: hub_view(&app.cfg, &persist, &nodes),
        sync: app.sync.lock().await.clone(),
        pull: app.pull.lock().await.clone(),
    }
}

fn summarize_stack(nodes: &[NodeView]) -> StackView {
    let total = nodes.len();
    let lm: Vec<_> = nodes.iter().filter_map(|n| n.mlx_lm.clone()).collect();
    let vlm: Vec<_> = nodes.iter().filter_map(|n| n.mlx_vlm.clone()).collect();
    StackView {
        mlx_lm: StackPkg {
            installed: lm.len(),
            total,
            version: lm.first().cloned(),
        },
        mlx_vlm: StackPkg {
            installed: vlm.len(),
            total,
            version: vlm.first().cloned(),
        },
    }
}

async fn api_status(State(app): State<App>) -> Json<StatusView> {
    Json(snapshot(&app).await)
}

async fn api_cluster_start(State(app): State<App>) -> Response {
    match cluster_up(&app).await {
        Ok(_) => Json(snapshot(&app).await).into_response(),
        Err(e) => (StatusCode::BAD_GATEWAY, e).into_response(),
    }
}

async fn api_cluster_stop(State(app): State<App>) -> Response {
    let _ = stop_serve(&app).await;
    {
        let mut p = app.persist.lock().await;
        p.desired_serving = false;
        p.autostart = false;
        let _ = p.save(&app.cfg.state_path);
    }
    Json(snapshot(&app).await).into_response()
}

#[derive(Deserialize)]
struct ServeStart {
    model_id: String,
    #[serde(default)]
    runtime: Option<String>,
    #[serde(default = "yes")]
    autostart: bool,
}

fn yes() -> bool {
    true
}

async fn api_serve_start(State(app): State<App>, Json(req): Json<ServeStart>) -> Response {
    let models = gather_models(&app).await;
    let found = models.into_iter().find(|m| {
        m.id == req.model_id || m.name == req.model_id || m.path == req.model_id
    });
    let Some(model) = found else {
        return (StatusCode::NOT_FOUND, "model not found on controller scan").into_response();
    };
    if !model.complete {
        return (StatusCode::BAD_REQUEST, "model is incomplete").into_response();
    }
    if !model.cluster_complete {
        return (
            StatusCode::BAD_REQUEST,
            format!(
                "模型只在 {} 上完整，先点「同步到四台」",
                model.source_node.as_deref().unwrap_or("一台")
            ),
        )
            .into_response();
    }
    let runtime = req.runtime.unwrap_or_else(|| model.profile.default_runtime.clone());
    if runtime != "mlx_lm" && runtime != "mlx_vlm" {
        return (StatusCode::BAD_REQUEST, "runtime must be mlx_lm or mlx_vlm").into_response();
    }
    if runtime == "mlx_lm" && !model.profile.allow_lm {
        return (
            StatusCode::BAD_REQUEST,
            format!(
                "{} 必须走 mlx-vlm（{}），当前 mlx-lm 没有该架构",
                model.profile.title, model.profile.load_vlm_label
            ),
        )
            .into_response();
    }
    let nodes = app.nodes.lock().await.clone();
    let missing: Vec<_> = nodes
        .iter()
        .filter(|n| {
            if runtime == "mlx_vlm" {
                n.mlx_vlm.as_ref().map(|s| s.is_empty()).unwrap_or(true)
            } else {
                n.mlx_lm.as_ref().map(|s| s.is_empty()).unwrap_or(true)
            }
        })
        .map(|n| n.name.clone())
        .collect();
    if !missing.is_empty() {
        return (
            StatusCode::CONFLICT,
            format!("{runtime} 未安装: {}", missing.join(", ")),
        )
            .into_response();
    }
    let mesh_ready = nodes
        .iter()
        .all(|n| n.reachable && n.mesh_ready.unwrap_or(false));
    if !mesh_ready {
        if let Err(e) = cluster_up(&app).await {
            return (StatusCode::BAD_GATEWAY, e).into_response();
        }
    }
    {
        let mut p = app.persist.lock().await;
        p.desired_serving = true;
        p.autostart = req.autostart;
        p.model_id = Some(model.id.clone());
        p.model_path = Some(model.path.clone());
        p.runtime = runtime.clone();
        let _ = p.save(&app.cfg.state_path);
    }
    match start_serve(&app, model.id, model.path, runtime, Some(model.profile)).await {
        Ok(_) => Json(snapshot(&app).await).into_response(),
        Err(e) => (StatusCode::BAD_GATEWAY, e).into_response(),
    }
}

async fn api_serve_stop(State(app): State<App>) -> Json<StatusView> {
    let _ = stop_serve(&app).await;
    let mut p = app.persist.lock().await;
    p.desired_serving = false;
    let _ = p.save(&app.cfg.state_path);
    drop(p);
    Json(snapshot(&app).await)
}

async fn api_models(State(app): State<App>) -> Json<Vec<LocalModel>> {
    Json(gather_models(&app).await)
}

fn hub_view(cfg: &Config, persist: &PersistentState, nodes: &[NodeView]) -> HubView {
    let hub = &persist.hub;
    let python = if hub.python.trim().is_empty() {
        cfg.python.display().to_string()
    } else {
        hub.python.clone()
    };
    let dest_dir = if hub.dest_dir.trim().is_empty() {
        cfg.model_roots
            .first()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| {
                crate::config::home_dir()
                    .join(".exo/models")
                    .display()
                    .to_string()
            })
    } else {
        hub.dest_dir.clone()
    };
    let node = if !hub.node.trim().is_empty()
        && nodes.iter().any(|n| n.name == hub.node)
    {
        hub.node.clone()
    } else {
        nodes
            .iter()
            .find(|n| n.huggingface_hub.is_some())
            .or_else(|| nodes.iter().find(|n| n.rank == 0))
            .or_else(|| nodes.first())
            .map(|n| n.name.clone())
            .unwrap_or_default()
    };
    HubView {
        token_set: !hub.token.trim().is_empty(),
        endpoint: hub.endpoint.clone(),
        python,
        dest_dir,
        node,
    }
}

async fn api_hub(State(app): State<App>) -> Json<HubView> {
    let persist = app.persist.lock().await.clone();
    let nodes = app.nodes.lock().await.clone();
    Json(hub_view(&app.cfg, &persist, &nodes))
}

#[derive(Deserialize)]
struct HubPut {
    #[serde(default)]
    token: Option<String>,
    #[serde(default)]
    token_clear: bool,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    python: Option<String>,
    #[serde(default)]
    dest_dir: Option<String>,
    #[serde(default)]
    node: Option<String>,
}

async fn api_hub_put(State(app): State<App>, Json(req): Json<HubPut>) -> Response {
    let mut persist = app.persist.lock().await;
    if req.token_clear {
        persist.hub.token.clear();
    } else if let Some(token) = req.token {
        let t = token.trim();
        if !t.is_empty() {
            persist.hub.token = t.to_string();
        }
    }
    if let Some(endpoint) = req.endpoint {
        persist.hub.endpoint = endpoint.trim().trim_end_matches('/').to_string();
    }
    if let Some(python) = req.python {
        persist.hub.python = python.trim().to_string();
    }
    if let Some(dest_dir) = req.dest_dir {
        persist.hub.dest_dir = dest_dir.trim().to_string();
    }
    if let Some(node) = req.node {
        persist.hub.node = node.trim().to_string();
    }
    if let Err(e) = persist.save(&app.cfg.state_path) {
        return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
    }
    let nodes = app.nodes.lock().await.clone();
    Json(hub_view(&app.cfg, &persist, &nodes)).into_response()
}

#[derive(Deserialize)]
struct PullReq {
    repo: String,
    #[serde(default)]
    nodes: Vec<String>,
    #[serde(default)]
    dest: Option<String>,
}

async fn api_pull(State(app): State<App>, Json(req): Json<PullReq>) -> Response {
    let repo = req.repo.trim().to_string();
    if repo.is_empty() {
        return (StatusCode::BAD_REQUEST, "repo is empty").into_response();
    }
    let persist = app.persist.lock().await.clone();
    let views = app.nodes.lock().await.clone();
    let hub = hub_view(&app.cfg, &persist, &views);
    let mut names = req.nodes.clone();
    if names.is_empty() {
        if hub.node.is_empty() {
            return (StatusCode::BAD_REQUEST, "没有指定下载机器").into_response();
        }
        names.push(hub.node.clone());
    }
    names.sort();
    names.dedup();
    let dest = req
        .dest
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(hub.dest_dir.as_str())
        .to_string();
    let mut logs = Vec::new();
    for name in names {
        let Some(node) = app.cfg.nodes.iter().find(|n| n.name == name) else {
            logs.push(serde_json::json!({"node": name, "error": "unknown node"}));
            continue;
        };
        let ready = views
            .iter()
            .find(|n| n.name == name)
            .and_then(|n| n.huggingface_hub.as_ref())
            .is_some();
        if !ready {
            logs.push(serde_json::json!({
                "node": name,
                "error": "this node has no huggingface_hub in the configured venv",
            }));
            continue;
        }
        let url = format!("{}/v1/models/pull", app.cfg.agent_url(node));
        let body = serde_json::json!({
            "repo": repo,
            "dest": dest,
            "token": persist.hub.token,
            "endpoint": persist.hub.endpoint,
            "python": hub.python,
        })
        .to_string();
        match agent_call(&app.cfg.token, &url, "POST", Some(&body), 30).await {
            Ok((status, body)) => {
                if (200..300).contains(&status) {
                    if let Ok(job) = serde_json::from_str::<serde_json::Value>(&body) {
                        if let Some(id) = job.get("id").and_then(|x| x.as_str()) {
                            *app.pull.lock().await = Some(PullView {
                                id: id.to_string(),
                                status: job
                                    .get("status")
                                    .and_then(|x| x.as_str())
                                    .unwrap_or("running")
                                    .to_string(),
                                node: node.name.clone(),
                                repo: repo.clone(),
                                dest: dest.clone(),
                                log: job
                                    .get("log")
                                    .and_then(|x| x.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                            });
                        }
                    }
                }
                logs.push(serde_json::json!({
                    "node": node.name,
                    "status": status,
                    "body": body,
                }));
            }
            Err(e) => logs.push(serde_json::json!({"node": node.name, "error": e})),
        }
    }
    Json(serde_json::json!({"repo": repo, "dest": dest, "nodes": logs})).into_response()
}

#[derive(Deserialize)]
struct DeleteReq {
    model_id: String,
}

async fn api_delete(State(app): State<App>, Json(req): Json<DeleteReq>) -> Response {
    let model_id = req.model_id.trim().to_string();
    if model_id.is_empty() {
        return (StatusCode::BAD_REQUEST, "model_id is empty").into_response();
    }
    {
        let serving = app.serving.lock().await;
        let loaded = serving.model_id.as_deref() == Some(model_id.as_str())
            || serving
                .model_path
                .as_ref()
                .is_some_and(|p| p.contains(&model_id));
        if loaded && serving.status != "stopped" {
            drop(serving);
            let _ = stop_serve(&app).await;
            let mut p = app.persist.lock().await;
            p.desired_serving = false;
            let _ = p.save(&app.cfg.state_path);
        }
    }
    let folder = model_id.replace('/', "--");
    let mut logs = Vec::new();
    for node in &app.cfg.nodes {
        let url = format!("{}/v1/models/delete", app.cfg.agent_url(node));
        let body = serde_json::json!({"model_id": folder}).to_string();
        match agent_call(&app.cfg.token, &url, "POST", Some(&body), 120).await {
            Ok((status, body)) => logs.push(serde_json::json!({
                "node": node.name,
                "status": status,
                "body": body,
            })),
            Err(e) => logs.push(serde_json::json!({"node": node.name, "error": e})),
        }
    }
    Json(serde_json::json!({"model_id": model_id, "nodes": logs})).into_response()
}

#[derive(Deserialize)]
struct SyncReq {
    model_id: String,
}

async fn api_sync(State(app): State<App>, Json(req): Json<SyncReq>) -> Response {
    if let Some(cur) = app.sync.lock().await.as_ref() {
        if cur.status == "running" {
            return (
                StatusCode::CONFLICT,
                format!("已有同步在跑：{} → {}", cur.source, cur.model_id),
            )
                .into_response();
        }
    }
    let models = gather_models(&app).await;
    let Some(model) = models.into_iter().find(|m| {
        m.id == req.model_id || m.name == req.model_id || m.path == req.model_id
    }) else {
        return (StatusCode::NOT_FOUND, "model not found").into_response();
    };
    let Some(source_name) = model.source_node.clone() else {
        return (StatusCode::BAD_REQUEST, "没有完整副本可作源").into_response();
    };
    let Some(src_node) = app.cfg.nodes.iter().find(|n| n.name == source_name).cloned() else {
        return (StatusCode::BAD_REQUEST, "source node missing from config").into_response();
    };
    let Some(src_replica) = model
        .replicas
        .iter()
        .find(|r| r.node == source_name && r.complete)
        .cloned()
    else {
        return (StatusCode::BAD_REQUEST, "source replica incomplete").into_response();
    };
    let dest_root = app
        .cfg
        .model_roots
        .first()
        .cloned()
        .unwrap_or_else(|| crate::config::home_dir().join(".exo/models"));
    let dest_path = dest_root.join(&model.id);
    let mut dests = Vec::new();
    for node in &app.cfg.nodes {
        if node.name == source_name {
            continue;
        }
        let Some(host) = app.cfg.hop_ip(&source_name, &node.name) else {
            continue;
        };
        dests.push(serde_json::json!({
            "name": node.name,
            "host": host,
            "path": dest_path.display().to_string(),
        }));
    }
    if dests.is_empty() {
        return (StatusCode::BAD_REQUEST, "没有需要同步的目标节点").into_response();
    }
    let body = serde_json::json!({
        "src_path": src_replica.path,
        "dests": dests,
    })
    .to_string();
    let url = format!("{}/v1/models/push", app.cfg.agent_url(&src_node));
    match agent_call(&app.cfg.token, &url, "POST", Some(&body), 20).await {
        Ok((code, text)) if (200..300).contains(&code) => {
            let job: serde_json::Value = serde_json::from_str(&text).unwrap_or(serde_json::json!({}));
            let id = job
                .get("id")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            if id.is_empty() {
                return (
                    StatusCode::BAD_GATEWAY,
                    format!("source agent 未返回 job: {text}"),
                )
                    .into_response();
            }
            let dest_names: Vec<String> = dests
                .iter()
                .filter_map(|d| d.get("name").and_then(|x| x.as_str()).map(|s| s.to_string()))
                .collect();
            let view = SyncView {
                id: id.clone(),
                status: "running".into(),
                source: source_name,
                model_id: model.id.clone(),
                dests: dest_names,
                log: job
                    .get("log")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
            };
            *app.sync.lock().await = Some(view.clone());
            Json(serde_json::json!({
                "ok": true,
                "sync": view,
            }))
            .into_response()
        }
        Ok((code, text)) => (
            StatusCode::BAD_GATEWAY,
            format!("push HTTP {code}: {text}"),
        )
            .into_response(),
        Err(e) => (StatusCode::BAD_GATEWAY, e).into_response(),
    }
}

async fn refresh_sync(app: &App) {
    let cur = app.sync.lock().await.clone();
    let Some(cur) = cur else {
        return;
    };
    if cur.status != "running" {
        return;
    }
    let Some(src) = app.cfg.nodes.iter().find(|n| n.name == cur.source) else {
        return;
    };
    let url = format!("{}/v1/jobs/{}", app.cfg.agent_url(src), cur.id);
    let Ok((200, body)) = agent_call(&app.cfg.token, &url, "GET", None, 8).await else {
        return;
    };
    let Ok(job) = serde_json::from_str::<serde_json::Value>(&body) else {
        return;
    };
    let mut g = app.sync.lock().await;
    if let Some(s) = g.as_mut() {
        if s.id == cur.id {
            s.status = job
                .get("status")
                .and_then(|x| x.as_str())
                .unwrap_or(&s.status)
                .to_string();
            if let Some(log) = job.get("log").and_then(|x| x.as_str()) {
                s.log = log.to_string();
            }
        }
    }
}

async fn refresh_pull(app: &App) {
    let cur = app.pull.lock().await.clone();
    let Some(cur) = cur else {
        return;
    };
    if cur.status != "running" {
        return;
    }
    let Some(node) = app.cfg.nodes.iter().find(|n| n.name == cur.node) else {
        return;
    };
    let url = format!("{}/v1/jobs/{}", app.cfg.agent_url(node), cur.id);
    let Ok((200, body)) = agent_call(&app.cfg.token, &url, "GET", None, 8).await else {
        return;
    };
    let Ok(job) = serde_json::from_str::<serde_json::Value>(&body) else {
        return;
    };
    let mut g = app.pull.lock().await;
    if let Some(s) = g.as_mut() {
        if s.id == cur.id {
            s.status = job
                .get("status")
                .and_then(|x| x.as_str())
                .unwrap_or(&s.status)
                .to_string();
            if let Some(log) = job.get("log").and_then(|x| x.as_str()) {
                s.log = log.to_string();
            }
        }
    }
}

async fn api_endpoints(State(app): State<App>) -> Json<EndpointView> {
    let persist = app.persist.lock().await;
    let ep = persist
        .endpoint
        .clone()
        .unwrap_or_else(|| app.cfg.endpoint.clone());
    Json(endpoint_view(&app.cfg, &ep))
}

async fn api_endpoints_put(State(app): State<App>, Json(ep): Json<EndpointConfig>) -> Response {
    if ep.port == 9090 || ep.port == 9091 || ep.port == app.cfg.internal_infer_port {
        return (StatusCode::BAD_REQUEST, "port collides with control plane").into_response();
    }
    {
        let mut p = app.persist.lock().await;
        p.endpoint = Some(ep.clone());
        let _ = p.save(&app.cfg.state_path);
    }
    let _ = app.endpoint_tx.send(ep.clone());
    Json(endpoint_view(&app.cfg, &ep)).into_response()
}

async fn api_logs(State(app): State<App>) -> Json<serde_json::Value> {
    let serve = serve::tail_log(&app.cfg.log_dir.join("serve.log"), 80_000);
    let sync = app.sync.lock().await.clone();
    let pull = app.pull.lock().await.clone();
    Json(serde_json::json!({ "serve": serve, "sync": sync, "pull": pull }))
}

async fn api_logs_clear(State(app): State<App>) -> Json<serde_json::Value> {
    let path = app.cfg.log_dir.join("serve.log");
    if let Err(e) = serve::clear_log(&path) {
        return Json(serde_json::json!({ "ok": false, "error": e.to_string() }));
    }
    {
        let mut sync = app.sync.lock().await;
        if sync.as_ref().map(|s| s.status.as_str()) != Some("running") {
            *sync = None;
        }
    }
    {
        let mut pull = app.pull.lock().await;
        if pull.as_ref().map(|p| p.status.as_str()) != Some("running") {
            *pull = None;
        }
    }
    Json(serde_json::json!({ "ok": true }))
}

async fn api_infer_chat(State(app): State<App>, req: Request<Body>) -> Response {
    let uri = format!(
        "http://127.0.0.1:{}/v1/chat/completions",
        app.cfg.internal_infer_port
    );
    let bytes = match axum::body::to_bytes(req.into_body(), 32 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    let mut payload = bytes.clone();
    if let Ok(mut v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
        let path = app.serving.lock().await.model_path.clone();
        if let Some(path) = path {
            v["model"] = serde_json::json!(path);
            payload = serde_json::to_vec(&v).unwrap_or_else(|_| bytes.to_vec()).into();
        }
    }
    let out = match Request::builder()
        .method("POST")
        .uri(&uri)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json, text/event-stream")
        .body(Body::from(payload))
    {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_GATEWAY, e.to_string()).into_response(),
    };
    let client = Client::builder(TokioExecutor::new()).build(HttpConnector::new());
    match client.request(out).await {
        Ok(resp) => resp.into_response(),
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            format!("model server unavailable: {e}"),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct StackInstallReq {
    package: String,
}

async fn api_stack_install(State(app): State<App>, Json(req): Json<StackInstallReq>) -> Response {
    if req.package != "mlx-vlm" && req.package != "mlx-lm" && req.package != "mlx" {
        return (StatusCode::BAD_REQUEST, "unsupported package").into_response();
    }
    let body = serde_json::json!({"package": req.package}).to_string();
    let mut logs = Vec::new();
    for node in &app.cfg.nodes {
        let url = format!("{}/v1/stack/install", app.cfg.agent_url(node));
        match agent_call(&app.cfg.token, &url, "POST", Some(&body), 300).await {
            Ok((status, text)) => logs.push(serde_json::json!({
                "node": node.name,
                "status": status,
                "body": text,
            })),
            Err(e) => logs.push(serde_json::json!({"node": node.name, "error": e})),
        }
    }
    refresh_nodes(&app).await;
    Json(serde_json::json!({"package": req.package, "nodes": logs, "stack": snapshot(&app).await.stack})).into_response()
}

async fn gather_models(app: &App) -> Vec<LocalModel> {
    let mut by_id: HashMap<String, LocalModel> = HashMap::new();
    let mut seen_any = false;
    for node in &app.cfg.nodes {
        let url = format!("{}/v1/models", app.cfg.agent_url(node));
        let Ok((200, body)) = agent_call(&app.cfg.token, &url, "GET", None, 12).await else {
            continue;
        };
        let Ok(list) = serde_json::from_str::<Vec<LocalModel>>(&body) else {
            continue;
        };
        seen_any = true;
        for mut m in list {
            crate::models::attach_profile(&mut m);
            let replica = crate::models::ModelReplica {
                node: node.name.clone(),
                complete: m.complete,
                size_bytes: m.size_bytes,
                path: m.path.clone(),
                downloading: m.downloading,
            };
            by_id
                .entry(m.id.clone())
                .and_modify(|acc| {
                    acc.replicas.push(replica.clone());
                    acc.downloading = acc.downloading || replica.downloading;
                    if replica.size_bytes > acc.size_bytes {
                        acc.size_bytes = replica.size_bytes;
                    }
                    if replica.complete
                        && (replica.size_bytes >= acc.size_bytes || !acc.complete)
                    {
                        acc.complete = true;
                        acc.path = replica.path.clone();
                        acc.kind = m.kind.clone();
                        acc.architecture = m.architecture.clone();
                        acc.profile = m.profile.clone();
                        acc.name = m.name.clone();
                    }
                })
                .or_insert_with(|| {
                    m.replicas = vec![replica];
                    m
                });
        }
    }
    if !seen_any {
        for m in crate::models::scan_models(&app.cfg.model_roots) {
            by_id.insert(m.id.clone(), m);
        }
        let mut v: Vec<_> = by_id.into_values().collect();
        for m in &mut v {
            m.cluster_complete = m.complete;
        }
        v.sort_by(|a, b| b.complete.cmp(&a.complete).then(a.name.cmp(&b.name)));
        return v;
    }
    let expected: Vec<String> = app.cfg.nodes.iter().map(|n| n.name.clone()).collect();
    for m in by_id.values_mut() {
        for name in &expected {
            if !m.replicas.iter().any(|r| r.node == *name) {
                m.replicas.push(crate::models::ModelReplica {
                    node: name.clone(),
                    complete: false,
                    size_bytes: 0,
                    path: String::new(),
                    downloading: false,
                });
            }
        }
        m.replicas.sort_by(|a, b| a.node.cmp(&b.node));
        m.downloading = m.replicas.iter().any(|r| r.downloading);
        m.cluster_complete = expected.iter().all(|name| {
            m.replicas
                .iter()
                .any(|r| r.node == *name && r.complete)
        });
        m.source_node = m
            .replicas
            .iter()
            .filter(|r| r.complete)
            .max_by_key(|r| r.size_bytes)
            .map(|r| r.node.clone());
        m.complete = m.source_node.is_some();
        crate::models::attach_profile(m);
    }
    let mut v: Vec<_> = by_id.into_values().collect();
    v.sort_by(|a, b| {
        b.cluster_complete
            .cmp(&a.cluster_complete)
            .then(b.complete.cmp(&a.complete))
            .then(a.name.cmp(&b.name))
    });
    v
}

async fn cluster_up(app: &App) -> Result<(), String> {
    for node in &app.cfg.nodes {
        let url = format!("{}/v1/network/up", app.cfg.agent_url(node));
        let (code, body) = agent_call(&app.cfg.token, &url, "POST", Some("{}"), 90)
            .await
            .map_err(|e| format!("{}: {e}", node.name))?;
        if !(200..300).contains(&code) {
            return Err(format!("{} network/up HTTP {code} {body}", node.name));
        }
    }
    refresh_nodes(app).await;
    Ok(())
}

async fn start_serve(
    app: &App,
    model_id: String,
    model_path: String,
    runtime: String,
    profile: Option<ServeProfile>,
) -> Result<(), String> {
    let _ = stop_serve(app).await;
    let port = app.cfg.internal_infer_port;
    if let Err(e) = serve::wait_port_free(port, Duration::from_secs(20)).await {
        return Err(format!("旧模型还没退出：{e}"));
    }
    {
        let mut s = app.serving.lock().await;
        s.status = "starting".into();
        s.error = None;
        s.model_id = Some(model_id.clone());
        s.model_path = Some(model_path.clone());
        s.runtime = runtime.clone();
        s.profile = profile;
        s.started_at = Some(chrono::Local::now().to_rfc3339());
    }
    let cfg = app.cfg.clone();
    let path = model_path.clone();
    let rt = runtime.clone();
    let pid = tokio::task::spawn_blocking(move || serve::spawn_launch(&cfg, &path, &rt))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    {
        let mut s = app.serving.lock().await;
        s.pid = Some(pid);
        let mut p = app.persist.lock().await;
        p.launch_pid = Some(pid);
        let _ = p.save(&app.cfg.state_path);
    }
    let serving = app.serving.clone();
    tokio::spawn(async move {
        match serve::wait_ready(port, Duration::from_secs(2700), Some(pid)).await {
            Ok(()) => {
                let mut s = serving.lock().await;
                if s.pid == Some(pid) {
                    s.status = "ready".into();
                }
            }
            Err(e) => {
                let mut s = serving.lock().await;
                if s.pid == Some(pid) {
                    s.status = "error".into();
                    s.error = Some(e.to_string());
                }
            }
        }
    });
    Ok(())
}

async fn stop_serve(app: &App) -> Result<(), String> {
    {
        let mut s = app.serving.lock().await;
        s.status = "stopping".into();
        if let Some(pid) = s.pid.take() {
            tokio::task::spawn_blocking(move || serve::stop_pid(pid))
                .await
                .ok();
        }
    }
    for node in &app.cfg.nodes {
        let url = format!("{}/v1/cleanup", app.cfg.agent_url(node));
        let _ = agent_call(&app.cfg.token, &url, "POST", Some("{}"), 8).await;
    }
    let mut s = app.serving.lock().await;
    s.status = "stopped".into();
    s.pid = None;
    Ok(())
}

async fn fallback_ui() -> Html<&'static str> {
    Html(include_str!("../web/index.html"))
}

