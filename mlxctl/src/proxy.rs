use crate::config::EndpointConfig;
use crate::anthropic;
use crate::responses;
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, Method, Request, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::Router;
use http_body_util::BodyExt;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{info, warn};

#[derive(Clone)]
struct ProxyState {
    inner_port: u16,
    endpoint: watch::Receiver<EndpointConfig>,
    client: Client<HttpConnector, Body>,
}

pub fn spawn(inner_port: u16, rx: watch::Receiver<EndpointConfig>) {
    tokio::spawn(async move {
        loop {
            let cfg = rx.borrow().clone();
            let bind_ip = if cfg.enabled {
                cfg.bind.clone()
            } else {
                "127.0.0.1".into()
            };
            let addr: SocketAddr = match format!("{}:{}", bind_ip, cfg.port).parse() {
                Ok(a) => a,
                Err(e) => {
                    warn!("invalid infer bind: {e}");
                    let mut rx2 = rx.clone();
                    let _ = rx2.changed().await;
                    continue;
                }
            };
            info!("inference proxy listening on {addr} (public={})", cfg.enabled);
            let listener = match TcpListener::bind(addr).await {
                Ok(l) => l,
                Err(e) => {
                    warn!("bind inference {addr}: {e}");
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    continue;
                }
            };
            let mut rx_stop = rx.clone();
            let state = ProxyState {
                inner_port,
                endpoint: rx.clone(),
                client: Client::builder(TokioExecutor::new()).build(HttpConnector::new()),
            };
            let app = Router::new()
                .route("/v1/messages", post(anthropic_messages))
                .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
                .route("/v1/responses", post(openai_responses))
                .route(
                    "/v1/responses/{id}",
                    get(openai_responses_get).delete(openai_responses_delete),
                )
                .fallback(any(forward))
                .with_state(state);
            let server = axum::serve(listener, app).with_graceful_shutdown(async move {
                let _ = rx_stop.changed().await;
            });
            if let Err(e) = server.await {
                warn!("inference proxy: {e}");
            }
        }
    });
}

fn authorized(headers: &HeaderMap, api_key: &str) -> bool {
    if api_key.is_empty() {
        return true;
    }
    let bearer_ok = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v == format!("Bearer {api_key}") || v == api_key);
    let x_ok = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v == api_key);
    bearer_ok || x_ok
}

fn anthropic_error(status: StatusCode, kind: &str, message: &str) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        json!({
            "type": "error",
            "error": {"type": kind, "message": message}
        })
        .to_string(),
    )
        .into_response()
}

fn is_anthropic_client(headers: &HeaderMap) -> bool {
    headers.contains_key("anthropic-version") || headers.contains_key("anthropic-beta")
}

async fn forward(State(st): State<ProxyState>, req: Request<Body>) -> Response {
    let cfg = st.endpoint.borrow().clone();
    if !authorized(req.headers(), &cfg.api_key) {
        if is_anthropic_client(req.headers()) {
            return anthropic_error(
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                "missing or invalid api key",
            );
        }
        return (StatusCode::UNAUTHORIZED, "missing or invalid api key").into_response();
    }
    let anthropic_models = req.method() == Method::GET
        && req.uri().path() == "/v1/models"
        && is_anthropic_client(req.headers());
    let path = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| req.uri().path().to_string());
    let uri = format!("http://127.0.0.1:{}{}", st.inner_port, path);
    let (parts, body) = req.into_parts();
    let Ok(uri) = uri.parse() else {
        return (StatusCode::BAD_GATEWAY, "bad upstream uri").into_response();
    };
    let mut out = Request::from_parts(parts, body);
    *out.uri_mut() = uri;
    out.headers_mut().remove(header::HOST);
    match st.client.request(out).await {
        Ok(resp) => {
            if anthropic_models {
                rewrite_models(resp).await
            } else {
                resp.into_response()
            }
        }
        Err(e) => {
            if anthropic_models {
                (
                    [(header::CONTENT_TYPE, "application/json")],
                    anthropic::anthropic_model_list(&json!({"data": []})).to_string(),
                )
                    .into_response()
            } else {
                (
                    StatusCode::BAD_GATEWAY,
                    format!("model server unavailable: {e}"),
                )
                    .into_response()
            }
        }
    }
}

async fn rewrite_models(resp: hyper::Response<hyper::body::Incoming>) -> Response {
    let status = resp.status();
    let bytes = match resp.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                &format!("read models: {e}"),
            )
        }
    };
    let openai: Value = serde_json::from_slice(&bytes).unwrap_or_else(|_| json!({"data": []}));
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        anthropic::anthropic_model_list(&openai).to_string(),
    )
        .into_response()
}

fn openai_error(status: StatusCode, kind: &str, message: &str) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        json!({
            "error": {
                "message": message,
                "type": kind,
                "code": Value::Null,
                "param": Value::Null
            }
        })
        .to_string(),
    )
        .into_response()
}

async fn resolve_model(st: &ProxyState, requested: &str) -> String {
    if !anthropic::looks_like_claude_model(requested)
        && !responses::looks_like_openai_alias(requested)
    {
        return requested.to_string();
    }
    let uri = format!("http://127.0.0.1:{}/v1/models", st.inner_port);
    let Ok(req) = Request::builder()
        .method("GET")
        .uri(&uri)
        .body(Body::empty())
    else {
        return requested.to_string();
    };
    let Ok(resp) = st.client.request(req).await else {
        return requested.to_string();
    };
    let Ok(bytes) = resp.into_body().collect().await else {
        return requested.to_string();
    };
    let Ok(v) = serde_json::from_slice::<Value>(&bytes.to_bytes()) else {
        return requested.to_string();
    };
    v.get("data")
        .and_then(|d| d.as_array())
        .and_then(|d| d.first())
        .and_then(|m| m.get("id"))
        .and_then(|id| id.as_str())
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| requested.to_string())
}

async fn post_chat(
    st: &ProxyState,
    payload: &Value,
) -> Result<hyper::Response<hyper::body::Incoming>, String> {
    let uri = format!("http://127.0.0.1:{}/v1/chat/completions", st.inner_port);
    let req = Request::builder()
        .method("POST")
        .uri(&uri)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json, text/event-stream")
        .body(Body::from(payload.to_string()))
        .map_err(|e| e.to_string())?;
    st.client.request(req).await.map_err(|e| e.to_string())
}

async fn anthropic_count_tokens(State(st): State<ProxyState>, req: Request<Body>) -> Response {
    let cfg = st.endpoint.borrow().clone();
    if !authorized(req.headers(), &cfg.api_key) {
        return anthropic_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "missing or invalid api key",
        );
    }
    let bytes = match axum::body::to_bytes(req.into_body(), 8 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &e.to_string(),
            )
        }
    };
    let body: Value = serde_json::from_slice(&bytes).unwrap_or_else(|_| json!({}));
    (
        [(header::CONTENT_TYPE, "application/json")],
        json!({"input_tokens": anthropic::estimate_tokens(&body)}).to_string(),
    )
        .into_response()
}

async fn anthropic_messages(State(st): State<ProxyState>, req: Request<Body>) -> Response {
    let cfg = st.endpoint.borrow().clone();
    if !authorized(req.headers(), &cfg.api_key) {
        return anthropic_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "missing or invalid api key",
        );
    }
    let bytes = match axum::body::to_bytes(req.into_body(), 64 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &e.to_string(),
            )
        }
    };
    let incoming: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("invalid json: {e}"),
            )
        }
    };
    let stream = incoming.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let requested_model = incoming
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let model = resolve_model(&st, &requested_model).await;
    let openai_req = match anthropic::to_openai_chat(&incoming, &model) {
        Ok(v) => v,
        Err(e) => {
            return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", &e)
        }
    };
    let upstream = match post_chat(&st, &openai_req).await {
        Ok(resp) => resp,
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                &format!("model server unavailable: {e}"),
            )
        }
    };
    if !upstream.status().is_success() {
        return map_upstream_error(upstream).await;
    }
    let content_type = upstream
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    if stream {
        return stream_anthropic(upstream, model, content_type).await;
    }
    let bytes = match upstream.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                &format!("read completion: {e}"),
            )
        }
    };
    if content_type.contains("text/event-stream") {
        return match collect_openai_sse(&bytes, &model) {
            Ok(v) => json_message(v),
            Err(e) => anthropic_error(StatusCode::BAD_GATEWAY, "api_error", &e),
        };
    }
    let openai: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                &format!("invalid upstream json: {e}"),
            )
        }
    };
    json_message(anthropic::from_openai_chat(&openai, &model))
}

async fn openai_responses(State(st): State<ProxyState>, req: Request<Body>) -> Response {
    let cfg = st.endpoint.borrow().clone();
    if !authorized(req.headers(), &cfg.api_key) {
        return openai_error(
            StatusCode::UNAUTHORIZED,
            "invalid_request_error",
            "missing or invalid api key",
        );
    }
    let bytes = match axum::body::to_bytes(req.into_body(), 64 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &e.to_string(),
            )
        }
    };
    let incoming: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("invalid json: {e}"),
            )
        }
    };
    let stream = incoming.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let requested_model = incoming
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let model = resolve_model(&st, &requested_model).await;
    let openai_req = match responses::to_openai_chat(&incoming, &model) {
        Ok(v) => v,
        Err(e) => {
            return openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", &e)
        }
    };
    let upstream = match post_chat(&st, &openai_req).await {
        Ok(resp) => resp,
        Err(e) => {
            return openai_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                &format!("model server unavailable: {e}"),
            )
        }
    };
    if !upstream.status().is_success() {
        return map_upstream_error_openai(upstream).await;
    }
    let content_type = upstream
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    if stream {
        return stream_responses(upstream, model, content_type).await;
    }
    let bytes = match upstream.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => {
            return openai_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                &format!("read completion: {e}"),
            )
        }
    };
    let openai = if content_type.contains("text/event-stream") {
        match collect_chat_sse(&bytes, &model) {
            Ok(v) => v,
            Err(e) => return openai_error(StatusCode::BAD_GATEWAY, "api_error", &e),
        }
    } else {
        match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => {
                return openai_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    &format!("invalid upstream json: {e}"),
                )
            }
        }
    };
    json_message(responses::from_openai_chat(&openai, &model))
}

async fn openai_responses_get(Path(id): Path<String>) -> Response {
    openai_error(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        &format!("Unknown response {id} (stateless proxy; set store=false)"),
    )
}

async fn openai_responses_delete(Path(_id): Path<String>) -> Response {
    StatusCode::NO_CONTENT.into_response()
}

fn json_message(body: Value) -> Response {
    (
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

async fn map_upstream_error(resp: hyper::Response<hyper::body::Incoming>) -> Response {
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let bytes = resp
        .into_body()
        .collect()
        .await
        .map(|b| b.to_bytes())
        .unwrap_or_default();
    let msg = String::from_utf8_lossy(&bytes);
    let message = serde_json::from_slice::<Value>(&bytes)
        .ok()
        .and_then(|v| {
            v.pointer("/error/message")
                .and_then(|m| m.as_str())
                .map(|s| s.to_string())
                .or_else(|| v.get("message").and_then(|m| m.as_str()).map(|s| s.to_string()))
        })
        .unwrap_or_else(|| msg.chars().take(500).collect());
    anthropic_error(status, "api_error", &message)
}

async fn map_upstream_error_openai(resp: hyper::Response<hyper::body::Incoming>) -> Response {
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let bytes = resp
        .into_body()
        .collect()
        .await
        .map(|b| b.to_bytes())
        .unwrap_or_default();
    let msg = String::from_utf8_lossy(&bytes);
    let message = serde_json::from_slice::<Value>(&bytes)
        .ok()
        .and_then(|v| {
            v.pointer("/error/message")
                .and_then(|m| m.as_str())
                .map(|s| s.to_string())
                .or_else(|| v.get("message").and_then(|m| m.as_str()).map(|s| s.to_string()))
        })
        .unwrap_or_else(|| msg.chars().take(500).collect());
    openai_error(status, "api_error", &message)
}

fn collect_openai_sse(bytes: &[u8], model: &str) -> Result<Value, String> {
    Ok(anthropic::from_openai_chat(
        &collect_chat_sse(bytes, model)?,
        model,
    ))
}

fn collect_chat_sse(bytes: &[u8], model: &str) -> Result<Value, String> {
    let text = String::from_utf8_lossy(bytes);
    let mut last: Option<Value> = None;
    let mut content = String::new();
    let mut tool_calls: HashMap<u64, Value> = HashMap::new();
    let mut finish = "stop".to_string();
    let mut id = String::new();
    let mut usage = json!({});
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        if let Some(v) = chunk.get("id").and_then(|v| v.as_str()) {
            id = v.to_string();
        }
        if let Some(u) = chunk.get("usage") {
            usage = u.clone();
        }
        let choice = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first());
        if let Some(reason) = choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
        {
            finish = reason.to_string();
        }
        if let Some(delta) = choice.and_then(|c| c.get("delta")) {
            if let Some(t) = delta.get("content").and_then(|v| v.as_str()) {
                content.push_str(t);
            }
            merge_tool_deltas(&mut tool_calls, delta.get("tool_calls"));
        }
        last = Some(chunk);
    }
    let mut message = json!({
        "role": "assistant",
        "content": content,
    });
    if !tool_calls.is_empty() {
        let mut calls: Vec<(u64, Value)> = tool_calls.into_iter().collect();
        calls.sort_by_key(|(i, _)| *i);
        message["tool_calls"] = Value::Array(calls.into_iter().map(|(_, v)| v).collect());
    }
    let openai = json!({
        "id": id,
        "model": last.as_ref().and_then(|v| v.get("model")).cloned().unwrap_or_else(|| json!(model)),
        "choices": [{
            "finish_reason": finish,
            "message": message,
        }],
        "usage": usage,
    });
    Ok(openai)
}

fn merge_tool_deltas(acc: &mut HashMap<u64, Value>, calls: Option<&Value>) {
    let Some(arr) = calls.and_then(|v| v.as_array()) else {
        return;
    };
    for call in arr {
        let idx = call.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
        let entry = acc.entry(idx).or_insert_with(|| {
            json!({
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""}
            })
        });
        if let Some(id) = call.get("id").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
            entry["id"] = json!(id);
        }
        if let Some(name) = call.pointer("/function/name").and_then(|v| v.as_str()) {
            if !name.is_empty() {
                entry["function"]["name"] = json!(name);
            }
        }
        if let Some(args) = call.pointer("/function/arguments").and_then(|v| v.as_str()) {
            let cur = entry["function"]["arguments"].as_str().unwrap_or("").to_string();
            entry["function"]["arguments"] = json!(format!("{cur}{args}"));
        }
    }
}

async fn stream_anthropic(
    upstream: hyper::Response<hyper::body::Incoming>,
    model: String,
    content_type: String,
) -> Response {
    if !content_type.contains("text/event-stream") {
        let bytes = match upstream.into_body().collect().await {
            Ok(b) => b.to_bytes(),
            Err(e) => {
                return anthropic_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    &format!("read completion: {e}"),
                )
            }
        };
        let openai: Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(_) => match collect_openai_sse(&bytes, &model) {
                Ok(v) => {
                    return sse_from_final(v);
                }
                Err(e) => return anthropic_error(StatusCode::BAD_GATEWAY, "api_error", &e),
            },
        };
        return sse_from_final(anthropic::from_openai_chat(&openai, &model));
    }

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);
    tokio::spawn(async move {
        let mut translator = AnthropicStream::new(model);
        for ev in translator.start() {
            if tx.send(Ok(ev)).await.is_err() {
                return;
            }
        }
        let mut rest = String::new();
        let mut body = upstream.into_body();
        while let Some(frame) = body.frame().await {
            let Ok(frame) = frame else {
                break;
            };
            let Ok(data) = frame.into_data() else {
                continue;
            };
            rest.push_str(&String::from_utf8_lossy(&data));
            while let Some(pos) = rest.find('\n') {
                let mut line: String = rest.drain(..=pos).collect();
                if line.ends_with('\n') {
                    line.pop();
                }
                if line.ends_with('\r') {
                    line.pop();
                }
                for ev in translator.on_line(&line) {
                    if tx.send(Ok(ev)).await.is_err() {
                        return;
                    }
                }
            }
        }
        if !rest.is_empty() {
            for ev in translator.on_line(&rest) {
                if tx.send(Ok(ev)).await.is_err() {
                    return;
                }
            }
        }
        for ev in translator.finish() {
            if tx.send(Ok(ev)).await.is_err() {
                return;
            }
        }
    });
    Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn sse_from_final(message: Value) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(16);
    tokio::spawn(async move {
        let events = final_to_sse(message);
        for ev in events {
            if tx.send(Ok(ev)).await.is_err() {
                return;
            }
        }
    });
    Sse::new(ReceiverStream::new(rx)).into_response()
}

fn event_named(name: &str, data: Value) -> Event {
    Event::default()
        .event(name)
        .json_data(data)
        .expect("sse json")
}

fn final_to_sse(message: Value) -> Vec<Event> {
    let id = message
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("msg_local")
        .to_string();
    let model = message
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("local")
        .to_string();
    let stop = message
        .get("stop_reason")
        .cloned()
        .unwrap_or_else(|| json!("end_turn"));
    let usage = message.get("usage").cloned().unwrap_or_else(|| json!({}));
    let blocks = message
        .get("content")
        .and_then(|c| c.as_array())
        .cloned()
        .unwrap_or_default();
    let mut events = vec![event_named(
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": {"input_tokens": usage.get("input_tokens").cloned().unwrap_or(json!(0)), "output_tokens": 0}
            }
        }),
    )];
    for (i, block) in blocks.iter().enumerate() {
        events.push(event_named(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": i,
                "content_block": empty_start_block(block)
            }),
        ));
        if block.get("type").and_then(|t| t.as_str()) == Some("text") {
            let text = block.get("text").and_then(|v| v.as_str()).unwrap_or("");
            if !text.is_empty() {
                events.push(event_named(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "text_delta", "text": text}
                    }),
                ));
            }
        } else if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
            events.push(event_named(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": block.get("input").cloned().unwrap_or_else(|| json!({})).to_string()
                    }
                }),
            ));
        }
        events.push(event_named(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": i}),
        ));
    }
    events.push(event_named(
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop, "stop_sequence": Value::Null},
            "usage": {"output_tokens": usage.get("output_tokens").cloned().unwrap_or(json!(0))}
        }),
    ));
    events.push(event_named(
        "message_stop",
        json!({"type": "message_stop"}),
    ));
    events
}

fn empty_start_block(block: &Value) -> Value {
    match block.get("type").and_then(|t| t.as_str()).unwrap_or("text") {
        "tool_use" => json!({
            "type": "tool_use",
            "id": block.get("id").cloned().unwrap_or_else(|| json!("")),
            "name": block.get("name").cloned().unwrap_or_else(|| json!("")),
            "input": {}
        }),
        _ => json!({"type": "text", "text": ""}),
    }
}

struct AnthropicStream {
    model: String,
    id: String,
    started: bool,
    text_open: bool,
    text_index: i64,
    next_index: i64,
    tools: HashMap<u64, ToolStream>,
    finish: String,
    output_tokens: u64,
    input_tokens: u64,
    closed: bool,
}

struct ToolStream {
    block_index: i64,
    started: bool,
}

impl AnthropicStream {
    fn new(model: String) -> Self {
        Self {
            model,
            id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
            started: false,
            text_open: false,
            text_index: 0,
            next_index: 0,
            tools: HashMap::new(),
            finish: "stop".into(),
            output_tokens: 0,
            input_tokens: 0,
            closed: false,
        }
    }

    fn start(&mut self) -> Vec<Event> {
        self.started = true;
        vec![event_named(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": self.id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }
            }),
        )]
    }

    fn on_line(&mut self, line: &str) -> Vec<Event> {
        let Some(data) = line.strip_prefix("data:") else {
            return Vec::new();
        };
        let data = data.trim();
        if data.is_empty() {
            return Vec::new();
        }
        if data == "[DONE]" {
            return self.finish();
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            return Vec::new();
        };
        if let Some(id) = chunk.get("id").and_then(|v| v.as_str()) {
            if self.id.starts_with("msg_") && id.starts_with("msg_") {
                self.id = id.to_string();
            }
        }
        if let Some(usage) = chunk.get("usage") {
            if let Some(n) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
                self.output_tokens = n;
            }
            if let Some(n) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
                self.input_tokens = n;
            }
        }
        let mut events = Vec::new();
        let choice = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first());
        if let Some(reason) = choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
        {
            self.finish = reason.to_string();
        }
        if let Some(delta) = choice.and_then(|c| c.get("delta")) {
            if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
                if !text.is_empty() {
                    events.extend(self.ensure_text());
                    let text_index = self.text_index;
                    events.push(event_named(
                        "content_block_delta",
                        json!({
                            "type": "content_block_delta",
                            "index": text_index,
                            "delta": {"type": "text_delta", "text": text}
                        }),
                    ));
                }
            }
            if let Some(calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                events.extend(self.close_text());
                for call in calls {
                    events.extend(self.on_tool_delta(call));
                }
            }
        }
        events
    }

    fn ensure_text(&mut self) -> Vec<Event> {
        if self.text_open {
            return Vec::new();
        }
        self.text_open = true;
        self.text_index = self.next_index;
        self.next_index += 1;
        vec![event_named(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": self.text_index,
                "content_block": {"type": "text", "text": ""}
            }),
        )]
    }

    fn close_text(&mut self) -> Vec<Event> {
        if !self.text_open {
            return Vec::new();
        }
        self.text_open = false;
        vec![event_named(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": self.text_index}),
        )]
    }

    fn on_tool_delta(&mut self, call: &Value) -> Vec<Event> {
        let idx = call.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
        let mut events = Vec::new();
        if !self.tools.contains_key(&idx) {
            let block_index = self.next_index;
            self.next_index += 1;
            self.tools.insert(
                idx,
                ToolStream {
                    block_index,
                    started: false,
                },
            );
        }
        let block_index = self.tools.get(&idx).map(|t| t.block_index).unwrap_or(0);
        let started = self.tools.get(&idx).map(|t| t.started).unwrap_or(false);
        if !started {
            if let Some(t) = self.tools.get_mut(&idx) {
                t.started = true;
            }
            let id = call.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let name = call
                .pointer("/function/name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            events.push(event_named(
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        "input": {}
                    }
                }),
            ));
        }
        if let Some(args) = call.pointer("/function/arguments").and_then(|v| v.as_str()) {
            if !args.is_empty() {
                events.push(event_named(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "input_json_delta", "partial_json": args}
                    }),
                ));
            }
        }
        events
    }

    fn finish(&mut self) -> Vec<Event> {
        if self.closed {
            return Vec::new();
        }
        self.closed = true;
        let mut events = self.close_text();
        let mut tools: Vec<(u64, i64)> = self
            .tools
            .iter()
            .map(|(k, v)| (*k, v.block_index))
            .collect();
        tools.sort_by_key(|(k, _)| *k);
        for (_, idx) in tools {
            events.push(event_named(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": idx}),
            ));
        }
        events.push(event_named(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": anthropic::map_stop_reason(&self.finish),
                    "stop_sequence": Value::Null
                },
                "usage": {"output_tokens": self.output_tokens}
            }),
        ));
        events.push(event_named(
            "message_stop",
            json!({"type": "message_stop"}),
        ));
        events
    }
}

async fn stream_responses(
    upstream: hyper::Response<hyper::body::Incoming>,
    model: String,
    content_type: String,
) -> Response {
    if !content_type.contains("text/event-stream") {
        let bytes = match upstream.into_body().collect().await {
            Ok(b) => b.to_bytes(),
            Err(e) => {
                return openai_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    &format!("read completion: {e}"),
                )
            }
        };
        let openai: Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(_) => match collect_chat_sse(&bytes, &model) {
                Ok(v) => v,
                Err(e) => return openai_error(StatusCode::BAD_GATEWAY, "api_error", &e),
            },
        };
        return responses_sse_from_final(responses::from_openai_chat(&openai, &model));
    }

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);
    tokio::spawn(async move {
        let mut translator = ResponsesStream::new(model);
        for ev in translator.start() {
            if tx.send(Ok(ev)).await.is_err() {
                return;
            }
        }
        let mut rest = String::new();
        let mut body = upstream.into_body();
        while let Some(frame) = body.frame().await {
            let Ok(frame) = frame else {
                break;
            };
            let Ok(data) = frame.into_data() else {
                continue;
            };
            rest.push_str(&String::from_utf8_lossy(&data));
            while let Some(pos) = rest.find('\n') {
                let mut line: String = rest.drain(..=pos).collect();
                if line.ends_with('\n') {
                    line.pop();
                }
                if line.ends_with('\r') {
                    line.pop();
                }
                for ev in translator.on_line(&line) {
                    if tx.send(Ok(ev)).await.is_err() {
                        return;
                    }
                }
            }
        }
        if !rest.is_empty() {
            for ev in translator.on_line(&rest) {
                if tx.send(Ok(ev)).await.is_err() {
                    return;
                }
            }
        }
        for ev in translator.finish() {
            if tx.send(Ok(ev)).await.is_err() {
                return;
            }
        }
    });
    Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn responses_sse_from_final(message: Value) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    tokio::spawn(async move {
        let events = responses_final_to_sse(message);
        for ev in events {
            if tx.send(Ok(ev)).await.is_err() {
                return;
            }
        }
    });
    Sse::new(ReceiverStream::new(rx)).into_response()
}

fn responses_final_to_sse(response: Value) -> Vec<Event> {
    let mut seq = 0u64;
    let mut events = Vec::new();
    let mut snapshot = response.clone();
    snapshot["status"] = json!("in_progress");
    snapshot["output"] = json!([]);
    events.push(seq_event(
        &mut seq,
        "response.created",
        json!({"type": "response.created", "response": snapshot.clone()}),
    ));
    events.push(seq_event(
        &mut seq,
        "response.in_progress",
        json!({"type": "response.in_progress", "response": snapshot}),
    ));
    let items = response
        .get("output")
        .and_then(|o| o.as_array())
        .cloned()
        .unwrap_or_default();
    for (output_index, item) in items.iter().enumerate() {
        events.push(seq_event(
            &mut seq,
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": item_started(item)
            }),
        ));
        if item.get("type").and_then(|t| t.as_str()) == Some("message") {
            let item_id = item.get("id").cloned().unwrap_or(json!(""));
            let part = item
                .get("content")
                .and_then(|c| c.as_array())
                .and_then(|c| c.first())
                .cloned()
                .unwrap_or_else(|| json!({"type": "output_text", "text": "", "annotations": []}));
            events.push(seq_event(
                &mut seq,
                "response.content_part.added",
                json!({
                    "type": "response.content_part.added",
                    "output_index": output_index,
                    "content_index": 0,
                    "item_id": item_id,
                    "part": {"type": "output_text", "text": "", "annotations": []}
                }),
            ));
            let text = part.get("text").and_then(|v| v.as_str()).unwrap_or("");
            if !text.is_empty() {
                events.push(seq_event(
                    &mut seq,
                    "response.output_text.delta",
                    json!({
                        "type": "response.output_text.delta",
                        "output_index": output_index,
                        "content_index": 0,
                        "item_id": item_id,
                        "delta": text
                    }),
                ));
            }
            events.push(seq_event(
                &mut seq,
                "response.output_text.done",
                json!({
                    "type": "response.output_text.done",
                    "output_index": output_index,
                    "content_index": 0,
                    "item_id": item_id,
                    "text": text
                }),
            ));
            events.push(seq_event(
                &mut seq,
                "response.content_part.done",
                json!({
                    "type": "response.content_part.done",
                    "output_index": output_index,
                    "content_index": 0,
                    "item_id": item_id,
                    "part": part
                }),
            ));
        } else if item.get("type").and_then(|t| t.as_str()) == Some("function_call") {
            let item_id = item.get("id").cloned().unwrap_or(json!(""));
            let args = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
            if !args.is_empty() {
                events.push(seq_event(
                    &mut seq,
                    "response.function_call_arguments.delta",
                    json!({
                        "type": "response.function_call_arguments.delta",
                        "output_index": output_index,
                        "item_id": item_id,
                        "delta": args
                    }),
                ));
            }
            events.push(seq_event(
                &mut seq,
                "response.function_call_arguments.done",
                json!({
                    "type": "response.function_call_arguments.done",
                    "output_index": output_index,
                    "item_id": item_id,
                    "arguments": args
                }),
            ));
        }
        events.push(seq_event(
            &mut seq,
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": item
            }),
        ));
    }
    events.push(seq_event(
        &mut seq,
        "response.completed",
        json!({"type": "response.completed", "response": response}),
    ));
    events
}

fn item_started(item: &Value) -> Value {
    let mut started = item.clone();
    started["status"] = json!("in_progress");
    if started.get("type").and_then(|t| t.as_str()) == Some("message") {
        started["content"] = json!([]);
    }
    if started.get("type").and_then(|t| t.as_str()) == Some("function_call") {
        started["arguments"] = json!("");
    }
    started
}

fn seq_event(seq: &mut u64, name: &str, mut data: Value) -> Event {
    *seq += 1;
    data["sequence_number"] = json!(*seq);
    Event::default()
        .event(name)
        .json_data(data)
        .expect("sse json")
}

struct ResponsesStream {
    model: String,
    response: Value,
    seq: u64,
    text_item_id: Option<String>,
    text_index: i64,
    text_buf: String,
    next_index: i64,
    tools: HashMap<u64, ToolOut>,
    finish: String,
    closed: bool,
}

struct ToolOut {
    item_id: String,
    call_id: String,
    name: String,
    args: String,
    output_index: i64,
    started: bool,
}

impl ResponsesStream {
    fn new(model: String) -> Self {
        let id = format!("resp_{}", uuid::Uuid::new_v4().simple());
        Self {
            response: responses::empty_response(&id, &model),
            model,
            seq: 0,
            text_item_id: None,
            text_index: 0,
            text_buf: String::new(),
            next_index: 0,
            tools: HashMap::new(),
            finish: "stop".into(),
            closed: false,
        }
    }

    fn emit(&mut self, name: &str, data: Value) -> Event {
        seq_event(&mut self.seq, name, data)
    }

    fn start(&mut self) -> Vec<Event> {
        vec![
            self.emit(
                "response.created",
                json!({"type": "response.created", "response": self.response.clone()}),
            ),
            self.emit(
                "response.in_progress",
                json!({"type": "response.in_progress", "response": self.response.clone()}),
            ),
        ]
    }

    fn on_line(&mut self, line: &str) -> Vec<Event> {
        let Some(data) = line.strip_prefix("data:") else {
            return Vec::new();
        };
        let data = data.trim();
        if data.is_empty() {
            return Vec::new();
        }
        if data == "[DONE]" {
            return self.finish();
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            return Vec::new();
        };
        if let Some(usage) = chunk.get("usage") {
            let input = usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
            let output = usage
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            self.response["usage"] = json!({
                "input_tokens": input,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": output,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": input + output,
            });
        }
        if let Some(model) = chunk.get("model").and_then(|v| v.as_str()) {
            if !model.is_empty() {
                self.response["model"] = json!(model);
            }
        }
        let mut events = Vec::new();
        let choice = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first());
        if let Some(reason) = choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
        {
            self.finish = reason.to_string();
        }
        if let Some(delta) = choice.and_then(|c| c.get("delta")) {
            if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
                if !text.is_empty() {
                    events.extend(self.on_text(text));
                }
            }
            if let Some(calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                events.extend(self.close_text());
                for call in calls {
                    events.extend(self.on_tool_delta(call));
                }
            }
        }
        events
    }

    fn on_text(&mut self, text: &str) -> Vec<Event> {
        let mut events = Vec::new();
        if self.text_item_id.is_none() {
            let item_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
            self.text_index = self.next_index;
            self.next_index += 1;
            self.text_item_id = Some(item_id.clone());
            let item = json!({
                "id": item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": []
            });
            events.push(self.emit(
                "response.output_item.added",
                json!({
                    "type": "response.output_item.added",
                    "output_index": self.text_index,
                    "item": item
                }),
            ));
            events.push(self.emit(
                "response.content_part.added",
                json!({
                    "type": "response.content_part.added",
                    "output_index": self.text_index,
                    "content_index": 0,
                    "item_id": item_id,
                    "part": {"type": "output_text", "text": "", "annotations": []}
                }),
            ));
        }
        self.text_buf.push_str(text);
        let item_id = self.text_item_id.clone().unwrap_or_default();
        events.push(self.emit(
            "response.output_text.delta",
            json!({
                "type": "response.output_text.delta",
                "output_index": self.text_index,
                "content_index": 0,
                "item_id": item_id,
                "delta": text
            }),
        ));
        events
    }

    fn close_text(&mut self) -> Vec<Event> {
        let Some(item_id) = self.text_item_id.take() else {
            return Vec::new();
        };
        let item = json!({
            "id": item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": self.text_buf,
                "annotations": []
            }]
        });
        let events = vec![
            self.emit(
                "response.output_text.done",
                json!({
                    "type": "response.output_text.done",
                    "output_index": self.text_index,
                    "content_index": 0,
                    "item_id": item_id,
                    "text": self.text_buf
                }),
            ),
            self.emit(
                "response.content_part.done",
                json!({
                    "type": "response.content_part.done",
                    "output_index": self.text_index,
                    "content_index": 0,
                    "item_id": item_id,
                    "part": {
                        "type": "output_text",
                        "text": self.text_buf,
                        "annotations": []
                    }
                }),
            ),
            self.emit(
                "response.output_item.done",
                json!({
                    "type": "response.output_item.done",
                    "output_index": self.text_index,
                    "item": item.clone()
                }),
            ),
        ];
        if let Some(arr) = self.response["output"].as_array_mut() {
            arr.push(item);
        }
        events
    }

    fn on_tool_delta(&mut self, call: &Value) -> Vec<Event> {
        let idx = call.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
        let mut events = Vec::new();
        if !self.tools.contains_key(&idx) {
            let output_index = self.next_index;
            self.next_index += 1;
            self.tools.insert(
                idx,
                ToolOut {
                    item_id: format!("fc_{}", uuid::Uuid::new_v4().simple()),
                    call_id: call
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    name: call
                        .pointer("/function/name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    args: String::new(),
                    output_index,
                    started: false,
                },
            );
        }
        let (item_id, output_index, started, name, call_id) = {
            let t = self.tools.get(&idx).unwrap();
            (
                t.item_id.clone(),
                t.output_index,
                t.started,
                t.name.clone(),
                t.call_id.clone(),
            )
        };
        if !started {
            if let Some(t) = self.tools.get_mut(&idx) {
                t.started = true;
                if let Some(id) = call.get("id").and_then(|v| v.as_str()) {
                    if !id.is_empty() {
                        t.call_id = id.to_string();
                    }
                }
                if let Some(name) = call.pointer("/function/name").and_then(|v| v.as_str()) {
                    if !name.is_empty() {
                        t.name = name.to_string();
                    }
                }
            }
            let call_id = self
                .tools
                .get(&idx)
                .map(|t| t.call_id.clone())
                .unwrap_or(call_id);
            let name = self.tools.get(&idx).map(|t| t.name.clone()).unwrap_or(name);
            events.push(self.emit(
                "response.output_item.added",
                json!({
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": call_id,
                        "name": name,
                        "arguments": ""
                    }
                }),
            ));
        }
        if let Some(args) = call.pointer("/function/arguments").and_then(|v| v.as_str()) {
            if !args.is_empty() {
                if let Some(t) = self.tools.get_mut(&idx) {
                    t.args.push_str(args);
                }
                events.push(self.emit(
                    "response.function_call_arguments.delta",
                    json!({
                        "type": "response.function_call_arguments.delta",
                        "output_index": output_index,
                        "item_id": item_id,
                        "delta": args
                    }),
                ));
            }
        }
        events
    }

    fn finish(&mut self) -> Vec<Event> {
        if self.closed {
            return Vec::new();
        }
        self.closed = true;
        let mut events = self.close_text();
        let mut tools: Vec<(u64, ToolOut)> = self.tools.drain().collect();
        tools.sort_by_key(|(k, _)| *k);
        for (_, t) in tools {
            events.push(self.emit(
                "response.function_call_arguments.done",
                json!({
                    "type": "response.function_call_arguments.done",
                    "output_index": t.output_index,
                    "item_id": t.item_id,
                    "arguments": t.args
                }),
            ));
            let item = json!({
                "id": t.item_id,
                "type": "function_call",
                "status": "completed",
                "call_id": t.call_id,
                "name": t.name,
                "arguments": t.args
            });
            events.push(self.emit(
                "response.output_item.done",
                json!({
                    "type": "response.output_item.done",
                    "output_index": t.output_index,
                    "item": item.clone()
                }),
            ));
            if let Some(arr) = self.response["output"].as_array_mut() {
                arr.push(item);
            }
        }
        let (status, incomplete) = match self.finish.as_str() {
            "length" => ("incomplete", json!({"reason": "max_output_tokens"})),
            "content_filter" => ("incomplete", json!({"reason": "content_filter"})),
            _ => ("completed", Value::Null),
        };
        self.response["status"] = json!(status);
        self.response["incomplete_details"] = incomplete;
        if self.response.get("model").and_then(|v| v.as_str()).unwrap_or("").is_empty() {
            self.response["model"] = json!(self.model);
        }
        let name = if status == "completed" {
            "response.completed"
        } else {
            "response.incomplete"
        };
        events.push(self.emit(
            name,
            json!({"type": name, "response": self.response.clone()}),
        ));
        events
    }
}

pub fn public_url(ep: &EndpointConfig) -> String {
    format!("{}/v1", claude_base_url(ep))
}

pub fn claude_base_url(ep: &EndpointConfig) -> String {
    let host = if ep.advertise_host.is_empty() {
        "127.0.0.1"
    } else {
        &ep.advertise_host
    };
    format!("http://{host}:{}", ep.port)
}
