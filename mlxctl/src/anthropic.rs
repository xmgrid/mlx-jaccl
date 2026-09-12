use serde_json::{json, Map, Value};

pub fn to_openai_chat(body: &Value, model: &str) -> Result<Value, String> {
    let mut messages = Vec::new();

    if let Some(system) = body.get("system") {
        if let Some(text) = flatten_system(system) {
            if !text.is_empty() {
                messages.push(json!({"role": "system", "content": text}));
            }
        }
    }

    let incoming = body
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or_else(|| "messages is required".to_string())?;

    for msg in incoming {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        push_openai_messages(&mut messages, role, msg.get("content"));
    }

    if messages.is_empty() {
        messages.push(json!({"role": "user", "content": " "}));
    }

    let mut out = json!({
        "model": model,
        "messages": messages,
        "max_tokens": body.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(4096),
    });

    copy_if_present(body, &mut out, "temperature");
    copy_if_present(body, &mut out, "top_p");
    copy_if_present(body, &mut out, "top_k");
    if let Some(stop) = body.get("stop_sequences") {
        out["stop"] = stop.clone();
    }
    if let Some(stream) = body.get("stream") {
        out["stream"] = stream.clone();
        if stream.as_bool() == Some(true) {
            out["stream_options"] = json!({"include_usage": true});
        }
    }
    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let mapped: Vec<Value> = tools.iter().filter_map(map_tool).collect();
        if !mapped.is_empty() {
            out["tools"] = Value::Array(mapped);
        }
    }
    if let Some(choice) = body.get("tool_choice") {
        out["tool_choice"] = map_tool_choice(choice);
    }
    if body
        .pointer("/metadata/user_id")
        .and_then(|v| v.as_str())
        .is_some()
    {
        // mlx ignores extra fields; keep request lean
    }
    Ok(out)
}

pub fn from_openai_chat(body: &Value, requested_model: &str) -> Value {
    let choice = body
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first());
    let message = choice.and_then(|c| c.get("message"));
    let finish = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|v| v.as_str())
        .unwrap_or("stop");

    let mut content = Vec::new();
    if let Some(text) = message.and_then(|m| m.get("content")).and_then(|c| as_text(c)) {
        if !text.is_empty() {
            content.push(json!({"type": "text", "text": text}));
        }
    }
    if let Some(calls) = message.and_then(|m| m.get("tool_calls")).and_then(|c| c.as_array()) {
        for call in calls {
            content.push(tool_use_block(call));
        }
    }
    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }

    let usage = body.get("usage").cloned().unwrap_or_else(|| json!({}));
    let id = body
        .get("id")
        .and_then(|v| v.as_str())
        .map(|id| {
            if id.starts_with("msg_") {
                id.to_string()
            } else {
                format!("msg_{id}")
            }
        })
        .unwrap_or_else(|| format!("msg_{}", uuid::Uuid::new_v4().simple()));
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .filter(|m| !m.is_empty())
        .unwrap_or(requested_model);

    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": map_stop_reason(finish),
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            "output_tokens": usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
        }
    })
}

pub fn map_stop_reason(finish: &str) -> &'static str {
    match finish {
        "length" => "max_tokens",
        "tool_calls" | "function_call" => "tool_use",
        "content_filter" => "refusal",
        _ => "end_turn",
    }
}

pub fn looks_like_claude_model(model: &str) -> bool {
    let m = model.trim();
    m.is_empty() || m.starts_with("claude") || m.contains("claude-")
}

pub fn anthropic_model_list(openai: &Value) -> Value {
    let mut data = Vec::new();
    if let Some(arr) = openai.get("data").and_then(|d| d.as_array()) {
        for m in arr {
            let id = m.get("id").and_then(|v| v.as_str()).unwrap_or("local");
            data.push(json!({
                "type": "model",
                "id": id,
                "display_name": id,
                "created_at": "2024-01-01T00:00:00Z",
                "object": "model",
            }));
        }
    }
    if data.is_empty() {
        data.push(json!({
            "type": "model",
            "id": "local",
            "display_name": "local",
            "created_at": "2024-01-01T00:00:00Z",
            "object": "model",
        }));
    }
    for alias in [
        "claude-sonnet-4-5",
        "claude-3-5-sonnet-latest",
        "claude-3-haiku-20240307",
        "claude-opus-4-5",
    ] {
        if !data.iter().any(|d| d.get("id").and_then(|v| v.as_str()) == Some(alias)) {
            data.push(json!({
                "type": "model",
                "id": alias,
                "display_name": alias,
                "created_at": "2024-01-01T00:00:00Z",
                "object": "model",
            }));
        }
    }
    let first = data.first().and_then(|d| d.get("id")).cloned();
    let last = data.last().and_then(|d| d.get("id")).cloned();
    json!({
        "object": "list",
        "data": data,
        "has_more": false,
        "first_id": first,
        "last_id": last,
    })
}

pub fn estimate_tokens(body: &Value) -> u64 {
    let s = body.to_string();
    ((s.len() as u64) / 4).max(1)
}

fn flatten_system(system: &Value) -> Option<String> {
    if let Some(s) = system.as_str() {
        return Some(s.to_string());
    }
    let arr = system.as_array()?;
    let mut parts = Vec::new();
    for block in arr {
        if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
            parts.push(t);
        } else if let Some(t) = block.as_str() {
            parts.push(t);
        }
    }
    Some(parts.join("\n"))
}

fn push_openai_messages(out: &mut Vec<Value>, role: &str, content: Option<&Value>) {
    let Some(content) = content else {
        out.push(json!({"role": role, "content": ""}));
        return;
    };
    if let Some(text) = content.as_str() {
        out.push(json!({"role": role, "content": text}));
        return;
    }
    let Some(blocks) = content.as_array() else {
        out.push(json!({"role": role, "content": content.clone()}));
        return;
    };

    let mut tool_results = Vec::new();
    let mut text_parts: Vec<Value> = Vec::new();
    let mut tool_calls = Vec::new();

    for block in blocks {
        match block.get("type").and_then(|t| t.as_str()).unwrap_or("text") {
            "tool_result" => {
                let id = block
                    .get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                tool_results.push(json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": flatten_tool_result(block.get("content")),
                }));
            }
            "tool_use" => {
                let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                let args = if input.is_string() {
                    input.as_str().unwrap_or("{}").to_string()
                } else {
                    input.to_string()
                };
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": args}
                }));
            }
            "image" => {
                if let Some(source) = block.get("source") {
                    text_parts.push(image_to_openai(source));
                }
            }
            "image_url" => text_parts.push(block.clone()),
            "thinking" | "redacted_thinking" => {}
            _ => {
                if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                    text_parts.push(json!({"type": "text", "text": t}));
                } else if let Some(t) = block.as_str() {
                    text_parts.push(json!({"type": "text", "text": t}));
                }
            }
        }
    }

    out.extend(tool_results);

    if role == "assistant" && !tool_calls.is_empty() {
        let content = flatten_text_parts(&text_parts);
        let mut msg = Map::new();
        msg.insert("role".into(), json!("assistant"));
        msg.insert(
            "content".into(),
            if content.is_empty() {
                Value::Null
            } else {
                json!(content)
            },
        );
        msg.insert("tool_calls".into(), Value::Array(tool_calls));
        out.push(Value::Object(msg));
        return;
    }

    if text_parts.is_empty() {
        if role != "assistant" || tool_calls.is_empty() {
            return;
        }
    }

    let openai_content = if text_parts.iter().all(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"))
    {
        json!(flatten_text_parts(&text_parts))
    } else if text_parts.len() == 1 && text_parts[0].get("type").and_then(|t| t.as_str()) == Some("text")
    {
        json!(text_parts[0].get("text").and_then(|v| v.as_str()).unwrap_or(""))
    } else {
        Value::Array(text_parts)
    };
    out.push(json!({"role": role, "content": openai_content}));
}

fn flatten_text_parts(parts: &[Value]) -> String {
    parts
        .iter()
        .filter_map(|p| {
            if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                p.get("text").and_then(|v| v.as_str()).map(|s| s.to_string())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

fn flatten_tool_result(content: Option<&Value>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    if let Some(s) = content.as_str() {
        return s.to_string();
    }
    if let Some(arr) = content.as_array() {
        return arr
            .iter()
            .filter_map(|b| {
                b.get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| b.as_str().map(|s| s.to_string()))
            })
            .collect::<Vec<_>>()
            .join("\n");
    }
    content.to_string()
}

fn image_to_openai(source: &Value) -> Value {
    match source.get("type").and_then(|t| t.as_str()).unwrap_or("base64") {
        "url" => {
            let url = source.get("url").and_then(|v| v.as_str()).unwrap_or("");
            json!({"type": "image_url", "image_url": {"url": url}})
        }
        _ => {
            let media = source
                .get("media_type")
                .and_then(|v| v.as_str())
                .unwrap_or("image/png");
            let data = source.get("data").and_then(|v| v.as_str()).unwrap_or("");
            json!({
                "type": "image_url",
                "image_url": {"url": format!("data:{media};base64,{data}")}
            })
        }
    }
}

fn map_tool(tool: &Value) -> Option<Value> {
    let name = tool.get("name")?.as_str()?;
    let description = tool.get("description").cloned().unwrap_or_else(|| json!(""));
    let parameters = tool
        .get("input_schema")
        .cloned()
        .or_else(|| tool.get("parameters").cloned())
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
    Some(json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    }))
}

fn map_tool_choice(choice: &Value) -> Value {
    match choice.get("type").and_then(|t| t.as_str()).unwrap_or("") {
        "any" => json!("required"),
        "none" => json!("none"),
        "tool" => {
            let name = choice.get("name").and_then(|v| v.as_str()).unwrap_or("");
            json!({"type": "function", "function": {"name": name}})
        }
        _ => json!("auto"),
    }
}

fn tool_use_block(call: &Value) -> Value {
    let id = call.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let name = call
        .pointer("/function/name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let args = call
        .pointer("/function/arguments")
        .and_then(|v| v.as_str())
        .unwrap_or("{}");
    let input = serde_json::from_str::<Value>(args).unwrap_or_else(|_| json!({}));
    json!({
        "type": "tool_use",
        "id": id,
        "name": name,
        "input": input,
    })
}

fn as_text(content: &Value) -> Option<String> {
    if content.is_null() {
        return Some(String::new());
    }
    if let Some(s) = content.as_str() {
        return Some(s.to_string());
    }
    if let Some(arr) = content.as_array() {
        return Some(
            arr.iter()
                .filter_map(|b| b.get("text").and_then(|v| v.as_str()))
                .collect::<Vec<_>>()
                .join(""),
        );
    }
    None
}

fn copy_if_present(src: &Value, dst: &mut Value, key: &str) {
    if let Some(v) = src.get(key) {
        dst[key] = v.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_system_and_text() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 32,
            "system": "be brief",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let out = to_openai_chat(&body, "local-model").unwrap();
        assert_eq!(out["model"], "local-model");
        assert_eq!(out["messages"][0]["role"], "system");
        assert_eq!(out["messages"][1]["content"], "hi");
    }

    #[test]
    fn converts_image_and_tools() {
        let body = json!({
            "max_tokens": 8,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what"},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "abcd"
                    }}
                ]
            }],
            "tools": [{
                "name": "Read",
                "description": "read a file",
                "input_schema": {"type": "object"}
            }]
        });
        let out = to_openai_chat(&body, "m").unwrap();
        assert_eq!(out["messages"][0]["content"][1]["type"], "image_url");
        assert!(out["messages"][0]["content"][1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/jpeg;base64,"));
        assert_eq!(out["tools"][0]["function"]["name"], "Read");
    }

    #[test]
    fn converts_tool_roundtrip() {
        let body = json!({
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "a.rs"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "fn main() {}"}
                ]}
            ]
        });
        let out = to_openai_chat(&body, "m").unwrap();
        assert_eq!(out["messages"][1]["tool_calls"][0]["id"], "toolu_1");
        assert_eq!(out["messages"][2]["role"], "tool");
        assert_eq!(out["messages"][2]["content"], "fn main() {}");
    }

    #[test]
    fn maps_openai_response() {
        let body = json!({
            "id": "chatcmpl-1",
            "model": "qwen",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "Read", "arguments": "{\"path\":\"a\"}"}
                    }]
                }
            }],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3}
        });
        let out = from_openai_chat(&body, "claude-sonnet-4-5");
        assert_eq!(out["stop_reason"], "tool_use");
        assert_eq!(out["content"][0]["type"], "tool_use");
        assert_eq!(out["content"][0]["input"]["path"], "a");
        assert_eq!(out["usage"]["input_tokens"], 9);
    }
}
