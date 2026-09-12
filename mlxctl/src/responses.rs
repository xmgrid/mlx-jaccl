use serde_json::{json, Value};

pub fn looks_like_openai_alias(model: &str) -> bool {
    let m = model.trim().to_ascii_lowercase();
    m.is_empty()
        || m.starts_with("gpt-")
        || m.starts_with("gpt_")
        || m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4")
        || m.starts_with("chatgpt")
        || m == "gpt"
}

pub fn to_openai_chat(body: &Value, model: &str) -> Result<Value, String> {
    let mut messages = Vec::new();
    if let Some(instructions) = flatten_instructions(body.get("instructions")) {
        if !instructions.is_empty() {
            messages.push(json!({"role": "system", "content": instructions}));
        }
    }

    match body.get("input") {
        None => messages.push(json!({"role": "user", "content": " "})),
        Some(Value::String(s)) => {
            messages.push(json!({"role": "user", "content": s}));
        }
        Some(Value::Array(items)) => push_input_items(&mut messages, items),
        Some(other) => {
            messages.push(json!({"role": "user", "content": other.to_string()}));
        }
    }

    if messages.is_empty() {
        messages.push(json!({"role": "user", "content": " "}));
    }

    let max_tokens = body
        .get("max_output_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(4096);

    let mut out = json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    });
    copy_if_present(body, &mut out, "temperature");
    copy_if_present(body, &mut out, "top_p");
    copy_if_present(body, &mut out, "top_k");
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

    let text = message
        .and_then(|m| m.get("content"))
        .and_then(as_text)
        .unwrap_or_default();
    let tool_calls = message
        .and_then(|m| m.get("tool_calls"))
        .and_then(|c| c.as_array())
        .cloned()
        .unwrap_or_default();

    let mut output = Vec::new();
    if !text.is_empty() || tool_calls.is_empty() {
        output.push(message_item(&text, "completed"));
    }
    for call in &tool_calls {
        output.push(function_call_item(call));
    }

    let usage_in = body.get("usage").cloned().unwrap_or_else(|| json!({}));
    let input_tokens = usage_in
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output_tokens = usage_in
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .filter(|m| !m.is_empty())
        .unwrap_or(requested_model);
    let id = body
        .get("id")
        .and_then(|v| v.as_str())
        .map(|id| {
            if id.starts_with("resp_") {
                id.to_string()
            } else {
                format!("resp_{id}")
            }
        })
        .unwrap_or_else(|| format!("resp_{}", uuid::Uuid::new_v4().simple()));

    let (status, incomplete) = match finish {
        "length" => (
            "incomplete",
            json!({"reason": "max_output_tokens"}),
        ),
        "content_filter" => ("incomplete", json!({"reason": "content_filter"})),
        _ => ("completed", Value::Null),
    };

    json!({
        "id": id,
        "object": "response",
        "created_at": chrono::Utc::now().timestamp(),
        "status": status,
        "error": Value::Null,
        "incomplete_details": incomplete,
        "instructions": Value::Null,
        "max_output_tokens": Value::Null,
        "model": model,
        "output": output,
        "parallel_tool_calls": true,
        "previous_response_id": Value::Null,
        "reasoning": {"effort": Value::Null, "summary": Value::Null},
        "store": false,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        },
        "user": Value::Null,
        "metadata": {}
    })
}

pub fn empty_response(id: &str, model: &str) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": chrono::Utc::now().timestamp(),
        "status": "in_progress",
        "error": Value::Null,
        "incomplete_details": Value::Null,
        "instructions": Value::Null,
        "max_output_tokens": Value::Null,
        "model": model,
        "output": [],
        "parallel_tool_calls": true,
        "previous_response_id": Value::Null,
        "reasoning": {"effort": Value::Null, "summary": Value::Null},
        "store": false,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": Value::Null,
        "user": Value::Null,
        "metadata": {}
    })
}

pub fn message_item(text: &str, status: &str) -> Value {
    json!({
        "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": []
        }]
    })
}

pub fn function_call_item(call: &Value) -> Value {
    let id = call.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let name = call
        .pointer("/function/name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let args = call
        .pointer("/function/arguments")
        .and_then(|v| v.as_str())
        .unwrap_or("{}");
    let call_id = if id.is_empty() {
        format!("call_{}", uuid::Uuid::new_v4().simple())
    } else {
        id.to_string()
    };
    json!({
        "id": format!("fc_{}", uuid::Uuid::new_v4().simple()),
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": name,
        "arguments": args
    })
}

fn flatten_instructions(v: Option<&Value>) -> Option<String> {
    let v = v?;
    if let Some(s) = v.as_str() {
        return Some(s.to_string());
    }
    if let Some(arr) = v.as_array() {
        let text = arr
            .iter()
            .filter_map(|b| {
                b.get("text")
                    .and_then(|t| t.as_str())
                    .or_else(|| b.as_str())
            })
            .collect::<Vec<_>>()
            .join("\n");
        return Some(text);
    }
    None
}

fn push_input_items(messages: &mut Vec<Value>, items: &[Value]) {
    let mut pending_calls: Vec<Value> = Vec::new();
    for item in items {
        let kind = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match kind {
            "function_call" => {
                pending_calls.push(responses_tool_call(item));
            }
            "function_call_output" | "tool_result" => {
                flush_tool_calls(messages, &mut pending_calls);
                let call_id = item
                    .get("call_id")
                    .or_else(|| item.get("tool_call_id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": flatten_tool_output(item.get("output").or_else(|| item.get("content"))),
                }));
            }
            "reasoning" | "item_reference" => {}
            _ => {
                flush_tool_calls(messages, &mut pending_calls);
                push_easy_message(messages, item);
            }
        }
    }
    flush_tool_calls(messages, &mut pending_calls);
}

fn flush_tool_calls(messages: &mut Vec<Value>, pending: &mut Vec<Value>) {
    if pending.is_empty() {
        return;
    }
    messages.push(json!({
        "role": "assistant",
        "content": Value::Null,
        "tool_calls": pending.split_off(0),
    }));
}

fn responses_tool_call(item: &Value) -> Value {
    let id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let args = item
        .get("arguments")
        .map(|v| {
            if let Some(s) = v.as_str() {
                s.to_string()
            } else {
                v.to_string()
            }
        })
        .unwrap_or_else(|| "{}".into());
    json!({
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": args}
    })
}

fn push_easy_message(messages: &mut Vec<Value>, item: &Value) {
    let role = match item.get("role").and_then(|r| r.as_str()).unwrap_or("user") {
        "developer" | "system" => "system",
        "assistant" => "assistant",
        "tool" => "tool",
        _ => "user",
    };
    let content = item.get("content").unwrap_or(item);
    if role == "tool" {
        messages.push(json!({
            "role": "tool",
            "tool_call_id": item.get("call_id").and_then(|v| v.as_str()).unwrap_or(""),
            "content": flatten_tool_output(Some(content)),
        }));
        return;
    }
    if let Some(text) = content.as_str() {
        messages.push(json!({"role": role, "content": text}));
        return;
    }
    let Some(parts) = content.as_array() else {
        if item.get("text").and_then(|v| v.as_str()).is_some() {
            messages.push(json!({
                "role": role,
                "content": item.get("text").and_then(|v| v.as_str()).unwrap_or("")
            }));
        }
        return;
    };
    let mut out_parts = Vec::new();
    for part in parts {
        match part.get("type").and_then(|t| t.as_str()).unwrap_or("text") {
            "input_image" | "image" | "image_url" => {
                let url = part
                    .get("image_url")
                    .and_then(|v| {
                        v.as_str()
                            .map(|s| s.to_string())
                            .or_else(|| v.get("url").and_then(|u| u.as_str()).map(|s| s.to_string()))
                    })
                    .or_else(|| part.get("url").and_then(|v| v.as_str()).map(|s| s.to_string()))
                    .unwrap_or_default();
                if !url.is_empty() {
                    out_parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
                }
            }
            "output_text" | "input_text" | "text" => {
                if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                    out_parts.push(json!({"type": "text", "text": t}));
                }
            }
            _ => {
                if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                    out_parts.push(json!({"type": "text", "text": t}));
                }
            }
        }
    }
    if out_parts.is_empty() {
        return;
    }
    let openai_content = if out_parts
        .iter()
        .all(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"))
    {
        json!(out_parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join(""))
    } else {
        Value::Array(out_parts)
    };
    messages.push(json!({"role": role, "content": openai_content}));
}

fn flatten_tool_output(content: Option<&Value>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    if let Some(s) = content.as_str() {
        return s.to_string();
    }
    content.to_string()
}

fn map_tool(tool: &Value) -> Option<Value> {
    let kind = tool.get("type").and_then(|t| t.as_str()).unwrap_or("function");
    if kind != "function" {
        return None;
    }
    let name = tool
        .get("name")
        .or_else(|| tool.pointer("/function/name"))
        .and_then(|v| v.as_str())?;
    let description = tool
        .get("description")
        .or_else(|| tool.pointer("/function/description"))
        .cloned()
        .unwrap_or_else(|| json!(""));
    let parameters = tool
        .get("parameters")
        .or_else(|| tool.pointer("/function/parameters"))
        .cloned()
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
    if let Some(s) = choice.as_str() {
        return json!(s);
    }
    match choice.get("type").and_then(|t| t.as_str()).unwrap_or("") {
        "required" | "any" => json!("required"),
        "none" => json!("none"),
        "function" => {
            let name = choice
                .get("name")
                .or_else(|| choice.pointer("/function/name"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            json!({"type": "function", "function": {"name": name}})
        }
        _ => json!("auto"),
    }
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
    fn converts_string_input() {
        let body = json!({
            "model": "gpt-4.1",
            "instructions": "be brief",
            "input": "hello",
            "max_output_tokens": 32
        });
        let out = to_openai_chat(&body, "local-model").unwrap();
        assert_eq!(out["model"], "local-model");
        assert_eq!(out["max_tokens"], 32);
        assert_eq!(out["messages"][0]["role"], "system");
        assert_eq!(out["messages"][1]["content"], "hello");
    }

    #[test]
    fn converts_vision_and_tools() {
        let body = json!({
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what"},
                    {"type": "input_image", "image_url": "data:image/jpeg;base64,abcd"}
                ]
            }],
            "tools": [{
                "type": "function",
                "name": "Read",
                "description": "read",
                "parameters": {"type": "object"}
            }]
        });
        let out = to_openai_chat(&body, "m").unwrap();
        assert_eq!(out["messages"][0]["content"][1]["type"], "image_url");
        assert_eq!(out["tools"][0]["function"]["name"], "Read");
    }

    #[test]
    fn converts_function_roundtrip() {
        let body = json!({
            "input": [
                {"role": "user", "content": "x"},
                {"type": "function_call", "call_id": "call_1", "name": "Read", "arguments": "{\"path\":\"a.rs\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "fn main() {}"}
            ]
        });
        let out = to_openai_chat(&body, "m").unwrap();
        assert_eq!(out["messages"][1]["tool_calls"][0]["id"], "call_1");
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
                    "content": "ok",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "Read", "arguments": "{\"path\":\"a\"}"}
                    }]
                }
            }],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3}
        });
        let out = from_openai_chat(&body, "gpt-4.1");
        assert_eq!(out["object"], "response");
        assert_eq!(out["status"], "completed");
        assert_eq!(out["output"][0]["type"], "message");
        assert_eq!(out["output"][1]["type"], "function_call");
        assert_eq!(out["output"][1]["call_id"], "call_1");
        assert_eq!(out["usage"]["total_tokens"], 12);
    }

    #[test]
    fn aliases_gpt_models() {
        assert!(looks_like_openai_alias("gpt-4.1"));
        assert!(looks_like_openai_alias("o3-mini"));
        assert!(!looks_like_openai_alias("mlx-community/qwen"));
    }
}
