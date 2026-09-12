const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

let lastStatus = null;

function bytes(n) {
  if (!n) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let x = n;
  while (x >= 1024 && i < u.length - 1) {
    x /= 1024;
    i += 1;
  }
  return `${x.toFixed(i === 0 ? 0 : 1)} ${u[i]}`;
}

function flash(msg) {
  const el = $("#flash");
  el.hidden = false;
  el.textContent = msg;
  clearTimeout(flash._t);
  flash._t = setTimeout(() => {
    el.hidden = true;
  }, 4000);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...opts,
  });
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!res.ok) {
    throw new Error(typeof data === "string" ? data : JSON.stringify(data));
  }
  return data;
}

function renderNodes(st) {
  $("#cluster-name").textContent = st.cluster_id;
  const pill = $("#serve-pill");
  const rt = st.serving.runtime ? ` · ${st.serving.runtime}` : "";
  pill.textContent = st.serving.status + (st.serving.model_id ? ` · ${st.serving.model_id}` : "") + (st.serving.status !== "stopped" ? rt : "");
  pill.className = `pill ${st.serving.status}`;
  syncUnloadButtons(st);
  $("#mesh-hint").textContent = st.mesh_ready
    ? "Thunderbolt 全互连就绪"
    : "有节点未连通，先点「启动网络」";
  const stack = st.stack || {};
  const lm = stack.mlx_lm || {};
  const vlm = stack.mlx_vlm || {};
  $("#stack-hint").textContent = `mlx-lm ${lm.installed || 0}/${lm.total || 0}${lm.version ? " " + lm.version : ""} · mlx-vlm ${vlm.installed || 0}/${vlm.total || 0}${vlm.version ? " " + vlm.version : ""}`;
  const vlmReady = (vlm.installed || 0) > 0 && vlm.installed === vlm.total;
  const installBtn = $("#btn-install-vlm");
  if (installBtn) installBtn.hidden = vlmReady;
  const vlmHint = $("#vlm-hint");
  if (vlmHint) {
    vlmHint.hidden = vlmReady;
    vlmHint.textContent = vlmReady ? "" : "有节点还没装 mlx-vlm，图/视频先点安装";
  }
  const modelHint = $("#model-hint");
  if (modelHint) {
    if (st.sync && st.sync.status === "running") {
      modelHint.textContent = `正在从 ${st.sync.source} 同步 ${st.sync.model_id} → ${(st.sync.dests || []).join("、")}。看日志末尾。`;
    } else if (st.sync && st.sync.status === "ok") {
      modelHint.textContent = `已从 ${st.sync.source} 同步完 ${st.sync.model_id}。四台齐全后即可加载。`;
    } else if (st.sync && st.sync.status === "error") {
      modelHint.textContent = `同步失败：${(st.sync.log || "").trim().split("\n").slice(-2).join(" ")}`;
    } else if (st.pull && st.pull.status === "running") {
      const last = (st.pull.log || "").trim().split("\n").slice(-1)[0] || "";
      modelHint.textContent = `正在 ${st.pull.node} 下载 ${st.pull.repo}。${last} 进度在「日志」页。`;
    } else if (st.pull && st.pull.status === "ok") {
      modelHint.textContent = `已在 ${st.pull.node} 下完 ${st.pull.repo}。点「同步到四台」再加载。`;
    } else if (st.pull && st.pull.status === "error") {
      modelHint.textContent = `拉取失败：${(st.pull.log || "").trim().split("\n").slice(-2).join(" ")}`;
    } else if (st.serving.status === "ready") {
      modelHint.textContent = `正在运行 ${st.serving.model_id || ""} · ${st.serving.runtime || ""}。换模型请先点「卸下」，或直接点另一行加载（会先卸下当前模型）。`;
    } else if (st.serving.status === "starting") {
      const elapsed = loadingElapsed(st.serving);
      modelHint.textContent = elapsed
        ? `正在四台加载（已 ${elapsed}）。「日志」页每 3 秒刷新。若停在 Loading model 且没有新行，是 JACCL 卡住，不是 mmap，点「中止加载」再重载。`
        : "正在四台加载。「日志」页每 3 秒刷新。若停在 Loading model 且没有新行，是 JACCL 卡住，点「中止加载」再重载。";
    } else if (st.serving.status === "stopping") {
      modelHint.textContent = "正在卸下当前模型，卸完后再点加载。";
    } else if (st.serving.status === "error") {
      modelHint.textContent = `加载失败：${st.serving.error || "看「日志」页"}`;
    } else {
      modelHint.textContent = "先保存 Hub 设置，选一台已装 huggingface_hub 的机器拉取，下完再同步到四台。";
    }
  }
  $("#nodes").innerHTML = st.nodes
    .map((n) => {
      const mem = n.memory_used_bytes
        ? `${bytes(n.memory_used_bytes)} / ${bytes(n.memory_total_bytes)}`
        : `${n.memory_gb} GB`;
      const links = (n.links || [])
        .map((l) => {
          const ms = l.ping_ms != null ? `${l.ping_ms.toFixed(2)} ms` : "";
          return `<li class="${l.peer_up ? "up" : ""}">${l.iface} ${l.ip} → ${l.peer_name} ${l.peer} ${ms}</li>`;
        })
        .join("");
      return `<article class="card">
        <h2><span class="dot ${n.reachable ? "ok" : ""}"></span>${n.name}</h2>
        <p class="meta">rank ${n.rank} · ${n.ssh}<br>${n.hostname || "离线"} · ${mem}</p>
        <p class="meta">RDMA ${n.rdma_enabled ? "on" : "off"} · mesh ${n.mesh_ready ? "ok" : "no"} · lm ${n.mlx_lm || "—"} · vlm ${n.mlx_vlm || "—"} · hf ${n.huggingface_hub || "—"}</p>
        ${n.python ? `<p class="meta">venv ${escapeHtml(n.python)}</p>` : ""}
        <ul class="links">${links || "<li>无链路数据</li>"}</ul>
        ${n.error ? `<p class="meta">${n.error}</p>` : ""}
      </article>`;
    })
    .join("");
}

function fillNodeSelect(sel, nodes, fallback) {
  if (!sel) return;
  const editing = document.activeElement === sel;
  const prev = sel.value;
  sel.innerHTML = (nodes || [])
    .map((n) => {
      const hf = n.huggingface_hub;
      const label = hf ? `${n.name} · hf ${hf}` : `${n.name} · 未安装 huggingface_hub`;
      const disabled = hf ? "" : " disabled";
      return `<option value="${escapeHtml(n.name)}"${disabled}>${escapeHtml(label)}</option>`;
    })
    .join("");
  const want = editing ? prev : (prev || fallback || "");
  const ok = [...sel.options].some((o) => o.value === want && !o.disabled);
  if (ok) {
    sel.value = want;
    return;
  }
  const ready = (nodes || []).find((n) => n.huggingface_hub);
  if (ready) sel.value = ready.name;
}

function selectedPullNode() {
  const pull = $("#pull-node");
  const hub = $("#hub-form")?.node;
  const node = pull?.value || hub?.value || "";
  const opt = pull?.selectedOptions?.[0] || hub?.selectedOptions?.[0];
  if (!node) return { node: "", error: "先在「拉取到」选一台已安装 huggingface_hub 的机器" };
  if (opt?.disabled) {
    return { node, error: "所选机器没有 huggingface_hub，换一台或先在该机 venv 里安装" };
  }
  return { node, error: "" };
}

function renderHub(st) {
  const f = $("#hub-form");
  if (!f) return;
  const hub = st.hub || {};
  const nodes = st.nodes || [];
  fillNodeSelect(f.node, nodes, hub.node);
  fillNodeSelect($("#pull-node"), nodes, $("#pull-node")?.value || hub.node);
  const current = document.activeElement && f.contains(document.activeElement);
  if (current) return;
  f.token.placeholder = hub.token_set ? "已保存，留空不改" : "hf_…";
  f.token.value = "";
  f.token_clear.checked = false;
  f.endpoint.value = hub.endpoint || "";
  f.dest_dir.value = hub.dest_dir || "";
  f.python.value = hub.python || "";
}

function renderEndpoints(st) {
  const ep = st.endpoint;
  const f = $("#endpoint-form");
  f.advertise_host.value = ep.advertise_host;
  f.bind.value = ep.bind;
  f.port.value = ep.port;
  f.api_key.value = ep.api_key || "";
  f.enabled.checked = ep.enabled;
  const key = ep.api_key || "sk-local";
  const auth = ep.api_key
    ? `  -H "Authorization: Bearer ${ep.api_key}" \\\n`
    : "";
  const vlm = st.serving.runtime === "mlx_vlm";
  const vision = `  -d '{"model":"${st.serving.model_id || "local"}","messages":[{"role":"user","content":[{"type":"text","text":"描述这张图"},{"type":"image_url","image_url":{"url":"https://example.com/frame.jpg"}}]}]}'`;
  const text = `  -d '{"model":"${st.serving.model_id || "local"}","messages":[{"role":"user","content":"你好"}]}'`;
  $("#snippet").textContent = `运行时: ${st.serving.runtime || "未加载"}

OpenAI（Cursor / OpenAI SDK）
${ep.public_url}

curl ${ep.public_url}/chat/completions \\
${auth}  -H "Content-Type: application/json" \\
${vlm ? vision : text}

OpenAI Responses（Codex / 新 SDK）
${ep.public_url}/responses

curl ${ep.public_url}/responses \\
${auth}  -H "Content-Type: application/json" \\
  -d '{"model":"${st.serving.model_id || "gpt-4.1"}","input":"你好"}'

Claude（Claude Code / Anthropic SDK）
ANTHROPIC_BASE_URL=${ep.claude_url}
ANTHROPIC_API_KEY=${key}

export ANTHROPIC_BASE_URL=${ep.claude_url}
export ANTHROPIC_API_KEY=${key}
claude

curl ${ep.claude_url}/v1/messages \\
  -H "x-api-key: ${key}" \\
  -H "anthropic-version: 2023-06-01" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"claude-sonnet-4-5","max_tokens":128,"messages":[{"role":"user","content":"你好"}]}'`;
}

function servingActive(st = lastStatus) {
  const s = st?.serving?.status;
  return s === "ready" || s === "starting" || s === "stopping" || s === "error";
}

function syncUnloadButtons(st) {
  const show = servingActive(st) && st?.serving?.status !== "stopped";
  const starting = st?.serving?.status === "starting";
  const stopping = st?.serving?.status === "stopping";
  const top = $("#btn-unload");
  const models = $("#btn-model-unload");
  if (top) {
    top.hidden = !show;
    top.disabled = stopping;
    top.textContent = starting ? "中止加载" : "卸下";
  }
  if (models) {
    models.hidden = !show;
    models.disabled = stopping;
    models.textContent = starting ? "中止加载" : "卸下当前模型";
  }
}

async function unloadCurrent() {
  try {
    flash("正在卸下当前模型…");
    await api("/api/serve/stop", { method: "POST", body: "{}" });
    flash("已卸下，可以加载其他模型");
    await refresh();
  } catch (e) {
    flash(e.message);
  }
}

function replicaLabel(m) {
  const reps = m.replicas || [];
  if (!reps.length) {
    return m.complete ? "本机完整" : "不完整";
  }
  const have = reps.filter((r) => r.complete);
  const downloading = reps.filter((r) => r.downloading);
  const missing = reps.filter((r) => !r.complete && !r.downloading).map((r) => r.node);
  const src = m.source_node || (have[0] && have[0].node) || (downloading[0] && downloading[0].node) || "";
  if (m.cluster_complete) {
    return `四台完整`;
  }
  if (downloading.length) {
    const who = downloading.map((r) => `${r.node} ${bytes(r.size_bytes)}`).join("、");
    return `下载中 · ${who}`;
  }
  if (have.length) {
    return `${have.length}/${reps.length} · ${src}${missing.length ? ` · 缺 ${missing.join("、")}` : ""}`;
  }
  return "不完整";
}

function renderModels(models, serving, stack) {
  const vlmReady = (stack?.mlx_vlm?.installed || 0) === (stack?.mlx_vlm?.total || 0) && (stack?.mlx_vlm?.installed || 0) > 0;
  const syncing = lastStatus?.sync?.status === "running";
  $("#model-rows").innerHTML = models
    .map((m) => {
      const loaded = serving.model_id === m.id || serving.model_path === m.path;
      const p = m.profile || {};
      let actions = "";
      if (m.downloading) {
        actions += `<span class="meta">下载中</span> `;
      }
      if (m.complete) {
        const syncLabel = m.cluster_complete ? "再同步" : "同步到四台";
        actions += `<button type="button" data-sync="${m.id}" ${syncing || m.downloading ? "disabled" : ""}>${syncLabel}</button> `;
      }
      if (m.cluster_complete) {
        const busy = servingActive({ serving });
        const thisLoaded = loaded && busy;
        if (thisLoaded) {
          const stopLabel = serving.status === "starting" ? "中止加载" : "卸下";
          actions += `<button type="button" class="danger" data-unload="${escapeHtml(m.id)}" ${serving.status === "stopping" ? "disabled" : ""}>${stopLabel}</button> `;
        }
        if (p.allow_vlm) {
          const primary = p.default_runtime === "mlx_vlm" ? "primary" : "";
          const label = thisLoaded ? "重新加载视觉" : (p.load_vlm_label || "加载视觉");
          actions += `<button type="button" class="${primary}" data-load="${m.id}" data-runtime="mlx_vlm" ${vlmReady ? "" : "disabled"}>${escapeHtml(label)}</button> `;
        }
        if (p.allow_lm) {
          const primary = p.default_runtime !== "mlx_vlm" ? "primary" : "";
          const label = thisLoaded ? "重新加载" : (p.load_lm_label || "加载");
          actions += `<button type="button" class="${primary}" data-load="${m.id}" data-runtime="mlx_lm">${escapeHtml(label)}</button>`;
        }
      } else if (m.complete) {
        actions += `<span class="meta">先同步再加载</span>`;
      }
      const del = `<button type="button" class="danger" data-del="${escapeHtml(m.id)}">删除四台</button>`;
      return `<tr>
        <td>${escapeHtml(m.name)}<br><span class="meta">${escapeHtml(m.path)}</span></td>
        <td><span class="family-tag">${escapeHtml(p.title || m.kind)}</span><br><span class="meta">${escapeHtml(m.kind)}${m.architecture ? ` · ${escapeHtml(m.architecture)}` : ""}</span></td>
        <td>${bytes(m.size_bytes)}</td>
        <td>${replicaLabel(m)}${loaded ? " · 已加载" : ""}</td>
        <td class="model-plan">${escapeHtml(p.hint || "")}</td>
        <td>${actions}</td>
        <td>${del}</td>
      </tr>`;
    })
    .join("");
}

$$(".tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tabs button").forEach((b) => b.classList.remove("on"));
    $$(".panel").forEach((p) => p.classList.remove("on"));
    btn.classList.add("on");
    $(`#tab-${btn.dataset.tab}`).classList.add("on");
    if (btn.dataset.tab === "logs") refreshLogs();
    if (btn.dataset.tab === "chat") {
      renderChat();
      $("#chat-input").focus();
    }
  });
});

$("#btn-cluster-up").addEventListener("click", async () => {
  try {
    flash("正在配置 Thunderbolt…");
    await api("/api/cluster/start", { method: "POST", body: "{}" });
    flash("网络已启动");
    await refresh();
  } catch (e) {
    flash(e.message);
  }
});

$("#btn-serve-stop").addEventListener("click", unloadCurrent);
$("#btn-unload").addEventListener("click", unloadCurrent);
$("#btn-model-unload").addEventListener("click", unloadCurrent);

$("#copy-url").addEventListener("click", async () => {
  if (!lastStatus) return;
  await navigator.clipboard.writeText(lastStatus.endpoint.public_url);
  flash("已复制 " + lastStatus.endpoint.public_url);
});

$("#copy-claude").addEventListener("click", async () => {
  if (!lastStatus) return;
  const ep = lastStatus.endpoint;
  const key = ep.api_key || "sk-local";
  const text = `export ANTHROPIC_BASE_URL=${ep.claude_url}\nexport ANTHROPIC_API_KEY=${key}`;
  await navigator.clipboard.writeText(text);
  flash("已复制 Claude 环境变量");
});

$("#btn-pull").addEventListener("click", async () => {
  const repo = $("#pull-repo").value.trim();
  if (!repo) return;
  const f = $("#hub-form");
  const picked = selectedPullNode();
  if (picked.error) {
    flash(picked.error);
    return;
  }
  try {
    flash(`在 ${picked.node} 开始下载 ${repo}，模型页会显示进度`);
    const out = await api("/api/models/pull", {
      method: "POST",
      body: JSON.stringify({
        repo,
        nodes: [picked.node],
        dest: f.dest_dir.value.trim() || undefined,
      }),
    });
    const err = (out.nodes || []).find((n) => n.error || (n.status && n.status >= 400));
    flash(err ? `${err.node}: ${err.error || err.body}` : `正在 ${picked.node} 下载，看模型页「下载中」和日志`);
    console.log(out);
  } catch (e) {
    flash(e.message);
  }
});

$("#pull-node")?.addEventListener("change", () => {
  const f = $("#hub-form");
  if (f?.node && $("#pull-node").value) f.node.value = $("#pull-node").value;
});

$("#hub-form")?.node?.addEventListener("change", () => {
  const pull = $("#pull-node");
  if (pull && $("#hub-form").node.value) pull.value = $("#hub-form").node.value;
});

$$(".pull-chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    const repo = btn.dataset.repo || "";
    $("#pull-repo").value = repo;
    const picked = selectedPullNode();
    if (picked.error) {
      flash(picked.error);
      $("#pull-node")?.focus();
      return;
    }
    $("#btn-pull").click();
  });
});

$("#btn-install-vlm").addEventListener("click", async () => {
  try {
    flash("四台安装 mlx-vlm，可能要一两分钟…");
    await api("/api/stack/install", {
      method: "POST",
      body: JSON.stringify({ package: "mlx-vlm" }),
    });
    flash("mlx-vlm 安装完成");
    await refresh();
  } catch (e) {
    flash(e.message);
  }
});

$("#model-rows").addEventListener("click", async (ev) => {
  const delId = ev.target?.dataset?.del;
  if (delId) {
    const name = ev.target.closest("tr")?.querySelector("td")?.innerText?.split("\n")[0] || delId;
    if (!window.confirm(`从四台删除 ${name}？不可恢复。若正在加载会先停止推理。`)) {
      return;
    }
    try {
      flash("正在四台删除…");
      await api("/api/models/delete", {
        method: "POST",
        body: JSON.stringify({ model_id: delId }),
        signal: AbortSignal.timeout(180000),
      });
      flash("已删除");
      await refresh();
    } catch (e) {
      flash(e.message);
    }
    return;
  }
  const syncId = ev.target?.dataset?.sync;
  if (syncId) {
    try {
      flash("开始从已有完整副本同步到其余三台（走 Thunderbolt）…");
      const out = await api("/api/models/sync", {
        method: "POST",
        body: JSON.stringify({ model_id: syncId }),
        signal: AbortSignal.timeout(30000),
      });
      flash(out?.sync?.source ? `已从 ${out.sync.source} 开始复制` : "同步已开始");
      await refresh();
    } catch (e) {
      flash(e.message);
    }
    return;
  }
  const unloadId = ev.target?.dataset?.unload;
  if (unloadId) {
    await unloadCurrent();
    return;
  }
  const id = ev.target?.dataset?.load;
  if (!id) return;
  const runtime = ev.target.dataset.runtime || undefined;
  const name = ev.target.closest("tr")?.querySelector("td")?.innerText?.split("\n")[0] || id;
  const serving = lastStatus?.serving;
  const busy = servingActive();
  const current = serving?.model_id || serving?.model_path || "";
  if (busy && serving?.status === "stopping") {
    flash("正在卸下当前模型，稍后再加载");
    return;
  }
  if (busy && current && current !== id) {
    const action = serving.status === "starting" ? "中止当前加载" : "先卸下";
    if (!window.confirm(`当前正在运行 ${current}。${action}再加载 ${name}？`)) {
      return;
    }
  } else if (busy && current === id) {
    if (!window.confirm(`${name} 已在运行或正在加载。卸下后重新加载？`)) {
      return;
    }
  }
  try {
    flash(runtime === "mlx_vlm" ? "正在卸下旧模型并按该方案加载（mlx-vlm）…" : "正在卸下旧模型并四台加载…");
    await api("/api/serve/start", {
      method: "POST",
      body: JSON.stringify({ model_id: id, runtime, autostart: true }),
      signal: AbortSignal.timeout(180000),
    });
    flash("已开始加载，状态栏会变成 ready");
    await refresh();
  } catch (e) {
    flash(e.name === "TimeoutError" ? "加载请求超时，看「日志」页" : e.message);
  }
});

$("#hub-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = ev.target;
  try {
    await api("/api/hub", {
      method: "PUT",
      body: JSON.stringify({
        token: f.token.value.trim(),
        token_clear: f.token_clear.checked,
        endpoint: f.endpoint.value,
        python: f.python.value.trim(),
        dest_dir: f.dest_dir.value.trim(),
        node: f.node.value,
      }),
    });
    f.token.value = "";
    f.token_clear.checked = false;
    flash("Hub 设置已保存");
    await refresh();
  } catch (e) {
    flash(e.message);
  }
});

$("#endpoint-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = ev.target;
  try {
    await api("/api/endpoints", {
      method: "PUT",
      body: JSON.stringify({
        advertise_host: f.advertise_host.value.trim(),
        bind: f.bind.value.trim() || "0.0.0.0",
        port: Number(f.port.value),
        api_key: f.api_key.value.trim(),
        enabled: f.enabled.checked,
      }),
    });
    flash("接口已更新（不必重载模型）");
    await refresh();
  } catch (e) {
    flash(e.message);
  }
});

async function refreshLogs() {
  try {
    const logs = await api("/api/logs");
    const sync = logs.sync
      ? `\n\n==== sync ${logs.sync.status || ""} ${logs.sync.source || ""} → ${logs.sync.model_id || ""} ====\n${logs.sync.log || ""}`
      : "";
    const pull = logs.pull
      ? `\n\n==== pull ${logs.pull.status || ""} ${logs.pull.node || ""} ${logs.pull.repo || ""} ====\n${logs.pull.log || ""}`
      : "";
    $("#log-box").textContent = (logs.serve || "暂无日志") + sync + pull;
  } catch (e) {
    $("#log-box").textContent = e.message;
  }
}

$("#btn-clear-logs")?.addEventListener("click", async () => {
  if (!window.confirm("清除推理日志？正在下载或同步的进度会保留。")) {
    return;
  }
  try {
    const out = await api("/api/logs/clear", { method: "POST", body: "{}" });
    if (out && out.ok === false) {
      throw new Error(out.error || "清除失败");
    }
    flash("日志已清除");
    await refreshLogs();
  } catch (e) {
    flash(e.message);
  }
});

async function refresh() {
  const [st, models] = await Promise.all([api("/api/status"), api("/api/models")]);
  lastStatus = st;
  renderNodes(st);
  renderHub(st);
  renderEndpoints(st);
  renderChatMeta(st);
  renderModels(models, st.serving, st.stack);
}

function logsTabOn() {
  return document.querySelector(".tabs button.on")?.dataset.tab === "logs";
}

refresh().catch((e) => flash(e.message));
setInterval(() => {
  api("/api/status")
    .then(async (st) => {
      const wasSync = lastStatus?.sync?.status;
      const wasPull = lastStatus?.pull?.status;
      lastStatus = st;
      renderNodes(st);
      renderHub(st);
      renderEndpoints(st);
      renderChatMeta(st);
      const syncBusy = st.sync?.status === "running" || (wasSync === "running" && st.sync?.status !== "running");
      const pullBusy = st.pull?.status === "running" || (wasPull === "running" && st.pull?.status !== "running");
      if (syncBusy || pullBusy) {
        const models = await api("/api/models");
        renderModels(models, st.serving, st.stack);
      }
      if (logsTabOn()) refreshLogs();
    })
    .catch(() => {});
}, 3000);
setInterval(() => {
  if (!lastStatus) return;
  api("/api/models")
    .then((models) => renderModels(models, lastStatus.serving, lastStatus.stack))
    .catch(() => {});
}, 5000);

const CHAT_LANG_KEY = "mlxctl.chat.lang";
const CHAT_LONG_KEY = "mlxctl.chat.long";
const CHAT_MAX_TOKENS_KEY = "mlxctl.chat.max_tokens";
const CHAT_THINKING_PROMPT =
  "请在 <think>...</think> 中写出简要推理，然后给出最终回答。不要省略 think 结束标签。";

const CHAT_LANGS = {
  auto: null,
  zh: "请始终用简体中文回复。即使用户用其他语言提问，也用中文回答。",
  en: "Always reply in English, even if the user writes in another language.",
  ja: "常に日本語で返信してください。ユーザーが他の言語で質問しても日本語で答えてください。",
  ko: "항상 한국어로 답하세요. 사용자가 다른 언어로 물어봐도 한국어로 답하세요.",
};

function servingProfile() {
  return lastStatus?.serving?.profile || null;
}

function familyKey(p) {
  return p?.family || "generic";
}

function familySettingsKey(family) {
  return `mlxctl.chat.family.${family}`;
}

function migrateLegacyChatSettings() {
  if (localStorage.getItem(familySettingsKey("qwen35"))) return;
  const lang = localStorage.getItem(CHAT_LANG_KEY);
  const long = localStorage.getItem(CHAT_LONG_KEY);
  const max = localStorage.getItem(CHAT_MAX_TOKENS_KEY);
  if (lang == null && long == null && max == null) return;
  localStorage.setItem(
    familySettingsKey("qwen35"),
    JSON.stringify({
      lang: lang || "auto",
      long: long === "1",
      max_tokens: parseInt(max, 10) || 2048,
      thinking: false,
    }),
  );
}

function defaultFamilySettings(p) {
  const longOn = false;
  return {
    lang: "auto",
    long: longOn,
    max_tokens: p?.chat_safe_max_tokens || 2048,
    thinking: false,
  };
}

function readFamilySettings(p) {
  const defaults = defaultFamilySettings(p);
  const raw = localStorage.getItem(familySettingsKey(familyKey(p)));
  if (!raw) return defaults;
  try {
    return { ...defaults, ...JSON.parse(raw) };
  } catch {
    return defaults;
  }
}

function tokenCapFor(p, longOn) {
  if (!p) return longOn ? 4096 : 2048;
  return longOn ? p.chat_long_max_tokens : p.chat_safe_max_tokens;
}

const chat = {
  turns: [],
  pendingImages: [],
  abort: null,
  busy: false,
  appliedFamily: null,
};

function chatLang() {
  const v = $("#chat-lang")?.value || "auto";
  return Object.prototype.hasOwnProperty.call(CHAT_LANGS, v) ? v : "auto";
}

function longCompletionOn() {
  return $("#chat-long")?.checked === true;
}

function thinkingOn() {
  return $("#chat-think")?.checked === true;
}

function tokenCap() {
  return tokenCapFor(servingProfile(), longCompletionOn());
}

function chatMaxTokens() {
  const p = servingProfile();
  const input = $("#chat-max-tokens");
  let n = parseInt(input?.value, 10);
  const fallback = longCompletionOn()
    ? p?.chat_long_default_tokens || 4096
    : p?.chat_safe_max_tokens || 2048;
  if (!Number.isFinite(n)) n = fallback;
  n = Math.max(16, Math.min(tokenCap(), n));
  if (input && String(input.value) !== String(n)) input.value = String(n);
  return n;
}

function persistChatGenSettings() {
  const p = servingProfile() || { family: "generic" };
  const stored = {
    lang: chatLang(),
    long: longCompletionOn(),
    max_tokens: chatMaxTokens(),
    thinking: thinkingOn(),
  };
  localStorage.setItem(familySettingsKey(familyKey(p)), JSON.stringify(stored));
}

function applyFamilySettings(p, { force } = {}) {
  const family = familyKey(p);
  if (!force && chat.appliedFamily === family) return;
  const stored = readFamilySettings(p);
  const sel = $("#chat-lang");
  const longBox = $("#chat-long");
  const thinkBox = $("#chat-think");
  const input = $("#chat-max-tokens");
  if (sel) sel.value = stored.lang || "auto";
  if (longBox) longBox.checked = !!stored.long;
  if (thinkBox) thinkBox.checked = !!stored.thinking;
  if (input) input.value = String(stored.max_tokens || p?.chat_safe_max_tokens || 2048);
  chat.appliedFamily = family;
  syncTokenLimits();
}

function syncTokenLimits({ bumpIfEnabling } = {}) {
  const input = $("#chat-max-tokens");
  const box = $("#chat-long");
  const p = servingProfile();
  if (!input) return;
  const long = longCompletionOn();
  const cap = tokenCapFor(p, long);
  input.max = String(cap);
  const hint = p?.chat_long_hint || (long ? `最长 ${cap}` : `关闭超长补全时上限 ${cap}`);
  input.title = hint;
  if (box) box.title = hint;
  let n = parseInt(input.value, 10);
  if (!Number.isFinite(n)) n = p?.chat_safe_max_tokens || 2048;
  if (long && bumpIfEnabling && p && n <= p.chat_safe_max_tokens) {
    n = p.chat_long_default_tokens;
  }
  if (!long && p && n > p.chat_safe_max_tokens) n = p.chat_safe_max_tokens;
  if (n > cap) n = cap;
  if (n < 16) n = 16;
  input.value = String(n);
  persistChatGenSettings();
}

function loadingElapsed(serving) {
  const started = Date.parse(serving?.started_at || "");
  if (!started) return "";
  const sec = Math.max(0, Math.floor((Date.now() - started) / 1000));
  if (sec < 60) return `${sec} 秒`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return s ? `${m} 分 ${s} 秒` : `${m} 分钟`;
}

function chatReady() {
  return lastStatus?.serving?.status === "ready";
}

function renderChatMeta(st) {
  const el = $("#chat-meta");
  if (!el) return;
  const s = st.serving || {};
  const p = s.profile;
  if (s.status === "ready") {
    el.textContent = `${s.model_id || "local"} · ${p?.title || s.runtime || "mlx-lm"} · 已就绪`;
  } else if (s.status === "starting") {
    const elapsed = loadingElapsed(s);
    el.textContent = elapsed ? `模型正在加载…（已 ${elapsed}）` : "模型正在加载…";
  } else {
    el.textContent = "尚未加载模型，先到「模型」页点加载";
  }
  const banner = $("#chat-family");
  if (banner) {
    if (p && (s.status === "ready" || s.status === "starting")) {
      banner.hidden = false;
      banner.textContent = `方案 ${p.title}：对话设置只作用于这一类模型，不和 Qwen3.5 / Flash-Next / V4 互相覆盖。`;
    } else {
      banner.hidden = true;
    }
  }
  const thinkWrap = $("#chat-think-wrap");
  if (thinkWrap) thinkWrap.hidden = !p?.thinking;
  if (p) applyFamilySettings(p);
  const vlm = s.runtime === "mlx_vlm" && s.status === "ready" && !!p?.vision;
  const attach = $("#chat-attach-wrap");
  if (attach) attach.hidden = !vlm;
  if (!vlm) chat.pendingImages = [];
  $("#chat-send").disabled = chat.busy || s.status === "starting";
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function visibleText(s) {
  const closed = s.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  if (/<think>/i.test(s) && !/<\/think>/i.test(s)) return closed ? closed + "\n思考中…" : "思考中…";
  return closed || s;
}

function turnHtml(turn, lastBusy) {
  const role = turn.role === "user" ? "你" : "模型";
  let body = "";
  if (Array.isArray(turn.content)) {
    for (const part of turn.content) {
      if (part.type === "text") body += escapeHtml(part.text || "");
      if (part.type === "image_url" && part.image_url?.url) {
        body += `<img src="${part.image_url.url}" alt="">`;
      }
    }
  } else {
    body = escapeHtml(visibleText(turn.content || ""));
  }
  const cls = `bubble ${turn.role}${lastBusy && turn.role === "assistant" ? " busy" : ""}`;
  return `<article class="${cls}"><span class="bubble-role">${role}</span>${body || " "}</article>`;
}

function renderChat() {
  const thread = $("#chat-thread");
  if (!chat.turns.length) {
    thread.innerHTML = `<p class="chat-empty" id="chat-empty">${
      chatReady() ? "跟当前加载的模型说话。" : "先在「模型」页加载，然后在这里直接跟后端模型对话。"
    }</p>`;
  } else {
    thread.innerHTML = chat.turns
      .map((t, i) => turnHtml(t, chat.busy && i === chat.turns.length - 1))
      .join("");
    thread.scrollTop = thread.scrollHeight;
  }
  $("#chat-send").hidden = chat.busy;
  $("#chat-stop").hidden = !chat.busy;
  renderPreviews();
}

function renderPreviews() {
  const box = $("#chat-previews");
  if (!chat.pendingImages.length) {
    box.hidden = true;
    box.innerHTML = "";
    return;
  }
  box.hidden = false;
  box.innerHTML = chat.pendingImages
    .map((url) => `<img src="${url}" alt="">`)
    .join("");
}

function historyForApi() {
  const msgs = chat.turns
    .filter((t) => t.role === "user" || (t.role === "assistant" && t.content))
    .map((t) => ({ role: t.role, content: t.content }));
  const prompt = CHAT_LANGS[chatLang()];
  const sys = [];
  if (prompt) sys.push(prompt);
  if (thinkingOn() && servingProfile()?.thinking) sys.push(CHAT_THINKING_PROMPT);
  if (sys.length) msgs.unshift({ role: "system", content: sys.join("\n") });
  return msgs;
}

function setBusy(on) {
  chat.busy = on;
  $("#chat-send").hidden = on;
  $("#chat-stop").hidden = !on;
  $("#chat-input").disabled = on;
}

async function sendChat(ev) {
  ev?.preventDefault();
  if (chat.busy) return;
  const text = $("#chat-input").value.trim();
  const images = chat.pendingImages.slice();
  if (!text && !images.length) return;
  if (!chatReady()) {
    flash("模型还没就绪");
    return;
  }
  let content;
  if (images.length) {
    content = [];
    if (text) content.push({ type: "text", text });
    for (const url of images) content.push({ type: "image_url", image_url: { url } });
  } else {
    content = text;
  }
  chat.turns.push({ role: "user", content });
  chat.turns.push({ role: "assistant", content: "" });
  chat.pendingImages = [];
  $("#chat-input").value = "";
  $("#chat-images").value = "";
  setBusy(true);
  renderChat();
  chat.abort = new AbortController();
  const assistant = chat.turns[chat.turns.length - 1];
  try {
    const res = await fetch("/api/infer/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      signal: chat.abort.signal,
      body: JSON.stringify({
        model: lastStatus.serving.model_path || lastStatus.serving.model_id || "default_model",
        messages: historyForApi(),
        stream: true,
        max_tokens: chatMaxTokens(),
        temperature: servingProfile()?.temperature ?? 0.7,
        enable_thinking: thinkingOn() && !!servingProfile()?.thinking,
      }),
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || res.statusText);
    }
    const ctype = res.headers.get("content-type") || "";
    if (!res.body || (!ctype.includes("event-stream") && !ctype.includes("octet-stream") && !ctype.includes("json"))) {
      assistant.content = await res.text();
    } else if (ctype.includes("application/json") && !ctype.includes("event-stream")) {
      const data = await res.json();
      assistant.content = data.choices?.[0]?.message?.content || JSON.stringify(data);
    } else {
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() || "";
        for (const line of lines) {
          const data = line.startsWith("data:") ? line.slice(5).trim() : "";
          if (!data || data === "[DONE]") continue;
          try {
            const chunk = JSON.parse(data);
            const delta = chunk.choices?.[0]?.delta?.content;
            if (delta) {
              assistant.content += delta;
              renderChat();
            }
          } catch {
            /* ignore malformed sse */
          }
        }
      }
    }
  } catch (e) {
    if (e.name === "AbortError") {
      if (!assistant.content) assistant.content = "（已停止）";
    } else {
      assistant.content = assistant.content || `出错：${e.message}`;
      flash(e.message);
    }
  } finally {
    chat.abort = null;
    setBusy(false);
    renderChat();
  }
}

$("#chat-form").addEventListener("submit", sendChat);
$("#chat-stop").addEventListener("click", () => chat.abort?.abort());
$("#chat-clear").addEventListener("click", () => {
  if (chat.busy) chat.abort?.abort();
  chat.turns = [];
  chat.pendingImages = [];
  renderChat();
});
{
  migrateLegacyChatSettings();
  const sel = $("#chat-lang");
  if (sel) {
    sel.addEventListener("change", () => persistChatGenSettings());
  }
  const longBox = $("#chat-long");
  const tokenInput = $("#chat-max-tokens");
  const thinkBox = $("#chat-think");
  if (longBox) {
    longBox.addEventListener("change", () => {
      syncTokenLimits({ bumpIfEnabling: longBox.checked });
    });
  }
  if (thinkBox) {
    thinkBox.addEventListener("change", () => persistChatGenSettings());
  }
  if (tokenInput) {
    tokenInput.addEventListener("change", () => syncTokenLimits());
    tokenInput.addEventListener("blur", () => syncTokenLimits());
  }
  syncTokenLimits();
}
$("#chat-input").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    sendChat();
  }
});
$("#chat-images").addEventListener("change", async (ev) => {
  const files = [...(ev.target.files || [])];
  for (const file of files) {
    const url = await new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result);
      r.onerror = reject;
      r.readAsDataURL(file);
    });
    chat.pendingImages.push(url);
  }
  renderPreviews();
});
