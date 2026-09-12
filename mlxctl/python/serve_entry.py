#!/usr/bin/env python3
"""Entry point for distributed mlx_lm / mlx_vlm serving.

Forces JACCL, and for mlx-vlm uses sharded_load so the language model is
split across ranks (vision tower stays replicated; it is small vs the MoE).
"""
from __future__ import annotations

import os
import sys
import threading
import time


try:
    sys.stderr.reconfigure(line_buffering=True)
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _raise_nofile(target: int = 65536) -> None:
    """macOS launchd/ssh sessions default to 256 FDs; Flash-Next has 131 shards."""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        cap = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if soft < cap:
            resource.setrlimit(resource.RLIMIT_NOFILE, (cap, hard))
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        _log(f"mlxctl: RLIMIT_NOFILE soft={soft} hard={hard}")
    except Exception as exc:
        _log(f"mlxctl: could not raise RLIMIT_NOFILE: {exc}")


_raise_nofile()

os.environ.setdefault("MLX_DISTRIBUTED_BACKEND", "jaccl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_argv_blob = " ".join(sys.argv).lower()
if any(tag in _argv_blob for tag in ("flash-next", "qwen4", "qwen3.8-flash")):
    # QSA KV cache cannot use KV_BITS with continuous batching.
    os.environ.pop("KV_BITS", None)
    os.environ.setdefault("MLX_VLM_MAX_NUM_SEQS", "1")

import mlx.core as mx  # noqa: E402

try:
    if mx.metal.is_available():
        rec = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(rec)
        _log(f"mlxctl: metal wired_limit={rec / 1e9:.1f}GB")
except Exception as exc:
    _log(f"mlxctl: wired_limit skipped: {exc}")

_orig_init = mx.distributed.init
_init_lock = threading.Lock()
_init_started = False


def _spawn_init_watchdog():
    """Heartbeat in a separate process: jaccl.init holds the GIL, so threads go silent."""
    import subprocess
    import tempfile

    marker = os.path.join(tempfile.gettempdir(), f"mlxctl-jaccl-init-{os.getpid()}")
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write("1")
    script = (
        "import os, sys, time\n"
        "marker, pid = sys.argv[1], sys.argv[2]\n"
        "t0 = time.time()\n"
        "while os.path.exists(marker):\n"
        "    time.sleep(5)\n"
        "    print(\n"
        "        f'mlxctl: still in mx.distributed.init '\n"
        "        f'({int(time.time()-t0)}s) pid={pid}',\n"
        "        file=sys.stderr,\n"
        "        flush=True,\n"
        "    )\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script, marker, str(os.getpid())],
        stdout=sys.stderr,
        stderr=sys.stderr,
        start_new_session=True,
    )
    return marker, proc


def _init(backend="any", *args, **kwargs):
    forced = os.environ.get("MLX_DISTRIBUTED_BACKEND")
    if forced:
        backend = forced
        kwargs.setdefault("strict", True)
    global _init_started
    first = False
    with _init_lock:
        if not _init_started:
            _init_started = True
            first = True
    marker = None
    if first:
        _log(f"mlxctl: mx.distributed.init backend={backend} strict={kwargs.get('strict')}")
        try:
            marker, _watch = _spawn_init_watchdog()
        except Exception as exc:
            _log(f"mlxctl: init watchdog failed: {exc}")
            marker = None
    try:
        if args:
            group = _orig_init(backend, *args, **kwargs)
        else:
            group = _orig_init(backend=backend, **kwargs)
        if first:
            _log(
                f"mlxctl: distributed ready rank={group.rank()} size={group.size()}"
            )
        return group
    finally:
        if marker:
            try:
                os.remove(marker)
            except FileNotFoundError:
                pass


mx.distributed.init = _init


def _wrap_tp_residual(cls) -> None:
    """Match mlx-lm: sum_gradients in, all_sum out, when sharding_group is set.

    GatedDeltaNet / SparseMoeBlock use shard_inplace (weight split only), so the
    block itself must all-sum. Dense MLP / softmax attention use shard_linear
    and already communicate inside the layer.
    """
    orig = cls.__call__
    if getattr(orig, "_mlxctl_tp_wrapped", False):
        return

    from mlx.nn.layers.distributed import sum_gradients

    def __call__(self, *args, **kwargs):
        group = getattr(self, "sharding_group", None)
        if group is not None:
            if args:
                args = (sum_gradients(group)(args[0]), *args[1:])
            else:
                for key in ("inputs", "x"):
                    if key in kwargs:
                        kwargs = dict(kwargs)
                        kwargs[key] = sum_gradients(group)(kwargs[key])
                        break
        out = orig(self, *args, **kwargs)
        if group is not None:
            out = mx.distributed.all_sum(out, group=group)
        return out

    __call__._mlxctl_tp_wrapped = True
    cls.__call__ = __call__


def _qwen35_shard(self, group=None):
    """Tensor-parallel the Qwen3.5 / Qwen3.5-MoE language stack. Vision is left whole."""
    from mlx.nn.layers.distributed import shard_inplace, shard_linear
    from mlx.utils import tree_map
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

    group = group or mx.distributed.init()
    N = group.size()
    rank = group.rank()
    if N <= 1:
        return

    def conv_sharding(key_dim):
        return lambda p, w: (0, [key_dim, 2 * key_dim])

    def repeat_kv_layer_inplace(layer, h):
        if N <= h:
            return

        def _repeat(p):
            s = p.shape
            p = p.reshape(h, s[0] // h, *s[1:])
            p = mx.repeat(p, N // h, axis=0)
            p = p.reshape(-1, *s[1:])
            return p

        layer.update(tree_map(_repeat, layer.parameters()))

    layers = self.layers
    if not layers:
        raise ValueError("Qwen3.5 shard: language model has no layers")

    for layer in layers:
        if getattr(layer, "is_linear", False):
            attn = layer.linear_attn
            if attn.num_k_heads % N != 0 or attn.num_v_heads % N != 0:
                raise ValueError(
                    f"GatedDeltaNet heads k={attn.num_k_heads} v={attn.num_v_heads} "
                    f"not divisible by tensor-parallel size {N}"
                )
            kd = attn.key_dim
            attn.sharding_group = group
            shard_inplace(attn.conv1d, conv_sharding(kd), group=group)
            attn.conv1d.groups //= N
            shard_inplace(
                attn.in_proj_qkv,
                "all-to-sharded",
                segments=[kd, 2 * kd],
                group=group,
            )
            shard_inplace(attn.in_proj_z, "all-to-sharded", group=group)
            shard_inplace(attn.in_proj_b, "all-to-sharded", group=group)
            shard_inplace(attn.in_proj_a, "all-to-sharded", group=group)
            attn.dt_bias = mx.contiguous(mx.split(attn.dt_bias, N)[rank])
            attn.A_log = mx.contiguous(mx.split(attn.A_log, N)[rank])
            shard_inplace(attn.out_proj, "sharded-to-all", group=group)
            attn.num_k_heads //= N
            attn.num_v_heads //= N
            attn.key_dim //= N
            attn.value_dim //= N
            attn.conv_dim //= N
            if hasattr(attn, "_qwen3_5_decode_conv_weight"):
                delattr(attn, "_qwen3_5_decode_conv_weight")
        else:
            attn = layer.self_attn
            if attn.num_attention_heads % N != 0:
                raise ValueError(
                    f"attention heads={attn.num_attention_heads} "
                    f"not divisible by tensor-parallel size {N}"
                )
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.q_proj = shard_linear(attn.q_proj, "all-to-sharded", group=group)
            repeat_kv_layer_inplace(attn.k_proj, attn.num_key_value_heads)
            repeat_kv_layer_inplace(attn.v_proj, attn.num_key_value_heads)
            attn.k_proj = shard_linear(attn.k_proj, "all-to-sharded", group=group)
            attn.v_proj = shard_linear(attn.v_proj, "all-to-sharded", group=group)
            attn.num_attention_heads //= N
            attn.num_key_value_heads = max(1, attn.num_key_value_heads // N)

        if isinstance(layer.mlp, Qwen3_5MLP):
            mlp = layer.mlp
            mlp.gate_proj = shard_linear(mlp.gate_proj, "all-to-sharded", group=group)
            mlp.down_proj = shard_linear(mlp.down_proj, "sharded-to-all", group=group)
            mlp.up_proj = shard_linear(mlp.up_proj, "all-to-sharded", group=group)
        else:
            mlp = layer.mlp
            mlp.sharding_group = group
            shard_inplace(mlp.shared_expert.gate_proj, "all-to-sharded", group=group)
            shard_inplace(mlp.shared_expert.down_proj, "sharded-to-all", group=group)
            shard_inplace(mlp.shared_expert.up_proj, "all-to-sharded", group=group)
            shard_inplace(mlp.switch_mlp.gate_proj, "all-to-sharded", group=group)
            shard_inplace(mlp.switch_mlp.down_proj, "sharded-to-all", group=group)
            shard_inplace(mlp.switch_mlp.up_proj, "all-to-sharded", group=group)

    if rank == 0:
        mod = type(self).__module__
        if "qwen4_exp" in mod:
            label = "Qwen4-Exp / Flash-Next"
            extra = "vision tower + PLE n-gram replicated/mmap"
        else:
            label = "Qwen3.5"
            extra = "vision tower replicated"
        print(
            f"mlx-vlm: sharded {label} language model across {N} ranks "
            f"({len(layers)} layers; {extra})",
            file=sys.stderr,
        )


def _patch_qwen35_tensor_parallel() -> bool:
    """Install mlx-lm's Qwen3.5 TP onto mlx-vlm's wrapper + language blocks."""
    if getattr(_patch_qwen35_tensor_parallel, "_done", False):
        return True
    try:
        from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet
        from mlx_vlm.models.qwen3_5.qwen3_5 import Model as Qwen35Model
    except Exception as exc:
        print(f"mlx-vlm: Qwen3.5 TP patch skipped: {exc}", file=sys.stderr)
        return False

    _wrap_tp_residual(Qwen3_5GatedDeltaNet)
    try:
        from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

        _wrap_tp_residual(Qwen3_5MoeSparseMoeBlock)
    except Exception as exc:
        print(f"mlx-vlm: MoE residual wrap skipped: {exc}", file=sys.stderr)

    Qwen35Model.shard = _qwen35_shard
    try:
        from mlx_vlm.models.qwen3_5_moe.qwen3_5_moe import Model as Qwen35MoeModel

        Qwen35MoeModel.shard = _qwen35_shard
    except Exception:
        pass

    _patch_qwen35_tensor_parallel._done = True
    print(
        "mlx-vlm: installed Qwen3.5 tensor-parallel shard() "
        "(language only; vision replicated)",
        file=sys.stderr,
    )
    return True


def _qwen4_shard(self, group=None):
    """Flash-Next TP: shard MoE only. Keep GDN and QSA attention replicated.

    4-bit packed GDN conv/in_proj does not split cleanly with shard_inplace,
    and QSA's indexer is not head-sharded. Either one garbles logits.
    """
    from mlx.nn.layers.distributed import shard_inplace, shard_linear
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

    group = group or mx.distributed.init()
    N = group.size()
    rank = group.rank()
    if N <= 1:
        return

    layers = self.layers
    if not layers:
        raise ValueError("Qwen4-Exp shard: language model has no layers")

    for layer in layers:
        if isinstance(getattr(layer, "mlp", None), Qwen3_5MLP):
            mlp = layer.mlp
            mlp.gate_proj = shard_linear(mlp.gate_proj, "all-to-sharded", group=group)
            mlp.down_proj = shard_linear(mlp.down_proj, "sharded-to-all", group=group)
            mlp.up_proj = shard_linear(mlp.up_proj, "all-to-sharded", group=group)
        elif getattr(layer, "mlp", None) is not None:
            mlp = layer.mlp
            mlp.sharding_group = group
            shard_inplace(mlp.shared_expert.gate_proj, "all-to-sharded", group=group)
            shard_inplace(mlp.shared_expert.down_proj, "sharded-to-all", group=group)
            shard_inplace(mlp.shared_expert.up_proj, "all-to-sharded", group=group)
            shard_inplace(mlp.switch_mlp.gate_proj, "all-to-sharded", group=group)
            shard_inplace(mlp.switch_mlp.down_proj, "sharded-to-all", group=group)
            shard_inplace(mlp.switch_mlp.up_proj, "all-to-sharded", group=group)

    if rank == 0:
        print(
            f"mlx-vlm: sharded Qwen4-Exp / Flash-Next language model across {N} ranks "
            f"({len(layers)} layers; MoE-only TP, GDN+QSA replicated; PLE n-gram mmap)",
            file=sys.stderr,
        )


def _patch_qwen4_tensor_parallel() -> bool:
    """Flash-Next (qwen4_exp): MoE-only TP; GDN + QSA replicated; PLE mmap."""
    if getattr(_patch_qwen4_tensor_parallel, "_done", False):
        return True
    if not _patch_qwen35_tensor_parallel():
        return False
    try:
        from mlx_vlm.models.qwen4_exp.qwen4_exp import Model as Qwen4Model
    except Exception as exc:
        print(f"mlx-vlm: Qwen4-Exp TP patch skipped: {exc}", file=sys.stderr)
        return False

    Qwen4Model.shard = _qwen4_shard
    try:
        from mlx_vlm.models.qwen4_exp.language import LanguageModel as Qwen4LM

        # Prefill uses Qwen4ExpModel.__call__; S=1 decode otherwise switches to
        # Qwen4ExpBatchInvariantForward. Those two paths do not share GDN / QSA /
        # PLE cache layout, so longer answers loop and then look like 乱码.
        Qwen4LM._supports_batch_invariant_decode = lambda self: False
        print(
            "mlx-vlm: Qwen4 decode uses the same forward as prefill "
            "(disabled batch-invariant split)",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"mlx-vlm: Qwen4 decode patch skipped: {exc}", file=sys.stderr)
    orig_sanitize = Qwen4Model.sanitize

    def sanitize(self, weights):
        n_layers = int(getattr(self.config.text_config, "num_hidden_layers", 0) or 0)
        missing = []
        for layer_idx in range(n_layers):
            prefix = f"model.language_model.layers.{layer_idx}.mlp"
            gate = f"{prefix}.experts.gate_up_proj"
            down = f"{prefix}.experts.down_proj"
            if gate in weights and down not in weights:
                missing.append(down)
        if missing:
            raise ValueError(
                "Qwen4-Exp / Flash-Next checkpoint is incomplete: missing "
                + ", ".join(missing[:3])
                + (f" (+{len(missing) - 3} more)" if len(missing) > 3 else "")
                + ". Wait until every model-*-of-*.safetensors shard and "
                "model.safetensors.index.json are on disk before loading."
            )
        return orig_sanitize(self, weights)

    Qwen4Model.sanitize = sanitize
    _patch_qwen4_tensor_parallel._done = True
    print(
        "mlx-vlm: installed Qwen4-Exp / Flash-Next tensor-parallel shard() "
        "(MoE-only TP; GDN+QSA replicated; PLE n-gram mmap)",
        file=sys.stderr,
    )
    return True


def _deepseek_v4_layers(model):
    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", lm)
    layers = getattr(inner, "layers", None) or getattr(lm, "layers", None)
    if not layers:
        raise ValueError("DeepSeek-V4 language model has no layers")
    return lm, layers


def _deepseek_v4_moe_only_shard(self, group=None):
    """Tensor-parallel routed/shared experts only; keep MLA replicated.

    mlx-vlm LanguageModel.shard() also splits attn.n_heads and shard_inplace()s
    quantized MultiLinear wo_a. This checkpoint's attention is 6-bit affine
    (packed codes do not split cleanly on the last axis), which garbles logits
    into 乱码 even with a correct chat template.
    """
    from mlx.nn.layers.distributed import shard_inplace

    group = group or mx.distributed.init()
    N = group.size()
    if N <= 1:
        return
    _, layers = _deepseek_v4_layers(self)
    first_ffn = getattr(layers[0], "ffn", None)
    if getattr(first_ffn, "sharding_group", None) is not None:
        return
    for layer in layers:
        ffn = getattr(layer, "ffn", None)
        if ffn is None:
            continue
        ffn.sharding_group = group
        shared = getattr(ffn, "shared_experts", None)
        if shared is not None:
            shard_inplace(shared.gate_proj, "all-to-sharded", group=group)
            shard_inplace(shared.down_proj, "sharded-to-all", group=group)
            shard_inplace(shared.up_proj, "all-to-sharded", group=group)
        switch = getattr(ffn, "switch_mlp", None)
        if switch is not None:
            shard_inplace(switch.gate_proj, "all-to-sharded", group=group)
            shard_inplace(switch.down_proj, "sharded-to-all", group=group)
            shard_inplace(switch.up_proj, "all-to-sharded", group=group)
    if group.rank() == 0:
        print(
            f"mlx-vlm: sharded DeepSeek-V4 MoE across {N} ranks "
            "(attention replicated; skip MLA 6-bit wo_a/n_heads split)",
            file=sys.stderr,
        )


def _deepseek_v4_pipeline(self, group):
    """Split layers across ranks without gaps.

    mlx-vlm PipelineMixin uses `start = (size-rank-1) * layers_per_rank`, which
    drops a layer when `n_layers % world_size != 0` (Flash is 43 layers, 4
    ranks → layer 10 never runs). Rank 0 still owns the last stage so recv/send
    order stays the same.
    """
    rank = group.rank()
    size = group.size()
    n = len(self.layers)
    counts = [n // size + (1 if i < n % size else 0) for i in range(size)]
    start = 0
    my_start, my_end = 0, n
    for i, count in enumerate(counts):
        owner = size - 1 - i
        if owner == rank:
            my_start, my_end = start, start + count
            break
        start += count
    self.pipeline_rank = rank
    self.pipeline_size = size
    self.start_idx = my_start
    self.end_idx = my_end
    self.layers = self.layers[:my_end]
    self.layers[:my_start] = [None] * my_start
    print(
        f"mlx-vlm: DeepSeek-V4 pipeline rank={rank} layers[{my_start}:{my_end}] of {n}",
        file=sys.stderr,
    )


def _patch_deepseek_v4_make_cache() -> None:
    """Keep mlx-vlm's RotatingKVCache.

    Pipeline parallelism wires `mx.depends(cache.keys, send(h))` so the send
    stays in the decode graph. KVCache often has empty `.keys` on the first
    decode step, which drops that edge and deadlocks recv/send.
    """
    return


def _patch_deepseek_v4_tensor_parallel() -> bool:
    if getattr(_patch_deepseek_v4_tensor_parallel, "_done", False):
        return True
    try:
        from mlx_vlm.models.deepseek_v4.deepseek_v4 import Model as DeepSeekV4Model
        from mlx_vlm.models.deepseek_v4.language import (
            DeepseekV4Model as DeepSeekV4Inner,
        )
        from mlx_vlm.models.deepseek_v4.language import LanguageModel
    except Exception as exc:
        print(f"mlx-vlm: DeepSeek-V4 TP patch skipped: {exc}", file=sys.stderr)
        return False

    DeepSeekV4Model.shard = _deepseek_v4_moe_only_shard
    LanguageModel.shard = _deepseek_v4_moe_only_shard
    DeepSeekV4Inner.pipeline = _deepseek_v4_pipeline

    orig_call = DeepSeekV4Inner.__call__
    if not getattr(orig_call, "_mlxctl_pipe_eval", False):
        orig_send = mx.distributed.send
        orig_recv_like = getattr(mx.distributed, "recv_like", None)

        def __call__(self, *args, **kwargs):
            if getattr(self, "pipeline_size", 1) <= 1:
                return orig_call(self, *args, **kwargs)

            def send(*a, **k):
                y = orig_send(*a, **k)
                mx.eval(y)
                return y

            def recv_like(*a, **k):
                y = orig_recv_like(*a, **k)
                mx.eval(y)
                return y

            mx.distributed.send = send
            if orig_recv_like is not None:
                mx.distributed.recv_like = recv_like
            try:
                return orig_call(self, *args, **kwargs)
            finally:
                mx.distributed.send = orig_send
                if orig_recv_like is not None:
                    mx.distributed.recv_like = orig_recv_like

        __call__._mlxctl_pipe_eval = True
        DeepSeekV4Inner.__call__ = __call__

    _patch_deepseek_v4_make_cache()
    _patch_deepseek_v4_tensor_parallel._done = True
    print(
        "mlx-vlm: installed DeepSeek-V4 pipeline() + MoE-only shard() fallback",
        file=sys.stderr,
    )
    return True


def _patch_vlm_distributed_generate() -> None:
    """mlx-vlm's server never shares work to other ranks; TP all_sum then hangs.

    Mirror mlx-lm.server: rank 0 HTTP still owns the client queue, but every
    generation thread participates in the same all_sum broadcast + forward.
    """
    if getattr(_patch_vlm_distributed_generate, "_done", False):
        return
    try:
        import pickle
        from queue import Queue

        import numpy as np
        from mlx_vlm.server.generation import QueuedGenerationRequest, ResponseGenerator
    except Exception as exc:
        print(f"mlx-vlm: distributed generate patch skipped: {exc}", file=sys.stderr)
        return

    orig_collect = ResponseGenerator._collect_pending_requests
    orig_drain = ResponseGenerator._drain_cancellations
    orig_init = ResponseGenerator._initialize_model

    def _share_object(obj):
        rank = mx.distributed.init().rank()
        if rank == 0:
            if obj is None:
                mx.eval(mx.distributed.all_sum(mx.array(0, dtype=mx.int64)))
                return None
            blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            data = mx.array(np.frombuffer(blob, dtype=np.uint8))
            mx.eval(mx.distributed.all_sum(mx.array(int(data.size), dtype=mx.int64)))
            mx.eval(mx.distributed.all_sum(data))
            return obj
        size = int(mx.distributed.all_sum(mx.array(0, dtype=mx.int64)).item())
        if size <= 0:
            return None
        if size > 256 * 1024 * 1024:
            raise RuntimeError(
                f"distributed request blob is {size} bytes; refusing to allocate"
            )
        data = mx.zeros((size,), dtype=mx.uint8)
        data = mx.distributed.all_sum(data)
        return pickle.loads(np.array(data, copy=False).tobytes())

    def _dump(obj):
        if isinstance(obj, mx.array):
            dt = str(obj.dtype)
            name = dt.split(".")[-1]
            if "int" in name or "uint" in name:
                value = np.array(obj)
            else:
                value = np.array(obj.astype(mx.float32))
            return {
                "__mx_array__": True,
                "dtype": dt,
                "value": value,
            }
        if isinstance(obj, dict):
            return {k: _dump(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_dump(v) for v in obj]
        if isinstance(obj, tuple):
            return {"__tuple__": True, "items": [_dump(v) for v in obj]}
        return obj

    def _load(obj):
        if isinstance(obj, dict) and obj.get("__mx_array__"):
            arr = mx.array(obj["value"])
            name = str(obj.get("dtype", "")).split(".")[-1]
            dt = getattr(mx, name, None)
            return arr.astype(dt) if dt is not None else arr
        if isinstance(obj, dict) and obj.get("__tuple__"):
            return tuple(_load(v) for v in obj["items"])
        if isinstance(obj, dict):
            return {k: _load(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_load(v) for v in obj]
        return obj

    def _collect_pending_requests(self, *args, **kwargs):
        group = mx.distributed.init()
        if group.size() <= 1:
            return orig_collect(self, *args, **kwargs)
        if group.rank() == 0:
            pending, should_stop = orig_collect(self, *args, **kwargs)
            serial = []
            for req in pending:
                serial.append(
                    {
                        "raw_inputs": _dump(req.raw_inputs),
                        "prompt_tokens": req.prompt_tokens,
                        "args": req.args,
                        "apc_semantic_hash": req.apc_semantic_hash,
                        "request_id": req.request_id,
                        "queued_at": req.queued_at,
                    }
                )
            _share_object({"pending": serial, "should_stop": should_stop})
            return pending, should_stop
        payload = _share_object(None)
        if not payload:
            return [], False
        restored = []
        for item in payload.get("pending") or []:
            restored.append(
                QueuedGenerationRequest(
                    rqueue=Queue(),
                    raw_inputs=_load(item["raw_inputs"]),
                    prompt_tokens=item["prompt_tokens"],
                    args=item["args"],
                    apc_semantic_hash=item.get("apc_semantic_hash"),
                    request_id=item.get("request_id"),
                    queued_at=item.get("queued_at") or 0.0,
                )
            )
        return restored, bool(payload.get("should_stop"))

    def _drain_cancellations(self):
        group = mx.distributed.init()
        if group.size() <= 1:
            return orig_drain(self)
        if group.rank() == 0:
            local = orig_drain(self)
            _share_object(list(local))
            return local
        payload = _share_object(None)
        return set(payload or [])

    def _initialize_model(self):
        orig_init(self)
        group = mx.distributed.init()
        if group.size() <= 1:
            return
        seed = mx.distributed.all_sum(mx.random.state[0]).view(mx.uint64).item()
        mx.random.seed(seed)
        if group.rank() == 0:
            print(
                f"mlx-vlm: generation broadcast enabled across {group.size()} ranks",
                file=sys.stderr,
            )

    ResponseGenerator._collect_pending_requests = _collect_pending_requests
    ResponseGenerator._drain_cancellations = _drain_cancellations
    ResponseGenerator._initialize_model = _initialize_model
    _patch_vlm_distributed_generate._done = True
    print(
        "mlx-vlm: installed distributed request broadcast (mlx-lm style)",
        file=sys.stderr,
    )


def _hw_memsize() -> int | None:
    try:
        import subprocess

        return int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        )
    except Exception:
        return None


def _cluster_size() -> int:
    """World size without calling distributed.init() (which can hang)."""
    from pathlib import Path
    import json

    for key in ("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE"):
        raw = os.environ.get(key)
        if raw and raw.isdigit() and int(raw) > 0:
            return int(raw)
    mesh = Path(__file__).resolve().parent / "jaccl-mesh.json"
    try:
        data = json.loads(mesh.read_text())
        if isinstance(data, list):
            return max(1, len(data))
        for key in ("hosts", "nodes"):
            hosts = data.get(key) if isinstance(data, dict) else None
            if isinstance(hosts, list) and hosts:
                return len(hosts)
    except Exception:
        pass
    return 1


def _assert_qwen4_fits_ram(model_path) -> None:
    """Refuse bf16 Flash-Next on 64/128GB ranks: PLE n-gram is replicated, not TP-split."""
    from pathlib import Path
    import json

    path = Path(str(model_path))
    if path.is_file():
        path = path.parent
    cfg_path = path / "config.json"
    if not cfg_path.is_file():
        return
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return
    if str(cfg.get("model_type") or "").lower() not in ("qwen4_exp", "qwen4exp"):
        return
    total = sum(p.stat().st_size for p in path.glob("*.safetensors") if p.is_file())
    ple = 0
    idx = path / "model.safetensors.index.json"
    if idx.is_file():
        try:
            wm = json.loads(idx.read_text()).get("weight_map") or {}
        except Exception:
            wm = {}
        files = {
            name
            for key, name in wm.items()
            if ".ple.ple_embedding.ngram_embedding." in key
        }
        ple = sum((path / name).stat().st_size for name in files if (path / name).is_file())
    mem = _hw_memsize()
    if mem is None or total <= 0:
        return
    # Do not call mx.distributed.init() here: a leftover JACCL socket
    # would hang the RAM check with no log output.
    n = max(1, _cluster_size())
    rest = max(0, total - ple)
    quant = cfg.get("quantization") if isinstance(cfg.get("quantization"), dict) else {}
    bits = quant.get("bits")
    name = path.name.lower() + " " + str(model_path).lower()
    fourbit = "4bit" in name or "4-bit" in name or (
        bits is not None and int(bits) <= 4
    )
    if fourbit:
        # PLE n-gram is mmap'd on 4bit; counting the full 37GB as wired
        # falsely refuses 64GB ranks (55GB "need" vs ~54GB headroom).
        need = rest // n
        _log(
            f"mlxctl: Flash-Next 4bit RAM estimate "
            f"disk={total / 1e9:.0f}GB ple_mmap={ple / 1e9:.0f}GB "
            f"sharded={need / 1e9:.0f}GB ranks={n} mem={mem / 1e9:.0f}GB"
        )
        return
    need = ple + rest // n
    headroom = int(mem * 0.78)
    if need <= headroom:
        return
    raise RuntimeError(
        "Qwen3.8-Flash-Next 放不进这台的统一内存："
        f"权重 {total / 1e9:.0f}GB（PLE n-gram {ple / 1e9:.0f}GB 每卡整份复制，"
        f"其余 {rest / 1e9:.0f}GB / {n} 卡切分）大约需要 {need / 1e9:.0f}GB，"
        f"本机只有 {mem / 1e9:.0f}GB。"
        "请改用 4bit Flash-Next（约 104GB，PLE 可 mmap），或先加载 Qwen3.8-27B-bf16。"
    )


def _assert_numbered_shards_complete(model_path) -> None:
    """Refuse HF snapshots that still have missing model-NNNNN-of-MMMMM shards."""
    from pathlib import Path

    path = Path(str(model_path))
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        return
    tmp = [
        p.name
        for p in path.iterdir()
        if p.name.startswith(".") and ".safetensors" in p.name
    ]
    found: dict[int, bool] = {}
    total = None
    for p in path.iterdir():
        name = p.name
        if not (name.startswith("model-") and name.endswith(".safetensors")):
            continue
        rest = name[len("model-") : -len(".safetensors")]
        if "-of-" not in rest:
            continue
        idx_s, total_s = rest.split("-of-", 1)
        try:
            idx, n = int(idx_s), int(total_s)
        except ValueError:
            continue
        if total is None:
            total = n
        elif n != total:
            raise ValueError(
                f"inconsistent shard totals in {path}: of-{total} vs of-{n}"
            )
        found[idx] = True
    if total is None:
        if tmp:
            raise ValueError(
                f"Hugging Face download still in progress in {path}: {tmp[0]}"
            )
        return
    have = len(found)
    if have < total or tmp:
        extra = f"; in-progress {tmp[0]}" if tmp else ""
        raise ValueError(
            f"incomplete Hugging Face snapshot in {path}: {have}/{total} shards{extra}. "
            "Wait for the download to finish (including model.safetensors.index.json) "
            "before loading."
        )


def _config_json(model_path) -> dict:
    from pathlib import Path
    import json

    path = Path(str(model_path))
    if path.is_file():
        path = path.parent
    cfg_path = path / "config.json"
    if not cfg_path.is_file():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return {}


def _model_type_of(model_path) -> str:
    cfg = _config_json(model_path)
    mt = str(cfg.get("model_type") or "").lower()
    if not mt:
        text = cfg.get("text_config")
        if isinstance(text, dict):
            mt = str(text.get("model_type") or "").lower()
    return mt


def _ensure_qwen4_ple_mmap(model_path) -> None:
    """Keep Flash-Next PLE n-gram on disk. The HF 4bit snapshot has no
    ple_storage, so mlx-vlm otherwise wires ~37GB of lookup tables on every
    64GB rank and the first prefill OOMs.
    """
    from pathlib import Path

    path = Path(str(model_path))
    if path.is_file():
        path = path.parent
    mt = _model_type_of(path).replace("-", "_")
    if mt not in ("qwen4_exp", "qwen4exp"):
        return
    cfg = _config_json(path)
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    existing = text.get("ple_storage") if isinstance(text.get("ple_storage"), dict) else {}
    manifest = Path(str(existing.get("manifest") or (path / "ple_manifest.json")))
    if not manifest.is_absolute():
        manifest = path / manifest
    if not manifest.is_file():
        from mlx_vlm.models.qwen4_exp.ple_storage import build_quantized_ple_manifest

        _log(f"mlxctl: building PLE mmap manifest at {manifest}")
        build_quantized_ple_manifest(path, manifest, cache_rows=16384)
        _log("mlxctl: PLE mmap manifest ready")
    else:
        _log(f"mlxctl: using PLE mmap manifest {manifest}")

    try:
        import mlx_vlm.utils as utils
    except Exception as exc:
        _log(f"mlxctl: could not patch load_config for PLE mmap: {exc}")
        return
    orig = getattr(utils, "load_config", None)
    if orig is None or getattr(orig, "_mlxctl_ple_mmap", False):
        return

    def load_config(model_path, *args, **kwargs):
        cfg = orig(model_path, *args, **kwargs)
        text = cfg.setdefault("text_config", {})
        if not isinstance(text, dict):
            return cfg
        if not isinstance(text.get("ple_storage"), dict):
            text["ple_storage"] = {}
        text["ple_storage"]["manifest"] = str(manifest)
        text["ple_storage"].setdefault("cache_rows", 16384)
        return cfg

    load_config._mlxctl_ple_mmap = True
    utils.load_config = load_config


def _wants_tensor_shard(model_path) -> bool:
    blob = _model_type_of(model_path).replace("-", "_")
    return any(
        key in blob
        for key in (
            "qwen3_5",
            "qwen35",
            "qwen4_exp",
            "qwen4exp",
            "deepseek_v4",
            "deepseekv4",
        )
    )


def _wants_pipeline_shard(model_path) -> bool:
    """Pipeline keeps quantized weights intact, but mlx-vlm decode deadlocks
    on send/recv (async next-token eval never pulls rank>0 send). Keep the
    helper for a later switch; DeepSeek currently uses MoE-only TP.
    """
    return False


def _vocab_is_byte_level(content: dict) -> bool:
    model = content.get("model") or {}
    vocab = model.get("vocab") or {}
    if not isinstance(vocab, dict) or len(vocab) < 100:
        return False
    keys = list(vocab.keys())[:4000]
    g_count = sum(1 for k in keys if isinstance(k, str) and k.startswith("Ġ"))
    if g_count >= 40:
        return True
    mapped = sum(
        1
        for k in keys
        if isinstance(k, str) and len(k) == 1 and 256 <= ord(k) < 356
    )
    return mapped >= 20


def _decoder_has_bytelevel(decoder) -> bool:
    if not isinstance(decoder, dict):
        return False
    if decoder.get("type") == "ByteLevel":
        return True
    return any(
        isinstance(d, dict) and d.get("type") == "ByteLevel"
        for d in (decoder.get("decoders") or [])
    )


def _patch_bytelevel_detokenizer() -> None:
    """DeepSeek-V4 tokenizer.json often looks like SPM but vocab is GPT-2 BPE.

    mlx-vlm then streams with SPMStreamingDetokenizer and Chinese comes out as
    mojibake (å¥½ / <0xE4> style 乱码) instead of UTF-8.
    """
    if getattr(_patch_bytelevel_detokenizer, "_done", False):
        return
    try:
        import json
        from pathlib import Path

        import mlx_vlm.tokenizer_utils as tu
    except Exception as exc:
        print(f"mlx-vlm: tokenizer patch skipped: {exc}", file=sys.stderr)
        return

    orig_is_bpe = tu._is_bpe_decoder
    orig_is_spm = tu._is_spm_decoder
    orig_is_spm_ns = tu._is_spm_decoder_no_space
    orig_load = tu.load_tokenizer

    def _is_bpe_decoder(decoder):
        return orig_is_bpe(decoder) or _decoder_has_bytelevel(decoder)

    def load_tokenizer(model_path, return_tokenizer=True, tokenizer_config_extra=None):
        extra = dict(tokenizer_config_extra or {})
        extra.setdefault("trust_remote_code", True)
        extra.setdefault("local_files_only", True)
        force_bpe = "deepseek" in _model_type_of(model_path)
        try:
            tok_file = Path(model_path) / "tokenizer.json"
            if tok_file.is_file():
                content = json.loads(tok_file.read_text())
                force_bpe = force_bpe or _vocab_is_byte_level(
                    content
                ) or _decoder_has_bytelevel(content.get("decoder"))
        except Exception as exc:
            print(f"mlx-vlm: tokenizer inspect skipped: {exc}", file=sys.stderr)
        if force_bpe:
            tu._is_spm_decoder = lambda _d: False
            tu._is_spm_decoder_no_space = lambda _d: False
            tu._is_bpe_decoder = lambda _d: True
            print(
                "mlx-vlm: using ByteLevel BPE detokenizer "
                f"for {model_path}",
                file=sys.stderr,
            )
        try:
            return orig_load(
                model_path,
                return_tokenizer=return_tokenizer,
                tokenizer_config_extra=extra,
            )
        finally:
            tu._is_spm_decoder = orig_is_spm
            tu._is_spm_decoder_no_space = orig_is_spm_ns
            tu._is_bpe_decoder = _is_bpe_decoder

    tu._is_bpe_decoder = _is_bpe_decoder
    tu.load_tokenizer = load_tokenizer
    try:
        import mlx_vlm.utils as utils

        utils.load_tokenizer = load_tokenizer
    except Exception:
        pass
    _patch_bytelevel_detokenizer._done = True
    print(
        "mlx-vlm: installed byte-level detokenizer detection",
        file=sys.stderr,
    )


def _force_bpe_on_loaded(loaded, model_path):
    """Replace a mis-selected SPM streamer after processor load."""
    if not isinstance(loaded, tuple) or len(loaded) < 2:
        return loaded
    processor = loaded[1]
    try:
        import mlx_vlm.tokenizer_utils as tu
    except Exception:
        return loaded
    det = getattr(processor, "detokenizer", None)
    tok = getattr(processor, "tokenizer", processor)
    if det is None:
        det = getattr(tok, "detokenizer", None)
    if isinstance(det, tu.BPEStreamingDetokenizer):
        return loaded
    mt = _model_type_of(model_path)
    force = "deepseek" in mt.replace("-", "_")
    if not force:
        try:
            from pathlib import Path
            import json

            path = Path(str(model_path))
            if path.is_file():
                path = path.parent
            tok_file = path / "tokenizer.json"
            if tok_file.is_file():
                content = json.loads(tok_file.read_text())
                force = _vocab_is_byte_level(content) or _decoder_has_bytelevel(
                    content.get("decoder")
                )
        except Exception:
            force = False
    if not force:
        return loaded
    inner = getattr(tok, "_tokenizer", tok)
    wrapped = tu.TokenizerWrapper(inner, tu.BPEStreamingDetokenizer)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not processor:
        processor.tokenizer = wrapped
    if hasattr(processor, "detokenizer"):
        try:
            processor.detokenizer = wrapped.detokenizer
        except Exception:
            pass
    print(
        "mlx-vlm: replaced processor detokenizer with ByteLevel BPE",
        file=sys.stderr,
    )
    return loaded


DEEPSEEK_V4_CHAT_TEMPLATE = (
    "{{- '<｜begin▁of▁sentence｜>' -}}"
    "{%- if messages and messages[0]['role'] == 'system' -%}"
    "{{- messages[0]['content'] -}}"
    "{%- set start = 1 -%}"
    "{%- else -%}"
    "{%- set start = 0 -%}"
    "{%- endif -%}"
    "{%- for m in messages[start:] -%}"
    "{%- if m['role'] == 'system' -%}"
    "{{- m['content'] -}}"
    "{%- elif m['role'] == 'user' -%}"
    "{{- '<｜User｜>' + m['content'] -}}"
    "{%- elif m['role'] == 'assistant' -%}"
    "{{- '<｜Assistant｜>' + (m['content'] or '') + '<｜end▁of▁sentence｜>' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<｜Assistant｜>' -}}"
    "{%- if enable_thinking -%}"
    "{{- '<think>' -}}"
    "{%- else -%}"
    "{{- '</think>' -}}"
    "{%- endif -%}"
    "{%- endif -%}"
)


def _set_chat_template(obj, template: str) -> None:
    if obj is None:
        return
    try:
        obj.chat_template = template
    except Exception:
        pass
    inner = getattr(obj, "_tokenizer", None)
    if inner is not None and inner is not obj:
        _set_chat_template(inner, template)
    tok = getattr(obj, "tokenizer", None)
    if tok is not None and tok is not obj:
        try:
            tok.chat_template = template
        except Exception:
            pass
        inner = getattr(tok, "_tokenizer", None)
        if inner is not None and inner is not tok:
            try:
                inner.chat_template = template
            except Exception:
                pass


def _ensure_deepseek_chat_template(loaded, model_path):
    """mlx-community DeepSeek-V4 has no chat_template; mlx-vlm then sends raw text.

    Official chat mode is `<｜Assistant｜></think>` so the model answers instead of
    emitting ungrounded tokens that look like 乱码.
    """
    if "deepseek_v4" not in _model_type_of(model_path).replace("-", "_"):
        return loaded
    if not isinstance(loaded, tuple) or len(loaded) < 2:
        return loaded
    processor = loaded[1]
    _set_chat_template(processor, DEEPSEEK_V4_CHAT_TEMPLATE)

    def apply(conversation, *args, tokenize=False, add_generation_prompt=True, **kwargs):
        if isinstance(conversation, dict):
            conversation = [conversation]
        text = _format_deepseek_v4_messages(
            conversation,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=bool(kwargs.get("enable_thinking")),
        )
        if tokenize:
            encode = getattr(processor, "encode", None)
            if encode is None:
                inner = getattr(processor, "_tokenizer", processor)
                encode = inner.encode
            return encode(text, add_special_tokens=False)
        return text

    try:
        processor.apply_chat_template = apply
    except Exception:
        pass
    inner = getattr(processor, "_tokenizer", None) or getattr(processor, "tokenizer", None)
    if inner is not None and inner is not processor:
        try:
            inner.apply_chat_template = apply
            inner.chat_template = DEEPSEEK_V4_CHAT_TEMPLATE
        except Exception:
            pass
    print(
        f"mlx-vlm: installed DeepSeek-V4 chat template on {type(processor).__name__}",
        file=sys.stderr,
    )
    return loaded


def _deepseek_content_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "".join(parts)
    return str(content)


def _format_deepseek_v4_messages(
    messages, add_generation_prompt: bool = True, enable_thinking: bool = False
) -> str:
    bos = "<｜begin▁of▁sentence｜>"
    user_t = "<｜User｜>"
    asst_t = "<｜Assistant｜>"
    eos = "<｜end▁of▁sentence｜>"
    parts = [bos]
    msgs = list(messages or [])
    start = 0
    if msgs and msgs[0].get("role") == "system":
        parts.append(_deepseek_content_text(msgs[0].get("content")))
        start = 1
    for msg in msgs[start:]:
        role = msg.get("role")
        text = _deepseek_content_text(msg.get("content"))
        if role == "user":
            parts.append(user_t + text)
        elif role == "assistant":
            parts.append(asst_t + text + eos)
        elif role == "system":
            parts.append(text)
    if add_generation_prompt:
        parts.append(asst_t)
        parts.append("<think>" if enable_thinking else "</think>")
    return "".join(parts)


def _config_model_type(config) -> str:
    if isinstance(config, dict):
        return str(config.get("model_type") or "").lower()
    return str(getattr(config, "model_type", "") or "").lower()


def _patch_deepseek_prompt_format() -> None:
    """Bypass mlx-vlm's missing-template fallback (raw user text → 乱码)."""
    if getattr(_patch_deepseek_prompt_format, "_done", False):
        return
    try:
        import sys

        import mlx_vlm.prompt_utils as pu
    except Exception as exc:
        print(f"mlx-vlm: DeepSeek prompt patch skipped: {exc}", file=sys.stderr)
        return

    orig_get = pu.get_chat_template
    orig_apply = pu.apply_chat_template

    def apply_chat_template(
        processor,
        config,
        prompt,
        add_generation_prompt=True,
        return_messages=False,
        num_images=0,
        num_audios=0,
        **kwargs,
    ):
        mt = _config_model_type(config).replace("-", "_")
        if "deepseek_v4" in mt and not return_messages:
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            elif isinstance(prompt, dict):
                messages = [prompt]
            elif isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": str(prompt)}]
            formatted = _format_deepseek_v4_messages(
                messages,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=bool(kwargs.get("enable_thinking")),
            )
            if mx.distributed.init().rank() == 0:
                print(
                    f"mlx-vlm: DeepSeek-V4 prompt chars={len(formatted)} "
                    f"thinking={bool(kwargs.get('enable_thinking'))}",
                    file=sys.stderr,
                )
            return formatted
        return orig_apply(
            processor,
            config,
            prompt,
            add_generation_prompt=add_generation_prompt,
            return_messages=return_messages,
            num_images=num_images,
            num_audios=num_audios,
            **kwargs,
        )

    pu.apply_chat_template = apply_chat_template
    try:
        import mlx_vlm
        import mlx_vlm.server as server_pkg

        mlx_vlm.apply_chat_template = apply_chat_template
        server_pkg.apply_chat_template = apply_chat_template
    except Exception:
        pass
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "apply_chat_template", None) is orig_apply:
                mod.apply_chat_template = apply_chat_template
        except Exception:
            pass
    _patch_deepseek_prompt_format._done = True
    print("mlx-vlm: patched DeepSeek-V4 prompt formatting", file=sys.stderr)


def _patch_vlm_sharded_load() -> None:
    try:
        import mlx_vlm.utils as utils
    except Exception:
        return

    _patch_qwen35_tensor_parallel()
    _patch_qwen4_tensor_parallel()
    _patch_deepseek_v4_tensor_parallel()
    _patch_bytelevel_detokenizer()
    plain = utils.load

    def load(model_path, *args, **kwargs):
        _log(f"mlxctl: load begin path={model_path}")
        _assert_numbered_shards_complete(model_path)
        _log("mlxctl: shard files complete, enabling PLE mmap if needed")
        _ensure_qwen4_ple_mmap(model_path)
        _log("mlxctl: checking RAM")
        _assert_qwen4_fits_ram(model_path)
        _log("mlxctl: RAM check ok, joining JACCL mesh")
        group = mx.distributed.init()
        _log(f"mlxctl: mesh joined rank={group.rank()} size={group.size()}")
        if group.size() <= 1 or not hasattr(utils, "sharded_load"):
            return _ensure_deepseek_chat_template(
                _force_bpe_on_loaded(plain(model_path, *args, **kwargs), model_path),
                model_path,
            )
        # Do not instantiate the full unsharded checkpoint just to probe
        # hasattr(model, "shard"). Flash-Next is ~104GB; that hangs 64GB ranks
        # and the UI stays on「加载中」until wait_ready times out.
        if _wants_pipeline_shard(model_path):
            print(
                "mlx-vlm: sharded_load pipeline-parallel "
                f"model_type={_model_type_of(model_path)!r} ranks={group.size()}",
                file=sys.stderr,
            )
            loaded = utils.sharded_load(
                model_path, tensor_group=None, pipeline_group=group
            )
        elif _wants_tensor_shard(model_path):
            print(
                "mlx-vlm: sharded_load tensor-parallel "
                f"model_type={_model_type_of(model_path)!r} ranks={group.size()}",
                file=sys.stderr,
            )
            loaded = utils.sharded_load(
                model_path, tensor_group=group, pipeline_group=None
            )
        else:
            print(
                "mlx-vlm: this checkpoint has no tensor/pipeline shard API; "
                "loading without sharding (needs enough RAM on rank 0)",
                file=sys.stderr,
            )
            loaded = plain(model_path, *args, **kwargs)
        return _ensure_deepseek_chat_template(
            _force_bpe_on_loaded(loaded, model_path), model_path
        )

    utils.load = load
    try:
        import mlx_vlm as pkg

        pkg.load = load
    except Exception:
        pass
    # generation.py binds `from ..utils import load` at import time, so wrap
    # utils.load first, then patch the server and rebind that name.
    _patch_vlm_distributed_generate()
    _patch_deepseek_prompt_format()
    try:
        import mlx_vlm.server.generation as gen

        gen.load = load
    except Exception:
        pass


def main() -> None:
    runtime = os.environ.get("MLXCTL_RUNTIME", "mlx_lm")
    if runtime == "mlx_vlm":
        _patch_vlm_sharded_load()
        try:
            from mlx_vlm.server.cli import main as server_main
        except ImportError:
            try:
                from mlx_vlm.server import main as server_main
            except Exception as exc:
                print(f"mlx-vlm is not available: {exc}", file=sys.stderr)
                sys.exit(2)
        server_main()
        return
    from mlx_lm.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
