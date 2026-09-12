#!/usr/bin/env python3
"""Entry point for distributed mlx_lm / mlx_vlm serving.

Forces JACCL, and for mlx-vlm uses sharded_load so the language model is
split across ranks (vision tower stays replicated; it is small vs the MoE).
"""
from __future__ import annotations

import os
import sys


os.environ.setdefault("MLX_DISTRIBUTED_BACKEND", "jaccl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import mlx.core as mx  # noqa: E402

_orig_init = mx.distributed.init


def _init(backend="any", *args, **kwargs):
    forced = os.environ.get("MLX_DISTRIBUTED_BACKEND")
    if forced:
        backend = forced
        kwargs.setdefault("strict", True)
    if args:
        return _orig_init(backend, *args, **kwargs)
    return _orig_init(backend=backend, **kwargs)


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


def _patch_qwen4_tensor_parallel() -> bool:
    """Flash-Next (qwen4_exp) inherits Qwen3.5 TP; PLE n-gram stays mmap/replicated."""
    if getattr(_patch_qwen4_tensor_parallel, "_done", False):
        return True
    if not _patch_qwen35_tensor_parallel():
        return False
    try:
        from mlx_vlm.models.qwen4_exp.qwen4_exp import Model as Qwen4Model
    except Exception as exc:
        print(f"mlx-vlm: Qwen4-Exp TP patch skipped: {exc}", file=sys.stderr)
        return False

    Qwen4Model.shard = _qwen35_shard
    _patch_qwen4_tensor_parallel._done = True
    print(
        "mlx-vlm: installed Qwen4-Exp / Flash-Next tensor-parallel shard() "
        "(MoE + attention; PLE n-gram replicated/mmap)",
        file=sys.stderr,
    )
    return True


def _deepseek_v4_model_shard(self, group=None):
    """mlx-vlm's wrapper has no shard(); LanguageModel.shard already exists."""
    lm = getattr(self, "language_model", None)
    if lm is None or not hasattr(lm, "shard"):
        raise ValueError("DeepSeek-V4 language_model has no shard()")
    lm.shard(group)
    group = group or mx.distributed.init()
    if group.rank() == 0:
        print(
            f"mlx-vlm: sharded DeepSeek-V4 across {group.size()} ranks "
            "(text MoE/MLA; no vision tower)",
            file=sys.stderr,
        )


def _patch_deepseek_v4_tensor_parallel() -> bool:
    if getattr(_patch_deepseek_v4_tensor_parallel, "_done", False):
        return True
    try:
        from mlx_vlm.models.deepseek_v4.deepseek_v4 import Model as DeepSeekV4Model
    except Exception as exc:
        print(f"mlx-vlm: DeepSeek-V4 TP patch skipped: {exc}", file=sys.stderr)
        return False

    DeepSeekV4Model.shard = _deepseek_v4_model_shard
    _patch_deepseek_v4_tensor_parallel._done = True
    print(
        "mlx-vlm: installed DeepSeek-V4 Model.shard() → LanguageModel.shard()",
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
                mx.eval(mx.distributed.all_sum(mx.array(0)))
                return None
            blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            data = mx.array(np.frombuffer(blob, dtype=np.uint8))
            mx.eval(mx.distributed.all_sum(mx.array(data.size)))
            mx.eval(mx.distributed.all_sum(data))
            return obj
        size = int(mx.distributed.all_sum(mx.array(0)).item())
        if size == 0:
            return None
        data = mx.zeros((size,), dtype=mx.uint8)
        data = mx.distributed.all_sum(data)
        return pickle.loads(np.array(data, copy=False).tobytes())

    def _dump(obj):
        if isinstance(obj, mx.array):
            dt = str(obj.dtype)
            return {
                "__mx_array__": True,
                "dtype": dt,
                "value": np.array(obj.astype(mx.float32)),
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


def _patch_vlm_sharded_load() -> None:
    try:
        import mlx_vlm.utils as utils
    except Exception:
        return

    _patch_qwen35_tensor_parallel()
    _patch_qwen4_tensor_parallel()
    _patch_deepseek_v4_tensor_parallel()
    plain = utils.load

    def load(model_path, *args, **kwargs):
        group = mx.distributed.init()
        if group.size() <= 1 or not hasattr(utils, "sharded_load"):
            return plain(model_path, *args, **kwargs)
        probe_load = getattr(utils, "load_model", None)
        get_path = getattr(utils, "get_model_path", None)
        tensor_group = None
        pipeline_group = None
        if probe_load is not None and get_path is not None:
            try:
                probe = probe_load(get_path(model_path), lazy=True, strict=False)
                if hasattr(probe, "shard"):
                    tensor_group = group
                else:
                    lm = getattr(probe, "language_model", None)
                    inner = getattr(lm, "model", lm) if lm is not None else None
                    if inner is not None and hasattr(inner, "pipeline"):
                        pipeline_group = group
            except Exception as exc:
                print(f"mlx-vlm shard probe failed: {exc}", file=sys.stderr)
        if tensor_group is None and pipeline_group is None:
            print(
                "mlx-vlm: this checkpoint has no tensor/pipeline shard API; "
                "loading without sharding (needs enough RAM on rank 0)",
                file=sys.stderr,
            )
            return plain(model_path, *args, **kwargs)
        return utils.sharded_load(
            model_path, tensor_group=tensor_group, pipeline_group=pipeline_group
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
