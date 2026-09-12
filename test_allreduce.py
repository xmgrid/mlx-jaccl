import mlx.core as mx

world = mx.distributed.init(backend="jaccl", strict=True)
x = mx.ones((8,), dtype=mx.float32)
y = mx.distributed.all_sum(x)
mx.eval(y)
print(f"rank={world.rank()} size={world.size()} sum0={float(y[0])}", flush=True)
