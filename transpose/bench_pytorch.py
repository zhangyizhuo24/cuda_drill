"""
PyTorch baseline for matrix transpose.
Run after building transpose_test to compare numbers.
"""

import torch
import time

# ── config ───────────────────────────────────────────────────────────────────
M, N     = 4096, 4096
DTYPE    = torch.float32
WARMUP   = 5
ITERS    = 20
DEVICE   = "cuda"

assert torch.cuda.is_available(), "CUDA not available"

# ── helpers ──────────────────────────────────────────────────────────────────

def bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters   # ms per call


def bandwidth_GBs(ms, bytes_moved):
    return bytes_moved / 1e9 / (ms / 1e3)


# ── data ─────────────────────────────────────────────────────────────────────
A = torch.randn(M, N, dtype=DTYPE, device=DEVICE)
bytes_moved = 2 * A.numel() * A.element_size()   # read + write

# ── correctness sanity ───────────────────────────────────────────────────────
B = A.t().contiguous()
assert torch.allclose(B, A.T.contiguous()), "sanity failed"
print(f"Correctness check: PASS")

# ── benchmark variants ───────────────────────────────────────────────────────
print(f"\nMatrix shape: ({M}, {N})  dtype={DTYPE}  device={torch.cuda.get_device_name()}")
print("-" * 55)

# 1. A.t().contiguous() — the standard path
ms = bench(lambda: A.t().contiguous())
print(f"torch .t().contiguous()   {ms:7.3f} ms   {bandwidth_GBs(ms, bytes_moved):6.1f} GB/s")

# 2. torch.transpose + contiguous
ms = bench(lambda: torch.transpose(A, 0, 1).contiguous())
print(f"torch.transpose+contiguous{ms:7.3f} ms   {bandwidth_GBs(ms, bytes_moved):6.1f} GB/s")

# 3. permute — same underlying op, kept for completeness
ms = bench(lambda: A.permute(1, 0).contiguous())
print(f"A.permute(1,0).contiguous {ms:7.3f} ms   {bandwidth_GBs(ms, bytes_moved):6.1f} GB/s")

print("-" * 55)
print("Compare these numbers against the CUDA kernel output above.")
