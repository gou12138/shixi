import time
import torch

from vector_add_torch import vector_add_torch
from vector_add_triton import vector_add_triton


def bench(fn, x, y, repeat=100):
    for _ in range(10):
        _ = fn(x, y)

    if x.is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = fn(x, y)
    if x.is_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000 / repeat


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    sizes = [2**20, 2**22, 2**24]

    for n in sizes:
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")

        out_torch = vector_add_torch(x, y)
        out_triton = vector_add_triton(x, y)

        max_diff = (out_torch - out_triton).abs().max().item()
        ok = torch.allclose(out_torch, out_triton, atol=1e-5, rtol=1e-5)

        t_torch = bench(vector_add_torch, x, y)
        t_triton = bench(vector_add_triton, x, y)

        print(f"n={n}")
        print(f"  correct   = {ok}")
        print(f"  max_diff  = {max_diff}")
        print(f"  torch_ms  = {t_torch:.6f}")
        print(f"  triton_ms = {t_triton:.6f}")
        print()


if __name__ == "__main__":
    main()