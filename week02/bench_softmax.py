import time
import torch

from softmax_torch import row_softmax_torch
from softmax_triton import row_softmax_triton



def bench(fn,x,repeats=100):
    # 预热
    for _ in range(10):
        fn(x)

    if x.is_cuda:
        torch.cuda.synchronize()  # 确保所有 CUDA 操作完成

    start_time = time.perf_counter()

    for _ in range(repeats):
        fn(x)

    if x.is_cuda:
        torch.cuda.synchronize()  # 确保所有 CUDA 操作完成

    end_time = time.perf_counter()

    avg_time_ms = (end_time - start_time) / repeats * 1000

    return avg_time_ms


def main():
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run this benchmark on a CUDA-enabled GPU.")
        return

    shapes = [(1024,128),(2048, 256),(4096,512)]

    for rows, cols in shapes:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

        out_torch = row_softmax_torch(x)
        out_triton = row_softmax_triton(x)

        ok = torch.allclose(out_torch, out_triton, atol=1e-5, rtol=1e-5)
        max_diff = (out_torch - out_triton).abs().max().item()
        t_torch = bench(row_softmax_torch, x)
        t_triton = bench(row_softmax_triton, x)

        print(f"shape=({rows}, {cols})")
        print(f"  correct   = {ok}")
        print(f"  max_diff  = {max_diff}")
        print(f"  torch_ms  = {t_torch:.6f}")
        print(f"  triton_ms = {t_triton:.6f}")
        print()


if __name__ == "__main__":
    main()