import argparse
import time
import torch


def vector_add_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024 * 1024)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(args.n, device=device)
    y = torch.randn(args.n, device=device)

    for _ in range(10):
        _ = vector_add_torch(x, y)

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    out = vector_add_torch(x, y)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    print("device:", device)
    print("n:", args.n)
    print("elapsed_ms:", (t1 - t0) * 1000)
    print("out[:5] =", out[:5])


if __name__ == "__main__":
    main()