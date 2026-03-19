import torch


def row_softmax_torch(x: torch.Tensor) -> torch.Tensor:
    """
    x: [M, N]
    row-wise stable softmax
    每一行分别做 softmax
    """
    # 1) 每行减最大值，保证数值稳定
    x_max = x.max(dim=1, keepdim=True).values
    z = x - x_max

    # 2) 求 exp
    numerator = torch.exp(z)

    # 3) 每行求和
    denominator = numerator.sum(dim=1, keepdim=True)

    # 4) 归一化
    return numerator / denominator


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 小矩阵，方便观察
    x = torch.randn(4, 8, device=device)

    out_manual = row_softmax_torch(x)
    out_torch = torch.softmax(x, dim=1)

    print("input x:")
    print(x)
    print()

    print("manual softmax:")
    print(out_manual)
    print()

    print("torch.softmax:")
    print(out_torch)
    print()

    print("allclose:", torch.allclose(out_manual, out_torch, atol=1e-6, rtol=1e-6))
    print("row sums (manual):", out_manual.sum(dim=1))
    print("row sums (torch):", out_torch.sum(dim=1))


if __name__ == "__main__":
    main()