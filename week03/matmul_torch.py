import torch

torch.manual_seed(0)

TEST_CASES = [
    (128, 256, 64),
    (256, 512, 128),
    (512, 512, 512),
]


def print_sep(title: str):
    print("\n" + "=" * 100)
    print(title)


def small_manual_demo():
    A = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    B = torch.tensor([
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ])
    C = torch.matmul(A, B)

    print_sep("small_manual_demo")
    print("A.shape:", tuple(A.shape))
    print("B.shape:", tuple(B.shape))
    print("C.shape:", tuple(C.shape))
    print("A =\n", A)
    print("B =\n", B)
    print("C = A @ B =\n", C)

    # 手算验证
    print("check C[0, 0] = 1*7 + 2*9 + 3*11 =", 1 * 7 + 2 * 9 + 3 * 11)
    print("check C[0, 1] = 1*8 + 2*10 + 3*12 =", 1 * 8 + 2 * 10 + 3 * 12)


def run_case(m: int, k: int, n: int, device: str):
    A = torch.randn(m, k, device=device, dtype=torch.float32)
    B = torch.randn(k, n, device=device, dtype=torch.float32)
    C = torch.matmul(A, B)

    print_sep(f"case: ({m}, {k}) x ({k}, {n})")
    print("A.shape:", tuple(A.shape))
    print("B.shape:", tuple(B.shape))
    print("C.shape:", tuple(C.shape))
    print("A.dtype:", A.dtype)
    print("B.dtype:", B.dtype)
    print("C.dtype:", C.dtype)
    print("device:", A.device)

    # 只打印一小部分，避免输出太长
    preview_cols = min(5, n)
    print(f"C[0, :{preview_cols}] =", C[0, :preview_cols].detach().cpu())


def linear_equivalence_demo(device: str):
    x = torch.randn(4, 8, device=device, dtype=torch.float32)
    linear = torch.nn.Linear(8, 16, bias=True).to(device)

    y1 = linear(x)
    y2 = x @ linear.weight.T + linear.bias
    max_diff = (y1 - y2).abs().max().item()

    print_sep("linear_equivalence_demo")
    print("x.shape:", tuple(x.shape))
    print("linear.weight.shape:", tuple(linear.weight.shape))
    print("linear.bias.shape:", tuple(linear.bias.shape))
    print("output.shape:", tuple(y1.shape))
    print("max_diff between linear(x) and x @ weight.T + bias:", max_diff)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print_sep("env")
    print("device:", device)
    print("torch version:", torch.__version__)

    small_manual_demo()

    for m, k, n in TEST_CASES:
        run_case(m, k, n, device)

    linear_equivalence_demo(device)


if __name__ == "__main__":
    main()