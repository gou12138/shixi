import torch


def describe(name, x):
    print(f"{name}:")
    print("shape:", x.shape)
    print("dtype:", x.dtype)
    print("device:", x.device)
    print(x)
    print()


def main():
    x = torch.arange(12).reshape(3,4).float()

    describe("x", x)

    y = x.t()
    describe("transpose", y)

    w = torch.randn(4,2)
    z = x @ w
    describe("matmul", z)

    if torch.cuda.is_available():
        x_cuda = x.cuda()
        describe("x_cuda", x_cuda)


if __name__ == "__main__":
    main()