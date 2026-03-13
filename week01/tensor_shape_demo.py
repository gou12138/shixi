import torch


def describe(name, x):
    print(f"{name}:")
    print("shape:", x.shape)
    print("dtype:", x.dtype)
    print("device:", x.device)
    print(x)
    print()


def main():
    a = torch.randn(2, 3, 4)
    describe("a", a)
    describe("a.reshape(6, 4)", a.reshape(6, 4))
    describe("a.transpose(1, 2)", a.transpose(1, 2))
    describe("a.unsqueeze(0)", a.unsqueeze(0))
    describe("a.squeeze()", a.unsqueeze(0).squeeze(0))
    describe("a.sum(dim=0)", a.sum(dim=0))
    describe("a.sum(dim=-1)", a.sum(dim=-1))


if __name__ == "__main__":
    main()