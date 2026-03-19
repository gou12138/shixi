import torch
import triton
import triton.language as tl


def next_power_of_2(x: int) -> int:
    return 1 if x == 1 else 2 ** (x - 1).bit_length()


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 当前程序实例处理第几行
    row_idx = tl.program_id(0)

    # 这行在输入矩阵中的起始位置
    row_start = input_ptr + row_idx * input_row_stride

    # 当前程序实例要访问这一行的哪些列
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start + col_offsets

    # 超过真实列数的位置要屏蔽
    mask = col_offsets < n_cols

    # 读入这一行；越界位置填 -inf，方便后面做 max / softmax
    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    # stable softmax: 先减最大值
    row_minus_max = row - tl.max(row, axis=0)

    # exp
    numerator = tl.exp(row_minus_max)

    # 分母：这一行 exp 的和
    denominator = tl.sum(numerator, axis=0)

    # softmax
    softmax_output = numerator / denominator

    # 写回输出
    output_row_start = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def row_softmax_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Triton softmax needs CUDA tensor"
    assert x.dim() == 2, "Expected 2D matrix [M, N]"

    x = x.contiguous()
    n_rows, n_cols = x.shape

    out = torch.empty_like(x)

    BLOCK_SIZE = next_power_of_2(n_cols)
    num_warps = 4 if BLOCK_SIZE <= 1024 else 8

    grid = (n_rows,)

    softmax_kernel[grid](
        out,
        x,
        x.stride(0),
        out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton softmax")

    torch.manual_seed(0)
    x = torch.randn(1024, 256, device="cuda", dtype=torch.float32)

    out_triton = row_softmax_triton(x)
    out_torch = torch.softmax(x, dim=1)

    print("input shape:", x.shape)
    print("triton output shape:", out_triton.shape)
    print("torch output shape:", out_torch.shape)
    print()

    ok = torch.allclose(out_triton, out_torch, atol=1e-5, rtol=1e-5)
    max_diff = (out_triton - out_torch).abs().max().item()

    print("allclose:", ok)
    print("max diff:", max_diff)
    print("row sums (first 5 rows):", out_triton[:5].sum(dim=1))


if __name__ == "__main__":
    main()