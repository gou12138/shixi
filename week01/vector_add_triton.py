import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def vector_add_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Triton version requires CUDA tensors"
    assert x.shape == y.shape

    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n_elements = out.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    vector_add_kernel[grid](
        x,
        y,
        out,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return out


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton demo")

    x = torch.randn(1024 * 1024, device="cuda")
    y = torch.randn(1024 * 1024, device="cuda")
    out = vector_add_triton(x, y)
    print("out[:5] =", out[:5])


if __name__ == "__main__":
    main()