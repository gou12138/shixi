import math
import torch
import torch.nn.functional as F


def causal_attention_manual(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    q, k, v: [B, T, D]
    返回:
      out:   [B, T, D]
      attn:  [B, T, T]
      scores:[B, T, T]
    """
    d = q.size(-1)

    # 1) 原始 attention score
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)

    # 2) causal mask
    T = q.size(-2)
    allow_mask = torch.tril(
        torch.ones(T, T, dtype=torch.bool, device=q.device)
    )
    scores = scores.masked_fill(~allow_mask, float("-inf"))

    # 3) softmax -> attention weights
    attn = torch.softmax(scores, dim=-1)

    # 4) attention output
    out = attn @ v
    return out, attn, scores


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # toy input
    B, T, D = 2, 8, 16
    x = torch.randn(B, T, D, device=device)

    # toy linear projections
    Wq = torch.randn(D, D, device=device)
    Wk = torch.randn(D, D, device=device)
    Wv = torch.randn(D, D, device=device)

    q = x @ Wq
    k = x @ Wk
    v = x @ Wv

    # manual implementation
    out_manual, attn_manual, scores_manual = causal_attention_manual(q, k, v)

    # official SDPA implementation
    out_sdpa = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True
    )

    max_diff = (out_manual - out_sdpa).abs().max().item()
    ok = torch.allclose(out_manual, out_sdpa, atol=1e-5, rtol=1e-5)

    print("q shape:", q.shape)
    print("k shape:", k.shape)
    print("v shape:", v.shape)
    print("manual output shape:", out_manual.shape)
    print("sdpa output shape:", out_sdpa.shape)
    print()

    print("manual vs sdpa allclose:", ok)
    print("max diff:", max_diff)
    print()

    print("manual out[0, 0, :5]:", out_manual[0, 0, :5])
    print("sdpa   out[0, 0, :5]:", out_sdpa[0, 0, :5])


if __name__ == "__main__":
    main()