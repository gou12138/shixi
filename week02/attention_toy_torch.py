import math
import torch


def causal_attention_toy(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    q, k, v: [B, T, D]
    """
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)

    T = q.size(-2)
    allow_mask = torch.tril(
        torch.ones(T, T, dtype=torch.bool, device=q.device)
    )
    scores = scores.masked_fill(~allow_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = attn @ v
    return out, attn, scores


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, T, D = 2, 8, 16
    x = torch.randn(B, T, D, device=device) 

    Wq = torch.randn(D, D, device=device)
    Wk = torch.randn(D, D, device=device)
    Wv = torch.randn(D, D, device=device)

    q = x @ Wq
    k = x @ Wk
    v = x @ Wv

    out, attn, scores = causal_attention_toy(q, k, v)

    print("x shape:", x.shape)
    print("q shape:", q.shape)
    print("k shape:", k.shape)
    print("v shape:", v.shape)
    print("scores shape:", scores.shape)
    print("attn shape:", attn.shape)
    print("out shape:", out.shape)
    print()

    print("attention weights for batch 0:")
    print(attn[0])
    print()

    print("row sums of batch 0:")
    print(attn[0].sum(dim=-1))


if __name__ == "__main__":
    main()