import torch


def build_causal_allow_mask(seq_len: int, device: str = "cpu") -> torch.Tensor:
    # True = allowed
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def main():
    seq_len = 6
    mask = build_causal_allow_mask(seq_len)

    print("causal allow mask (True means allowed):")
    print(mask.int())
    print()

    scores = torch.randn(seq_len, seq_len)
    print("raw scores:")
    print(scores)
    print()

    masked_scores = scores.masked_fill(~mask, float("-inf"))
    print("masked scores:")
    print(masked_scores)
    print()

    probs = torch.softmax(masked_scores, dim=-1)
    print("softmax after masking:")
    print(probs)
    print()

    print("row sums:")
    print(probs.sum(dim=-1))


if __name__ == "__main__":
    main()