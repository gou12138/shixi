from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TEXTS = [
    "你好，今天我们学习 tokenizer。",
    "Large language models generate text token by token.",
    "今天我们 mixed 一下 English 和 中文。"
]


def print_special_tokens(tokenizer):
    print("special tokens:")
    print("  bos_token:", repr(tokenizer.bos_token), "id =", tokenizer.bos_token_id)
    print("  eos_token:", repr(tokenizer.eos_token), "id =", tokenizer.eos_token_id)
    print("  pad_token:", repr(tokenizer.pad_token), "id =", tokenizer.pad_token_id)
    print("  unk_token:", repr(tokenizer.unk_token), "id =", tokenizer.unk_token_id)


def inspect_single_text(tokenizer, text):
    enc = tokenizer(text, return_tensors="pt")
    ids = enc["input_ids"][0]
    mask = enc["attention_mask"][0]
    tokens = tokenizer.convert_ids_to_tokens(ids.tolist())


    print("\n" + "=" * 100)


    
    print("原文:", repr(text))
    print("tokens (raw):", tokens)
    print("ids:", ids.tolist()) #day2新增内容，观察 tokenizer 输出的 token ids
    print("input_ids.shape:", tuple(enc["input_ids"].shape))
    print("attention_mask.shape:", tuple(enc["attention_mask"].shape))
    print("token_count:", ids.numel())
    print("-" * 100)
    print("逐 token 观察:")
    for i, (tok, tid, m) in enumerate(zip(tokens, ids.tolist(), mask.tolist())):
        print(f"{i:02d} | id={tid:<8} | mask={m} | token={repr(tok)}")
    
    print("-" * 100)
    print("decode(skip_special_tokens=False):", repr(tokenizer.decode(ids, skip_special_tokens=False)))
    print("decode(skip_special_tokens=True): ", repr(tokenizer.decode(ids, skip_special_tokens=True)))


def inspect_batch_padding(tokenizer):
    texts = [
        "短句。",
        "这是一个更长一点的句子，用来观察 padding 和 attention_mask。"
    ]

    # 有些 decoder-only tokenizer 没有单独的 pad_token，这里做个兜底
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    batch = tokenizer(texts, padding=True, return_tensors="pt")

    print("\n" + "=" * 100)
    print("批量编码 + padding 观察")
    print("batch input_ids.shape:", tuple(batch["input_ids"].shape))
    print("batch attention_mask.shape:", tuple(batch["attention_mask"].shape))
    print("batch input_ids:")
    print(batch["input_ids"])
    print("batch attention_mask:")
    print(batch["attention_mask"])


def inspect_offset_mapping(tokenizer):
    print("\n" + "=" * 100)
    print("可选：offset mapping 观察")
    try:
        enc = tokenizer("tokenizer 很重要。", return_offsets_mapping=True)
        print("offset_mapping:", enc["offset_mapping"])
    except Exception as e:
        print("当前 tokenizer 没有直接返回 offset_mapping，跳过。")
        print("error:", e)


def inspect_chat_template(tokenizer):
    print("\n" + "=" * 100)
    print("chat template 观察")

    if getattr(tokenizer, "chat_template", None) is None:
        print("当前 tokenizer 没有 chat template，跳过。")
        return

    messages = [
        {"role": "system", "content": "You are a helpful and concise assistant."},
        {"role": "user", "content": "请用两句话解释 tokenizer 是什么。"},
    ]

    plain_user = messages[-1]["content"]
    plain_enc = tokenizer(
        plain_user,
        return_tensors="pt",
        add_special_tokens=False,
    )

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    chat_enc = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )

    ids = chat_enc["input_ids"][0]

    print("messages:", messages)
    print("-" * 100)
    print("plain_user:", repr(plain_user))
    print("plain_user_token_count:", plain_enc["input_ids"][0].numel())
    print("-" * 100)
    print("rendered_chat_text:")
    print(rendered)
    print("-" * 100)
    print("chat input_ids.shape:", tuple(chat_enc["input_ids"].shape))
    print("chat attention_mask.shape:", tuple(chat_enc["attention_mask"].shape))
    print("chat token_count:", ids.numel())
    print("extra_tokens_from_template:", ids.numel() - plain_enc["input_ids"][0].numel())
    print("chat ids:", ids.tolist())
    print("chat decode(skip_special_tokens=False):", repr(tokenizer.decode(ids, skip_special_tokens=False)))
    print("chat decode(skip_special_tokens=True): ", repr(tokenizer.decode(ids, skip_special_tokens=True)))

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    print("model_name:", MODEL_NAME)
    print("tokenizer class:", tokenizer.__class__.__name__)
    print_special_tokens(tokenizer)

    for text in TEXTS:
        inspect_single_text(tokenizer, text)

    inspect_batch_padding(tokenizer)
    inspect_offset_mapping(tokenizer)
    inspect_chat_template(tokenizer)


if __name__ == "__main__":
    main()