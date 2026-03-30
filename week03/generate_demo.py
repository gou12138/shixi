import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 16
TOPK_TO_SHOW = 5

MESSAGES = [
    {"role": "system", "content": "You are a helpful and concise assistant."},
    {
        "role": "user",
        "content": (
            "请用三句话解释 tokenizer 和 Hugging Face 里的 model.generate() 的关系。"
            "不要把 generate() 解释成 Python 生成器。"
        ),
    },
]


def to_list(x: torch.Tensor):
    return x.detach().cpu().tolist()


def print_sep(title: str):
    print("\n" + "=" * 100)
    print(title)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print_sep("环境信息")
    print("model_name:", MODEL_NAME)
    print("device:", device)
    print("dtype:", dtype)

    # 1) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    model.to(device)
    model.eval()

    # 3) chat template
    rendered_prompt = tokenizer.apply_chat_template(
        MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 4) tokenize
    batch = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    input_token_count = batch["input_ids"].shape[1]

    print_sep("原始 messages")
    for m in MESSAGES:
        print(f"{m['role']}: {m['content']}")

    print_sep("rendered_prompt")
    print(rendered_prompt)

    print_sep("输入张量")
    print("input_ids.shape:", tuple(batch["input_ids"].shape))
    print("attention_mask.shape:", tuple(batch["attention_mask"].shape))
    print("input_token_count:", input_token_count)

    # 5) 一次 forward，观察 logits
    with torch.no_grad():
        outputs = model(**batch, use_cache=True, return_dict=True)
        logits = outputs.logits                  # [batch, seq_len, vocab_size]
        last_logits = logits[:, -1, :]          # [batch, vocab_size]
        probs = torch.softmax(last_logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=TOPK_TO_SHOW, dim=-1)
        greedy_next_id = torch.argmax(last_logits, dim=-1)

    print_sep("前向输出")
    print("logits.shape:", tuple(logits.shape))
    print("last_logits.shape:", tuple(last_logits.shape))
    print("has_past_key_values:", outputs.past_key_values is not None)

    print_sep(f"最后一个位置 top-{TOPK_TO_SHOW} 候选")
    for rank, (tid, prob) in enumerate(zip(to_list(top_ids[0]), to_list(top_probs[0])), start=1):
        token_text = tokenizer.decode([tid], skip_special_tokens=False)
        print(f"{rank:02d} | id={tid:<8} | prob={prob:.6f} | token={repr(token_text)}")

    greedy_id = to_list(greedy_next_id)[0]
    greedy_text = tokenizer.decode([greedy_id], skip_special_tokens=False)

    print_sep("greedy 选择的下一个 token")
    print("greedy_next_token_id:", greedy_id)
    print("greedy_next_token_text:", repr(greedy_text))

    # 6) 只生成 1 个 token，和 greedy 对齐
    one_step_cfg = GenerationConfig(
        max_new_tokens=1,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        one_step_ids = model.generate(**batch, generation_config=one_step_cfg)

    first_generated_ids = one_step_ids[0, input_token_count:]
    first_generated_list = to_list(first_generated_ids)

    print_sep("max_new_tokens=1 的 generate 结果")
    print("first_generated_ids:", first_generated_list)
    print(
        "first_generated_text:",
        repr(tokenizer.decode(first_generated_list, skip_special_tokens=False)),
    )

    # 7) 完整 generate
    full_cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        output_ids = model.generate(**batch, generation_config=full_cfg)

    total_output_token_count = output_ids.shape[1]
    new_token_ids = output_ids[0, input_token_count:]
    new_token_list = to_list(new_token_ids)

    prompt_text = tokenizer.decode(to_list(batch["input_ids"][0]), skip_special_tokens=True)
    new_text = tokenizer.decode(new_token_list, skip_special_tokens=True)
    full_text = tokenizer.decode(to_list(output_ids[0]), skip_special_tokens=True)

    print_sep("完整 generate 结果")
    print("total_output_token_count:", total_output_token_count)
    print("new_token_count:", len(new_token_list))
    print("new_token_ids:", new_token_list)

    print_sep("原始 prompt（decode 后）")
    print(prompt_text)

    print_sep("新生成内容")
    print(new_text)

    print_sep("完整输出（prompt + 新生成）")
    print(full_text)

    print_sep("你这次要重点看什么")
    print("1. logits.shape 应该是 [batch, seq_len, vocab_size]")
    print("2. last_logits = logits[:, -1, :] 表示“下一个 token”的打分")
    print("3. greedy_next_token_id 应该和 max_new_tokens=1 时生成的第一个 token 对齐")
    print("4. 完整输出 = 原始 prompt + 新生成内容")


if __name__ == "__main__":
    main()