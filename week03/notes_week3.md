# Day 1 Notes

1. tokenizer 的作用是把文本变成模型可处理的 token ids，并且可以把 ids decode 回文本。
2. token 不等于字，也不一定等于单词，很多模型使用 subword tokenizer。
3. tokenizer(text, return_tensors="pt") 返回的核心结果是 input_ids 和 attention_mask。
4. input_ids 是 token 对应的整数序列。
5. attention_mask 用来标记哪些位置有效，哪些位置是 padding。
6. special tokens 可能出现在输入或输出里，decode 时可以用 skip_special_tokens=True 跳过。
7. generate 的本质是根据已有 token 继续预测下一个 token。



# Day 2 Notes

1. token 是模型真正处理的离散单位。
2. tokenizer 负责 text ↔ ids。
3. decode 不一定严格恢复原字符串格式。
4. token 数量会直接影响后面推理成本。



# Day 3 Notes

1. generate() 不是一次性直接输出整句，而是一次生成一个 token。
2. prompt 会先经过 tokenizer 变成 input_ids。
3. 模型做一次 forward，会输出 logits，shape 是 [batch_size, sequence_length, vocab_size]。
4. logits 是 softmax 之前对整个词表的打分。
5. 生成第一个新 token 时，最关键的是最后一个位置的 logits，也就是 logits[:, -1, :]。
6. 如果 do_sample=False，就用 greedy decoding，本质上就是选分数最高的 token。
7. 选出的新 token 会拼回输入，再进入下一轮生成。
8. use_cache / past_key_values 的作用，是加速后续逐 token 的 decode。