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