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


# Day 4 Notes - 我理解的 generate()

1. generate() 不是一次性直接输出整句，而是一次生成一个 token。
2. prompt 会先经过 tokenizer 变成 input_ids。
3. 模型做一次 forward，会输出 logits，shape 是 [batch_size, sequence_length, vocab_size]。
4. logits 是 softmax 之前对整个词表的打分。
5. 生成第一个新 token 时，最关键的是最后一个位置的 logits，也就是 logits[:, -1, :]。
6. 如果 do_sample=False，就用 greedy decoding，本质上就是选分数最高的 token。
7. 选出的新 token 会拼回输入，再进入下一轮生成。
8. output_ids 通常包含原始输入和新增生成两部分，所以要按 input_token_count 切出 new_token_ids。
9. use_cache / past_key_values 的作用，是加速后续逐 token 的 decode。


# Day 5 Notes

1. matmul 的基本规则是 [M, K] @ [K, N] -> [M, N]。
2. A 的列数必须等于 B 的行数，否则不能做矩阵乘。
3. 输出矩阵 C 的每个元素，都是 A 的一行和 B 的一列做点积得到的。
4. 今天先写 PyTorch reference，是为了给 Day 6 的 Triton matmul 提供 correctness 基准。
5. LLM 里的 linear 层，本质上是矩阵乘再加 bias。
6. 如果输入 X.shape = [B, K]，权重 W.shape = [K, N]，那么输出 Y.shape = [B, N]。
7. 在 PyTorch 的 nn.Linear 中，常见实现等价于 x @ weight.T + bias。
8. 今天的重点不是 benchmark，也不是追性能，而是把 shape 规则和 linear = matmul 彻底看懂。