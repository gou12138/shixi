day2:scaled_dot_product_attention 文档的等价实现
attention 的计算链路：
QK^T * scale -> 加 attn_bias / mask -> softmax -> dropout -> @V


triton算子需要load内存需要连续，同时要满足对齐要求：2的幂次，要对mask位置置-inf等
读地址，计算offset，load，计算，写回内存


先对X进行线性变换得到 Q、K、V
Q、K、V 的维度是 [batch_size, seq_len, token表示维度]
scores = Q @ K^T / sqrt(d_k) d：token表示维度
scores 的维度是 [batch_size, seq_len, seq_len]
scores 经过 mask 和 softmax 处理后得到 attention 权重矩阵，维度也是 [batch_size, seq_len, seq_len]
最后 attention 输出 = attention 权重矩阵 @ V，维度是 [batch_size, seq_len, token表示维度]