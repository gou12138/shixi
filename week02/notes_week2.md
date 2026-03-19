day2:scaled_dot_product_attention 文档的等价实现
attention 的计算链路：
QK^T * scale -> 加 attn_bias / mask -> softmax -> dropout -> @V


triton算子需要load内存需要连续，同时要满足对齐要求：2的幂次，要对mask位置置-inf等
读地址，计算offset，load，计算，写回内存