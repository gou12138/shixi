day2:scaled_dot_product_attention 文档的等价实现
attention 的计算链路：
QK^T * scale -> 加 attn_bias / mask -> softmax -> dropout -> @V