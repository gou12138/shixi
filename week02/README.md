scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)

它的核心意思就是：给你 Q、K、V，外加可选 mask 和一些控制参数，算出 attention 输出。

Q:Q、K、V、mask、dropou_P、is_causal分别是什么，有什么含义





PyTorch SDPA 的核心链路：QK^T -> scale -> mask -> softmax -> @V

# 你只需要记住这个逻辑，不用背官方原文
L = query 的长度
S = key 的长度

scale = 1 / sqrt(head_dim)

bias = 全 0 的 [L, S] 矩阵

如果 is_causal=True:
    做一个下三角允许矩阵
    不允许的位置填成 -inf

如果传了 attn_mask:
    如果是 bool mask:
        不允许的位置填成 -inf
    如果是 float mask:
        直接把它加到 bias 上

scores = q @ k.T * scale
scores = scores + bias
probs = softmax(scores, dim=-1)
probs = dropout(probs)
out = probs @ v



缩放系数 1 / sqrt(d)，q @ k^T 的数值会随着维度变大而变大

如果不缩放，softmax 很容易变得太尖锐

所以通常会先除以 sqrt(d) 再送进 softmax



score=qk^T×scale+bias,这个 bias 就是 mask 或别的偏置统一作用的位置



如果 is_causal=True，就构造下三角 mask,如果 is_causal=True，它会构造一个下三角的 bool mask，然后把不允许的位置填成 -inf；而且此时要求 attn_mask 为空，也就是不能同时又给自定义 mask、又开 causal




文档说得很明确：attn_mask 支持两种类型。

bool mask：True 表示这个位置可以参与 attention

float mask：直接把这个数加到 attention score 上

如果是 bool mask，内部逻辑仍然是：把“不允许”的位置填成 -inf。如果是 float mask，就把它直接加到 attn_bias 上。




attn_weight=q@k^T×scale

然后再加上前面准备好的 attn_bias。也就是说你可以把整个过程理解成两层：

第一层：看 query 和 key 有多像
第二层：再用 mask/bias 去修正这些分数

所以 attention score 不是只有相似度，还是相似度 + 规则约束。这里的规则约束最常见的就是 causal mask。





对最后一维做 softmax,attn_weight=softmax(attn_weight,dim=−1)

dim=-1 表示沿最后一个维度归一化。在 attention 里，这通常意味着：对于每一个 query，去看“它对所有 key 的注意力分布”

softmax 不是在算“哪个位置绝对重要”，而是在算“对当前 query 来说，各个 key 的相对权重怎么分配”。PyTorch Softmax 文档也明确说，softmax 会把指定维度上的元素重标定到 [0,1]，并且这一维总和为 1。





文档最后会把注意力权重做 dropout，然后再乘 value：随机 “丢弃” 一部分权重（置为 0），再做相应缩放

output=attn_weight@𝑣

这一步的直觉是：

attn_weight 是“我该看谁、看多少”的分布

v 是“我真正要取回来的内容”

两者相乘，就是“按注意力分布加权求和后的信息聚合结果”

还有一个你很容易踩坑的点：PyTorch 文档明确警告，这个函数会严格按照 dropout_p 参数应用 dropout。所以你做手写实现和官方对齐时，一定要传 dropout_p=0.0，否则就算逻辑对了，数值也可能对不上。