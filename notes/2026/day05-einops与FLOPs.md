# CS336 第 5 天：einops 与 FLOPs

建议用时：约 25 分钟阅读，留 5 分钟做文末自测。

材料来源：Stanford CS336 Spring 2026 官方讲义 [`lecture_02.py`](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py) 中的 `tensor_einops()`（含 `einsum` / `reduce` / `rearrange`）和 `tensor_operations_flops()`。昨天把“东西存在哪里、占多少字节、用什么精度”弄清楚了；今天进入 compute accounting：用带名字的维度把张量运算写清楚，并开始数 FLOPs。算术强度（arithmetic intensity）和屋顶线（roofline）明天再看——讲义在介绍完 MFU 之后正好问“为什么它不更接近 1”，那就是明天的入口。课程总入口：[cs336.stanford.edu](https://cs336.stanford.edu/)。

今天这一段官方讲义几乎没有配图，所以用讲义里的例子和文献链接来看。

---

## 先把今天放回主线

上一讲结束在张量、数值精度，以及 `C = 6ND`、每参数 12 字节那样的草稿纸估算。今天还在系统这一条线上，只是从“显存怎么记账”转到“计算量怎么记账”。

讲义提醒你带走的仍是三类知识：

- **Mechanics**：einops 怎么写，以及一次矩阵乘法有多少 FLOPs。
- **Mindset**：算计算量和算显存一样，动手前先记账。
- **Intuitions**：一次训练大概有多少 FLOPs、硬件声称能跑多快、实际又能用上多少。

## 1. 为什么要用 einops

先看传统的 PyTorch 写法：

```python
x = torch.ones(2, 2, 3)      # batch seq hidden
y = torch.ones(2, 2, 3)      # batch seq hidden
z = x @ y.transpose(-2, -1)  # batch seq seq
```

`transpose(-2, -1)` 里的 `-2`、`-1` 很容易写错。einops 的办法是给每个维度起名字。它受到爱因斯坦求和约定（[Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation)，1916）的启发，教程在 [Einops tutorial](https://einops.rocks/1-einops-basics/)。

### einsum：带记账的广义矩阵乘法

一个最简单的矩阵乘：

```python
x = torch.ones(3, 4)  # seq1 hidden
y = torch.ones(4, 3)  # hidden seq2
z = x @ y             # 旧写法
z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")
```

两边都写了 `hidden`，箭头右边却没有它：这个维度会被对上并加总，也就做成了点积。再看一个更接近注意力的、带 batch 的例子：

```python
x = torch.ones(2, 3, 4)  # batch seq1 hidden
y = torch.ones(2, 3, 4)  # batch seq2 hidden
z = x @ y.transpose(-2, -1)  # 旧写法
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
```

规则就这一句：**输出里没有写出的维度，会被加总掉。** 也可以用 `...` 表示任意多个被广播的前导维度：

```python
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
```

### reduce：按名字做归约

```python
x = torch.ones(2, 3, 4)  # batch seq hidden
y = x.sum(dim=-1)                           # 旧写法
y = reduce(x, "... hidden -> ...", "sum")   # einops
```

`sum`、`mean`、`max`、`min` 都可以这样写，不必去记 `dim=-1` 到底是哪一维。

### rearrange：把一个维度拆开，或再拼回去

有时一个维度其实装着两个维度。比如 `total_hidden` 是把 `heads * hidden1` 压扁后的结果，你只想对其中的 `hidden1` 做变换：

```python
x = torch.ones(3, 8)  # seq total_hidden，这里 total_hidden = heads * hidden1
w = torch.ones(4, 4)  # hidden1 hidden2

# 把 total_hidden 拆成 heads 和 hidden1（heads=2，所以 hidden1=4）
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)

# 用 w 把 hidden1 变到 hidden2
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")

# 再把 heads 和 hidden2 拼回去
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
```

括号表示这两个名字共享同一个物理维度。这正是多头注意力里常见的“拆头、变换、再拼回”的写法。

## 2. 开始算 FLOPs

运算写清楚以后，就可以问：这次计算要做多少次浮点运算？

一个 **FLOP**（floating-point operation）就是一次基本运算，比如加法或乘法。

两个缩写读音一样，非常容易混：

- **FLOPs**：浮点运算的次数，衡量“做了多少计算”。
- **FLOP/s**：每秒多少次浮点运算（也有人写成 FLOPS），衡量硬件有多快。

### 先建立数量级

讲义给了两个参照：

- 训练 [GPT-3](https://arxiv.org/abs/2005.14165)（2020）用了 3.14×10²³ FLOPs，讲义引用的是 [这篇估算](https://lambdalabs.com/blog/demystifying-gpt-3)。
- 训练 GPT-4（2023）被推测大约要 2×10²⁵ FLOPs，见[这篇整理](https://patmcguinness.substack.com/p/gpt-4-details-revealed)。注意这是推测，不是官方公布的数。

一张 H100 的标称峰值：带稀疏是 1979 TFLOP/s，不带稀疏大约只有一半，所以讲义和昨天一样，先写成 1979×10¹² / 2。这个峰值还严重依赖于数据类型，可以在 [H100 规格书](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306)里对照。

若按这个不带稀疏的峰值满载来算，8 张 H100 跑 2 周大约能提供 9.58×10²¹ FLOPs，大约只有 GPT-3 训练量的 3%。这还只是“卡纸面上能提供的峰值算力”，没有乘 MFU。

### 一次矩阵乘法有多少 FLOPs

看一个线性层：`y = x @ w`，其中 `x` 的形状是 (B, D)，`w` 是 (D, K)。对每一个 `(i, j, k)` 三元组，都有一次乘法 `x[i][j] * w[j][k]`，再加一次。所以

FLOPs = 2 B D K

这就是昨天说“前向里，每个参数对每个 token 大约 2 次运算”的微观来源：乘一次、加一次，计 2 个 FLOPs。

把这次运算实际跑一遍、记下时间，就能得到实际 FLOP/s = 实际 FLOPs / 实际耗时。规格书给的则是峰值（promised FLOP/s）。

### MFU

**MFU（model FLOPs utilization，模型算力利用率）= 实际 FLOP/s ÷ 标称峰值 FLOP/s**。讲义这里先不把通信和额外开销算进去。

讲义说：MFU ≥ 0.5 已经相当不错。昨天估算 70B、15T token 时取的 0.5，就是这个意思：不是随便打的折扣，而是真实训练常常只能吃到峰值的一半左右。

但为什么不能更接近 1？讲义把这个问题留到下一节：要看计算是怎么在 GPU 上发生的。那就是明天的算术强度和屋顶线。

---

## 今天可以先停在这里

如果抓住了下面几条，今天就够了：

- einops 用名字标记维度：`einsum` 做带记账的乘法式组合，`reduce` 按维归约，`rearrange` 拆开或拼回维度。输出里没写的维度会被加总。
- FLOPs 是做了多少计算，FLOP/s 是硬件每秒能做多少；两个缩写读音相同，不要混。
- 形状 (B, D) 与 (D, K) 的矩阵乘法是 2BDK 个 FLOPs。
- MFU = 实际 FLOP/s / 峰值 FLOP/s；≥ 0.5 已经很好。它为什么往往到不了 1，明天用算术强度来回答。

## 5 分钟自测

先不要往回翻，用自己的话写一写：

1. 和 `x @ y.transpose(-2, -1)` 比，einops 主要解决什么问题？在 `einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")` 里，哪个维度被加总了，为什么？
2. FLOPs 和 FLOP/s 有什么区别？形状为 (B, D) 和 (D, K) 的一次矩阵乘法，FLOPs 是多少，这个 2 是从哪来的？
3. 用自己的话写出 MFU 的定义。结合昨天 70B 那道估算题，你现在怎么理解当时取的 0.5？不必解释它为什么到不了 1，那是明天的问题。

写完发给我，我们对照讲义对一下。明天仍在同一份 `lecture_02.py`，从 `arithmetic_intensity()` 开始，看算术强度和屋顶线，回答“为什么 MFU 往往到不了 1”，以及为什么大矩阵乘法是计算受限、推理时的矩阵-向量乘却常常是访存受限。
