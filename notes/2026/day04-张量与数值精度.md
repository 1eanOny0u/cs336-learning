# CS336 第 4 天：资源估算入门，张量与数值精度

建议用时：约 25 分钟阅读，留 5 分钟做文末自测。

材料来源：Stanford CS336 Spring 2026 官方讲义 [`lecture_02.py`](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py) 中的引言、`motivating_questions()`、`tensors_basics()`、`tensors_memory()`、`tensors_on_gpus()`。第 2 讲的主题是 resource accounting（把计算和显存算清楚）；今天先把“东西存在哪里、占多少字节、用什么精度”搞清楚，einops 和 FLOPs 明天再看。课程总入口：[cs336.stanford.edu](https://cs336.stanford.edu/)。

图来自官方讲义，文献可以点开。

---

## 先把这堂课放回主线

上一讲是概览和分词。今天进入系统这一条线：给定计算和显存预算，怎样做出最好的模型，也就是最大化效率。前提是：对一次计算要用掉的资源和内存，先能算清楚。

讲义提醒你带走的仍是第 1 天那三类知识：

- **Mechanics**：主要是 PyTorch 里这些对象到底是什么意思，比较直。
- **Mindset**：记得做 resource accounting，不要算都不算就开训。
- **Intuitions**：对资源花在哪里建立感觉；今天没有什么机器学习上的“魔法”。

顺便一提：讲义开头展示了 Marin 一次 10²³ FLOPs 的训练已经跑完，而且[和事先登记的预测对上了](https://x.com/WilliamBarrHeld/status/2039373983632814318)。这正是第 2 天说的：可预期性至少和最优性同样重要。

![Marin 10²³ FLOPs 训练跑完，实测与预先登记的预测相符](https://pbs.twimg.com/media/HE1P1HmaUAAjLXF?format=jpg&name=medium)

*官方讲义引用的 Marin 结果图：灰点是预先登记的预测，彩色曲线是实际训练。今天先不必读图里的每个超参，只要记住“先预测、再开跑、最后能对上”这件事是真的发生过的。*

## 1. 先做两道草稿纸估算

讲义一上来就问了两个问题，数字不必死记，先感受这种“餐巾纸计算”的味道。

**问题 1：用 1024 张 H100，把一个 700 亿参数的模型在 15 万亿 token 上训完，大概要多久？**

训练的总计算量仍用第 2 天见过的粗算：C = 6ND。

C = 6 × 70×10⁹ × 15×10¹² = 6.3×10²⁴ FLOPs

一张 H100 的标称峰值，带稀疏是 1979 TFLOP/s，不带稀疏大约只有一半，所以讲义先写成 1979×10¹² / 2。真正训练时还跑不到峰值，再乘一个 **MFU（model FLOPs utilization，模型算力利用率）**，这里取 0.5。于是 1024 张卡一天能提供的有效算力是：

（1979×10¹² / 2）× 0.5 × 1024 × 86400 ≈ 4.38×10²² FLOPs/天

两者相除，大约 **144 天**。

**问题 2：8 张 H100，用 AdamW，最大能训多大的模型？**

每张 H100 按 80 GB 显存算。讲义给每个参数预留了 12 字节：参数 2 字节 + 梯度 2 字节 + 优化器状态 4 + 4 字节。于是

（80×10⁹ × 8）/ 12 ≈ **533 亿参数**

这只是上限：激活值还没算进去，而激活值和 batch size、序列长度有关。这 12 字节是从哪来的，看完下面的混合精度就会对上。

这些都是很粗的估算，但已经能让你在动手前先对资源有数。

## 2. 一切都先是张量

参数、梯度、优化器状态、激活，以及数据，全都存在张量里。例如 [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) 的参数，可以在 [Hugging Face 的权重索引](https://huggingface.co/deepseek-ai/DeepSeek-V3.2?show_file_info=model.safetensors.index.json) 里直接看到一块块张量。

张量有阶（rank），也就是有多少个维度：`torch.zeros(4)` 是 1 阶（向量），`torch.zeros(4, 8)` 是 2 阶（矩阵），`torch.zeros(4, 8, 2)` 是 3 阶。Transformer 里常会见到 4 阶张量，比如：

- B = 32：batch size
- S = 16：序列长度
- H = 16：头数
- D = 64：每个头的隐层宽度

对应 `torch.zeros(B, S, H, D)`。

## 3. 显存由“有多少个数”和“每个数几字节”决定

张量里的元素一般是浮点数。占用的显存 = 元素个数 × 每个元素的字节数。默认类型是 fp32，一个数 4 字节，所以 `torch.zeros(4, 8)` 占 4 × 8 × 4 = 128 字节。

这个数会涨得很快。GPT-3 前馈层里的一个矩阵，形状是 (12288 × 4, 12288)，用 fp32 存就要 **2.3 GB**。

### fp32、fp16、bf16

![fp32：1 位符号、8 位指数、23 位尾数](https://raw.githubusercontent.com/stanford-cs336/lectures/main/images/fp32.png)

*官方讲义图（[IEEE 单精度](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)）：fp32 共 32 位，其中指数占 8 位、尾数占 23 位。科学计算里它常常是默认精度；深度学习里，我们可以“马虎”得多。*

![fp16：1 位符号、5 位指数、10 位尾数](https://raw.githubusercontent.com/stanford-cs336/lectures/main/images/fp16.png)

*官方讲义图（[IEEE 半精度](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)）：fp16 把每个数砍到 2 字节，显存立刻减半。但指数只剩 5 位，能表示的动态范围变窄，尤其是很小的数。讲义里 `torch.tensor([1e-8], dtype=torch.float16)` 会下溢成 0。训练时如果发生这种事，就可能不稳定。*

![bf16：1 位符号、8 位指数、7 位尾数](https://raw.githubusercontent.com/stanford-cs336/lectures/main/images/bf16.png)

*官方讲义图（[bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)）：2018 年 Google Brain 提出的 brain floating point。它和 fp16 一样只占 2 字节，但指数位数与 fp32 相同（8 位），所以动态范围和 fp32 一样；`1e-8` 在 bf16 里不会变成 0。代价是尾数只剩 7 位，分辨率更差，但对深度学习来说，这一点通常没有动态范围重要。*

### 混合精度

于是训练时会面对这样的取舍：

- 全程 fp32 能训，但显存很贵。
- 全程 fp16 甚至全程 bf16 都有风险，可能不稳定。

办法是 **混合精度训练**（[Micikevicius et al., 2017](https://arxiv.org/pdf/1710.03740.pdf)）。讲义里的用法是：

- 参数、激活、梯度用 bf16（每个 2 字节）；
- 优化器状态留在 fp32（AdamW 常见是两份状态，各 4 字节）。

这正好就是前面那道估算题里的 **2 + 2 + (4 + 4) = 12 字节/参数**。

PyTorch 有自动混合精度库 [AMP](https://pytorch.org/docs/stable/amp.html)：在安全的时候把计算转到 bf16（例如矩阵乘法），像 `exp` 这类运算则不随便转。

### 再往下：fp8 和 fp4

2022 年，面向机器学习负载，fp8 被标准化（[论文](https://arxiv.org/pdf/2209.05433.pdf)）。H100 支持两种变体：

- E4M3：范围约 [-448, 448]
- E5M2：范围约 [-57344, 57344]

名字里的数字就是指数位和尾数位：一种更偏向分辨率，一种更偏向动态范围。讲义里引用的那张 NVIDIA 格式图，目前原链接已经打不开，所以这里只保留论文和上述范围。

2025 年，NVIDIA 又做了 [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)：每个值只占 4 位，能取的数是 -6、-4、-3、-2、-1.5、-1、-0.5、0、0.5、1、1.5、2、3、4、6。再给每个块一个单独的缩放因子，动态范围会比这 16 个数值看起来更大，只是邻近值不能完全自由变化。[Nemotron 3 Super](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf) 就是用 NVFP4 训练的。其中一些细节发生在 NVIDIA 的库里，并不完全由用户代码直接控制。

## 4. 张量默认在 CPU 上，要计算得搬到 GPU

默认情况下，`torch.zeros(32, 32)` 存在 CPU 内存里。GPU 要发挥大规模并行，必须把数据移到 GPU 显存，例如 `x = x.to(device)`，或者创建时就指定 device。

![CPU 内存与 GPU 显存经 PCI 总线相连](https://raw.githubusercontent.com/stanford-cs336/lectures/main/images/cpu-gpu.png)

*官方讲义图：左边是 CPU 和它的 RAM，右边是 GPU。GPU 里有许多 streaming multiprocessor（SM，计算单元），数据则放在下面的 DRAM（也就是前面说的 HBM 一类显存）里。两边通过 PCI 总线来回搬。这和第 2 天那张“计算和显存不在一处、中间隔着带宽”的图是同一件事。*

---

## 今天可以先停在这里

如果抓住了下面几条，今天就够了：

- 资源估算不必精确到个位，但要能很快写出 C = 6ND，以及“每张卡一天能提供多少有效 FLOPs”。
- 训练里几乎所有东西都是张量；显存由元素个数和数据类型共同决定。
- fp16 省显存，但动态范围窄，可能下溢；bf16 同样 2 字节，却保住了和 fp32 一样的动态范围，只是分辨率更粗。
- 讲义中的混合精度：参数、激活、梯度用 bf16，优化器状态留在 fp32，所以 AdamW 下每个参数大约 12 字节。用这个数估“最大能训多大模型”时，别忘了激活还没算，结果只是上限。
- 张量默认在 CPU 上，真正计算前要搬到 GPU。

## 5 分钟自测

先不要往回翻，用自己的话写一写：

1. 为什么训练时往往会选 bf16 而不是 fp16？和 fp32 比，bf16 又牺牲了什么？
2. 按讲义里的混合精度，哪些量用 bf16，哪些量留在 fp32？这和“8 张 H100 上用 AdamW 最多能训多大模型”里每个参数 12 字节，是怎么对上的？
3. 用自己的话写出“70B 模型、15T token、1024 张 H100”这道估算题的公式，并说明 6ND、峰值算力、MFU 各自是什么。不必记住精确天数。为什么“8 张 H100 的最大模型”只是一个上限？

写完发给我，我们对照讲义对一下。明天仍在同一份 `lecture_02.py`，从 `tensor_einops()` 开始，进入用 einops 把张量运算写清楚，并开始算 FLOPs。
