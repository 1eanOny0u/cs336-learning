# CS336 第 2 天：五条主线，以及贯穿全课的效率视角

建议用时：约 25 分钟阅读，留 5 分钟做文末自测。

材料来源：Stanford CS336 Spring 2026 官方讲义 [`lecture_01.py`](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py) 中的 `course_syllabus()`，以及它依次展开的 `basics()`、`systems()`、`scaling_laws()`、`data()`、`alignment()`。这仍是第 1 讲的后半段大纲，还没进入分词细节。课程总入口：[cs336.stanford.edu](https://cs336.stanford.edu/)。

图均来自官方讲义（仓库配图，或讲义里引用的原图）；文献链接也按讲义中的引用给出，可以直接点开。

---

## 先抓住一条主线

整门课可以看成五块，也对应五份作业：basics、systems、scaling laws、data、alignment。Percy 在大纲结尾反复强调：记住，这一切都是在谈 **efficiency（效率）**。

你拥有的资源是：数据，再加上硬件里的计算、内存和通信带宽。问题是：在资源固定时，怎样训练出最好的模型？

眼下我们主要受计算约束，所以许多设计都是为了把给定硬件挤干：

- **Systems**：最直接就是在谈效率。
- **Tokenization**：直接在原始字节上建模很优雅，但以今天的模型结构来看，计算上并不划算。
- **Model architecture**：很多改动是为了降低显存或 FLOPs，例如共享 KV cache、sliding window attention。
- **Data filtering**：不要把宝贵的计算浪费在糟糕或无关的数据上。
- **Scaling laws**：用较小模型、较少计算先做超参选择，再放大。

讲义也提了一句：将来我们可能会变成数据约束，那时具体取舍会变，但“先想效率”这个心态还在。下面按五块把大纲过一遍，先建立地图，不必一次记住所有论文名。

## 1. Basics：先能从零训练出一个语言模型

目标：能够训练一个基本的语言模型。零件是三块：分词、模型架构、训练。

**分词**问的是：模型操作的“原子”是什么？形式上，tokenizer 在原始输入（字节）和整数 token 序列之间转换。常用的是 **Byte-Pair Encoding（BPE）**（[Sennrich et al., 2016](https://arxiv.org/abs/1508.07909)），直觉是把输入切成经常一起出现的块。从效率看，它做了两件事：缩短上下文（讲义里的量级是 1000 字节大约变成 250 个 token），以及自适应计算（更稀有、更有信息的部分保留成更多 token，常见部分压成更少 token）。终极梦想是不要 tokenizer、直接在字节上建模，相关工作有 [ByT5](https://arxiv.org/abs/2105.13626)、[MegaByte](https://arxiv.org/pdf/2305.07185.pdf)、[BLT](https://arxiv.org/abs/2412.09871)、[T-Free](https://arxiv.org/abs/2406.19223)、[H-Net](https://arxiv.org/abs/2507.07955) 等，但还没有做到前沿规模，所以这门课仍然要讲分词。细节我们留到第 3 天。

![Tokenizer 把句子编码成 token 序列，再解码回句子](https://raw.githubusercontent.com/stanford-cs336/lectures/main/images/tokenized-example.png)

*官方讲义图：tokenize 就是在字符串和整数 token 之间做 encode / decode。图中 “Stanford was founded in 1885.” 被切成带颜色的若干块，并对应到一串 ID。*

**架构**的起点是 2017 年的 Transformer（[Vaswani et al., 2017](https://arxiv.org/pdf/1706.03762.pdf)），之后有一长串 refinement：激活（ReLU、[SwiGLU](https://arxiv.org/pdf/2002.05202.pdf)）、位置编码（正弦、[RoPE](https://arxiv.org/pdf/2104.09864.pdf)）、归一化（LayerNorm、RMSNorm、QK norm，以及 pre-norm 和 post-norm）、注意力（全注意力、稀疏/局部注意力、[GQA](https://arxiv.org/pdf/2305.13245.pdf)、[MLA（DeepSeek-V2）](https://arxiv.org/abs/2405.04434)）、状态空间/线性注意力（[Mamba-2](https://arxiv.org/abs/2405.21060)、[Gated DeltaNet](https://arxiv.org/abs/2412.06464)）、MLP（稠密或 [mixture of experts](https://arxiv.org/pdf/1701.06538.pdf)、[Switch Transformers](https://arxiv.org/abs/2101.03961)），还有模型形状（隐层宽度、深度、头数、专家数）。这些名字现在先认个脸，第 3、4 讲会展开。

![Transformer 整体结构与单个 Block 内部](https://raw.githubusercontent.com/stanford-cs336/lectures/main/images/transformer-architecture.png)

*官方讲义图：左边是模型全貌，右边是一个 Transformer Block。输入输出张量形状都是 (batch_size, seq_len, d_model)。注意这里画出的是 pre-norm 风格（先 Norm 再做 Attention / FFN），这和 2017 年论文里最初的 post-norm 已经有所不同，后面讲架构时会回到这一点。*

**训练**则是如何把参数定下来：损失函数（例如 [multi-token prediction](https://arxiv.org/abs/2404.19737)）、优化器（[Adam](https://arxiv.org/pdf/1412.6980.pdf)、[AdamW](https://arxiv.org/pdf/1711.05101.pdf)、[SOAP](https://arxiv.org/abs/2409.11321)、Muon）、初始化（[Xavier](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)、[muP](https://arxiv.org/abs/2203.03466)）、学习率日程（[cosine](https://arxiv.org/pdf/1608.03983.pdf)、[WSD（MiniCPM）](https://arxiv.org/pdf/2404.06395.pdf)）、正则（dropout、weight decay）、batch size，以及 MoE 特有的负载均衡。看上去像一堆超参，但讲义的重点是：这些选择会直接决定一次训练是稳定跑完，还是直接炸掉。

贯穿这三块的平衡是：**表达能力**（能不能刻画数据里的复杂依赖）、**稳定性**（参数和梯度范数要落在不太爆、也不消失的区间）、**效率**（训练和推理都要在硬件上跑得快）。

（对应 Assignment 1：实现 BPE、Transformer、交叉熵、AdamW 和训练循环，做资源估算，并在 TinyStories、OpenWebText 上训练。我们这个月先不做作业。）

## 2. Systems：把硬件挤干

目标：从 GPU 或 TPU 里挤出尽量多的性能。零件是：kernels、并行，以及推理。

先要会做 **resource accounting**：算清模型的计算量和显存开销。一个会反复出现的粗算是：训练 700 亿参数、用 1 万亿 token，大约是 6 × 70×10⁹ × 10¹² = 4.2×10²³ FLOPs。还要记住：参数必须先从显存（HBM）搬到计算单元（SM）上才能算。以 B200 为例，bf16 算力约 2.25 PFLOP/s，显存带宽约 8 TB/s。Roofline 分析用来判断一次计算是算力受限还是内存受限；再用 profiling 看实际发生了什么。

![计算单元与显存分离、中间靠带宽相连](https://raw.githubusercontent.com/stanford-cs336/lectures/main/images/compute-memory.png)

*官方讲义图：Compute 和 Memory 不在同一个地方，中间那条“细管子”就是带宽。很多加速，本质上是在减少来回搬运。*

![NVIDIA DGX B200 系统拓扑：8 张 GPU、CPU、NVSwitch 与网络](https://docs.nvidia.com/dgx/dgxb200-user-guide/_images/dgx-b200-system-topology.png)

*官方讲义引用的 [DGX B200](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html) 拓扑图：两侧各 4 张 GPU，经 NVSwitch 互连，再经由 CPU、网卡和存储对外通信。后面讲并行时，会反复回到“数据要经过哪几条链路”。*

**Kernel** 是跑在 GPU 上的函数。普通 PyTorch 运算已经在调标准 kernel；要更快，往往得自己写，原则是：**尽量减少数据搬运**。朴素做法是读显存、算 A、写回，再读、算 B、再写回；融合（fusion）则是读一次，把 A 和 B 一起算完再写回。同类思路还有 tiling（如 FlashAttention）。课上会接触 Triton 等工具。

如果有很多 GPU（讲义里举的是 1024 张），GPU 之间搬数据更慢，但“减少搬运”的原则不变。这时要用 gather、reduce、all-reduce 这类集合通信，并把参数、激活、梯度、优化器状态切分到不同 GPU 上。切分方式包括 data、tensor、pipeline、sequence、expert 并行。

**推理**是根据 prompt 生成 token，不只是聊天要用，强化学习、测试时计算和评测也要用。它分两段：prefill 和 decode。Prefill 时 token 都已给定，可以一次处理，通常偏算力受限；decode 必须一个 token 一个 token 生成，通常偏内存受限。加速办法包括用更便宜的模型（剪枝、量化、蒸馏）、推测解码（先用小模型起草，再用大模型并行校验，结果仍是精确解码），以及融合 kernel、连续批处理等系统优化。

![推理的 Prefill 与 Decoding 两阶段，以及 KV Cache](https://raw.githubusercontent.com/stanford-cs336/lectures/main/images/prefill-decode.png)

*官方讲义图：左边 Prefill 把 prompt（“Computer science is”）一次性写入 KV-Cache；右边 Decoding 再逐个生成 “discipline”、“.”、“&lt;EOS&gt;”。加速 decode 的许多方法，都是在应对“一次只出 1 个 token、却要反复读缓存”这个瓶颈。*

（对应 Assignment 2：用 Triton 写融合 RMSNorm、实现分布式数据并行和优化器状态切分，并做基准测试。推荐阅读：[How to Scale Your Model](https://jax-ml.github.io/scaling-book/)。）

## 3. Scaling laws：还没在目标规模上训练，就要能做决定

设定是：假如你有 10²⁵ FLOPs，该用怎样的超参去训练一个好模型？在那个规模上做超参搜索太贵了。

关键转变是：不要只想“某一个规模上的一个模型”，而要想到一条 **scaling recipe（缩放配方）**，也就是从 FLOPs 映射到超参。做法是：先在较小规模上跑实验、记下 loss，再拟合一条缩放定律，用来预测目标规模上的 loss。这样既可以在小规模上为大规模选配方，也可以在真跑之前先做出预测。

缩放定律不是自然定律，不会自动出现，必须把配方设计好。常见目标是 **hyperparameter transfer**：小规模上好用的超参，到更大规模仍然适用，或可以按可预期的方式调整（例如 [muP](https://arxiv.org/abs/2203.03466)）。讲义特别强调：**可预期性至少和最优性同样重要**。如果每个规模的学习率都在乱跳，你到了大尺度就会猜不准。

一个经典问题是：给定 FLOPs 预算（粗略有 C = 6ND），该把模型做大（增大 N），还是在更多 token 上训练（增大 D）？[Kaplan et al., 2020](https://arxiv.org/pdf/2001.08361.pdf) 和 [Hoffmann et al., 2022（Chinchilla）](https://arxiv.org/pdf/2203.15556.pdf) 的计算最优缩放定律给出过经验规则：大约 D = 20N。例如 700 亿参数的模型，大约应训练 1.4 万亿 token。但要注意：这还没计入推理成本——现在很多模型会训得比“计算最优”更久、同时保持更小，因为推理时更想要小模型。

![Chinchilla ISOFLOP 曲线，以及最优参数量和 token 数随 FLOPs 的变化](https://raw.githubusercontent.com/stanford-cs336/lectures/main/images/chinchilla-isoflop.png)

*官方讲义中的 Chinchilla 图。左图是 ISOFLOP 曲线：同一计算预算下，横轴是模型大小，纵轴是训练 loss，最低点就是该预算下较好的 N。中图、右图再把这些最优点连起来外推：参数量和 token 数如何随 FLOPs 增长。右图标出的 1.4T tokens，对应讲义里 “70B 大约配 1.4T tokens” 的经验规则。*

![Marin Delphi Scaling Suite：小规模拟合后对外推点做预先登记](https://pbs.twimg.com/media/HDuErvvbsAAQ5Yt?format=jpg&name=4096x4096)

*官方讲义引用的 Marin 图（[Percy Liang 的说明](https://x.com/percyliang/status/2034367256277533100)）。彩色曲线是不同 FLOPs 预算下的实验结果，灰色虚线是计算最优前沿，叉号是外推预测。图中较大的点还标了实测与预测的 Δloss；最右侧写着 “Currently training!”，正是“先预测、再开跑”的例子。*

（对应 Assignment 3：通过一个训练 API 在计算预算内取点、拟合缩放定律并外推。我们同样先只掌握思路。）

## 4. Data：模型能做什么，很大程度上由数据决定

先问：你希望模型具备哪些能力？多语言？会对话？能做智能体式的编程？这些都会回到数据上。

**评测**有两种目的，不要混在一起。内部评测用来指导研发，关键是跨尺度要平滑、相对比较要可靠；外部评测用来衡量真实用途上的绝对质量，更看重生态效度。例子包括困惑度（最好用未公开、不在互联网上的文档，以减少污染），以及更接近真实用途的基准，如 GPQA、HLE、SWE-Bench、Terminal-Bench。语言模型号称通用，因此需要一组多样化的评测，而不是只看一个平均数。

**数据不会从天上掉下来**，必须主动收集：网页抓取、书籍、arXiv 论文、GitHub 代码等。这里还有版权和许可问题（讲义提到了对 [fair use 的讨论](https://arxiv.org/pdf/2303.15715.pdf)，以及可能需要授权）。而且原始数据往往是 HTML、PDF 或目录，不是现成的纯文本，必须处理。

![The Pile 数据集按类别构成的矩形树图](https://ar5iv.labs.arxiv.org/html/2101.00027/assets/pile_chart2.png)

*官方讲义引用的 The Pile 构成图（来自 [Gao et al., 2020](https://arxiv.org/pdf/2101.00027.pdf)）。可以看到学术（PubMed、arXiv、法律等）、互联网（Common Crawl、OpenWebText2、Wikipedia、StackExchange）、书籍、对话和代码等被按比例拼在一起。今天的数据配比已经不同，但“多源、要混合、比例是设计出来的”这一点没有变。*

处理通常包括：把 HTML/PDF 转成文本并抽出正文；过滤，留下高质量、去掉有害内容；去重（节省计算、减少死记硬背，常用 Bloom filter 或 MinHash）；混合不同来源的配比（[RegMix](https://arxiv.org/abs/2407.01492)、[OLMix](https://arxiv.org/abs/2602.12237)）；以及用语言模型改写或生成合成数据，使数据更接近下游任务（[WRAP](https://arxiv.org/abs/2401.16380)）。数据也可以分成预训练数据（量大、多样）、中期训练数据（更高质量，常包括长上下文），以及后训练数据（对话、带工具调用的智能体轨迹）。

（对应 Assignment 4：从 Common Crawl 出发做清洗、过滤和 MinHash 去重。）

## 5. Alignment：在已有合理模型上，用弱监督再推进一步

到这里，模型已经在“预测下一个 token”这种全监督下训练过，本身应当已经像样。下一步可以用 **弱监督** 再改进。为什么是弱监督？因为有时批评一个回答，比直接写出正确答案更容易。

基本模板是三步：先让模型生成回答；再用人类、验证器或语言模型裁判来打分；然后更新模型，让它更偏好更好的回答。算法上，常见的有强化学习里的 [PPO](https://arxiv.org/pdf/1707.06347.pdf)（以及 [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf)）、针对偏好数据、形式更简单的 [DPO](https://arxiv.org/pdf/2305.18290.pdf)，以及去掉价值函数的 [GRPO（DeepSeekMath）](https://arxiv.org/pdf/2402.03300.pdf)。

难点在于：强化学习算法不稳定、难调；做到规模后需要大量新基建（例如带异步 rollout 的推理）；而且还要在系统效率和 on-policy 程度之间不断权衡。

（对应 Assignment 5：实现 DPO 和 GRPO。我们这个月同样只先把握这条链路。）

---

## 今天可以先停在这里

如果抓住了下面几条，今天就够了：

- 17 讲正课可以先记成一张地图：分词与架构和训练 → 系统与推理 → 缩放定律 → 数据与评测 → 对齐。
- 贯穿全课的问题是：资源固定时，怎样把模型做好，也就是最大化效率。
- 当前主要是计算约束，所以分词、架构、过滤数据、缩放定律，都可以从“别浪费计算”来理解。
- Scaling 的关键不是只在大模型上死调，而是先有一条可外推的配方，可预期性非常重要。
- 对齐把“会续写”推进到“更偏好更好的回答”，监督信号变弱了，系统难度也上去了。

## 5 分钟自测

先不要往回翻，用自己的话写一写：

1. 用一张短名单画出这门课的五条主线，并给每条用一句话说明它在解决什么问题。
2. 为什么说分词、架构改动和数据过滤，都可以放到“效率”这同一个镜头下看？
3. 什么是 scaling recipe？为什么讲义认为可预期性至少和最优性同样重要？可以把上面那张 Chinchilla ISOFLOP 图也一起想进去。

写完发给我，我们对照讲义对一下。明天仍在同一份 `lecture_01.py`，从 `tokenization()` 开始，进入第 1 讲的技术部分：字符、字节、词，以及 BPE。
