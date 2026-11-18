# CS336 第 1 天：课程定位与后勤

建议用时：约 25 分钟阅读，留 5 分钟做文末自测。

材料来源：Stanford CS336 Spring 2026 官方讲义 [`lecture_01.py`](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py) 中的 `welcome()`、`why_this_course_exists()`、`current_lm_landscape()`、`what_is_this_program()`、`course_logistics()`。今天读到课程后勤为止，五大部分大纲和分词（tokenization）明天再看。课程总入口：[cs336.stanford.edu](https://cs336.stanford.edu/)。

---

## 1. 这门课是什么

CS336: Language Models From Scratch（从零做语言模型），2026 年春季是第 3 次开课。讲师是 Tatsunori Hashimoto 和 Percy Liang。

没变的是 from scratch 的哲学：靠自己把关键部分搭起来，才能真正理解它怎么工作。变的是更强调单位时间里的高价值概念，不要只见树木不见森林；也会多讲一些更现代的成分，比如 mixture of experts、长上下文，以及 agents。

2025 年春季的录像仍在 [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)。我们这轮用的是 2026 年春季讲义。

## 2. 为什么要开这门课

讲义里指出的问题是：研究者正在和底层技术脱节。

- 2016 年左右：研究者自己实现并训练模型。
- 2018 年左右：下载已有模型（例如 BERT）再微调。
- 今天：直接去 prompt API 模型（例如 GPT / Claude / Gemini）。

抽象层级上升会提高生产力，但这些抽象是漏的（leaky），不像编程语言或操作系统那样可靠。如果还想做需要掀开整个技术栈的基础研究，只停留在调 API 会把设计空间收得很窄。

这门课的哲学是：**understanding via building（靠搭建来理解）**。

但有一个麻烦：语言模型已经工业化了。前沿模型非常贵——讲义里写到，2023 年 GPT-4 据称训练成本约 1 亿美元；2025 年 xAI 为训练 Grok 搭建了约 23 万 GPU 的集群。前沿模型怎么做出来的，公开细节也几乎没有。GPT-4 技术报告就明确表示，出于竞争和安全考虑，不会分享模型是怎么建成的。

这些前沿模型我们够不到。也可以做小模型（小于 10 亿参数），但小模型不一定代表大模型。讲义给了两个例子：

1. Attention 和 MLP 各自占多少 FLOPs，会随规模变化。小规模时花在 attention 上的比例，放大以后可能完全不同。
2. 有些能力会随规模“涌现”（emergence）。小规模上看似没用的现象，到临界规模才突然出现。

### 三类知识：哪些能迁移？

那这门课还能学到什么、并且能迁移到前沿模型上？讲义把知识分成三类：

- **Mechanics（机理）**：东西是怎么工作的，例如 Transformer 是什么、模型并行是怎么做的。
- **Mindset（心态/方法）**：怎么把硬件挤到极限、怎么认真对待 scaling。
- **Intuitions（直觉）**：哪些数据和建模决策会带来好效果。

Mechanics 和 mindset 可以教，而且**能够迁移**。Intuitions 只能部分教，因为它们**不一定能跨尺度迁移**。很多设计决策目前还无法从原理上说清，只能靠实验。一个著名例子是 Noam Shazeer 提出 SwiGLU 的论文：作者在结论里几乎是坦白地说，给不出解释，只能把成功归为 “divine benevolence”（上天慈悲）。

### 正确理解 bitter lesson

常见的错误理解是：规模是唯一重要的，算法不重要。

正确理解是：**能规模化的算法才重要**。可以写成：

**accuracy = efficiency × resources**

在更大的规模上，效率其实更重要，因为浪费不起。OpenAI 2020 年的一篇论文指出，2012 到 2019 年间，ImageNet 上的算法效率提升了约 44 倍。

这门课的总提问是：给定一定的计算和数据预算，能做出的最好模型是什么？换句话说：**maximize efficiency（最大化效率）**。

## 3. 语言模型走到了哪一步

可以按时间把图景分成几段（不必一次记住所有论文名，先抓住主线）：

**神经网络出现之前（2010 年代以前）**

- Shannon（1950）用语言模型衡量英语的熵。
- N-gram 语言模型，曾用在机器翻译和语音识别里。

**神经网络时期的关键零件（2010 年代）**

- LSTM、Bengio 的早期神经语言模型、seq2seq、Adam、attention、Transformer、mixture of experts，以及模型并行（GPipe、ZeRO、Megatron-LM）等。注意：Transformer 最初也是为机器翻译提出的。

**早期基础模型（2010 年代末）**

- ELMo、BERT：先预训练，再微调，下游任务会明显变好。那时语言模型被理解成“拿来微调的东西”。
- Google 的 T5（110 亿参数）：把各种任务都写成 text-to-text。

**开始认真做 scaling**

- GPT-2（15 亿）：文本更流畅，出现零样本（zero-shot）的苗头。
- Scaling laws：给放大规模带来可预期性。
- GPT-3（1750 亿）：出现 in-context learning。
- PaLM（5400 亿）：规模很大，但训练不充分。
- Chinchilla（700 亿）：提出计算最优的 scaling laws。

**开放模型**

早期是在复现 GPT-3：EleutherAI 的 The Pile / GPT-J，Meta 的 OPT，BigScience 的 BLOOM。这些尝试很重要，但还不够强。

近三年出现了更可信的 **open-weight** 模型（权重大多配有论文），例如 Llama、Mistral / Mixtral、DeepSeek、Qwen、Kimi、GLM 等，已经在接近闭源前沿模型。还有一类更开放的 **open-source** 模型，会同时公开权重、论文、代码和数据，例如 AI2 的 OLMo、NVIDIA 的 Nemotron，以及 Marin。讲义强调：开放很重要，因为正是这些公开工作，才让 CS336 教得下去。

**“语言模型”这个词的含义也在变**

- 2018（BERT）：拿来微调的东西
- 2020（GPT-3）：拿来 prompt 的东西
- 2022（ChatGPT）：可以对话的东西
- 2026（agents）：可以自主行动的东西

基本零件大体还在（attention、kernels、优化方法），但规格变了：上下文更长，推理效率变得更关键。

## 4. 这份讲义本身是什么

第 1 讲的讲义是一份 **executable lecture（可执行讲义）**：它本身就是一个 Python 程序，运行起来就是这堂课的内容。这样做的好处是可以直接看见并运行代码，也能看清讲座的层次结构。有的讲次是这种 `.py`，有的是 `.pdf`。我们接下来都会按官方讲义来切。

## 5. 课程怎么安排（后勤）

这是一门 **5 学分** 的课。2024 年春季有学生在评教里写过：第一份作业的工作量，大约相当于 CS224N 全部 5 个作业再加期末项目。讲义也提醒：这个说法或许夸张，但时间预算宁可估得保守一些。

**适合来学的人**

- 有一种非搞清楚不可的好奇心，想理解事情到底怎么工作。
- 想把研究工程（research engineering）的肌肉练起来。

**不适合把它当第一选择的人**

- 这个学期其实更想把研究做出来（先和导师谈清楚）。
- 主要想追最新热点技巧（多模态、RAG 等更适合去研讨课；这门课也不深讲 agents）。
- 只想在自己的应用上尽快做出效果（更应该先 prompt 或微调已有模型；自己预训练通常又贵又痛苦，是最后手段）。

**作业形态**

一共 5 份作业，对应 basics、systems、scaling laws、data、alignment。不提供脚手架代码，但会给单元测试和接口，方便检查正确性。建议先在本地实现并验证正确性，再到集群上做准确率或速度方面的评测。部分作业有排行榜（例如在给定训练预算下尽量降低困惑度）。

对自学来说，更现实的目标是先把 17 讲主线理清；官方作业非常重，一个月每天约 30 分钟通常做不完。

**关于用 AI**

编码代理现在已经强到可以直接把作业做出来，但那样学不到东西。AI 很适合用来提问和辅导。课程要求：如果要用 AI，必须配合他们提供的 `AGENTS.md`（让 AI 按教学方式回答，而不是直接把 Transformer 写出来）。政策说明在课程的 [AI policy guide](https://docs.google.com/document/d/1SZAlExB1qAc9izHt54gwunNpjKE6wXb8Y7yA_e-baK8/edit?tab=t.0)。

**计算资源**

课内计算由 Modal 赞助。课程写了一份[使用指南](https://docs.google.com/document/d/1cHE0iKVyXLJ3XpIs2XuXTmZ-HMmPk2hIPeCvy-AydMg/edit?tab=t.otis27tacaef)。自学如果要做训练或评测，也可以自己找云 GPU；这个月我们先以讲义为主，不必现在准备算力。

---

## 今天可以先停在这里

你如果抓住了下面几条，今天这 25 分钟就够了：

- 这门课要解决的是“只会用抽象、却掀不开技术栈”的问题，办法是靠搭建来理解。
- 能稳定带走的是 mechanics 和 mindset；关于“什么技巧效果好”的直觉，不一定能从小管到大。
- bitter lesson 不是“只靠堆规模”，而是“要找能规模化的算法”；准确度来自效率乘以资源。
- 语言模型的使用方式在变（微调 → prompt → 对话 → agent），但底层主线还在。
- 正式课很重；我们这个月先把主线走完。

## 5 分钟自测

先不要往回翻，用自己的话写一写：

1. 为什么只会被 API、不够做基础研究？这门课打算用什么办法让你真正理解？
2. mechanics、mindset、intuitions 分别指什么？哪些更可能迁移到更大的模型上，为什么？
3. 用一两句话写出：你怎么理解 bitter lesson？为什么在更大规模上，效率反而更重要？

写完发给我，我们对照讲义对一下。明天继续同一份 `lecture_01.py`，从 `course_syllabus()` 开始，把 basics、systems、scaling laws、data、alignment 五条主线过完。
