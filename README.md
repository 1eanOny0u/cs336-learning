# CS336 自学笔记

Stanford [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)（Spring 2026）的中文自学笔记。

正式课是 5 学分，五份实现型作业非常重。这份材料面向想在大约一个月里把正课主线理清的人：每天约 25 分钟读一小节官方讲义整理，再用约 5 分钟做文末自测。大约 30 天、合计约 15 小时，目标是搞明白一条语言模型从零搭起来要经过什么，而不是做完 Assignment 1–5。

## 怎么学

- **中文讲解**，依据官方讲义（`lecture_XX.py` / PDF）拆分，并附上官方配图和关键论文链接。
- **每天约 30 分钟**：25 分钟阅读 + 5 分钟自测。
- **大约一个月走完 17 讲主线**：从分词、架构、训练与系统优化，到缩放定律、推理与评测、数据和对齐。最后两天做总复习，用自己的话把整条链路串起来。

| 周次 | 安排 |
| --- | --- |
| 第 1 周（第 1–7 天） | Lecture 1–4：概览与分词、PyTorch / 资源估算、架构与超参、Attention 变体与 MoE；第 7 天复习 |
| 第 2 周（第 8–14 天） | Lecture 5–8：GPU / TPU、Kernel 与 Triton、并行训练；第 14 天复习 |
| 第 3 周（第 15–21 天） | Lecture 9–14：缩放定律、推理、评测、预训练数据；第 21 天复习 |
| 第 4 周（第 22–30 天） | Lecture 15–17：SFT / RLHF、RLVR、对齐与多模态；第 29–30 天总复习 |

当前已更新到第 4 天（第 2 讲：张量与数值精度）。后续按天追加到 [notes/2026/](notes/2026/)。

## 官方入口

- 课程主页：[cs336.stanford.edu](https://cs336.stanford.edu/)
- 讲义仓库：[stanford-cs336/lectures](https://github.com/stanford-cs336/lectures)
- Spring 2026 录像：[YouTube 播放列表](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
- 本仓库中的官方讲义快照与索引：[materials/](materials/README.md)

## 仓库目录

```text
cs336-learning/
├── materials/      # 官方讲义快照、配图和辅助文件
├── notes/          # 按天拆分的中文自学笔记（见 notes/2026/）
├── assignments/    # 按课程年份整理的作业实践
├── experiments/    # 小实验、配置和结果摘要
└── .gitignore
```

## 说明

这是个人自学笔记，不是课程官方材料，也不包含作业答案。文中的课件图通过官方仓库的公开链接引用。如与官方材料冲突，以课程主页和官方讲义为准。
