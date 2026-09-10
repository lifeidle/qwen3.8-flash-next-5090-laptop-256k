# Qwen3.8-Flash-Next 177B on an RTX 5090 Laptop — 24 GB VRAM + 64 GB RAM · 256K Context

**中文**：一台 RTX 5090 笔记本（24G 显存）+ 64G 内存，跑通 Qwen3.8-Flash-Next 177B 的 GGUF 量化版：**256K 上下文、23–28 tok/s**，以及完整的选型、踩坑与调优记录。
**English**: Running **Qwen3.8-Flash-Next 177B** (GGUF quantized) on a single **RTX 5090 Laptop (24 GB VRAM) + 64 GB RAM** — **256K context at 23–28 tok/s**, with the full story of quant selection, pitfalls, and tuning.

> 📄 **完整实录** → [中文](./docs/deploy-log.zh.md) ｜ [English](./docs/deploy-log.en.md)
> 📊 **模型与量化参考** → [models & quants](./docs/model-reference.md)
> 🛠 **可复用测试工具** → [tools/](./tools/bench_single_instance.py)

---

## 成果一览 / Results at a glance

| 指标 / Metric | 起点 / Baseline | 最终 / Final |
|---|---|---|
| 生成速度 Generation | ~11 tok/s（社区同配参考 / community reference） | **21.7（32K）/ 24.6–28.2（64K）/ 23.4 tok/s（256K）** |
| 上下文 Context | 8K | **256K = 262,144 tokens（模型全长 / full model length）** |
| 量化 Quant | — | AtomicChat **AD-4.27bpw**（主力）/ AD-3.84bpw（速度备选） |
| 显存占用 VRAM | — | 21.6–21.9 / 24 GiB（按档位 / per config） |
| 单实例内存峰值 RAM peak | — | 82–96%（稳定，无失控 / stable） |

## 最终配置 / Final config

```bash
llama-server \
  -m Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf \
  -ngl 99 --n-cpu-moe 42 \
  -fa on -fit off \
  -c 262144 -np 1 \
  --jinja
```

| 需求 Need | ncmoe | ctx | 实测 Measured |
|---|---|---|---|
| 极速 Speed-first | 34 | 65536 | 24.6–28.2 tok/s |
| **均衡 Balanced（推荐）** | **42** | **262144** | **23.4 tok/s** |
| 多会话 Multi-slot | 48 | 262144 | 17.7–22.5 tok/s |

> 跷跷板规律 / The see-saw rule：**每 +2 层专家回内存 ≈ 腾出 2.06 GiB 显存 ≈ 上下文翻一倍**
> Moving 2 expert layers back to RAM frees ≈2.06 GiB VRAM ≈ doubles the context window.

## 目录 / Repository layout

```
├── README.md                     ← 本页 / this page
├── docs/
│   ├── deploy-log.zh.md          ← 完整实录（中文，十节）
│   ├── deploy-log.en.md          ← Full write-up (English)
│   └── model-reference.md        ← 候选量化对照、架构参数、引擎与运行时
├── tools/
│   └── bench_single_instance.py  ← 单实例纪律的基准测试驱动（可复用于任意 GGUF）
└── LICENSE
```

## 适用场景 / Who this is for

- 你有一台 **24 GB 显存的消费级显卡 + 64 GB 内存** 的机器，想跑 100B+ 级别的 MoE 大模型；
- 你在纠结**选哪个量化档位**（AD 4.27 / 3.84 / 5.00，unsloth UD-Q3_K_XL / UD-IQ4_XS）；
- 你想知道**上下文能开多大**、`--n-cpu-moe` 怎么调、MTP 投机解码为什么在你这儿没用；
- 你想避免"**静默失败**"：推理悄悄退回 CPU、显存悄悄溢出到内存——本文都有针对性的诊断方法。

- You have a **24 GB VRAM consumer GPU + 64 GB RAM** and want to run 100B+ MoE models locally;
- You're choosing between quant variants and want **measured** numbers, not guesses;
- You want the maximum usable **context length**, and the `--n-cpu-moe` tuning recipe;
- You want to detect **silent failures** (CPU fallback, VRAM oversubscription) instead of guessing why it's slow.

## 硬件与软件 / Tested on

| 项目 Item | 规格 Spec |
|---|---|
| GPU | RTX 5090 **Laptop** GPU，24 GB 显存（24435 MiB 可见 / reported），compute capability **12.0 (sm_120)** |
| 内存 RAM | 64 GB |
| 存储 Storage | NVMe SSD（模型约 88 GB / model ≈ 88 GB） |
| 引擎 Engine | llama.cpp（Unsloth `b10840-mix-d5c17a0`，`cuda12-portable` 构建）+ 补装的 CUDA 12.8 运行时 DLL |
| 模型 Model | Qwen3.8-Flash-Next GGUF，总参数 176.9B（含 51.2B N-gram 表） |

## 关键结论速览 / Key takeaways

1. **布局比位宽重要** — 大权重表（N-gram 表，35.76 GiB）是否独占分片，决定这套方案能不能在这台机器上成立；
2. **两类静默失败**要防 — CUDA 运行时缺失会静默退回 CPU；显存溢出会静默走 PCIe（慢 30 倍、无报错）；
3. **MoE + CPU 专家 = 投机解码负收益** — 验证批的专家激活并集会放大内存流量；
4. **KV 缓存小得出奇**（每 token 仅 33 KiB）→ 显存应尽量让给上下文，把专家还给内存；
5. **`--n-cpu-moe` 是唯一的旋钮**，且存在悬崖（本例 28 层即崩）。

## 免责声明 / Disclaimer

- 所有数据均为**单机实测**，硬件/驱动/构建版本不同结果会有差异；
- 模型权重与量化文件版权归各自发布方，本仓库仅提供部署经验与方法；模型许可见 [Qwen 社区许可](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF)；
- 测试脚本会**自动结束 llama-server 进程**，请勿在有其他推理服务运行时使用。

- All numbers are **measured on one machine**; your mileage may vary with driver/build versions;
- Model weights belong to their respective publishers — this repo only documents deployment methodology;
- The benchmark script **kills `llama-server` processes**; do not run it while other inference services are active.

## License

MIT（仅适用于本仓库的文档与脚本 / applies to this repository's docs and scripts only）
