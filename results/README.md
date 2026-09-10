# 原始实测数据 / Raw measurements

本目录保存文章中所有数字的**原始输出**，供核对与复现。
Raw outputs behind every number in the write-up, kept for verification.

| 文件 File | 内容 Content | 来源 Source |
|---|---|---|
| `llama-bench-ncmoe-sweep.txt` | `--n-cpu-moe` 扫描原始输出（44/40/36/34/32/28） | `llama-bench -p 512 -n 128 -r 1 -ncmoe ...` |
| `single-instance-matrix.txt` | 单实例纪律下的上下文扫描 + MTP 对照（7 组） | 单实例测试驱动（同 `tools/bench_single_instance.py`） |
| `long-context-and-384.txt` | 长上下文三层分配（64K / 256K）+ AD-3.84bpw 对照 | 同上，含 10,455 token 长 prompt 预填充 |
| `mtp-draft-acceptance.txt` | MTP 投机解码的草案接受率（4 组配置） | 服务器日志中的 `draft acceptance` 行 |
| `vram-ledger-8192.txt` | ncmoe=32 / ctx=8192 的显存与内存分配账本 | `-lv 4` 详细日志 |
| `oom-evidence-ncmoe40-ctx256k.txt` | 256K + ncmoe=40 的显存不足原始报错 | 服务器启动日志 |
| `kv-quant-ab.txt` | KV 缓存量化 A/B：f16 vs q8_0 vs q4_0（+3.4%） | `tools/ab_bench.py`，6 轮取中位数 |
| `needle-accuracy.txt` | 长上下文召回精度：三种 KV 配置均 12/12 | `tools/needle_test.py`，4 组随机化 |
| `engine-build-ab.txt` | 引擎构建 A/B：b10840 vs b10889（**无提升**） | llama-bench r=3/r=5 + 服务端 6 轮 |

---

## 阅读提示 / How to read these

**口径 / Measurement protocol**

1. 每组配置独立进程：`kill → 校验进程数=0 → 启动 → 就绪 → 预热 ×2 → 测量 → kill → 校验`；
2. 采样请求固定（`max_tokens=450`，同一中文 prompt），取其 `timings.predicted_per_second`；
3. 全程每秒采样系统内存，报告案例时间窗内的峰值；
4. **以热态数据为准**——冷启动第一发明显偏低（权重未入页缓存）。

**关于被排除的数据 / On excluded data**

调试早期有一批数据作废：测试脚本的进程清理静默失效，最多 4 个 `llama-server` 实例叠加、显存与内存同时溢出，测出的 1.2~6 tok/s 全部失真，**未收录在本目录**。

An early batch of measurements was discarded — a silent failure in the process cleanup left up to four instances running at once, oversubscribing VRAM and RAM. Those 1.2–6 tok/s figures were artifacts, deliberately **not** included here.

> **基准测试的第一步不是测模型，而是确认测试环境干净。**
> Before benchmarking the model, make sure the test environment is clean.

**硬件 / Hardware**：RTX 5090 Laptop（24 GB VRAM，24435 MiB 可见）+ 64 GB RAM + NVMe SSD
**引擎 / Engine**：llama.cpp，Unsloth `b10840-mix-d5c17a0`（`cuda12-portable` 构建）+ CUDA 12.8 运行时 DLL
**模型 / Model**：Qwen3.8-Flash-Next GGUF，AD-4.27bpw（主力）/ AD-3.84bpw（对照）

**本目录数据由以下工具产出 / Produced with these tools**：
[`bench_single_instance.py`](../tools/bench_single_instance.py)（ncmoe 扫描）、
[`ab_bench.py`](../tools/ab_bench.py)（多轮取中位数的服务端 A/B）、
[`needle_test.py`](../tools/needle_test.py)（长上下文召回精度测试）。
后两者可直接拿去测你自己的配置。
