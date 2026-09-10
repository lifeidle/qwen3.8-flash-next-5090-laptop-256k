# 模型与量化参考 / Model & Quant Reference

> 本页汇总部署中涉及的全部模型资料、量化对照、引擎与运行时要求。
> All model facts, quant comparisons, engine and runtime requirements gathered during this deployment.

---

## 1. 模型本体 / The model

**Qwen3.8-Flash-Next** — 稀疏 MoE 大模型，官方许可 `qwen-community-1.0`。

### 1.1 架构参数（从 GGUF 元数据实测读取 / read from GGUF metadata）

| 键 Key | 值 Value | 说明 |
|---|---|---|
| `general.architecture` | `qwen4exp` | 架构标识 |
| `general.size_label` | `512x56B` | 512 专家 / 56B 级 |
| 总参数 Total params | **176.94 B** | 含 N-gram 表 |
| ├ N-gram 表 Table | **51.2 B** | 占 35.76 GiB（4.27 档，Q5_1） |
| └ MoE 主体 Body | 125.74 B | 48 层 |
| `block_count` | **48** | 层数 |
| `context_length` | **262 144** | 原生 256K |
| `embedding_length` | 2560 | 隐藏维度 |
| `attention.head_count` | 24 | 注意力头 |
| `attention.head_count_kv` | **2** | KV 头（极致 GQA，KV 极小的原因） |
| `attention.key_length` / `value_length` | 256 / 256 | |
| `expert_count` | **512** | 每层专家数 |
| `expert_used_count` | **10** | 每 token 激活专家数 |
| `expert_feed_forward_length` | 640 | 专家中间维 |
| `expert_shared_feed_forward_length` | 640 | 共享专家中间维 |
| `ssm.conv_kernel` | 4 | **GatedDeltaNet 混合架构**（部分层为线性循环层，状态固定） |
| `rope.dimension_sections` | [11, 11, 10, 0] | 多段 RoPE |
| `rope.freq_base` | 10 000 000 | |
| `general.tags` | `["image-text-to-text"]` | **多模态模型**，视觉需额外 mmproj 文件 |

**为什么这些参数重要 / why they matter:**

- **2 个 KV 头 + GatedDeltaNet** → KV 缓存每 token 仅 **33 KiB**（实测：8192 ctx = 264 MiB；256K = 8.25 GiB），这是能把上下文开到全长的根本原因；
- **512 选 10 的 MoE** → 每 token 只读 2% 的专家权重，因此"专家放显存 / 内存"的分层才有意义；
- **51.2B 的 N-gram 表** → 每 token 只查 16 行（KB 级），所以放 SSD 几乎零成本。

### 1.2 视觉能力（可选）/ Vision (optional)

模型标签含 `image-text-to-text`，需要 mmproj 视觉编码器才能识图：

| 文件 File | 大小 Size | 说明 |
|---|---|---|
| `mmproj-Qwen3.8-Flash-Next-F16.gguf` | 0.842 GiB | 推荐 / recommended |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 0.845 GiB | 等价 / equivalent |

启用方式：`llama-server ... --mmproj <mmproj 文件>`。代价约 1 GiB 显存，长上下文档位下需相应调高 `--n-cpu-moe`。

---

## 2. 量化版本对照 / Quant candidates

### 2.1 候选总表 / All candidates

| 发布方 Publisher | 档位 Variant | 总大小 Total | 分片 Shards | N-gram 表 Table | 主体 bpw Body | 质量 Quality | 本机可行性 Feasible |
|---|---|---|---|---|---|---|---|
| AtomicChat | **AD-4.27bpw-Q4_K_M-M64** | **88.03 GiB** | 33 | 独占分片 35.76 GiB（Q5_1, 6bpw） | 3.57 | **KLD 0.0842 / Top-1 89.49%** | ✅ **主力** |
| AtomicChat | AD-3.84bpw-IQ4_XS-M64 | 79.10 GiB | 28 | 独占分片 35.76 GiB（同上） | ≈2.92 | 未公布 | ✅ 速度备选（实测 +5~9.5%） |
| AtomicChat | AD-5.00bpw-Q5_K_M-M64 | 102.93 GiB | 33 | 独占分片 50.66 GiB（8.5bpw） | 3.57 | — | ⚠️ 表更大，SSD 压力上升 |
| unsloth | UD-IQ4_XS | 87.25 GiB | 3 | **混装**（IQ4_NL ≈26.8 GiB 与专家同片） | 4.13 | — | ❌ 布局导致整片锁内存 |
| unsloth | UD-Q3_K_XL | 83.80 GiB | 3 | 推定混装 | ≈3.6 | — | ❌ 同上（推定） |

### 2.2 分片结构明细 / Shard layout detail

```
AD-4.27bpw-Q4_K_M-M64（33 片，M64 布局）
  00001  0.646 GiB   头（输出层等）
  00002 35.763 GiB   ★ N-gram 表独占 —— 可被 mmap 独立换出
  00003~00033        专家与注意力（每片 1.5~1.9 GiB）

AD-3.84bpw-IQ4_XS-M64（28 片，同构）
  00001  0.646 GiB   头
  00002 35.763 GiB   ★ N-gram 表（与 4.27 完全同尺寸 → 同一张表）
  00003~00028        主体（更小）

unsloth UD-IQ4_XS（3 片）
  00001  0.010 GiB   元数据
  00002 46.413 GiB   表 + 专家混装  ← 问题所在
  00003 40.826 GiB   主体
```

### 2.3 为什么 "M64" 布局是决定性的 / Why the M64 layout is decisive

| | 表独占分片（M64） | 表与专家混装 |
|---|---|---|
| mmap 行为 | 表可独立按需分页、可被换出 | 分片内有 GPU 张量 → 整片锁定进内存 |
| 常驻内存需求 | ≈ 67.4 GiB（映射工作集） | 最坏 ≈ 89.6 GiB（> 64 GiB 物理内存） |
| 结果 | ✅ 三层分级成立 | ❌ 爆内存，或退化为全 CPU |

### 2.4 为什么没有选 NVFP4 / Why not NVFP4

RTX 5090 是 Blackwell 架构，原生支持 **NVFP4**（4 位浮点 + 分块缩放）张量核，社区也常有"Blackwell 就该上 NVFP4"的说法。本次部署**没有采用**，有三个层次的原因：

**① 事实层：这个模型压根没有 NVFP4 版本可下**

把两个发布方的仓库文件全量枚举过（AtomicChat 104 个文件 / unsloth 60 个文件，关键字 `fp4`、`nvfp`、`mx` 零命中）：

| 发布方 | 可用量化档位 |
|---|---|
| AtomicChat | AD-3.84bpw-IQ4_XS-M64、AD-4.27bpw-Q4_K_M-M64、AD-5.00bpw-Q5_K_M-M64 |
| unsloth | BF16、Q8_0、UD-IQ1_M/S、UD-IQ3_XXS、UD-IQ4_XS、UD-Q2_K_XL、UD-Q3_K_XL、UD-Q4_K_XL、UD-Q5_K_XL、UD-Q6_K_XL |

全是 K-quant / I-quant（整数/混合量化），**没有任何 NVFP4 发布**。没有的东西谈不上选择。

**② 原理层：就算有，对这台机器的生成速度也没帮助——瓶颈不在算力**

NVFP4 加速的是**矩阵乘法算力**（张量核），而本机生成阶段的瓶颈是**内存带宽**（每 token 要读一遍激活的专家权重）。实测证据：

| 阶段 | 实测速度 | 瓶颈类型 | NVFP4 能否帮上 |
|---|---|---|---|
| 生成（tg） | 20~28 tok/s，对应约 20~25 GB/s 内存读取 | **内存带宽受限** | ❌ 算力再快也要等内存 |
| 预填充（pp） | 89~139 tok/s（1 万 token 长文） | 算力/批处理 | ✅ 能帮上，但只影响"读长文"的首字延迟 |

同样的证据还出现在 `--n-cpu-moe` 扫描里：把专家层从 8 层减到 4 层（ncmoe 40→44），预填充速度从 92.3 掉到 53.5（算力相关，变化剧烈），而生成速度只从 25.0 变成 22.6（带宽相关，平缓变化）。

**③ 数学层：NVFP4 的等效位宽反而比现在更大**

NVFP4 = 4 bit 权重 + FP8 缩放因子（每 16 个元素一个）≈ **4.5 bpw 等效**。

| 方案 | 主体等效位宽 | 每 token 专家字节 |
|---|---|---|
| 现用 AD-4.27bpw | **3.57 bpw** | 基准 |
| AD-3.84bpw（速度备选） | ≈2.92 bpw | 少 17% → 实测快 5~9.5% |
| 假想 NVFP4 | ≈4.5 bpw | **多约 26% → 更慢** |

在这个"模型比总内存还大"的场景里，**要的是更少的字节，不是更快的数学**。所以选型方向一直是"同布局下取更小位宽"（4.27 → 3.84 就是这么来的），而不是追求更强的算力格式。

**什么情况下 NVFP4 才值得考虑**：① 出现官方 NVFP4 量化、且等效位宽 ≤ 现有档位；② 工作负载转向**算力密集**（如大批量并发推理、长文预填充为主）；③ llama.cpp 对 `qwen4exp` + 混合 SSM 架构的 FP4 内核完整可用（目前路径覆盖有限，混合架构风险较高）。

### 2.5 为什么用 llama.cpp，而不是 vLLM / TensorRT-LLM / KTransformers

三个理由，按决定性排序：

1. **内存分级能力**——只有 llama.cpp 提供 `--n-cpu-moe` 这种"按层把专家权重留在内存、其余进显存"的精细控制（配合 `-ot` 可做张量级覆盖），这正是本机 24G+64G 跑 88 GiB 模型的前提。vLLM/TensorRT-LLM 面向"权重能全进显存"的场景；
2. **mmap 分片按需分页**——N-gram 表放 SSD 的方案依赖 mmap 的按需换页，这在此前的工作流里是天然支持；
3. **配套齐备**——MTP 投机解码头、mmproj 视觉编码器、混合 SSM 架构的 CUDA 内核，这个构建都已具备（即使我们最终没用 MTP）。

---

## 3. 引擎与运行时 / Engine & runtime

| 项目 Item | 内容 Content |
|---|---|
| 引擎 Engine | llama.cpp，Unsloth `b10840-mix-d5c17a0` 构建 |
| 变体 Variant | `cuda12-portable`（含 sm_120 / Blackwell 内核，压缩包约 355 MB） |
| 必需补件 Required extras | `cudart64_12.dll`、`cublas64_12.dll`、`cublasLt64_12.dll`、`nvblas64_12.dll` |
| 补件来源 Source | PyPI 官方 wheel：`nvidia-cuda-runtime-cu12==12.8.90`（0.9 MB）、`nvidia-cublas-cu12==12.8.5.5`（391 MB） |
| 验证方法 Verify | 迷你模型跑 `llama-bench`，输出须含 `backend = CUDA` 与 `compute capability 12.0` |
| 常见误判 Misdiagnosis | 缺运行时 → **静默退回 CPU**，无任何报错 |

---

## 4. 实测资源账本 / Measured resource ledger

配置 `--n-cpu-moe 42 -c 262144`、模型 AD-4.27bpw：

| 资源 Resource | 占用 Usage |
|---|---|
| 显存 VRAM | 模型 ~10.6 GiB（6 层专家 6.18 + 注意力常驻 4.4）+ KV 8.25 GiB + 计算缓冲 1.98 GiB + 循环态 0.11 GiB ≈ **20.9 / 24 GiB** |
| 内存 RAM（映射） | **≈ 79 GiB 映射工作集**（42 层专家 43.3 + 表 35.76），物理 64 GiB，比值 1.23，稳态占用 94~96% |
| 固态 SSD | 模型文件 88 GiB，表的冷页按需读入 |

`--n-cpu-moe 32 -c 8192`（速度基准）参考账本：

| 项 Item | 大小 Size |
|---|---|
| 显存模型缓冲 Model buffer | 20.87 GiB |
| KV 缓存 KV cache | 0.26 GiB |
| 循环态 RS buffer | 0.11 GiB |
| 计算缓冲 Compute buffer | 0.64 GiB |
| CPU 映射 CPU_Mapped（47 段） | 67.36 GiB |

---

## 5. 参数速查 / Parameter cheat sheet

| 参数 | 含义 | 本机推荐值 |
|---|---|---|
| `-ngl 99` | 全部层上卡 | 99（再由 `--n-cpu-moe` 做张量级分流） |
| `--n-cpu-moe N` | 前 N 层的专家权重留在 CPU | 34（速度）/ **42（256K 全长）** / 48（多会话） |
| `-c N` | 上下文长度 | 65536 ~ 262144 |
| `-fa on` | Flash Attention | 长上下文必开 |
| `-fit off` | 关闭自动显存适配 | 本架构自动测算不准，手动控制 |
| `-np N` | 并行槽位 | 1（默认）；显存有余时可提高 |
| `--jinja` | 启用模型自带对话模板 | 开启（含思维链模板） |
| `--no-warmup` | 跳过启动预热 | 大模型必开，否则启动极慢 |
| `--mmproj <file>` | 挂载视觉编码器 | 需要识图时加 |
| `-md <file>` + `--spec-type draft-mtp` | MTP 投机解码 | **不推荐**（见实录第六节） |

---

## 6. 相关链接 / Links

| 对象 | 位置 |
|---|---|
| 模型权重（主力） | `AtomicChat/Qwen3.8-Flash-Next-GGUF` → `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64/` |
| 模型权重（速度备选） | 同上 → `Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64/` |
| MTP 头 | `unsloth/Qwen3.8-Flash-Next-GGUF` → `MTP/` |
| 引擎构建 | Unsloth llama.cpp releases（`b10840-mix` 系列，选 `cuda12-portable`） |
| CUDA 运行时 | PyPI：`nvidia-cuda-runtime-cu12` / `nvidia-cublas-cu12`（Windows wheel） |

> 模型与量化文件的版权及许可归各自发布方所有；本仓库仅记录部署方法与实测数据。
> Model weights and quants remain under their publishers' licenses; this repository only documents methodology and measurements.
