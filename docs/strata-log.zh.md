# Strata 引擎实录：从 25 tok/s 到 93.5 tok/s —— MTP 修复、三量化对比与 GPU/CPU 双满载

> 续篇：[llama.cpp 部署实录](./deploy-log.zh.md)（22-25 tok/s 时代）之后，换用第三方引擎 [Strata](https://github.com/Niko1221/Strata) v0.1.27→0.1.28 的完整实测记录。
> 硬件不变：RTX 5090 Laptop 24GB VRAM + 64GB RAM + Core Ultra 9 275HX。

## 结论速览

| 模型（Qwen3.8-Flash-Next GSQ-RCO） | 有效精度 | llama.cpp 时代 | Strata（MTP 修复后最优） | 定位 |
|---|---|---|---|---|
| Coder IQ1_M（256 专家） | 1.89 bpw | — | 62.5 tok/s | 最省内存（23.4GB arena） |
| Q2_0 完整版（512 专家） | ~2.2 bpw | 25 tok/s（3.84bpw AD 量化） | **93.5 tok/s** | 速度王 |
| IQ3_XXS 完整版（512 专家） | ~3.1 bpw | — | **77.4 tok/s**（256K+Vision 下 74.4） | 质量优先 |

**3.7 倍速度提升**，且是 2-bit 量化的 Q2_0 达成——同时 GPU 与 CPU 占用率几乎全部占满。

---

## 一、最重要的发现：GPU 和 CPU 第一次同时占满

llama.cpp 时代的老大难：**CPU 占用率常年 100%（单大核瓶颈），GPU 利用率上不去**（20-40%），CPU 成为绝对瓶颈，GPU 的 24GB 显存里躺着权重却在等饭吃。

Strata 的三层架构解决了这个结构性问题：

```
显存（24GB）：热专家缓存（~9,300-13,200 slots）+ 32K KV 热窗 + MTP 草稿层 + vision
内存（64GB）：全量专家 arena（pinned）+ KV 冷数据（streaming）
SSD：        PLE n-gram 查表（mmap，按需）
```

- **热专家走 GPU**：按路由频次预填显存，命中即 GPU 计算（命中率 92-98%+）
- **冷专家走 CPU 池**：23 个 worker 线程 + 主线程并行计算，miss 即时补
- **MTP 草稿层在 GPU 上独立跑**：与主模型流水线重叠
- **结果：CPU 池与 GPU 同时满载，互不等待**。这是速度从 25 → 93 的根本原因，比任何单点调参都重要。

## 二、MTP 坏死之谜：20/31 个权重文件下载损坏

### 现象
Strata 默认开启 MTP（Multi-Token Prediction 推测解码），官方数据 2.4-3.2 tokens/round。但我们实测：

```
drafts accepted 0 of 0   ← 默认配置下草稿从未被提出
drafts accepted 0 of 765 ← 强制开窗后提出 765 个草稿，100% 被否决
```

### 排查路径（层层递进）
1. 猜测采样参数/温度/上下文 → 全排除（温度 0/0.7/seed 固定均无效）
2. 猜测 Coder 版 256 专家与草稿层错配 → 换 512 专家完整版仍 0
3. 读引擎源码（安装包自带完整 C++/CUDA 源码）发现：`drafts accepted 0 of 0` 的第二个 0 是 **offered（提出数）**——窗口 T 恒为 1，草稿从未被提出过。此前的推断（"草稿全被拒"）是误读
4. 强制开窗（`spec_min_p=0`）→ `0 of 765`：草头提出的候选全部错误
5. **自编译引擎加诊断探针**（编译链 2 分钟跑通）：`dprob=NaN, drafts=0,0,0`——草稿层前向输出 NaN，argmax 恒 0
6. 逐张量检查权重文件：`dense.bin` 从第 11 个张量起数值 2.1e37 级（垃圾）
7. 追到源头：**31 个 MTP 权重文件里 20 个的内容根本不是权重，而是 safetensors shard 文件的开头（JSON header + 无关数据）**

### 根因
`tools/mtp_fetch.py` 用 HTTP Range 请求从镜像站拉取 BF16 checkpoint 里的 mtp.* 张量。**部分请求的 Range 头被服务器忽略**（返回 200 全文件），工具只存前 n_bytes——存下的恰是 shard 的头部。而 manifest 里的 sha256 校验的是"下载到的字节"，**无法发现这种错误**。

### 修复
1. 检测：正确的张量 raw 文件前 10 字节不会是 `[u64 header_len][b'{"']`（见 `tools/check_dense.py` / `tools/refetch_mtp.py`）
2. 重拉 20 个坏文件（~110MB，两个 5GB 专家大文件幸免）
3. 重新打包 → 全部 29 张量 0 NaN/Inf → **MTP 立刻复活**

| 阶段 | drafts accepted | 速度 |
|---|---|---|
| 修复前（强制开窗） | 0 of 381 (0.0%) | 42.7 tok/s |
| **修复后（强制开窗）** | 62 of 195 (31.8%) | 69.9 tok/s |

**教训**：下载类工具必须校验 Range 响应是否真的返回 206（`Content-Range`），sha256 对"下载到的字节"算 hash 毫无意义。

### 修复后各模型 MTP 草稿接受率
- Q2_0：44-54%（spec_min_p=0.5）
- IQ3_XXS：59-71%（草稿质量最高，见第四节）

## 三、IQ3 的反直觉调参：spec_min_p 越高越快

MTP 的窗口 T 由上一轮草稿置信度门控（`spec_min_p`）。两个模型的峰值完全相反：

| spec_min_p | Q2_0（草稿一般） | IQ3_XXS（草稿优秀） |
|---|---|---|
| 0.20 | — | 60.4 tok/s（崩） |
| 0.30 | **93.5（峰）** | 66.8 |
| 0.45-0.5 | ~90 | 79.4 |
| **0.70** | ~87 | **83.6 热 / 74.4 防缓存（峰）** |
| 1.0（MTP off） | ~59.7 | 63.7 |

**解释**：草稿质量高 → 引擎可以"高标准严要求"，只验证高置信度窗口，错误草稿几乎不出现（接受率 70.8%）。草稿质量一般 → 需要放宽门槛多试。**换模型后必须重扫这个参数**，照搬别的模型的值会亏 10-15%。

## 四、256K 上下文阶梯（IQ3_XXS + Vision + MTP）

Strata 的 KV streaming：KV 放 pinned RAM，显存只留 32K token 热窗。专家缓存几乎无损：

| 上下文 | 专家缓存 slots | KV 进内存 | 速度（中位） | 备注 |
|---|---|---|---|---|
| 32K | 9,588 | 0（全显存） | **77.4 tok/s** | 最快 |
| 65K | 9,559 | 0.77 GiB | 65.7 | |
| 128K | 9,487 | 1.55 GiB | 64.5 | prefill 2,527 tok/s（1.4 万 token 6.7s） |
| **256K** | 9,350 | 3.09 GiB | **65.1** | ✅ 官方 setup 不给这个组合（保守），实测能跑 |

- 65K/128K/256K 速度几乎相同（streaming 固有开销 ~-16%，与长度无关）→ **要么 32K（最快），要么直接 256K**
- MTP 全程健康（46-59% 接受率），专家缓存损失仅 2.5%
- 64GB 内存下 256K 稳定：arena 42.9GB pinned + KV 3.1GB + 系统，余量充足

## 五、Vision 挂载

- mmproj BF16 编码器 0.9GB，GPU 模式预留 ~1.4GB 显存（专家缓存相应缩小）
- 每图 ≤1,024 token（640×480 照片 ≈300），编码 0.1-0.5s/张
- 配置加两处即可（无需重跑 setup）：引擎 args 加 `--vision`（布尔开关）；config 顶层加 `"vision": {"exe": ..., "mmproj": ..., "model": ..., "gpu": true, "max_tokens": 1024}`
- **256K 上下文理论上可一次塞 250+ 张图**

## 六、最终配置（IQ3_XXS · 256K · Vision · MTP）

```
--pack packs/iq3_xxs --expert-cache auto --prefill auto
--spec 4 --spec-min-p 0.7 --mtp <rt-dir>
--max-context 262144 --kv int8 --kv-resident 32768
--vision --vram-reserve-mib 700
```

**实测：防缓存 median 74.4 tok/s（服务端单请求 84.0），MTP 接受率 70.8%，GPU+CPU 双满载。**

尝试过并否决的方向：spec 6（大窗口验证成本>收益，-8%）；k8v4 KV 压缩（省 streaming 开销但专家缓存 -14.6%，打平后回退）；pcie_frac 0.35/0.75（默认 0.55 最优）；spec_min_p 0.2（阈值过低，-18%）。

## 七、给后来者的检查清单

1. 装完引擎先验证 MTP：看 log 里 `drafts accepted` 是否非 0——若是 `0 of 0`，先查草稿层权重是否下载损坏（本文第二节）
2. 换模型必须重扫 `spec_min_p`（峰值位置因草稿质量而异）
3. 上下文 ≥65K 记得 KV streaming（`--kv-resident 32768`），官方 setup 对某些组合过于保守，可自行实测放宽
4. 基准测试要防 prompt cache 污染（同文本 reused 会让速度虚高，请求加随机后缀）
5. 吞吐瓶颈先看 GPU/CPU 是否双满载：CPU 单核满 + GPU 闲 = 引擎架构问题，调参无用
