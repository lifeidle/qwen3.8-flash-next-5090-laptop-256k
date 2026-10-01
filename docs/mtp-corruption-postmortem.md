# MTP 草稿层权重损坏事故复盘 / MTP Drafter Corruption Postmortem

> **TL;DR (EN)**: MTP acceptance was 0% on every configuration with no visible error. Root cause: 20 of 31 drafter weight files were silently corrupted at download time — an HF mirror ignored the HTTP Range header, and the fetch tool saved the first n_bytes of the response, which was the safetensors shard's *header*, not the tensors. Length and sha256 checks both passed (they verify "the bytes you got", not "you got the right bytes"). Detection: real tensor files never start with a safetensors header. Fix: re-fetch (see `tools/refetch_mtp.py`), acceptance went 0% → 70.8%. Filed upstream as [Niko1221/Strata#327](https://github.com/Niko1221/Strata/issues/327).
>
> 中文完整复盘如下。English readers: the TL;DR above plus the [upstream issue](https://github.com/Niko1221/Strata/issues/327) carries the essentials; the Chinese sections add the full investigation trail.
>
> **UPDATE 2026-10-02**: confirmed by the author and **fixed in [Strata v0.1.32](https://github.com/Niko1221/Strata/releases/tag/v0.1.32)** — `mtp_fetch` now requires HTTP 206 with a matching `Content-Range`, checks per-tensor SHA-256 against the pinned revision, and re-fetches any corrupt tensor. Our report's two suggested checks both made it into the fix.

- **上游 issue**：[Niko1221/Strata#327](https://github.com/Niko1221/Strata/issues/327)
- **影响**：Q2_0 / IQ3_XXS / Coder IQ1_M 三个量化版共用同一份草稿层权重，全部中招
- **结果**：修复后 MTP 接受率 0% → 70.8%，decode +64%

---

## 一、现象：一个"看不见"的故障

引擎一切正常：模型加载、对话流畅、专家缓存命中率 92-98%。唯一的异常是推测解码完全无效：

```
strata serve: ... drafts accepted 0 of 0, 1 checkpoints      ← 默认配置
strata serve: ... drafts accepted 0 of 765, 1 checkpoints    ← spec_min_p=0 强制开窗
```

最阴险的地方：**没有任何报错**。如果用户不做基准对比，根本不会发现 MTP 从未工作过——只会觉得"这个引擎的 MTP 没什么用"。

## 二、排查时间线（十余个假设的排除过程）

| # | 假设 | 验证方法 | 结果 |
|---|---|---|---|
| 1 | 采样参数问题 | temperature 0/0.7/top_p/seed 全组合 | ❌ 排除 |
| 2 | prompt 太短 | 1,630 token 长生成 | ❌ 排除 |
| 3 | Coder 256 专家与草稿层错配 | 换 512 专家完整版 | ❌ 排除（仍 0） |
| 4 | draft_vocab 缺 CJK（issue #137） | 检查文件为修复版 + 英文测试 | ❌ 排除 |
| 5 | IQ 内核宽度分歧（issue #152） | `STRATA_NO_IQ*` 绕过实测 | ❌ 排除 |
| 6 | 草稿从未被提出？（读源码） | `draft_offered` 是第二个 0 | ✅ **关键转折：T 恒为 1** |
| 7 | 强制开窗看草稿质量 | `spec_min_p=0` | ✅ `0 of 765`：提出即全错 |
| 8 | 草稿层 2-bit 量化精度不足？ | 自实现 q2_0 量化对比 | ❌ 误差 31-43% 是有界值，不产生 NaN |
| 9 | **自编译引擎加探针** | `dprob=NaN, drafts=0,0,0` | ✅ **前向输出 NaN** |
| 10 | NaN 从哪层开始？ | 逐缓冲探针 | ✅ 第一个 matmul（fc_embedding）输出即 NaN |
| 11 | 权重文件数值检查 | numpy 逐张量 | ✅ 第 11 个张量起 f32 权重 2.1e37 级 |
| 12 | 下载损坏？ | 文件头检测 | ✅ **20/31 个文件是 safetensors shard 头部** |

### 关键转折点的教训

第 6 步读源码发现 `drafts accepted 0 of 0` 的第二个 0 是 **offered（提出数）** 而非"被拒数"——这推翻了此前所有"草稿被拒"的推断。**日志语义的精确理解比参数扫描更重要。**

第 9 步自编译引擎（安装包自带完整源码，nvcc + MSVC 全量编译仅 2 分钟）加探针，把"草稿层输出什么"从黑盒变成可观测——直接给出 NaN 这个决定性证据。

## 三、根因：三层校验全部失效

### 下载流程
```
mtp_fetch.py --range 请求 shard 的 [start, end] 字节
    ↓ 代理/镜像忽略了 Range 头，返回 200 + 整个文件（或代理截断到指定长度）
mtp_fetch.py 存下响应的前 n_bytes     ← 恰好是 shard 文件的开头（JSON header 区）
mtp_pack.py / mtp_rt.py 正常打包      ← 垃圾也是"合法字节"
引擎正常加载                          ← 垃圾是有界浮点数，不崩，只是错误
```

### 为什么三道防线全漏

| 防线 | 设计意图 | 为什么失效 |
|---|---|---|
| 长度校验（`len(data) != end-start+1` 则 raise） | 防 short read | 响应长度恰好等于请求范围（代理截断），长度"正确" |
| sha256（manifest 记录） | 防内容损坏 | hash 是**对下载到的字节**算的——错位内容也能自洽通过 |
| 引擎加载校验 | 防格式错误 | 垃圾在数值上是有界浮点数，格式完全合法 |

**根本缺陷**：没有任何一环验证"你拿到的字节就是你要的字节"。HTTP 语义上这只需要一个检查——**响应状态码必须是 206 Partial Content**（或存在 `Content-Range` 头）。`urllib` 默认不检查，`200` 会被当作成功。

### 检测方法（可复用）

正确的张量 raw 文件**永远不会**以 safetensors 文件头开头。判别式：

```python
def looks_like_shard_head(path):
    with open(path, "rb") as f:
        head = f.read(16)
    return len(head) >= 10 and head[8:10] == b'{"'   # [u64 header_len] + '{"...'
```

31 个文件中 20 个命中（`tools/check_dense.py` 做逐张量数值体检，`tools/refetch_mtp.py` 做检测+重拉）。

## 四、修复与验证

1. **重拉**：从 inventory.json 取 range，curl 显式 `-r start-end`，逐文件校验（大小 + 非 shard 头）→ 20/20 OK（~110MB；两个 5GB 专家大文件幸免）
2. **重新打包**：`mtp_pack.py --experts q2_0` → `mtp_rt.py` → `check_dense.py` 全绿（29 张量 0 NaN/Inf）
3. **验证**：

```
修复前（强制开窗）:  drafts accepted 0 of 381 (0.000),  decode 42.7 tok/s
修复后（强制开窗）:  drafts accepted 62 of 195 (0.318), decode 69.9 tok/s  (+64%)

各模型最终接受率:
  Q2_0:       44-54%     IQ3_XXS: 59-71%     Coder: 41-61%
```

## 五、给下载类工具作者的三条建议

1. **Range 请求必须验证 206**（或 `Content-Range` 头），收到 200 应当报错或整文件走完再切片——"长度恰好对"不等于"内容对"
2. **sha256 要对着"权威源"算**：要么服务器提供 per-file checksum，要么下载完做内容特征断言（如 safetensors 头检测）。对自产数据算 hash 是同义反复
3. **静默降级是最贵的失败模式**：这次故障的全部成本不在损坏本身（重拉 110MB），而在"引擎正常但 MTP 永远无效"的排查上。宁可 fail loud

## 六、相关材料

- 上游 issue（英文，含修复建议代码）：[Niko1221/Strata#327](https://github.com/Niko1221/Strata/issues/327)
- 本仓库：[Strata 引擎实录](./strata-log.zh.md) ｜ [English](./strata-log.en.md) ｜ [原始数据](../results/strata-mtp-repair.txt)
- 工具：[check_dense.py](../tools/check_dense.py)（权重数值体检）· [refetch_mtp.py](../tools/refetch_mtp.py)（检测+重拉）· [diag-window.py](../tools/diag-window.py)（spec_min_p 扫描基准）
