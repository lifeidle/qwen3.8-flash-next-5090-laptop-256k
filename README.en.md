# Qwen3.8-Flash-Next 177B on an RTX 5090 Laptop — 24 GB VRAM + 64 GB RAM · 256K Context

**English** | [中文首页 →](./README.md)

Running **Qwen3.8-Flash-Next 177B** (GGUF quantized) on a single **RTX 5090 Laptop (24 GB VRAM) + 64 GB RAM** — reaching the **full 256K context at 23–28 tok/s**, with the complete story of quant selection, pitfalls, and tuning.

> 📄 **Full write-up** → [English](./docs/deploy-log.en.md) ｜ [中文](./docs/deploy-log.zh.md)
> 📊 **Models & quants** → [reference](./docs/model-reference.md)
> 🔧 **Port to your hardware** → [porting guide](./docs/porting-guide.md)
> 🧪 **Raw measurements** → [results/](./results/README.md)
> 🛠 **Reusable bench tool** → [tools/](./tools/bench_single_instance.py)

---

## Results at a glance

| Metric | Baseline | Final |
|---|---|---|
| Generation | ~11 tok/s (community reference on comparable hardware) | **21.7 (32K) / 24.6–28.2 (64K) / 23.4 tok/s (256K)** |
| Context | 8K | **256K = 262,144 tokens (full model length)** |
| Quant | — | AtomicChat **AD-4.27bpw** (primary) / AD-3.84bpw (speed option) |
| VRAM used | — | 21.6–21.9 of 24 GiB (per config) |
| RAM peak (single instance) | — | 82–96% (stable, no runaway) |

## Final config

```bash
llama-server \
  -m Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf \
  -ngl 99 --n-cpu-moe 42 \
  -fa on -fit off \
  -c 262144 -np 1 \
  --jinja
```

| Need | ncmoe | ctx | Measured |
|---|---|---|---|
| Speed-first | 34 | 65536 | 24.6–28.2 tok/s |
| **Balanced (recommended)** | **42** | **262144** | **23.4 tok/s** |
| Multi-slot | 48 | 262144 | 17.7–22.5 tok/s |

> **The see-saw rule:** moving 2 expert layers back to RAM frees ≈2.06 GiB of VRAM ≈ doubles the context window.

## Repository layout

```
├── README.md / README.en.md      ← 中文 / English homepages
├── docs/
│   ├── deploy-log.zh.md          ← 完整实录（中文）
│   ├── deploy-log.en.md          ← Full write-up (English, 10 sections)
│   ├── model-reference.md        ← Architecture params, quant comparison, why not NVFP4
│   └── porting-guide.md          ← Formulas + hardware lookup table
├── results/
│   ├── llama-bench-ncmoe-sweep.txt
│   ├── single-instance-matrix.txt
│   ├── long-context-and-384.txt
│   ├── mtp-draft-acceptance.txt
│   ├── vram-ledger-8192.txt
│   └── oom-evidence-ncmoe40-ctx256k.txt
├── tools/
│   └── bench_single_instance.py  ← single-instance benchmark driver (works with any GGUF)
└── LICENSE
```

## Who this is for

- You have a **24 GB VRAM consumer GPU + 64 GB RAM** and want to run 100B+ MoE models locally;
- You're choosing between quant variants and want **measured** numbers, not guesses;
- You want the maximum usable **context length** and the `--n-cpu-moe` tuning recipe;
- You want to detect **silent failures** (CPU fallback, VRAM oversubscription) instead of wondering why it's slow.

## Tested on

| Item | Spec |
|---|---|
| GPU | RTX 5090 **Laptop** GPU, 24 GB VRAM (24435 MiB reported), compute capability **12.0 (sm_120)** |
| RAM | 64 GB |
| Storage | NVMe SSD (model ≈ 88 GB) |
| Engine | llama.cpp (Unsloth `b10840-mix-d5c17a0`, `cuda12-portable` build) + manually added CUDA 12.8 runtime DLLs |
| Model | Qwen3.8-Flash-Next GGUF, 176.9B total params (incl. a 51.2B N-gram table) |

## Key takeaways

1. **Layout beats bit-width.** Whether the big weight table (the 35.76 GiB N-gram table) gets its own shard decides if the whole approach is viable on this machine;
2. **Guard against two silent failures** — a missing CUDA runtime silently falls back to CPU; VRAM overflow silently spills over PCIe (30× slower, no error);
3. **MoE + CPU-resident experts = speculative decoding is a net loss** — verification batches read the union of activated experts, amplifying memory traffic;
4. **The KV cache is surprisingly small** (only 33 KiB per token) → give VRAM to context and hand experts back to RAM;
5. **`--n-cpu-moe` is the one knob**, and it has a cliff (28 layers collapses in this case).

## FAQ

**Q: Why not NVFP4? The RTX 5090 is Blackwell.**
A: Three reasons: ① **no NVFP4 quant exists** for this model (full enumeration of both publishers: 164 files, zero FP4 hits); ② generation is **memory-bandwidth bound**, not compute bound — FP4 tensor cores accelerate matmul, not the per-token expert weight reads; ③ NVFP4 is ≈**4.5 bpw** effective, *larger* than the 3.57 bpw body we run — on a machine where the model exceeds total memory that means reading ~26% more bytes per token. See [model-reference §2.4](./docs/model-reference.md).

**Q: Why llama.cpp and not vLLM / TensorRT-LLM?**
A: Only llama.cpp offers `--n-cpu-moe` (per-layer control of which experts stay in RAM) plus mmap demand paging — the two prerequisites for running an 88 GiB model on 24 GB VRAM. See [model-reference §2.5](./docs/model-reference.md).

**Q: How large a context can I run on my GPU?**
A: Use the formula in the [porting guide](./docs/porting-guide.md): `VRAM ≈ 4.4 + (48−ncmoe)×1.03 + ctx×33KiB + compute buffer`. On a 32 GB card, roughly ncmoe 34–36 with 256K.

**Q: Can generation go faster?**
A: Two levers: ① a smaller body quant (AD-3.84bpw, measured +5–9.5%); ② more VRAM so more expert layers fit (≈+1–3% per 2 layers). MTP speculative decoding is a **net loss** in this "experts live in RAM" configuration — leave it off.

**Q: Will it blow up my RAM?**
A: Single instance peaks at 82–96%, stable. The real risk is running **multiple llama-server instances** — each demands 13–16 GiB of VRAM and two will always overflow. The bench tool enforces single-instance discipline.

## Disclaimer

- All numbers are **measured on one machine**; results vary with driver and build versions;
- Model weights belong to their respective publishers — this repo documents deployment methodology only;
- The benchmark script **kills `llama-server` processes**; do not run it while other inference services are active.

## License

MIT (applies to this repository's docs and scripts only)
