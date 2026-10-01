# Strata Engine Log: 25 → 93.5 tok/s — MTP Repair, Three Quants Compared, GPU+CPU Both Saturated

> Continuation of the [llama.cpp deploy log](./deploy-log.en.md) (22-25 tok/s era). This is the full measured record after switching to the third-party [Strata](https://github.com/Niko1221/Strata) engine v0.1.27→0.1.28.
> Same hardware: RTX 5090 Laptop 24GB VRAM + 64GB RAM + Core Ultra 9 275HX.

## TL;DR

| Quant (Qwen3.8-Flash-Next GSQ-RCO) | Effective | llama.cpp era | Strata (post-MTP-fix, tuned) | Role |
|---|---|---|---|---|
| Coder IQ1_M (256 experts) | 1.89 bpw | — | 62.5 tok/s | lowest RAM (23.4 GB arena) |
| Q2_0 full (512 experts) | ~2.2 bpw | 25 tok/s (3.84bpw AD quant) | **93.5 tok/s** | fastest |
| IQ3_XXS full (512 experts) | ~3.1 bpw | — | **77.4 tok/s** (74.4 at 256K+Vision) | quality-first |

**3.7x speedup**, achieved on the 2-bit Q2_0 — with GPU and CPU utilization both near 100%.

## 1. The headline finding: GPU and CPU saturate together, for the first time

The llama.cpp era chronic pain: **CPU pinned at 100% (single-core bottleneck) while GPU idled at 20-40%**. The 24 GB of VRAM sat mostly waiting.

Strata's three-tier architecture fixes this structurally:

```
VRAM (24 GB): hot-expert cache (~9,300-13,200 slots) + 32K KV hot window + MTP drafter + vision
RAM (64 GB):  full expert arena (pinned) + cold KV (streaming)
SSD:          PLE n-gram table (mmap, on demand)
```

- **Hot experts on the GPU**: pre-filled by routing frequency, 92-98%+ hit rate
- **Cold experts on a CPU pool**: 23 worker threads + the host thread, refilled on miss
- **The MTP drafter runs on the GPU**, overlapping the main model pipeline
- **Result: the CPU pool and the GPU saturate simultaneously and never wait on each other.** This — not any single knob — is why 25 became 93.

## 2. The dead-MTP mystery: 20 of 31 weight files were corrupted downloads

### Symptom
Strata ships with MTP (speculative decoding) on; upstream reports 2.4-3.2 tokens/round. We measured:

```
drafts accepted 0 of 0   ← the drafter never even proposed, on defaults
drafts accepted 0 of 765 ← with windows forced open, 765 proposals, 100% rejected
```

### How it was found
1. Sampling params / temperature / context → ruled out
2. Coder's 256 experts vs the drafter → ruled out (full 512-expert model still 0)
3. Reading the engine source (the installer ships full C++/CUDA sources) revealed that the second zero in `drafts accepted 0 of 0` is the **offered** count — the window T was always 1, so the drafter was never given a chance. The earlier reading ("all drafts rejected") was wrong
4. Forcing windows open (`spec_min_p=0`) → `0 of 765`: every proposal was wrong
5. **Self-compiled the engine with diagnostic probes** (2-minute build): `dprob=NaN, drafts=0,0,0` — the draft layer's forward pass produced NaNs, argmax pinned at 0
6. Per-tensor inspection: `dense.bin` turned to ~2.1e37-magnitude garbage from the 11th tensor on
7. Root cause: **20 of the 31 MTP weight files did not contain weights at all — they contained the beginning of the safetensors shard (JSON header + unrelated data)**

### Root cause
`tools/mtp_fetch.py` pulls the mtp.* tensors from a BF16 checkpoint via HTTP Range requests. **Some mirrors ignored the Range header** (returning 200 with the whole file); the tool kept the first n_bytes — which is the shard's header. The manifest's sha256 hashes "the bytes that were downloaded", which **cannot catch this failure mode**.

### The fix
1. Detection: a real tensor's raw file does NOT start with `[u64 header_len][b'{"']` (see `tools/check_dense.py` / `tools/refetch_mtp.py`)
2. Re-fetched the 20 bad files (~110 MB; the two 5 GB expert tensors were fine)
3. Re-packed → all 29 tensors 0 NaN/Inf → **MTP came alive immediately**

| Stage | drafts accepted | Speed |
|---|---|---|
| Before the fix (windows forced) | 0 of 381 (0.0%) | 42.7 tok/s |
| **After the fix (windows forced)** | 62 of 195 (31.8%) | 69.9 tok/s |

**Lesson: download tools must verify the Range request actually returned 206 (check `Content-Range`). A sha256 over "the bytes you got" proves nothing.**

### Post-fix MTP acceptance rates
- Q2_0: 44-54% (spec_min_p=0.5)
- IQ3_XXS: 59-71% (best drafter of the three)

## 3. The counter-intuitive knob: higher spec_min_p is faster (on IQ3)

The speculative window T is gated by the previous draft's confidence (`spec_min_p`). The two models peak at opposite ends:

| spec_min_p | Q2_0 (mediocre drafts) | IQ3_XXS (excellent drafts) |
|---|---|---|
| 0.20 | — | 60.4 tok/s (collapse) |
| 0.30 | **93.5 (peak)** | 66.8 |
| 0.45-0.5 | ~90 | 79.4 |
| **0.70** | ~87 | **83.6 warm / 74.4 cache-busted (peak)** |
| 1.0 (MTP off) | ~59.7 | 63.7 |

**Why**: with an excellent drafter the engine can afford high standards — it only verifies high-confidence windows, and wrong drafts nearly vanish (70.8% acceptance). With a mediocre drafter you must cast a wider net. **Re-sweep this knob whenever you change models**; reusing another model's optimum costs 10-15%.

## 4. The 256K context ladder (IQ3_XXS + Vision + MTP)

Strata's KV streaming keeps the KV in pinned RAM and only a 32K-token hot window in VRAM, so the expert cache barely shrinks:

| Context | Expert slots | KV in RAM | Speed (median) | Note |
|---|---|---|---|---|
| 32K | 9,588 | 0 (all VRAM) | **77.4 tok/s** | fastest |
| 65K | 9,559 | 0.77 GiB | 65.7 | |
| 128K | 9,487 | 1.55 GiB | 64.5 | prefill 2,527 tok/s (14K tokens in 6.7 s) |
| **256K** | 9,350 | 3.09 GiB | **65.1** | ✅ setup refuses this combo (conservative); it works |

- 65K/128K/256K are all within noise (~-16% fixed streaming cost, independent of length) → **either 32K (fastest) or straight to 256K**
- MTP stayed healthy throughout (46-59% acceptance); the expert cache lost only 2.5%
- On 64 GB RAM, 256K is comfortable: 42.9 GB pinned arena + 3.1 GB KV + OS

## 5. Vision

- mmproj BF16 encoder, 0.9 GB; GPU mode reserves ~1.4 GB VRAM (expert cache shrinks accordingly)
- ≤1,024 tokens per image (a 640×480 photo ≈ 300); 0.1-0.5 s/image
- Two config edits, no re-setup: add `--vision` to the engine args (bare flag) and a top-level `"vision": {...}` entry
- **256K of context fits 250+ images in one conversation**

## 6. Final configuration (IQ3_XXS · 256K · Vision · MTP)

```
--pack packs/iq3_xxs --expert-cache auto --prefill auto
--spec 4 --spec-min-p 0.7 --mtp <rt-dir>
--max-context 262144 --kv int8 --kv-resident 32768
--vision --vram-reserve-mib 700
```

**Measured: cache-busted median 74.4 tok/s (84.0 server-side on a single request), MTP acceptance 70.8%, GPU+CPU both saturated.**

Tried and rejected: spec 6 (verification cost > gain, -8%); k8v4 KV (drops streaming cost but -14.6% expert slots, a wash); pcie_frac 0.35/0.75 (default 0.55 wins); spec_min_p 0.2 (-18%).

## 7. Checklist for newcomers

1. After installing, verify MTP first: `drafts accepted` must be non-zero. `0 of 0` means check the drafter weights for corrupted downloads (section 2)
2. Re-sweep `spec_min_p` for every model — the peak moves with draft quality
3. At ≥65K context use KV streaming (`--kv-resident 32768`); setup is conservative for some combos, test and widen
4. Bust the prompt cache in benchmarks (same text gets reused and inflates tok/s — append a random suffix)
5. When throughput is low, first check whether GPU and CPU are BOTH saturated: one CPU core pegged + idle GPU is an engine-architecture problem, no knob will fix it
