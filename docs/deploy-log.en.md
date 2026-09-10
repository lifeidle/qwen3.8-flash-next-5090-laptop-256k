# Running Qwen3.8-Flash-Next 177B Locally on an RTX 5090 Laptop (24 GB VRAM + 64 GB RAM) — Quant Selection, Three-Tier Memory Split, and 256K Context

**中文版 / Chinese version → [deploy-log.zh.md](./deploy-log.zh.md)** ｜ [Repository home](../README.md)

> **Hardware:** RTX 5090 Laptop GPU (24 GB VRAM) / 64 GB RAM / NVMe SSD
> **Model:** Qwen3.8-Flash-Next, GGUF quantized — 176.9B total parameters, of which **51.2B form an N-gram embedding table**; 125.7B MoE body; ~3B active per token
> **Results:** **21–28 tok/s** generation (community reference on comparable hardware: ~11 tok/s), context extended from 8K to the **full 256K**
> **Date:** September 2026

---

## 1. Why this is an interesting problem

Qwen3.8-Flash-Next is an unusual hybrid: a 176.9B-parameter MoE model in which **51.2B parameters are a single N-gram embedding table** (used for speculative decoding and multi-token prediction), plus 48 MoE layers (512 experts, top-10) with a GatedDeltaNet hybrid attention scheme.

For local deployment that produces a brutal arithmetic problem:

```
Total model ≈ 88 GiB (at 4.27 bpw)
VRAM 24 GiB + RAM 64 GiB = 88 GiB    ← exactly the model size; not one GiB to waste
```

One wrong allocation decision and the machine falls into a paging death spiral. But the same structure is also an opportunity: the model's three parts — N-gram table, expert weights, attention — have completely different access patterns, so in principle each can live where it belongs:

| Part | Access pattern | Ideal location |
|---|---|---|
| N-gram table (35.76 GiB) | 16 rows read per token ≈ a few KB | **SSD**, mmap'd on demand |
| Expert weights (~1 GiB/layer, 10 of 512 active per token) | A few experts per layer per token | **VRAM first**, overflow to RAM |
| Attention / resident tensors / KV | Read in full every token | VRAM |

Whether this three-tier layout works depends on a detail that is easy to overlook: **the GGUF shard layout**. That turned out to be the single most important discovery of the whole project.

---

## 2. Choosing the quantization: three rounds of elimination — layout beats bit-width

Two publishers dominate the available quants: **AtomicChat** (the "AD" series) and **unsloth** (the "UD" series). Five candidates in total:

| Candidate | Total size | Shards | N-gram table placement |
|---|---|---|---|
| AtomicChat AD-5.00bpw-Q5_K_M-M64 | 102.93 GiB | 33 | **dedicated shard** (50.66 GiB, 8.5 bpw) |
| AtomicChat AD-4.27bpw-Q4_K_M-M64 | 88.03 GiB | 33 | **dedicated shard** (35.76 GiB, Q5_1 6 bpw) |
| AtomicChat AD-3.84bpw-IQ4_XS-M64 | 79.10 GiB | 28 | **dedicated shard** (35.76 GiB, same) |
| unsloth UD-IQ4_XS | 87.25 GiB | 3 (0.01 / 46.41 / 40.83) | **mixed in** (IQ4_NL ≈26.8 GiB sharing a shard with experts) |
| unsloth UD-Q3_K_XL | 83.80 GiB | 3 (0.01 / 46.55 / 37.25) | presumably mixed |

### Round 1: unsloth is out — for layout reasons, not bit-width

unsloth's 3-shard layout puts the N-gram table and expert weights **in the same 46 GiB shard**. On a 128 GB machine that's irrelevant (everything fits anyway). On 24 GB + 64 GB it is fatal:

- If *any* tensor in that shard is assigned to the GPU, the **entire 46 GiB mapped region is pinned in RAM** (cannot be evicted), requiring up to 89.6 GiB resident — instant memory exhaustion;
- Avoiding that pinning requires keeping *all* experts on the CPU — which throws away the VRAM tier entirely and drops you back to baseline speed.

**Lesson 1: when the model is larger than your RAM, the shard layout matters far more than the quant bit-width.** unsloth's default layout targets 128 GB unified-memory / large-RAM machines.

### Round 2: within the AD family, 4.27 bpw becomes the daily driver

All three AD variants share the same 6 bpw N-gram table; they differ in body quantization (and table size):

| Variant | Body size | Body bpw | Quality metric |
|---|---|---|---|
| 5.00bpw | 51.6 GiB | 3.57 | — |
| **4.27bpw** | 51.6 GiB | **3.57** | **KLD 0.0842 / Top-1 89.49% (publisher-measured)** |
| 3.84bpw | 42.7 GiB | ≈2.92 | not published |

The 4.27 and 5.00 variants share an identical body (3.57 bpw) and differ only in table quantization; 3.84 compresses the body to ≈2.92 bpw — faster, but with no published quality evidence. **The daily driver is 4.27** (KLD-backed); 3.84 was later downloaded and benchmarked as a speed alternative (section 8).

### Round 3: why NVFP4 is not on the list

The RTX 5090 is Blackwell and supports NVFP4 tensor cores natively, so "should we use FP4?" is a fair question. The answer is no, for three reasons:

1. **It doesn't exist for this model** — a full enumeration of both publishers' repositories (AtomicChat: 104 files; unsloth: 60 files) shows only K-quant / I-quant variants; zero hits for `fp4` / `nvfp`;
2. **Compute isn't the bottleneck** — NVFP4 accelerates matrix multiplication, whereas generation here is limited by **memory bandwidth** (each token reads its activated expert weights, saturating ~20–25 GB/s). Compute has plenty of headroom: prefill reaches 89–139 tok/s;
3. **Its effective bit-width is larger** — NVFP4 = 4-bit values + block scales ≈ **4.5 bpw**, versus the 3.57 bpw body we already run. On a machine where the model exceeds total memory, that means reading ~26% more bytes per token — strictly slower.

Full argument and supporting data: [model-reference §2.4](./model-reference.md).

---

## 3. Engine and runtime: two "silent failure" traps

### Trap 1: the engine must support Blackwell (sm_120) natively

The 5090 is sm_120; older llama.cpp CUDA builds don't contain the kernels. I used Unsloth's `b10840-mix` build (`cuda12-portable` variant, whose `supported_sms` list includes 120). **Verify** by checking `llama-bench --help` for `-ncmoe` and reading `BUILD_INFO` for the sm list.

### Trap 2 (the nastiest): a "portable" build that doesn't include the CUDA runtime

On the first benchmark run, `llama-bench` reported `backend = CPU` — **with no error message at all**. Inference was silently running on the CPU. Diagnosis:

1. `ctypes.WinDLL("ggml-cuda.dll")` → "module or one of its dependencies not found";
2. A small hand-written PE import-table parser showed `ggml-cuda.dll` statically imports `cudart64_12.dll` and `cublas64_12.dll` — **neither exists on the machine**;
3. The "portable" package only embeds the CUDA **kernels** (a 338 MB `.nv_fatb` section) into the DLL; the runtime libraries are still your responsibility.

**Fix:** instead of installing the multi-GB CUDA Toolkit, pull NVIDIA's official Windows wheels from PyPI (versions align exactly with the CUDA 12.8 build):

```
nvidia-cuda-runtime-cu12==12.8.90  → cudart64_12.dll
nvidia-cublas-cu12==12.8.5.5       → cublas64_12.dll / cublasLt64_12.dll / nvblas64_12.dll
```

A wheel is just a zip; extract the four DLLs next to the engine. After the fix:

```
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090 Laptop GPU, compute capability 12.0, VRAM: 24435 MiB
backend = CUDA
```

**Lesson 2: the most dangerous failure in GPU deployment isn't an error — it's a silent CPU fallback.** After every engine or model change, run `llama-bench` on a 1 MB toy model and confirm `backend = CUDA` and the `compute capability` line before doing anything else.

---

## 4. Downloading 88 GiB through a throttled mirror

The model was pulled through hf-mirror. It applies a **rolling token bucket**: after 2–4 GB of continuous download, throughput collapses to zero (0.03–0.5 MB/s), recovers by itself after 5–10 minutes of idling, then runs at 14.5 MB/s single-connection and 20–26 MB/s multi-connection. A control test against a different mirror (a steady 26 MB/s) confirmed site-side throttling rather than a local network problem.

What worked:

- A purpose-written multi-threaded chunked downloader: **fixed 256 MiB chunks** (independent of thread count, so resuming with a different thread count never writes misaligned data), resume support, unlimited retries with exponential backoff capped at 60 s;
- ModelScope hosts a mirror of the same repo but benchmarked much slower (1.3 MB/s multi-threaded) — rejected;
- After downloading, every file was verified against HuggingFace's LFS sha256 (note: the API field is `lfs.sha256`, and requests need a browser User-Agent or they return 403).

All 33 shards verified — not a byte off.

---

## 5. Tuning: the `--n-cpu-moe` sweep, and one painful data-contamination incident

### 5.1 The one knob that matters: `--n-cpu-moe N` (keep the experts of the first N layers on CPU)

Sweep with `llama-bench` (`-ngl 99 -fa on -p 512 -n 128`):

| ncmoe | Expert layers in VRAM | pp512 | **tg128 (generation)** |
|---|---|---|---|
| 44 | 4 | 53.5 | 22.57 |
| 40 | 8 | 92.3 | 24.99 |
| 36 | 12 | 79.4 | 17.78 |
| 34 | 14 | 60.2 | 24.89 |
| **32** | 16 | 55.7 | **26.88 ← best** |
| 28 | 20 | 20.6 | **4.65 ← the cliff** |

At 28 layers VRAM overflows and the driver **silently** moves the excess into system RAM over PCIe — throughput collapses. The shape of this curve (gentle decline, then a cliff) is the signature of every VRAM-tiering setup.

### 5.2 Painful lesson: process leakage invalidated an entire round of data

Mid-tuning, "mysteriously slow" results kept appearing: MTP dropping to 1.2 tok/s, 16K context at 1.4 tok/s. The root cause turned out to be **a process-kill command in my script silently failing** (a Git Bash argument-conversion issue), so `llama-server` instances accumulated — up to four at once. Each instance demands 13–16 GiB of VRAM; together they overflowed VRAM, pushed RAM to 99%, and every measurement taken during that period was junk.

The **single-instance test discipline** I adopted (now a reusable Python driver):

1. Kill processes via Python `subprocess` with an argument array (bypasses shell escaping), then **verify with `tasklist` that the count is zero**;
2. One dedicated process per test case: start → wait for ready → 2 warm-up requests → measure → kill → verify;
3. Sample RAM every second throughout, and report the peak within each case's time window;
4. Only trust **warm-state** numbers (first request after a cold start is consistently slower).

### 5.3 Memory behaviour: mmap working set > physical RAM is by design, not a bug

After the fix, the memory curve of a healthy config:

```
idle          52.6 GiB free (17%)
model load  → 30.9 GiB free (51%)   ← 32 layers of experts enter RAM
generation  → 12.3 GiB free (80%)   ← mmap page cache grows (reclaimable, but visible)
steady      →  9.5 GiB free (85%)   ← this is normal for this setup
```

**An mmap working set of 67.4 GiB against 64 GiB of physical RAM is intrinsic to this approach** (it's the precondition for "table on SSD"); the OS balances it by evicting cold pages. With a single instance, RAM peaks stay at 82–96% and never run away; **every "99% memory" event came from multiple instances stacked on top of each other.**

Two safety fuses worth setting:

1. **NVIDIA Control Panel → "CUDA - Sysmem Fallback Policy" → Prefer No Sysmem Fallback.** By default, VRAM overflow is silently absorbed by system RAM over PCIe (30× slower, no error). With this setting, you get an explicit OOM instead — turning an invisible failure into a visible one;
2. **Enlarge the page file** (9 GiB → 32–64 GiB): the commit limit is RAM + page file; give peaks some headroom.

---

## 6. MTP speculative decoding: mechanically negative here — disabled

The model ships an MTP speculative-decoding head (2.60 GiB shared / 3.85 GiB self-contained), advertised at 1.3–1.7× speedup in the official setup. Measured single-instance (draft acceptance 50–63%, confirming it really runs):

| ctx | No MTP | MTP (ncmoe=35) |
|---|---|---|
| 8192 | 20.08 | 21.44 (shared) / 18.37 (self-contained) / 19.20 (self-contained + forced into VRAM) |
| 32768 | 21.66 | 14.55 |

**A wash at 8K, 33% slower at 32K.** The reason is structural: in an MoE model, a speculative verification pass forward-runs 3 tokens at once, and the activated experts are the **union** of those 3 tokens' experts — every verification step reads 2–3× the expert bytes from RAM. Divided by the 2.1-token acceptance rate, per-token traffic actually *increases* by 1.4–3×. **Speculative decoding only pays off when all experts reside in VRAM** (where reading more weights in one batch is free); the official 1.3–1.7× numbers come from exactly such 128 GB everything-fits setups.

Conclusion: on a tiered layout with experts in RAM, MTP is a net loss. Disabled. The self-contained head and `-ot` overrides forcing the draft head's experts onto the GPU were both tried — no improvement.

---

## 7. Long context: give the VRAM back to the KV cache

### 7.1 A counter-intuitive fact: the KV cache is tiny

The VRAM ledger from a verbose log (ncmoe=32, ctx=8192):

| VRAM item | Size |
|---|---|
| Model weights (16 expert layers + attention + embeddings) | 20.87 GiB |
| **KV cache** | **0.26 GiB** |
| Recurrent state (GatedDeltaNet, **fixed regardless of context**) | 0.11 GiB |
| Compute buffer | 0.64 GiB |

This model has only **2 KV heads** (extreme GQA) and mostly GatedDeltaNet layers with fixed-size state — each token adds just **33 KiB** of KV. Even at the full 256K context, KV is only 8.25 GiB.

**The context limit was never about KV — it was about VRAM being occupied by experts.** The fix follows directly: move 2 expert layers back to RAM (freeing 2.06 GiB) and the context window doubles.

### 7.2 The see-saw, measured all the way to the full 256K

| Config | Context | ncmoe | Generation | After 8K-token prefill | RAM peak |
|---|---|---|---|---|---|
| Baseline | 32K | 32 | 21.66 | — | 83% |
| C3 | **64K** | 34 | 24.61 | **26.88** (prefill 139 tok/s) | 96% |
| **C2b** | **256K (full)** | **42** | **23.43** | **22.81** (prefill 111 tok/s) | 95% |
| C1 | 256K (full) | 48 (all experts in RAM) | 17.67 | 22.50 (prefill 89 tok/s) | 94% |

Three observations:

1. **The full 256K context works, with no speed penalty** — the "sparse attention + DeltaNet" design pays off exactly when the three-tier split gives the KV cache room;
2. **Generation speed doesn't degrade once a large KV cache is built** (prefilling 10k tokens takes ~90 seconds);
3. At 256K the prefill **compute buffer** alone needs 1.98 GiB, which kills one configuration tier (ncmoe=40 OOMs, 42 works) — **every context step up requires re-checking the compute buffer.**

### 7.3 The final frontier (speed ↔ context, measured)

| Use case | Config | Measured |
|---|---|---|
| Speed-first | ncmoe=34, ctx=64K | 24.6–28.2 tok/s |
| **Balanced (chosen)** | **ncmoe=42, ctx=256K** | **22.8–23.4 tok/s** |
| Multi-slot | ncmoe=48, ctx=256K | 17.7–22.5 tok/s, 8.4 GiB VRAM spare |

The see-saw rule: **+2 expert layers back in RAM ≈ 2.06 GiB VRAM freed ≈ context doubles**, and speed decays gently along the whole curve — because the bottleneck is *how many expert bytes a token must read*, and moving layers only changes *where* those bytes live.

---

## 8. AD-3.84bpw comparison: +5–9.5% speed, quality unverified

The 3.84 variant (79.10 GiB / 28 shards, all sha256 verified) with a 17% smaller body:

| Config | 3.84 | 4.27 | Delta |
|---|---|---|---|
| ncmoe=34, ctx=64K | 25.59 (28.24 after large KV) | 26.88 (after large KV) | **+5%** |
| ncmoe=42, ctx=256K | 18.66 (24.97 after large KV) | 23.43 / 22.81 | **+9.5%** |

Warm-state comparison: 3.84 is 5–9.5% faster — the right direction, but below the theoretical 17%, indicating the bottleneck is a mix of "expert bytes + memory bandwidth + page misses". Bonus: the smaller body leaves more VRAM headroom (4.27 OOMs at ncmoe=40 with 256K; 3.84 doesn't).

**But a ≈2.92 bpw body has no published quality evidence.** My approach: keep both, run real tasks on each for a few days before choosing; default to 4.27 for quality-sensitive work.

---

## 9. Results summary

| Metric | Baseline | Final |
|---|---|---|
| Generation | ~11 tok/s | **21.7 (32K) / 24.6–28.2 (64K) / 23.4 tok/s (256K)** |
| Usable context | 8K (default) | **262,144 (full model length)** |
| Quant | — | AD-4.27bpw (primary) / AD-3.84bpw (speed option) |
| Engine | — | llama.cpp (Unsloth b10840-mix, sm_120) + CUDA 12.8 runtime DLLs |
| Stability | — | single instance, RAM peak 82–96%, no runaway |

Launch command (balanced):

```
llama-server -m <model> -ngl 99 --n-cpu-moe 42 -fa on -fit off \
  -c 262144 -np 1 --jinja --no-warmup
```

---

## 10. Reusable lessons

1. **Layout beats bit-width.** When the model exceeds RAM, check the GGUF shard structure (does the big weight table get its own shard?) before comparing quant levels;
2. **Watch for two kinds of silent failure:** missing CUDA runtime → silent CPU fallback; VRAM overflow → silent PCIe spill. Countermeasures: verify `backend=CUDA` with a toy model, and set "Prefer No Sysmem Fallback" so overflow becomes an explicit error;
3. **MoE + CPU-resident experts = speculative decoding is a net loss.** Verification batches read the union of activated experts, amplifying memory traffic. MTP only helps when everything is in VRAM;
4. **KV is far smaller than you think.** With GQA plus hybrid attention (fixed recurrent state), context costs next to nothing — give VRAM to the KV cache;
5. **The see-saw formula makes planning trivial:** 2 expert layers moved between VRAM and RAM ≈ 1 GiB ≈ 32K of context capacity;
6. **Data hygiene:** multi-instance leakage invalidates whole benchmark rounds. Verify process counts on every start/stop, codify the driver, trust only warm-state numbers;
7. **Mirror throttling is normal:** chunked resume + unlimited retry + long backoff beats switching sources;
8. **Never skip sha256 verification** — download tools rename files and shards go missing silently; HuggingFace's LFS hashes are a free source of truth.

---

## Appendix: key parameters

| Parameter | Value | Notes |
|---|---|---|
| `-ngl 99` | all layers on GPU | per-tensor split is then decided by ncmoe |
| `--n-cpu-moe N` | 32–48 | keep first N layers' experts on CPU; 32 = speed-first, 42 = 256K full context, 48 = multi-slot |
| `-fa on` | Flash Attention | required for long context |
| `-fit off` | disable auto-fit | auto-fitting mis-measures on this architecture (worse with a draft head); manual control is more reliable |
| `-c N` | 32768–262144 | KV costs only 33 KiB/token — be bold |
| `--no-warmup` | skip startup warm-up | cold warm-up is very slow on large models; let the first request pay for it |

*— End —*
