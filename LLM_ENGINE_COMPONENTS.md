# LLM Engine Components - Paper Reference Map (2024-2026)

> **Target engine:** Mellum2.1 (12B MoE, 2.5B active, 64 experts, top-8, 28 layers, hidden=2304, GQA 32Q/4KV, SWA 3/4 layers, MTP head, 131K context, YaRN, RoPE). **Platform:** Windows + Vulkan (RDNA4/RDNA2/Arc iGPU). **Scope:** single-user CLI, MoE-first, expert disk paging v0.1, correctness-first against llama.cpp oracle.

---

## Table of Contents

1. [Core Components (what you actually build)](#1-core-components-what-you-actually-build)
2. [Papers by Component](#2-papers-by-component)
3. [What Mellum2.1 Actually Uses (build only these)](#3-what-mellum21-actually-uses-build-only-these)
4. [What to Skip for v0.1](#4-what-to-skip-for-v01)
5. [Gap Papers (no direct paper, build from adjacent systems)](#5-gap-papers-no-direct-paper-build-from-adjacent-systems)
6. [Citations Note](#6-citations-note)

---

## 1. Core Components (what you actually build)

For a local single-model MoE engine, the component list is much shorter than the full PRISM taxonomy. Here is the real build list in dependency order, each with its engine concern and the papers that inform it.

| # | Component | Engine Concern | Why It's Needed | Mellum2.1 Relevance |
|---|---|---|---|---|
| 1 | **GGUF loader** | Parse GGUF, map tensors by name/offset, verify shapes | Model entry point | Mellum2.1 ships GGUF (7GB Q4_K_M). Loader must dump tensor list and match llama.cpp's tensor names/shapes. |
| 2 | **Tokenizer** | BPE/byte-level tokenization, matching llama.cpp id-for-id | Gate 0 - tokenizer bugs look like model bugs | JetBrains tokenizer (from HuggingFace repo). Must produce identical ids to llama.cpp on fixed corpus. |
| 3 | **CPU reference forward (f32)** | Full forward pass in float32, greedy decode | Oracle for all validation (Gates A/B/C) | Reference against which Vulkan kernels are validated, and against llama.cpp for Gate A/C. |
| 4 | **MoE router** | For each token: score all 64 experts, pick top-8, compute gating weights | Mellum2.1 is MoE - this is the heart | 64 experts, top-8 per token. Must sort selected experts by expert ID before weighted sum (fixes 2607.28097 divergence). |
| 5 | **Expert FFN** | For each selected expert: load weights, run FFN, produce output | MoE compute core | Each expert is a feed-forward network. Weights loaded from disk or cached in RAM. |
| 6 | **Expert store + LRU hot cache** | File-offset addressed expert weights. LRU cache with byte budget. Resident mode = all in RAM. | Mellum2.1 won't fit fully resident - paging is mandatory | **This is v0.1.** Same code path, different budget. Track hit/miss counts. |
| 7 | **KV cache (contiguous)** | Allocate contiguous buffer sized to context. One session. No paging for v0.1. | Store K/V per layer for autoregressive decode | GQA 32Q/4KV. Sliding window attention on 3/4 layers. |
| 8 | **Attention** | GQA matmul + sliding window attention where applicable | Core transformer operation | 4 KV heads. 3 of 4 layers use SWA (1024 window). 1 of 4 uses full attention. |
| 9 | **RMSNorm** | Pre/Post normalization per the Mellum2.1 config | Standard transformer norm | Check config for exact variant. |
| 10 | **RoPE positional encoding** | Apply rotary embeddings to Q/K | Positional information | Mellum2.1 uses RoPE (YaRN-extrapolated to 131K). Check exact base dim/freq factor from config. |
| 11 | **MTP head (optional, v0.1 OUT)** | Multi-token prediction head | JetBrains built-in draft for speculative decoding | Present in architecture but "coming soon" for vLLM. Skip for v0.1. |
| 12 | **Greedy argmax decoder** | Pick token with highest logit | Base sampling for gate validation | Start with greedy only. Other samplers later. |
| 13 | **Vulkan backend kernels** | GPU implementations of matmul, rmsnorm, rope, attention, router, expert FFN, argmax | Target hardware (RDNA4/RDNA2/Arc) | Each kernel diffed against CPU reference before end-to-end use. |
| 14 | **CLI (prompt in, tokens out, gate report)** | Single-file interface for validation | No server needed for v0.1 | Outputs gate report to file. |
| 15 | **Gates harness (0, A, B, C)** | Validation framework with defined pass criteria | Correctness enforcement | Gate 0=tokenizer, A=resident CPU vs llama.cpp, B=paged vs resident bit-identical, C=paged vs llama.cpp. |
| 16 | **Telemetry** | Per-token timing, expert cache hit/miss counts | Basic observability | Enough for v0.1. No Prometheus/OpenTelemetry. |

---

## 2. Papers by Component

Each component below lists the most relevant 2024-2026 papers. Papers are **leads, not facts** - verify against llama.cpp behavior.

### 1. GGUF Loader

| Paper/Resource | When | Relevance |
|---|---|---|
| GGUF format spec (ggml/llama.cpp) | 2023+ | The format itself. llama.cpp's GGUF parser is the reference. |
| llama.cpp source (ggml-tensor, gguf.c) | ongoing | Actual parsing code. Mellum2.1 GGUF on HuggingFace loads with llama.cpp. |
| FORGE (2610.16890) | 🟢 Oct 2026 | Radical CAS - regenerate weights from seeds. Not needed but shows where storage is heading. |
| Kilobyte Models (2608.00860) | 🟢 Aug 2026 | Seed+latent models. Far end of the "recipe not table" direction. |

**Build approach:** Parse GGUF directly (it's a simple keyed-tensor format). Validate tensor names and shapes against what llama.cpp expects. No paper to read - use llama.cpp source as reference.

### 2. Tokenizer

| Paper/Resource | When | Relevance |
|---|---|---|
| GraphTok (2610.13325) | 🟢 Oct 2026 | Tokenization off the host as CUDA graph. Interesting but CUDA-only, for server throughput. For v0.1, just match llama.cpp. |
| "Are you going to finish that?" (2601.23223) | 🟢 Jan 2026 | Partial token problem study. Token healing shows mixed effectiveness. Relevant if you add token healing later. |
| BPE exploitable gaps (2607.01239) | 🟢 Jul 2026 | Safety/correctness argument for byte-fallback. |
| How BPE Creates Exploitable Gaps (2607.01239) | 🟢 Jul 2026 | Same as above. |

**Build approach:** Use the exact same tokenizer as llama.cpp (likely SentencePiece or HuggingFace tokenizers with the same vocab file). Gate 0: identical token IDs on fixed corpus. This is a test harness problem, not a research problem.

### 3. CPU Reference Forward (f32)

| Paper/Resource | When | Relevance |
|---|---|---|
| llama.cpp reference implementation | ongoing | The practical oracle. Use it to validate your CPU forward pass. |
| Mellum2 Technical Report (2605.31268) | 🟢 May 2026 | Architecture details: 28 layers, hidden=2304, 64 experts top-8, GQA 32Q/4KV, SWA 3/4, MTP head, RoPE + YaRN, Muon-trained FP8. Your spec document. |
| HiNa-MoE (2610.05123) | 🟢 Oct 2026 | Intel AMX MoE kernels. CPU-side optimization ideas (NUMA-aware, matvec->matmul for decode). For later CPU optimization, not v0.1. |
| BITCOS (2609.16338) | 🟢 Sep 2026 | Ternary CPU kernels (AVX-512/Xe2). CPU-side quant ideas. For later. |

**Build approach:** Implement the standard transformer forward pass in C99/C11 float32. Layer structure: attention (QKV projection, GQA, RoPE, attention matmul, output projection) -> RMSNorm -> MoE (router, select top-8 experts, FFN, weighted sum with canonical ordering) -> RMSNorm -> final output projection. Validate layer-by-layer if possible.

### 4. MoE Router

| Paper/Resource | When | Relevance |
|---|---|---|
| Mellum2 Technical Report (2605.31268) | 🟢 May 2026 | Router is a standard gating network over 64 experts, top-8. Exact implementation from paper/config. |
| Expert-aggregation determinism (2607.28097) | 🟢 Jul 2026 | **Critical:** Expert reduction order causes behavioral divergence. Sort selected experts by expert ID before weighted sum. No float atomics. |
| SeqMoE (2609.12978) | 🟢 Sep 2026 | Forecast-driven prefetch+eviction. 96.97% expert hit at 45% residency. Relevant to your LRU cache design. |
| SSD-LLaMA (2609.18110) | 🟢 Sep 2026 | Expert-granularity SSD reads, pinned-RAM transfers. Tiering decoupled from compute. Flagship validation for your expert paging approach. |
| ReMoE (2605.27081) | 🟢 May 2026 | Make routing cache-friendly at train time (+26% expert reuse). Not relevant for inference but shows the reuse principle. |
| Cache-Aware Router Adaptation (2609.04895) | 🟢 2026 | Similar to ReMoE. |

**Build approach:** Router is a linear layer (or small MLP) from hidden state to 64 expert logits. Take top-8, compute softmax over those 8, then for each selected expert: load FFN weights, run FFN on the token, multiply by gating weight, sum. **Sort the 8 selected expert IDs ascending before the weighted sum** to ensure deterministic results regardless of selection order.

### 5. Expert FFN

| Paper/Resource | When | Relevance |
|---|---|---|
| Mellum2 Technical Report (2605.31268) | 🟢 May 2026 | Expert FFN structure from config. Standard MoE FFN. |
| SSD-LLaMA (2609.18110) | 🟢 Sep 2026 | Expert weights loaded from disk. Your expert store uses the same concept (file-offset addressed). |
| SeqMoE (2609.12978) | 🟢 Sep 2026 | Expert caching patterns. |

**Build approach:** Each expert FFN is likely a standard two-layer MLP (up-projection + activation + down-projection). Load weights from the GGUF by name/offset. If paged, load from disk via pread/mmap when not in LRU cache.

### 6. Expert Store + LRU Hot Cache

| Paper/Resource | When | Relevance |
|---|---|---|
| SSD-LLaMA (2609.18110) | 🟢 Sep 2026 | **Primary reference.** Expert-granularity SSD reads. Pinned-RAM transfers. Residency decoupled from compute. Your approach mirrors this but simpler (LRU with byte budget, no prefetch forecasts). |
| SeqMoE (2609.12978) | 🟢 Sep 2026 | **Secondary reference.** 96.97% hit at 45% residency. Forecast-driven prefetch. Your v0.1 LRU doesn't need forecasts - simple LRU with byte budget. |
| Mira (2609.38090) | 🟢 Sep 2026 | HOT+STAGE two-tier cache. Up to 5.71x throughput. Overkill for v0.1 but shows tiered caching patterns. |
| SpecPrefetch (2607.24787) | 🟢 Jul 2026 | Adapter-based expert prioritization. +20% decode on Snapdragon. For later if you add prefetch. |
| When Fancy Eviction Fails (2609.28870) | 🟢 Oct 2026 | **Cautionary.** 20B-token production trace, 14 policies. LRU robust for prefix reuse. Keep scorer simple. |
| ICaRus (2026) | 🟢 Sep 2026 | Identical cache reuse for multi-model inference. LRU caching, learned pinning. Standard engineering patterns. |
| NVIDIA model cache reuse (2026) | 🟢 Sep 2026 | Model cache reuse as operational pattern. Confirms LRU/caching as standard engineering. |

**Build approach:**
```
ExpertStore:
  - experts[] indexed by expert ID (0..63)
  - each expert has: file_offset, size_in_bytes, weight_names[]
  - on access(expert_id):
      if in_lru_cache(expert_id): return cached_ptr
      else:
        read_from_disk(file_offset, size)  // pread or mmap
        if lru_has_room(byte_budget): insert
        else: evict_one_lru()
        return ptr
  - resident_mode: byte_budget = total_expert_size (or large enough)
  - track: hit_count, miss_count, evict_count
```
Use the same code path for resident and paged modes. Only the budget differs.

### 7. KV Cache (Contiguous)

| Paper/Resource | When | Relevance |
|---|---|---|
| PagedAttention (2309.06180) | ⚪ 2023 | Foundational block-based KV. Not needed for v0.1 (contiguous is simpler) but understand the concept for later. |
| SGLang RadixAttention (2312.07104) | ⚪ 2023 | Prefix caching. Not needed for single session. |
| HiSparse (2608.07009) | 🟢 Aug 2026 | Hierarchical window. Fused CUDA-graph kernels. Merged into SGLang. Overkill for v0.1. |
| KVBuffer (2605.19049) | 🟢 May 2026 | IO-aware serving for linear attention. Not relevant (Mellum2.1 is standard attention, not linear). |
| Sliding Window Attention papers | various | SWA is used in 3/4 Mellum2.1 layers. Simple to implement - just mask out positions > window. |

**Build approach:** For v0.1, allocate one contiguous buffer per layer: `[num_layers][context_len][num_kv_heads][head_dim]`. For 28 layers, 4 KV heads, 64 head dim (2304/32=72... check config), context 131K: that's a lot of memory. You may need to limit effective context for v0.1, or use a smaller test context. KV paging (block-based) can come later if needed.

Note: Mellum2.1 has SWA on 3/4 layers with 1024 window. For those layers, you only need to store the last 1024 positions per layer. The remaining 1/4 layers need full context. This affects memory planning.

### 8. Attention (GQA + SWA)

| Paper/Resource | When | Relevance |
|---|---|---|
| Mellum2 Technical Report (2605.31268) | 🟢 May 2026 | GQA 32Q/4KV. SWA on 3/4 layers (1024 window). Full attention on 1/4 layers. This is your spec. |
| Multi-Scale Dequant (2605.13915) | 🟢 May 2026 | Avoid materializing BF16 weights. For quant-aware kernels later. Not v0.1 (you start f32 CPU, then f16/f8 Vulkan). |
| QUILT (2610.11134) / PageWeaver (2610.11201) | 🟢 Oct 2026 | Shared-query sparse prefill. For prefill optimization later. Not v0.1 (single-user greedy). |
| FlashAttention-3 | ⚪ 2024 | GQA/TMA optimizations. Conceptual reference for attention kernel structure, but your Vulkan backend is from scratch. |

**Build approach:**
- Standard GQA: project hidden to Q (32 heads), K (4 heads), V (4 heads). RoPE on Q and K. For each query head, attend over the corresponding KV head (32/4 = 8 query heads share 1 KV head).
- SWA layers (3/4 of them): mask positions where `pos - k_pos > 1024`.
- Full attention layers (1/4): standard causal mask.
- Output: concatenate heads, project back to hidden.

Validate: each attention kernel diffed elementwise against CPU reference.

### 9. RMSNorm

| Paper/Resource | When | Relevance |
|---|---|---|
| llama.cpp GEMMLowp/RMSNorm impl | ongoing | Reference implementation. |
| Mellum2 Technical Report (2605.31268) | 🟢 May 2026 | Confirm norm type from config. Mellum2.1 likely uses RMSNorm (standard for Llama-family MoE). |
| HiNa-MoE (2610.05123) | 🟢 Oct 2026 | AMX RMSNorm patterns for CPU. For later CPU optimization. |

**Build approach:** RMSNorm: `norm(x) = x * rsqrt(mean(x^2) + eps) * weight`. Standard. Validate against CPU reference.

### 10. RoPE Positional Encoding

| Paper/Resource | When | Relevance |
|---|---|---|
| Mellum2 Technical Report (2605.31268) | 🟢 May 2026 | RoPE with YaRN extrapolation to 131K. Base freq dim and factor from config. |
| DaRoPE (2609.34556) | 🟢 Sep 2026 | Data-aware positional encodings. Not relevant - Mellum2.1 uses standard RoPE. |
| GAPE (2605.10414) | 🟢 May 2026 | Gated adaptive positional encoding. Not relevant. |
| RoPE extensions (NTK, YaRN, etc.) | various | YaRN is used for context extrapolation. Read the YaRN paper if you need to understand the scaling, but for inference you just apply the pre-computed frequencies. |

**Build approach:** Apply rotary embeddings to Q and K based on position. Mellum2.1 uses YaRN-scaled RoPE up to 131K. Get base_freq_dim and factor from model config. For inference, you apply the rotation matrix per position - standard. Validate against CPU reference.

### 11. MTP Head (skip for v0.1)

| Paper/Resource | When | Relevance |
|---|---|---|
| JetSpec (2606.18394) | 🟢 Jun 2026 | One-pass parallel tree drafting with MTP. |
| P-EAGLE (2602.01469) | 🟢 Feb 2026 | Parallel-draft EAGLE with scalable training. |
| DFlash (2602.06036) | 🟢 Feb 2026 | Block-diffusion drafter + parallel verify. |
| Mellum2 Technical Report (2605.31268) | 🟢 May 2026 | Mellum2.1 has MTP head as built-in draft for speculative decoding. "Coming soon" for vLLM. Skip for v0.1. |

**Build approach:** Skip. MTP head is present in the architecture but not needed for v0.1. If you add speculative decoding later, this head can serve as the drafter.

### 12. Greedy Argmax Decoder

| Paper/Resource | When | Relevance |
|---|---|---|
| llama.cpp sampler impl | ongoing | Reference for greedy decoding. |
| Latент Distilling (2604.24927) | 🟢 Apr 2026 | LLM exploration via latent distilling. Sampling theory, but not needed for greedy. |
| RL-Guided Adaptive Sampling (2026) | 🟢 2026 | Adaptive sampling for test-time scaling. Not needed for v0.1 (greedy only). |

**Build approach:** `token = argmax(logits)`. Simple. Validate: greedy decode from CPU reference should match llama.cpp greedy decode on same prompt (after tokenizer gate passes).

### 13. Vulkan Backend Kernels

| Paper/Resource | When | Relevance |
|---|---|---|
| RLX (2609.37916) | 🟢 Sep 2026 | One IR/runtime across Vulkan, Metal, WebGPU, CUDA, ROCm. Notable because it targets Vulkan - but you're building custom Vulkan stack (VAiSt/ninfer style), not using RLX. Read for inspiration only. |
| Meganeura (2608.01563) | 🟢 Aug 2026 | Cross-vendor kernel portability. |
| GraphTok (2610.13325) | 🟢 Oct 2026 | Command replay as CUDA graph. You'll have a similar concept for Vulkan (command buffer replay). |
| HiSparse (2608.07009) | 🟢 Aug 2026 | Fused CUDA-graph kernels for hierarchical KV. Fused kernel concept applies to Vulkan. |
| RESOLVE (2610.05683) | 🟢 Oct 2026 | Kernel correctness validation. Methodology for testing kernels. |
| D2K-Bench (2610.03226) / MetaKernelBench (2610.05014) | 🟢 Oct 2026 | Agent-generated kernel reality check. |
| TRANSIT (2610.07593) / MemFerry (2610.09657) | 🟢 Oct 2026 | Zero-copy CPU-GPU transfer. Relevant to your Vulkan buffer management. |
| FusionML (2607.22785) | 🟢 Jul 2026 | Co-execution helps prefill, not decode. Phase-aware bandwidth model. |

**Build approach:** Write Vulkan compute shaders for each operation:
1. Matmul (for projections, FFN, attention)
2. RMSNorm
3. RoPE application (can be fused into matmul or separate)
4. Attention (GQA matmul: Q @ K^T, softmax, @ V)
5. MoE router (linear + top-8 selection)
6. Expert FFN (per expert: load, matmul, activation, matmul, store/accumulate)
7. Argmax (greedy)

Each kernel validated elementwise against CPU reference (f32). Use floating-point tolerance appropriate to your precision (f16 has ~1e-4 relative error vs f32).

For RDNA4 specifically: leverage AMD's matrix cores (if available in Vulkan via extensions), wavefront size 64. For RDNA2: wavefront 64, no matrix cores. For Arc: check Xe driver support.

### 14-16. CLI, Gates, Telemetry

These are engineering, not research. No papers needed.

---

## 3. What Mellum2.1 Actually Uses (build only these)

From the Mellum2 Technical Report (2605.31268) and the Shattered.io analysis:

| Feature | Mellum2.1 Config | v0.1 Action |
|---|---|---|
| **Architecture** | 12B MoE, 2.5B active, 28 layers, hidden=2304 | Build for exactly this |
| **Experts** | 64 total, top-8 activated per token | Router selects 8 of 64. Expert FFN for each. |
| **Attention** | GQA: 32 query heads, 4 KV heads | Implement GQA (8 Q heads per KV head) |
| **SWA** | Sliding window 1024 on 3/4 layers | Implement SWA masking on those layers |
| **Full attention** | 1/4 layers (every 4th) | Standard causal attention |
| **RoPE** | Rotary embeddings, YaRN-extrapolated to 131K | Apply RoPE per position. Get freq params from config. |
| **Norm** | RMSNorm (assumed, verify from config) | Implement RMSNorm |
| **MTP head** | Present (1 future token prediction) | **Skip for v0.1** - "coming soon" even for vLLM |
| **Context** | 131,072 tokens | Plan memory accordingly. May need to cap for v0.1. |
| **Training** | Muon optimizer, FP8 hybrid precision, WHD schedule | Not relevant to inference |
| **Quantization** | FP8 training, but inference via GGUF quants (Q4_K_M etc.) | Use whatever GGUF ships. Q4_K_M is ~7GB. |
| **Activation** | SwiGLU or similar (check config) | Standard FFN activation. Verify from config. |
| **Bias** | Likely no bias (Llama-family convention) | Check config |

**Critical unknown:** Exact FFN structure (hidden multiplier, activation function), exact RoPE parameters (base freq, factor), exact norm epsilon, whether there are biases. **Read the GGUF metadata / config.json to get these.**

---

## 4. What to Skip for v0.1

These are **out of scope** for the corrected v0.1 plan:

| Category | Items | Reason |
|---|---|---|
| **Server infrastructure** | HTTP API, OpenAI compat, async runtime (tokio/axum), continuous batching, multi-session, prefix/radix cache, Paged KV | Single user, single session, CLI only |
| **Decoding acceleration** | Speculative decoding (any form), MTP, self-speculation, draft models, adaptive depth, tree verification | Add after gates pass |
| **Sampling** | Temperature, top-p, top-k, min-p, typical, CD, Gumbel | Greedy only for v0.1 |
| **Backends** | CUDA, ROCm, Metal, OpenVINO, WebGPU | Vulkan only for v0.1 (covers RDNA4, RDNA2, Arc) |
| **Quantization** | Other quant types beyond what GGUF ships | Use Q4_K_M (or whatever Mellum2.1 GGUF provides) |
| **Observability** | Prometheus, OpenTelemetry, structured logging beyond basics | Per-token timing + expert cache hit/miss counts only |
| **Advanced memory** | Paged KV cache, prefix cache, KV compression, span store, latent context memory | Contiguous KV for v0.1 |
| **MoE features** | Hot expert prediction, expert prefetch forecasts, load balancing across experts | Simple LRU cache. No predictors. |
| **Compiler/IR** | MLIR, AOT compilation, kernel generation, plan cache | Write kernels directly in Vulkan SPIR-V or GLSL |
| **Security/provenance** | CAS, SBOM, signatures, sandboxing, plugin system, model registry | Not needed for local single-model engine |
| **dLLM/Diffusion** | Block diffusion, approximate KV caching for bidirectional attention | Only if Mellum2.1 uses it (it doesn't - it's autoregressive MoE) |
| **Shape-class batching** | Microbatch shape classes, plan-cache optimization by batch shape | Irrelevant for single-user. Also can't be "novelty" and "already in vLLM/SGLang/TensorRT-LLM" at once. |
| **Multi-modal** | Vision, audio, ASR, TTS, cross-modal handoff | Mellum2.1 is text-only |
| **Platforms** | Linux, macOS, other GPUs | Windows + Vulkan. Drop CUDA/ROCm/Metal from plan. |
| **Distributed** | Multi-card, P/D disaggregation, cross-model KV handoff, net-tier KV, collaborative inference | Single GPU/CPU. Not applicable. |

---

## 5. Gap Papers (no direct paper, build from adjacent systems)

These are the items from the PRISM coverage gaps list that apply to your engine, with the adjacent-system papers to consult:

| Gap | Your Action | Adjacent Papers |
|---|---|---|
| **CAS tensor store + provenance** | Skip for v0.1 (not needed) | FORGE (2610.16890), Kilobyte (2608.00860), Drop-by-Drop (2606.12876) for concept only |
| **Compaction/defrag/arena** | Skip for v0.1 (contiguous KV, no block pool) | When Fancy Eviction Fails (2609.28870) for eviction philosophy - keep it simple |
| **Microbatch shape classes** | **Skip entirely.** Single-user, no batching. | If you ever add batching: vLLM BatchDescriptor/ShapeKey, SGLang ShapeKey, FlashInfer plan reuse, BucketServe, TensorRT-LLM bucketing |
| **Sampler theory** | Skip (greedy only) | Latent Distilling (2604.24927), RL-Guided Adaptive Sampling (2026) - sampling research, not needed for greedy |
| **Streaming delta assembly** | Skip (CLI, not streaming server) | Protocol engineering if you ever add streaming |
| **AOT packaging / signed bundles** | Skip for v0.1 | Reproducible-build practice. Not an engine concern. |
| **Audio/reranker/reward heads** | Skip (text-only model) | NVIDIA model cache reuse, Ouroboros SRAM CIM - hardware feasibility only |
| **Net-tier P2P KV transport** | Skip (no network) | PrfaaS (2604.15039), CacheLink (2610.15509) - KV handoff patterns, not needed |

---

## 6. Citations Note

All arXiv IDs and paper claims in this document are **leads to verify, not ground truth**. The Mellum2 Technical Report (2605.31268) is the authoritative source for Mellum2.1's architecture. llama.cpp is the authoritative oracle for tokenizer and output validation. Every paper-listed technique should be checked against actual behavior before relying on it.

**Priority order for reading:**
1. Mellum2 Technical Report (2605.31268) - know your target model
2. llama.cpp source - know your oracle and format
3. Expert-aggregation determinism (2607.28097) - know why ordering matters
4. SSD-LLaMA (2609.18110) + SeqMoE (2609.12978) - understand expert tiering concepts (adapt to your simpler LRU)
5. When Fancy Eviction Fails (2609.28870) - know why simple LRU is the right call
6. Everything else - only if you expand scope beyond v0.1

---

*Built for: Mellum2.1 MoE engine, Windows + Vulkan, single-user CLI, correctness-first against llama.cpp oracle. v0.1 scope only.*
