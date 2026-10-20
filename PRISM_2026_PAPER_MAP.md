# PRISM — 2026 (with 2025 Fallback) Paper Map
### Mapping the latest optimal papers onto every PRISM component

> **Compiled:** 15 October 2026
> **Spec being mapped:** PRISM — Pipelined Residency & Incremental Streaming Model
> (clean-room LLM inference engine; components A–P + Design Pillars, per the component
> specification this document accompanies)

**Sources:**
- Repo-curated digests: `ML-NEW-TECH.md` Parts 8–9 (Aug 2026 systems batch + Oct 6–14 2026 arrivals),
  `ENGINE_MASTER_SPEC.md` §11–13 (Sep–Oct 2026 fresh paper layer + consolidated index),
  `data/arxiv_latest_20261015.json` (raw Oct 6–13 snapshot, 122 papers).
- Fresh web research (Oct 2026 window) for clusters underrepresented in the digests
  (mixed-precision allocation, P/D-disaggregation serving, self-speculation, positional
  encodings, grammar-constrained decoding, tokenization).

**Conventions:**
- 🟢 **2026** — first submitted January–October 2026 (primary evidence).
- 🟠 **2025** — fallback, used only where no strong 2026 paper exists for the sub-component.
- ⚪ **≤2024** — foundational, kept only when nothing newer directly replaces it.
- All performance numbers are **author-reported**, mostly from preprints; treat as
  directional, not apples-to-apples across papers.
- "Optimal pick" = the single best starting reference for that component today, per the
  evidence in this window.

---

## 0. Pillar-Level Verdicts (map to §1 Design Pillars)

| # | PRISM Pillar | 2026 verdict | Strongest 2026 evidence |
|---|---|---|---|
| 1 | Content-addressed tensor store (not one monolithic file) | **Partially validated — mostly an engineering gap** | Multi-bitwidth single-artifact codecs ([Drop-by-Drop 2606.12876]🟢), procedural weight regeneration ([FORGE 2610.16890]🟢, [Kilobyte 2608.00860]🟢). No 2026 paper does full CAS + provenance; PRISM is ahead here. |
| 2 | Explicit tiered residency (HBM→DRAM→NVMe→net) | **Strongly validated** | [SSD-LLaMA 2609.18110]🟢, [py-kvcache 2609.11744]🟢, [TempoKV 2609.35065]🟢, [TierKV 2609.21172]🟢, [HiSparse 2608.07009]🟢, [mzCache 2609.01338]🟢 |
| 3 | Latent context memory (recurrent summary + sparse spans), not linear paged KV | **Strongly validated — hybrid architectures are now mainstream serving targets** | [KVBuffer 2605.19049]🟢, [GateSight 2610.14754]🟢, [KVFetch 2610.08811]🟢, [ReadKV 2610.11245]🟢, [Orthros 2610.13735]🟢, [Bridging KV-Quant & Linear Attention 2610.11214]🟢 |
| 4 | Compiled dataflow (AOT+JIT), not eagerly interpreted op graph | **Validated** | [RLX 2609.37916]🟢 (one IR → Vulkan/Metal/WebGPU/CUDA/ROCm), [GraphTok 2610.13325]🟢 (replayable CUDA-graph pipeline), [RESOLVE 2610.05683]🟢 (kernel correctness), [HiSparse 2608.07009]🟢 (fused CUDA-graph kernels) |
| 5 | Lane-based decode + P/D disaggregation, not token-step continuous batching | **Validated direction; the "per-lane token clock" abstraction itself is still PRISM's novelty** | [Vesta 2610.14686]🟢 (session-pinned KV holder, per-request executor), [AMPD 2602.14516]🟢, [PrfaaS 2604.15039]🟢, [Multi-Tier SLA 2608.16336]🟢, [AgentWeave 2610.14804]🟢, [Prorsa 2610.16592]🟢 |
| 6 | Self-speculative decoding (no second model) | **Strongly validated — the hottest decode-acceleration cluster of 2026** | [SEED 2609.36590]🟢, [KnapSpec 2602.20217]🟢, [BitNest 2610.02800]🟢, [SP²ec 2610.18735]🟢, [DLoop 2610.07659]🟢 |
| 7 | Per-tensor adaptive codecs (not one global block quant) | **Strongly validated — allocation is now the research frontier** | [GAMMA 2605.18475]🟢, [SLQ 2605.02404]🟢, [MultiAlloc 2610.18642]🟢, [DuoRate 2610.13328]🟢, [SAGE 2610.14745]🟢, [FidelityQ 2610.14934]🟢 |

**Unifying-need confirmation:** the bandwidth thesis is now consensus. 2026 work attacks
*bytes moved* at every layer — store-read gap for MoE ([DuoRate]🟢), bits-fetched-per-query KV
([ReadKV]🟢), zero-prefill KV handoff (Cluster §G-6), host-tokenization removal ([GraphTok]🟢),
weight regeneration instead of storage ([FORGE]🟢).

---

## A. Artifact & Storage Layer

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **Codec registry + quality-targeted allocation** | [GAMMA: Global Bit Allocation under Arbitrary Budgets](https://arxiv.org/abs/2605.18475) | 🟢 May 2026 | Learns module-level precision preference, solves exact-budget assignment via integer programming; +7.0 over search-based mixed precision; matches 3-bit quality at 2.5-bit average. **Optimal pick for the registry's allocation core.** |
| Quality targets for codec assignment | [Statistically-Lossless Quantization (SLQ)](https://arxiv.org/abs/2605.02404) | 🟢 May 2026 | Makes fidelity targets explicit (task-lossless 3.3 bpw; distribution-lossless 5–6 bpw, 1.7–3.7× speedup). The calibration store's objective function. |
| Sensitivity store (KV side) | [MultiAlloc](https://arxiv.org/abs/2610.18642) | 🟢 Oct 2026 | Offline profiling of layer sensitivity + token importance; allocation without online search. Same pattern PRISM needs for the per-tensor sensitivity store. |
| Store-read gap (MoE) | [DuoRate](https://arxiv.org/abs/2610.13328) | 🟢 Oct 2026 | Names what every allocator got wrong: storage bits ≠ decode-traffic bits. Per-workload Lagrangian allocation; at 2.3 bpw beats 2-bit baselines by 6.3–47 points. |
| **Codec plugins: VQ / codebooks** | [SAGE: 2-Bit VQ toward usability](https://arxiv.org/abs/2610.14745) | 🟢 Oct 2026 | Energy-stratified grouping + adaptive bit split; 95.3% of BF16 on Qwen3-30B-A3B, 2.55× e2e vs FP16. **Optimal VQ plugin reference.** |
| Multi-bitwidth single artifact | [Drop-by-Drop (additive codebooks)](https://arxiv.org/abs/2606.12876) | 🟢 Jun 2026 | One model, Matryoshka-style progressive bit rates — a CAS object that serves multiple quality points without duplicate storage. |
| Lattice VQ plugin | [GLVQ](https://arxiv.org/abs/2510.20984) | 🟠 Oct 2025 | Grouped lattice quantizers, simple matvec decode; NeurIPS 2025. (No stronger 2026 lattice-weight paper; E8 work [2610.15384]🟢 covers KV.) |
| Low-bit PTQ plugin | [FidelityQ](https://arxiv.org/abs/2610.14934) / [DualQuant](https://arxiv.org/abs/2610.13767) | 🟢 Oct 2026 | W2 lookup-free decoding at 8× compression; calibration-free W4A4 dual-scaling on MX/NVFP4. |
| Residual-correction codec | [ARCQuant](https://arxiv.org/abs/2601.07475) | 🟢 Jan 2026 | NVFP4 + augmented residual channels staying on one GEMM path; 3× vs FP16 on RTX 5090. ACL 2026. |
| Variable-length coding plugin | [VLCQ](https://arxiv.org/abs/2610.15092) | 🟢 Oct 2026 | Data-free PTQ with variable-length codes; up to 12.25× compression <4% degradation. |
| Ternary plugin | [BITCOS](https://arxiv.org/abs/2609.16338) | 🟢 Sep 2026 | Distribution-adaptive ternary layout; AVX-512/Xe2 kernels; 1.18× CPU / 1.27× GPU decode. |
| Procedural weights (radical CAS) | [FORGE](https://arxiv.org/abs/2610.16890) / [Kilobyte Models](https://arxiv.org/abs/2608.00860) | 🟢 Oct / Aug 2026 | Regenerate expert FFNs from O(log²d) seeds (<1 MB replaces GBs); seed+quantized-latent models. The far end of the "store a recipe, not a table" direction. |
| Calibration methodology | [D²Quant](https://arxiv.org/abs/2602.02546) / [Quantize by Drift](https://arxiv.org/abs/2610.09227) | 🟢 2026 | Down-projection dual scales + deviation correction; label-free drift signal for mixed precision (embedders). |
| **Fetch transport / tiered objects** | [py-kvcache](https://arxiv.org/abs/2609.11744) / [LM-CXD](https://arxiv.org/abs/2609.26828) | 🟢 Sep 2026 | Async direct NVMe I/O, bounded staging (2.0× LMCache at 80K); CXL-SSD chunk-visible KV (4.03× lower TTFT). Ranged/resumable fetch design patterns. |
| Adapter / delta store | [Edge0 recovery LoRA](https://arxiv.org/abs/2609.18063) / [CoDe-LoRA](https://arxiv.org/abs/2610.08312) | 🟢 2026 | Deltas as first-class runtime objects (served recovery LoRA; consolidation of continual updates). Thin coverage — LoRA hot-swap as CAS objects remains mostly engineering. |
| **Provenance / signatures / SBOM** | *(no strong 2025–2026 paper found)* | — | **Gap:** supply-chain verification is an engineering adoption (sigstore/TUF patterns), not a research question this year. |

**Pillar verdict:** the *storage format* research has moved past uniform block quant to
allocation, multi-rate artifacts, and regeneration; PRISM's CAS + provenance layer remains
ahead of the literature and should be built on the codec plugins above.

---

## B. Lexical Codec

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **Tokenizer as runtime component** | [Taking Tokenization off the Host (GraphTok)](https://arxiv.org/abs/2610.13325) | 🟢 Oct 2026 | Whole encode pipeline as one replayable CUDA graph, bit-exact across 5 model families; tokenization is the largest TTFT term on agentic prefix-cached traffic; 518 req/s frontend vs 18.5 default. **Optimal pick — and a direct blueprint for the command-replay pillar applied to the codec.** |
| Token healing / partial tokens | [Are you going to finish that? A Practical Study of the Partial Token Problem](https://arxiv.org/abs/2601.23223) | 🟢 Jan 2026 | First systematic study of partial-token artifacts; token healing shows **mixed** effectiveness, ByteSampler proposed. Informs PRISM's boundary-repair stage. |
| End-to-end tokenization | [You Can Learn Tokenization End-to-End](https://arxiv.org/abs/2602.13940) | 🟢 Feb 2026 | Learnable boundaries instead of frozen BPE — relevant if PRISM ever trains/distills its own tokenizer. |
| Byte-patch / boundaries analysis | [How BPE Creates Exploitable Gaps](https://arxiv.org/abs/2607.01239) | 🟢 Jul 2026 | Sub-word fragmentation of safety-critical words; a correctness + safety argument for byte-fallback and healing. |
| Byte-level fallback (2025 line) | [H-Net (dynamic hierarchical chunking)](https://arxiv.org/abs/2503.23462) · [Byte Latent Transformer](https://arxiv.org/abs/2412.18668) · [Exact Byte-Level Probabilities](https://arxiv.org/abs/2410.09303) | 🟠 2025 / ⚪ Dec 2024 | The tokenizer-free/byte-patch line. No strong 2026 successor; keep as the fallback design basis for a codec-independent text boundary layer. |
| Multimodal spans | see §P | — | Image/audio placeholder handling lives in the extensions layer. |

---

## C. Residency & Memory Layer

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **Tiered manager (SSD→RAM→VRAM)** | [SSD-LLaMA](https://arxiv.org/abs/2609.18110) | 🟢 Sep 2026 | Expert-granularity SSD reads, pinned-RAM transfers, residency decoupled from compute; trillion-param MoE >1 tok/s on RTX 5090 with ≤32 GB RAM. **Optimal pick — the flagship validation of explicit tiering on consumer hardware.** |
| Heat / reuse model + prefetch | [SeqMoE](https://arxiv.org/abs/2609.12978) / [Mira](https://arxiv.org/abs/2609.38090) | 🟢 Sep 2026 | Forecast-driven prefetch+eviction: **96.97% expert hit at 45% residency** (closest single number to any "98% hit" goal); Mira's HOT+STAGE two-tier cache + telemetry rebalance, up to 5.71× throughput. |
| Router-side reuse (fewer fetches) | [ReMoE](https://arxiv.org/abs/2605.27081) / [Cache-Aware Router Adaptation](https://arxiv.org/abs/2609.04895) | 🟢 2026 | Make routing cache-friendly at train/adapt time (+26% expert reuse; 4.6–53.3% lower expert traffic) — the layer *below* the residency manager. |
| Edge prefetch | [SpecPrefetch](https://arxiv.org/abs/2607.24787) | 🟢 Jul 2026 | Adapter-based expert prioritization for async transfer; +20% decode on Snapdragon 8 Elite. |
| Prefetch scheduler (KV) | [OasisKV](https://arxiv.org/abs/2608.08097) | 🟢 Aug 2026 | Lookahead sparse prefetch driven by speculative-decoding tokens; 1.69–2.1× throughput at 6.5–9.7× less KV footprint. Draft tokens are a free prefetch oracle. |
| **Eviction scoring** | [When Fancy Eviction Fails](https://arxiv.org/abs/2609.28870) | 🟢 Oct 2026 | 20B-token production trace, 14 policies: LRU robust for prefix reuse; add quick-demotion of one-hit prompts + compute-aware granularity. **The cautionary optimal pick — keep the scorer simple.** |
| Timing-aware staging | [TempoKV](https://arxiv.org/abs/2609.35065) | 🟢 Sep 2026 | Commit fast-tier capacity only when time-to-use ≈ time-to-ready; 63–91% lower protected byte-time, up to 48% lower p95 TTFT. |
| Spill / restore (memory pressure) | [mzCache](https://arxiv.org/abs/2609.01338) / [TierKV](https://arxiv.org/abs/2609.21172) | 🟢 Sep 2026 | Mobile OS pressure → eviction+restore with CPU-side concurrency (2.1–5.5× lower TTFT); predictive multi-tier KV (17.6× prefill on mobile SoCs). |
| Hierarchical resident window | [HiSparse](https://arxiv.org/abs/2608.07009) | 🟢 Aug 2026 | Fixed small GPU cache + full history in host RAM, fused CUDA-graph kernels for hit detection/LRU/layer-wise prefetch; 4.7× long-context throughput; merged into SGLang. **Optimal pick for the window/history split.** |
| Zero-copy / DMA-equivalent paths | [TRANSIT](https://arxiv.org/abs/2610.07593) / [MemFerry](https://arxiv.org/abs/2610.09657) / [FusionML](https://arxiv.org/abs/2607.22785) | 🟢 2026 | User-space interposition, zero-copy CPU↔GPU (TRANSIT +42–68% per-GPU throughput); Direct Host Access scheduling; and FusionML's measured result that **co-execution helps prefill, not decode** — the bandwidth model must be phase-aware. |
| Compactor / defrag, arena | *(no direct 2025–2026 paper)* | — | **Gap:** block-pool/defrag design is systems engineering; borrow from databases, not this year's LLM literature. |

---

## D. Compiler & Execution Layer

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **Multi-backend IR/runtime** | [RLX](https://arxiv.org/abs/2609.37916) | 🟢 Sep 2026 | One Rust IR/runtime across Vulkan, Metal, WebGPU, CUDA, ROCm. **Optimal pick — closest thing to PRISM's frontend-IR + codegen story; verify LLM workload coverage before adopting.** |
| Command replay / plan cache | [GraphTok](https://arxiv.org/abs/2610.13325) / [HiSparse](https://arxiv.org/abs/2608.07009) | 🟢 2026 | Both run steady-state pipelines as replayed CUDA graphs (tokenization; hierarchical-KV bookkeeping). Production evidence for PRISM's command-replay pillar. |
| Quant-aware fused kernels (avoid dequant) | [Multi-Scale Dequant](https://arxiv.org/abs/2605.13915) | 🟢 May 2026 | Decompose activations instead of materializing BF16 weights; error bounds + simulated 2.5× KV-traffic reduction. The kernel/codec boundary need not reconstruct floats. |
| Fused low-bit GEMM evidence | [Green AI energy study](https://arxiv.org/abs/2610.17531) | 🟢 Oct 2026 | 10 PTQ × 5 models × 4 GPUs: fused-kernel int8 saves up to 4× energy vs FP32; quantization peaks at long-input/short-output. |
| Sparse/structured kernels | [UnionSparse](https://arxiv.org/abs/2608.09291) | 🟢 Aug 2026 | W4A4 sparse index-bottleneck fix on edge GPUs; 1.43–2.30× over SpMM baselines. |
| Attention kernels (prefill) | [QUILT](https://arxiv.org/abs/2610.11134) / [PageWeaver](https://arxiv.org/abs/2610.11201) | 🟢 Oct 2026 | Shared-query sparse prefill (−55% kernel latency, −36.8% TTFT); KV-guided query unions keeping tensor-core tiles full (1.70× on H200). |
| **Kernel generation & validation** | [D2K-Bench](https://arxiv.org/abs/2610.03226) · [MetaKernelBench](https://arxiv.org/abs/2610.05014) · [RESOLVE](https://arxiv.org/abs/2610.05683) | 🟢 Oct 2026 | The agent-generated-kernel reality check: guidance lifts correctness 93.1→98.5%; DSL transfer is double-edged; RESOLVE found **4 previously unreported mega-kernel bugs** via testing/reduction/proof. **Optimal pick for the kernel-validation harness PRISM must run in CI.** |
| CPU GEMM/matvec | [HiNa-MoE](https://arxiv.org/abs/2610.05123) | 🟢 Oct 2026 | Intel AMX MoE kernels with standard layouts, NUMA-aware partitioning, decode matvec→matmul; 3.37× FFN / 2.09× e2e. |
| Numerics policy | [W4A4 analysis](https://arxiv.org/abs/2609.21450) / [Losslessness under finite precision](https://arxiv.org/abs/2609.15504) | 🟢 Sep 2026 | Separates weight-compensable error from orthogonal residual; and BF16-vs-FP32 trajectory divergence audits that every "lossless" claim needs. |
| Frontend IR / pass pipeline | *(no single 2026 paper owns this)* | ⚪ | MLIR/torch-export adoption remains engineering; RLX is the nearest integrated artifact. |

---

## E. Context Memory Subsystem (replaces paged KV)

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **Latent summary state (recurrent/hybrid)** | [KVBuffer](https://arxiv.org/abs/2605.19049) | 🟢 May 2026 | IO-aware serving for linear attention (SGLang, Qwen3-Next): batch recurrent-state updates, speculative verification without temp state; 45.17% lower decode latency. **Optimal pick for hybrid-model serving.** |
| Recurrent state utilities | [GateSight](https://arxiv.org/abs/2610.14754) | 🟢 Oct 2026 | In hybrid models, the recurrent gate's prefill telemetry **anticipates future KV utility before any query exists** — the importance scorer for both the latent state and the exact window. |
| KV-quant ↔ linear-attention bridge | [Bridging KV-Cache Quantization and Linear Attention](https://arxiv.org/abs/2610.11214) | 🟢 Oct 2026 | Proves the bridge between discrete KV compression and recurrent aggregation; migrates 9 pretrained models in 500M tokens. The theory backing PRISM's "compressive memory" claim. |
| Long-context in linear attention | [History Replays Itself](https://arxiv.org/abs/2610.15715) / [HLA](https://arxiv.org/abs/2610.05842) / [Orthros](https://arxiv.org/abs/2610.13735) | 🟢 Oct 2026 | Unlocked long-context capability in linear attention; chunk-wise dynamic mixing beyond training context; phase-aware module (21.9× prefill, 128K needle). |
| Hybrid position mechanics | [Mechanics of Long-Context Hybrid Models 1.1](https://arxiv.org/abs/2610.10114) | 🟢 Oct 2026 | How hybrid attention induces hybrid positional behavior — needed when the working window and latent state have different positional semantics. |
| **Span store (compressed cold tiers)** | [KVFetch](https://arxiv.org/abs/2610.08811) | 🟢 Oct 2026 | Score-based compression destroys verbatim copying; a quantized cold tier with positional recall recovers it (RULER-16K copying 0.8→78.4, +8.4 avg/13 tasks). **Optimal pick — the single most important idea for this subsystem.** |
| Retrieval & reinjection gate | [ReadKV ("Read What Matters")](https://arxiv.org/abs/2610.11245) | 🟢 Oct 2026 | Splits the budget into bits-stored vs bits-fetched-per-query over a progressive code; 4-of-8-bit reads cost ≤0.66% perplexity at ¼ the reads. This *is* the reinjection gate. |
| Span compressor (low-rank) | [BELT](https://arxiv.org/abs/2610.17966) / [AttSVD](https://arxiv.org/abs/2610.06927) | 🟢 Oct 2026 | Behavior-aligned low-rank compression (70% compression, +23.3 LongBench, 3.3× decode); prompt-adaptive attention-guided SVD at half cache memory. |
| Persistent-cache codec | [AlignPred](https://arxiv.org/abs/2610.17970) | 🟢 Oct 2026 | Error-budgeted codec for **long-lived agent caches** (42–72% storage cut in SGLang HiCache) — the format for PRISM's persistent span store. |
| Eviction timing | [DeferKV](https://arxiv.org/abs/2610.06286) | 🟢 Oct 2026 | Defer irreversible eviction until the first decode query — no predictor needed. Cheap default when the importance scorer isn't worth its cost. |
| Budget allocation | [MultiAlloc](https://arxiv.org/abs/2610.18642) / [ChampionKV](https://arxiv.org/abs/2610.18026) / [BreadthKV](https://arxiv.org/abs/2610.05685) | 🟢 Oct 2026 | Layer-vs-token precision allocation; head-wise adaptive budgets; precision-vs-count trade for long-CoT reasoning. |
| Aggressive quant ends | [E8 Lattice KV](https://arxiv.org/abs/2610.15384) / [Dual-QK](https://arxiv.org/abs/2610.09827) / [TaSQ](https://arxiv.org/abs/2610.03027) / [ResidualQuant](https://arxiv.org/abs/2610.10381) | 🟢 2026 | k2v2 lattice (~6×, throughput parity); 2-bit + pruning (6.8×, 3.75× decode); 1-bit VQ (14× batch, 1.87× peak); looped-transformer 2-bit residuals (4.15× peak). |
| **Negative result to respect** | [EdgeLLM-Nano audit](https://arxiv.org/abs/2610.14475) | 🟢 Oct 2026 | On a 33.6 MB on-device LM, **no tested compressor beat FP16 on energy/token** (Raspberry Pi 5). Compression must be chosen per tier, not universally. |
| Exact fallback mode | [KV4-RL](https://arxiv.org/abs/2610.14448) | 🟢 Oct 2026 | INT4 KV with **selective INT8 protection of attention sinks and recent context** — the hybrid "aggressive spans + exact recent window" configuration PRISM wants, validated in async RLVR. |
| Legacy eviction lineage | AnchorKV 2608.02901 · QEvict 2608.05326 · RestoreKV 2608.01247 · SAKI 2608.03228 | 🟢 Aug 2026 | The Aug-digest eviction/compression cohort the above succeed. |

---

## F. Attention & Positional

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **Positional abstraction (beyond RoPE)** | [DaRoPE: Data-aware Positional Encodings](https://arxiv.org/abs/2609.34556) | 🟢 Sep 2026 | Keep RoPE fast bands; replace slow bands with bounded data-aware coordinates; best-or-par on-par for language + length extrapolation up to 50B. **Optimal pick — the strongest RoPE-alternative evidence this year.** |
| Length extrapolation (gated) | [GAPE: Gated Adaptive Positional Encoding](https://arxiv.org/abs/2605.10414) | 🟢 May 2026 | Content-aware query/key gates; extrapolation to 8× training length, retrofittable to a pretrained 7B. |
| NoPE theory | [How Local Mixing Encodes Relative Position in Global NoPE](https://arxiv.org/abs/2609.38109) | 🟢 Sep 2026 | Why hybrid local+global NoPE models get implicit relative position that persists long-range — the design rationale for a NoPE working window. |
| Relative-bias fast paths | [Positional LSH](https://arxiv.org/abs/2605.09472) | 🟢 May 2026 | ALiBi bias realized as sampled block-diagonal masks → near-linear randomized short-window attention. |
| Exploratory | [Jordan-RoPE](https://arxiv.org/abs/2605.04217) · [Multiscreen](https://arxiv.org/abs/2604.01178) | 🟢 2026 | Non-semisimple rotary+polynomial basis (limited NLP evidence); bounded-relevance softmax replacing key competition (haystack robustness). |
| FIRE line | FIRE (Fourier relative-position encoding, 2024) | 🟠 2024 | No strong 2026 FIRE successor; DaRoPE/GAPE supersede it as the learned-relative references. |
| **Attention variants: GQA/MLA-class** | [Virtual Head Attention](https://arxiv.org/abs/2610.17959) / [MLAfication](https://arxiv.org/abs/2610.17202) | 🟢 Oct 2026 | Two linear ops recover virtual-head diversity — beats GQA at all scales with **2 KV heads instead of 4**; retrofitting MLA onto modern LLMs. **Optimal picks for KV-head efficiency.** |
| Block-sparse | [MASA](https://arxiv.org/abs/2610.10871) · [SPIN](https://arxiv.org/abs/2610.09025) · [QUILT](https://arxiv.org/abs/2610.11134) · [LongCat Sparse](https://arxiv.org/abs/2608.01662) | 🟢 2026 | Sparse attention as matrix-product approximation (the math fix); history-predicted indexing (no full rescan, +14.9% in vLLM); shared-query prefill; production sparse-attention architecture. |
| Linear/SSM hybrid | [Gated DeltaNet](https://arxiv.org/abs/2406.06484) · [Linearizing LMs comparison](https://arxiv.org/abs/2504.14366) · [Bole tree speculation](https://arxiv.org/abs/2608.01651) | 🟡 2024–25 + 🟢 | GDN remains the hybrid kernel of choice (see §E for 2026 serving layers on top); the linearization comparison is the 2025 fallback survey; Bole adds tree speculation for hybrids. |
| Mask/window policy | [Reasoning traces as De Bruijn paths](https://arxiv.org/abs/2610.09186) / [S3R](https://arxiv.org/abs/2610.13202) | 🟢 Oct 2026 | Task traces are sparse DAGs — <15%-length windows keep >75% accuracy (evidence for window policies); sparse-proposal + exact full-attention verification pattern. |
| Causal/prefix correctness | [Dichotomy…](https://arxiv.org/abs/2610.09186) + SAFE-KV (§G-6) | 🟢 | Visibility policies must respect context-conditioned causal state, not just position. |

---

## G. Scheduling & Batch Pipeline

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **P/D disaggregation** | [AMPD](https://arxiv.org/abs/2602.14516) / [PrfaaS](https://arxiv.org/abs/2604.15039) / [FluidPD](https://arxiv.org/abs/2610.06917) | 🟢 2026 | Multi-round (agentic) disaggregated prefill with SLO gains 67–340%; cross-datacenter prefill-as-a-service (54% over homogeneous PD); in-place elastic P/D pools. **AMPD is the optimal pick for agentic traffic; PrfaaS for the net-tier residency link (§K).** |
| **Priority / SLA manager** | [Prorsa](https://arxiv.org/abs/2610.16592) / [Multi-Tier SLA Scheduling](https://arxiv.org/abs/2608.16336) / [Pascal](https://arxiv.org/abs/2602.11530) / [GoodServe](https://arxiv.org/abs/2605.16867) | 🟢 2026 | Prediction-guided prefill-decode resource allocation for agentic RL; forecasts short-horizon P/D demand per trajectory and triggers role switching; 1.11–4.05× E2E rollout speedup over PD colocation (sync and async). Multi-Tier SLA Scheduling: up to 10 SLA tiers with migration (8.3× prefill, 3.1× p99); reasoning-vs-answer phase awareness (−72% tail TTFT); predicted-length routing + risk-triggered migration (+27.4% goodput). GoodServe for the baseline. |

| Queueing-aware lane dispatch | [Vesta](https://arxiv.org/abs/2610.14686) | 🟢 Oct 2026 | **The closest published design to PRISM's lane model:** stable session-pinned KV holder + per-request executor choice; backlog + prefix-reuse + KV-transfer + queueing-externality objective; +14% throughput, −57.8% mean TTFT. **Optimal pick.** |
| Token-clock backpressure | [Stable Token-level Scheduling](https://arxiv.org/abs/2610.17689) | 🟢 Oct 2026 | Admission at token granularity via closed-loop Token Window Pacing over ITL/queue depth; +45% serverless throughput, stabilized TTFT — the admission/backpressure primitive. |
| Baseline policy evidence | [Lightweight Scheduling Policies study](https://arxiv.org/abs/2610.15899) / [Quota-Based Aging](https://arxiv.org/abs/2610.17795) | 🟢 Oct 2026 | 1,152-run study: **aging-shortest-prompt-first is the strongest practical baseline**; even oracle length knowledge loses to batching/hardware effects. Quota-bounded promotion of aged long requests. |
| Lifecycle FSM phases | [Pascal](https://arxiv.org/abs/2602.11530) | 🟢 | Requests have more phases than prefill/decode (reasoning → answer), each with different preemption tolerance — the FSM needs ≥4 states, matching PRISM's design. |
| Microbatch shape classes | *(no direct paper)* | — | **Gap / PRISM novelty** — shape-class packing for plan-cache hits is unclaimed territory; cite plan-cache hits (§D) as the motivation. |
| Elastic pools | [FastEEP](https://arxiv.org/abs/2610.18451) / [Zepp](https://arxiv.org/abs/2610.11158) | 🟢 Oct 2026 | Resize EP groups in-place (4.5–11.7× faster scaling); relaxed-balance MoE serving optimizing the real bottleneck (1.86× geomean). |
| State sharing (see also §L cross-model cluster) | [AgentWeave](https://arxiv.org/abs/2610.14804) | 🟢 Oct 2026 | Flow decomposition of long agent requests with semantics-preserving units; +1.81× throughput. The lane-splitting mechanism for agentic workloads. |

---

## H. Decoding Acceleration

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **Self-speculation** | [SEED](https://arxiv.org/abs/2609.36590) | 🟢 Sep 2026 | Reuse deep contextual reps from verification; a light decoder drafts between target passes — **no second model, no full-depth drafting**; up to 2.7× on 4B, 28% faster than EAGLE-3. **Optimal pick.** |
| Training-free self-speculation | [KnapSpec](https://arxiv.org/abs/2602.20217) / [SP²ec](https://arxiv.org/abs/2610.18735) | 🟢 2026 | Layer selection as knapsack (context- and hardware-aware, 1.47×); VLM bandit over draft configs with wall-time-regret guarantees (+11%). |
| Shared-weight draft+target | [BitNest](https://arxiv.org/abs/2610.02800) | 🟢 Oct 2026 | Low-precision draft nested inside the target's weights + KV; 95.2% acceptance, 1.48–1.61× e2e — the memory-free draft-head design for low-VRAM tiers. |
| **Multi-token heads + tree verify** | [JetSpec](https://arxiv.org/abs/2606.18394) / [P-EAGLE](https://arxiv.org/abs/2602.01469) / [DCOCO](https://arxiv.org/abs/2610.12570) | 🟢 2026 | One-pass parallel tree drafting (9.64× MATH-500); parallel-draft EAGLE with scalable training (1.10–1.36× over EAGLE-3); contextual correction of diffusion blocks (up to 6.6×). |
| Draft from block diffusion | [DFlash](https://arxiv.org/abs/2602.06036) | 🟢 Feb 2026 | Block-diffusion drafter + parallel verify; >6× lossless, 2.5× EAGLE-3's speedup. |
| **Adaptive depth controller** | [DSpark](https://arxiv.org/abs/2607.05147) / [EntMTP](https://arxiv.org/abs/2606.27550) / [LEDE](https://arxiv.org/abs/2606.03113) | 🟢 2026 | Confidence-scheduled verification length in production (60–85% faster per-user at matched throughput, DeepSeek-V4 serving); entropy-guided tree-topology selection; RL-chosen exit layer × draft length (2.0–2.7×). |
| Execution-loop optimization | [DLoop](https://arxiv.org/abs/2610.07659) | 🟢 Oct 2026 | Draft longer, verify less often — 5–41% higher wall-clock speedup across 5 spec schemes, lossless. Free win for any scheme PRISM runs. |
| Offload-aware speculation | [SpecFit](https://arxiv.org/abs/2610.15417) / [SpecStream](https://arxiv.org/abs/2609.33184) / [Adaptive DHiSparse](https://arxiv.org/abs/2610.13708) | 🟢 2026 | When experts/KV live in host memory, maximize predicted throughput not acceptance length (21.6% over EAGLE-3); overlap drafting with KV streaming (1.41× over offload baseline); speculative decoding over hierarchical sparse KV at load (1.45× HiSparse). **The spec↔residency co-design trio PRISM uniquely needs.** |
| Lookahead / overlap | [Saguaro ("Speculative Speculative Decoding")](https://arxiv.org/abs/2603.03251) / [DPara](https://arxiv.org/abs/2609.27396) | 🟢 2026 | Prepare likely continuations while verification runs (+30% over optimized SD); precomputed draft reps for acceptance boundaries (3.21–3.52×). |
| Drafter KV-free / serving-scale | [H-Spec](https://arxiv.org/abs/2609.24197) / [NebulaSD](https://arxiv.org/abs/2609.29364) | 🟢 Sep 2026 | Target-KV reuse for drafters (+5–13.3% acceptance); draft/target worker pools (50.4% over disaggregated baseline). |
| Draft-source selection | [SwitchSD](https://arxiv.org/abs/2609.20186) / [TLAR](https://arxiv.org/abs/2610.07350) | 🟢 2026 | Probe-gated copy-vs-draft (AUC >0.99, +15% over EAGLE3); trajectory-retrieval drafts combined with neural drafts in one tree. |
| **Training objectives for drafters** | [EDR](https://arxiv.org/abs/2610.10411) / [Acceptance-Aware Training](https://arxiv.org/abs/2609.24150) / [VAT](https://arxiv.org/abs/2608.30135) | 🟢 2026 | Train for expected decoding rounds (Markov-reward objective); EAL/WTV direct acceptance-length training; verification-aware loss weighting (+11.4% accepted length). **If PRISM trains an MTP head, use these, not surrogate token-loss.** |
| **S³R self-sparse speculative decoding** | [S³R](https://arxiv.org/abs/2610.13202) | 🟢 Oct 2026 | Uses sparse attention to propose tokens, then generates responses through full-attention verification and exact speculative sampling. Drafting and verification share the same policy parameter, eliminating the need to train a separate drafter. Maintains dense-policy distribution; with a KV-aware scheduler limits cache repeated offloading/restoration. Achieves 1.39×–1.35× end-to-end training throughput on Qwen3 with comparable task performance. |
| Early-exit / MoD systems | [Continuous Depth Batching](https://arxiv.org/abs/2608.09444) | 🟢 Aug 2026 | Per-iteration token scheduling for looped/adaptive-depth models; realizes 99% of theoretical adaptive-depth speedup. The scheduling companion to layer-skipping. |
| Classic baselines | Speculative Sampling 2302.01318 · Lookahead Reasoning (2025) · HiSpec 2510.01336 · PARD-2 2605.08632 | ⚪/🟠/🟢 | Verification-math baseline; 2025 lookahead line; 2026 CPU-friendly and PARD successors from the Sep digest. |

---

## I. Logit & Output Layer

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **Grammar / JSON-schema → automaton** | [XGrammar-2](https://arxiv.org/abs/2601.04426) | 🟢 Jan 2026 | Second-generation dynamic structured generation for tool calling / response protocols; XGrammar is already the default GCD backend in vLLM/SGLang/TensorRT-LLM/MLC. **Optimal pick — build on it or match its interface.** |
| CFG masking efficiency | [Efficient Grammar-Constrained Decoding via Parser Stack](https://arxiv.org/abs/2608.03065) | 🟢 Aug 2026 | Parser-stack state to prune grammar masks without full automaton evaluation. |
| Preprocessing-minimal GCD (2025 fallback) | [Flexible and Efficient Grammar-Constrained Decoding](https://arxiv.org/abs/2502.05111) | 🟠 Feb 2025 | Reduces grammar preprocessing overhead while keeping guarantees; ICML 2025. The fallback when no incremental-compile path exists for a new grammar class. |
| Structured execution at the API layer | [SGLang structured execution](https://arxiv.org/abs/2312.07104) | ⚪ 2023 | Still the systems reference for composing grammar constraints + radix reuse in one program. |
| Schema-guided parallel decoding | [SchemaWeaver](https://arxiv.org/abs/2610.17436) | 🟢 Oct 2026 | On-device schema-guided **parallel** decoding — constraints as a speedup, not just a mask. Relevant to PRISM's multi-token heads interacting with grammars. |
| **S³R self-sparse speculative decoding** | [S³R](https://arxiv.org/abs/2610.13202) | 🟢 Oct 2026 | Uses sparse attention to propose tokens, then generates responses through full-attention verification and exact speculative sampling. Shown to maintain dense-policy distribution with KV-aware scheduling; relevant to speculative+constrained decoding interplay. |
| Samplers & penalties | *(no strong 2026 theory paper; min-p/typical remain community practice)* | — | **Gap:** PRISM's sampler chain is engineering; seed manager + determinism live in §N. |
| Constrained + speculative interplay | [Speedbumps](https://arxiv.org/abs/2610.10929) + [Secure SD](https://arxiv.org/abs/2610.08678) (see §O) | 🟢 Oct 2026 | Rejection attacks against speculative sampling; hardened verification — any lossy/constrained fast path must survive these. |
| Logprob/stream assembly | *(no direct paper)* | — | **Gap:** streaming delta assembly is protocol engineering. |

---

## J. Backend & Hardware

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **Platform abstraction** | [RLX](https://arxiv.org/abs/2609.37916) · [Numbat](https://arxiv.org/abs/2609.10632) · [Meganeura](https://arxiv.org/abs/2608.01563) | 🟢 2026 | Unified multi-backend runtime; self-contained verified ML stack; cross-vendor kernel portability. **Optimal pick: evaluate RLX + Meganeura as the two abstraction candidates (§12.10 takeaway).** |
| WebGPU | [GPU Goldmines](https://arxiv.org/abs/2610.17321) · [WebGPU dispatch overhead](https://arxiv.org/abs/2608.08730) · [dispatch 2604.02344](https://arxiv.org/abs/2604.02344) | 🟢 2026 | Tunable edge-WebGPU kernels; measured batch-1 dispatch overhead bounds for the WebGPU backend's expectations. |
| Metal / unified memory | [BaseRT](https://arxiv.org/abs/2607.00501) / [EdgeAgent](https://arxiv.org/abs/2610.03394) | 🟢 2026 | Native Metal runtime: 1.56× llama.cpp, 1.35× MLX decode; zero-copy CPU–GPU tensor parallelism on Apple M4 UMA (1.29×). |
| CPU path | [HiNa-MoE](https://arxiv.org/abs/2610.05123) · [BITCOS](https://arxiv.org/abs/2609.16338) · [WIDE](https://arxiv.org/abs/2607.28418) | 🟢 2026 | AMX MoE (3.37× FFN); ternary AVX-512 kernels; wide-ISA CPU execution. |
| Edge NPUs | [HA-NPU](https://arxiv.org/abs/2609.32114) | 🟢 Sep 2026 | Hybrid-attention dataflow reorganization on NPUs: 35.95× linear-attention kernel speedup, 2.03× e2e. |
| Capability DB / dispatch policy | [TraceDSE](https://arxiv.org/abs/2610.07191) / [Transition-Aware Backend Dispatch](https://arxiv.org/abs/2607.17415) / [HeteroMosaic](https://arxiv.org/abs/2607.12839) | 🟢 2026 | Proposer–critic placement across CPU/GPU/NPU (6–9× fewer evaluations); phase-transition-aware dispatch; heterogeneous tiling. The planner inputs PRISM's capability DB needs. |
| Vendor math libs / precision dispatch | ExaGEMM 2607.14622 · PolyQ 2607.14618 · Early-Stop Accumulations 2608.06177 | 🟢 Jul–Aug 2026 | Exascale GEMM autotuning; multi-precision kernel selection; accumulation early-stopping — the precision-ISA dispatch menu. |
| Comm layer | [Parallelism characterization](https://arxiv.org/abs/2610.05305) · Multi-Node B300 field report 2608.05944 | 🟢 2026 | Compute–communication trade-off model separating pipeline bubbles per phase; ops power/failure playbook. |

---

## K. Placement & Distribution

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **Placement solver (cost-model DSE)** | [TraceDSE](https://arxiv.org/abs/2610.07191) | 🟢 Oct 2026 | Joint workload placement + hardware settings from execution-trace feedback; up to 35% hypervolume over NSGA-II with 6–9× fewer evaluations. **Optimal pick for the solver's search loop.** |
| Cost-model inputs | [Compositional Performance Modeling](https://arxiv.org/abs/2610.15366) / [Parallelism characterization](https://arxiv.org/abs/2610.05305) / [FusionML](https://arxiv.org/abs/2607.22785) | 🟢 2026 | Transferable 1P-1D-calibrated cost models (1.34% latency error vs SGLang); tensor-vs-pipeline trade-offs per phase; measured co-execution boundaries. |
| **Expert / piece offload** | [SSD-LLaMA](https://arxiv.org/abs/2609.18110) / [SeqMoE](https://arxiv.org/abs/2609.12978) / [Edge0](https://arxiv.org/abs/2609.18063) | 🟢 2026 | Runtime-only tiering at trillion-param scale; 96.97% hit at 45% residency; trained prerouter that *becomes* the routing decision + recovery LoRA (~20 tok/s int4-from-SSD on 24 GB). |
| **Multi-card weak-GPU** | [CoMoE](https://arxiv.org/abs/2610.09424) | 🟢 Oct 2026 | Host-as-routing-hub over PCIe without P2P: 1.46× on RTX 5090s, approaching NVLink A800 at 23.4% hardware cost. **Optimal pick — the exact "consumer multi-card as one" design PRISM's pillar 5 needs.** |
| Edge–cloud split | [DySCo](https://arxiv.org/abs/2610.08268) / [Cascadia](https://arxiv.org/abs/2609.38697) | 🟢 2026 | Depth-synchronized batching over edge-prefix/cloud-suffix cuts (up to 275% vs FIFO); control-plane-free commodity-NPU fleets (3.10–4.06×). |
| Parallelism strategy | [FastEEP](https://arxiv.org/abs/2610.18451) / [Zepp](https://arxiv.org/abs/2610.11158) / [Logit-Aware AirComp](https://arxiv.org/abs/2610.03741) | 🟢 Oct 2026 | Intra-instance elastic EP groups; communication-first distributed MoE scheduling (6.68× MoE-layer); wireless-edge aggregation weighted by logit sensitivity. |
| Net-tier / cross-cluster | [PrfaaS](https://arxiv.org/abs/2604.15039) (§G) · [CacheLink](https://arxiv.org/abs/2610.15509) (§L) | 🟢 2026 | KV as the unit that crosses cluster boundaries — the net tier of the residency ladder, already in production use-cases. |
| Split/federated lineage (weak nodes) | GQ-FSL 2607.29659 · Gecko 2608.02378 · FedRings 2608.03436 · Collaborative MEC 2608.02031 | 🟢 Jul–Aug 2026 | Asymmetric-precision split points; crypto-cheap public/private split; link-aware ring aggregation; PPO subtask scheduling for edge pools. |

---

## L. Service & API

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **Control plane / routing** | [TokenRouter](https://arxiv.org/abs/2610.12242) / [WRP vision (vLLM Semantic Router)](https://arxiv.org/abs/2603.21354) | 🟢 2026 | Token-level routing runtime with throughput-model-tuned delayed batching (2.01–64× decode); workload–router–pool fleet architecture. |
| **TokenRouter token-level routing** | [TokenRouter](https://arxiv.org/abs/2610.12242) | 🟢 Oct 2026 | Efficient serving system for token-level routed LLM inference; request-centric programming, model-centric execution; delayed-batching scheduler with optimal hyperparameters; achieves 2.01–64.15× higher decoding throughput than existing systems across diverse routing algorithms and model pairs. |
| Session/executor lifecycle | [Vesta](https://arxiv.org/abs/2610.14686) (§G) | 🟢 | Session-pinned KV holder + request-scoped executor = the registry's session model. |
| **Cross-model / cross-agent context handoff — the breakout 2026 API primitive** | [Swarm Telepathy](https://arxiv.org/abs/2610.15095) · [CacheLink](https://arxiv.org/abs/2610.15509) · [RaReCache](https://arxiv.org/abs/2610.11358) · [Attention-map reuse](https://arxiv.org/abs/2610.14688) · [SAFE-KV](https://arxiv.org/abs/2610.15186) · [LatentRemedy](https://arxiv.org/abs/2610.18014) · [Think Before You Speak](https://arxiv.org/abs/2610.16165) | 🟢 Oct 2026 | Prefill-free KV transfer across families (40% accuracy, 4× wall-clock); **fidelity is the wrong objective** — behavior-coupled correction (≤7.27× lower handoff latency); rank-disagreement selective recomputation (3.04× prefill, −6.4× P99 TTFT); non-prefix reuse for RAG/multi-agent with causal-suffix repair / latent remedies; training-free KV interpolation (+8.6 pts, 12.2× TTFT). **PRISM's multi-model registry should expose context handoff as a first-class API.** |
| Batch / offline + determinism runtime | [OpRAG](https://arxiv.org/abs/2608.08340) | 🟢 Aug 2026 | Resource-deterministic multi-stage GPU runtime (embedding → search → decode) — the template for offline/batch jobs with predictable resource envelopes. |
| Flow control / SLO | [Stable Token-level Scheduling](https://arxiv.org/abs/2610.17689) · [Carbon-SLO tension](https://arxiv.org/abs/2610.14571) | 🟢 Oct 2026 | Token-window admission control; geo-distributed SLO-vs-carbon scheduler ladder (MILP: −60% carbon at ~3% violations). |
| Health / observability hooks | [TELLER](https://arxiv.org/abs/2608.01975) (§N) | 🟢 | Cross-layer request-span root-cause analysis — the readiness/debug surface behind /health. |
| Tool calling | XGrammar-2 (§I) + SchemaWeaver (§I) | 🟢 2026 | Schema-constrained tool calls with streaming partial JSON are now table stakes; these are the two 2026 engines. |

---

## M. Build, Planning & Tuning

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **Cost / roofline model** | [Compositional Performance Modeling](https://arxiv.org/abs/2610.15366) | 🟢 Oct 2026 | Calibrated discrete-event models that transfer across workers without refitting (1.34% mean error, 15.3× faster replay). **Optimal pick — PRISM's pre-run planner, validated against SGLang.** |
| Empirical tuning baseline | [Lightweight Scheduling Policies study](https://arxiv.org/abs/2610.15899) | 🟢 Oct 2026 | Scheduler choice alone moves every metric; oracle length knowledge is insufficient — justifies autotuning over static policy. |
| Auto-tuner search efficiency | [TraceDSE](https://arxiv.org/abs/2610.07191) / [ATTUNE](https://arxiv.org/abs/2610.17528) | 🟢 Oct 2026 | 6–9× fewer evaluations to a better config; agentic throughput tuning for training runs. |
| Kernel autotuning | ExaGEMM 2607.14622 · FireQ 2505.20839 🟠 | 🟢/🟠 | Autotuned exascale GEMM selection; 2025 fallback: FireQ's auto-tuned FP8 kernel pipelines for quantized serving. |
| **Quality/speed harness** | [D2K-Bench](https://arxiv.org/abs/2610.03226) · [RESOLVE](https://arxiv.org/abs/2610.05683) · [EdgeLLM-Nano](https://arxiv.org/abs/2610.14475) · [Green AI study](https://arxiv.org/abs/2610.17531) | 🟢 2026 | Kernel-correctness benchmarking + proof-backed validation; new fidelity metric separating equal-perplexity backends; unified energy study. The codec/plan sweep harness's acceptance tests. |
| AOT build / packaging / signed bundles | *(no paper — engineering)* | — | **Gap:** adopt standard reproducible-build + signing practice; nothing in this year's literature changes it. |

---

## N. Observability & Reliability

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **Per-op root cause** | [TELLER](https://arxiv.org/abs/2608.01975) | 🟢 Aug 2026 | Non-intrusive cross-layer analysis spanning engine + CUDA + comms — the tracer for a multi-backend stack. **Optimal pick.** |
| Determinism / error taxonomy | [Expert-aggregation determinism](https://arxiv.org/abs/2607.28097) | 🟢 Jul 2026 | Tests that must pass before any MoE sharding or nondeterministic expert ordering ships. |
| Stage-replay divergence | [Stage-Replay Divergence](https://arxiv.org/abs/2607.28495) | 🟢 Jul 2026 | Fresh-prefill vs retained-cache continuation is not exact in BF16 at stage boundaries — constrains the plan-cache/debug replay design. |
| Losslessness auditing | [Losslessness under Finite Precision](https://arxiv.org/abs/2609.15504) | 🟢 Sep 2026 | 45% trajectory divergence in BF16 with no downstream degradation — the audit protocol for spec-decoding "lossless" claims. |
| Sparse-path auditing | [Sparse-attention selectivity audit](https://arxiv.org/abs/2608.01676) | 🟢 Aug 2026 | Counterfactual probe framework proving sparse routing *causes* output changes (13/16 cells) — required before exposing any sparse mode. |
| Safety drift under quant | [RTSI](https://arxiv.org/abs/2606.10154) | 🟢 Jun 2026 | Measure refusal/safety per quant tier, not just perplexity. |
| Fidelity metric (new) | [EdgeLLM-Nano](https://arxiv.org/abs/2610.14475) | 🟢 Oct 2026 | Long-Context Decode Fidelity separates backends that perplexity scores identically. |
| Memory snapshots / warmup / model-cache LRU | *(no direct paper)* | — | **Gap:** standard engineering; When Fancy Eviction Fails (§C) covers the disk/model-cache LRU evidence. |

---

## O. Security & Governance

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **Spec-decoding attack surface** | [Secure Speculative Decoding](https://arxiv.org/abs/2610.08678) / [Speedbumps](https://arxiv.org/abs/2610.10929) | 🟢 Oct 2026 | Stricter verification for early draft tokens improves jailbreak/prompt-injection posture; rejection attacks demonstrate the lossy-SD threat model. **PRISM's self-speculation module needs this hardening by design.** |
| Depth-skipping safety | [Jailbreak vulnerability across depth-based dynamic inference](https://arxiv.org/abs/2610.15207) | 🟢 Oct 2026 | Shallow exits are *more* robust; adversarial suffixes optimize easiest against deep exits — a real constraint on the adaptive-depth controller (§H). |
| Context-handoff safety | [To Cache or Not to Cache (VReparo)](https://arxiv.org/abs/2610.15721) | 🟢 Oct 2026 | Cross-model KV reuse perturbs safety behavior; value reuse far worse than keys; learned affine repair cuts HarmBench ASR 57%→39% while keeping 85.6% TTFT reduction. Mandatory for the §L handoff API. |
| Input/tokenization surface | [BPE exploitable gaps](https://arxiv.org/abs/2607.01239) (§B) | 🟢 | Sub-word fragmentation bypasses keyword filters; healing/byte-fallback as mitigation. |
| Isolation (federated/edge) | [PrivPair](https://arxiv.org/abs/2609.09794) | 🟢 Sep 2026 | Activations leak input text; obfuscate-and-recover pairs for split inference — relevant if PRISM's distributed tier (§K) crosses trust boundaries. |
| Artifact signatures / SBOM / sandboxing | *(no 2025–2026 paper found)* | — | **Gap:** adopt sigstore/TUF-style verification; the research literature this year does not address model-artifact supply chain. |

---

## Q. Quality & Benchmarking

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **LLM evaluation frameworks** | [Systematic Evaluation of LLM Benchmarks](https://arxiv.org/abs/2601.03986) · [Bench360: Benchmarking Local LLM Inference](https://arxiv.org/abs/2511.16682) | 🟢 Jan 2026 / 🟠 2025 | Comprehensive benchmark suites spanning mathematics, reasoning, and knowledge domains; 11 LLMs across 15 benchmarks; establishes baseline quality before deployment. |
| **Inference-time latency evaluation** | [Human-less LLM Serving: Quantifying the Human Tax](https://arxiv.org/abs/2606.20577) · [Prompt-Aware Scheduling for Low-Latency LLM Serving](https://arxiv.org/html/2510.03243v2) | 🟢 May 2026 / 🟠 Oct 2025 | Measures TTFT/TPOT SLOs as perceived by human users; framework for quantifying the "human tax" on latency across serving systems. |
| **Carbon-aware evaluation** | [A Reference Framework for LLM Inference Carbon](https://arxiv.org/abs/2603.02949v1) | 🟢 Mar 2026 | Guiding principles for LLM inference carbon framework; measures energy-proportional efficiency per-token across models and hardware. |
| **LLM-as-a-Judge evaluation** | [LLM Evaluation: Frameworks, Metrics, and Best Practices](https://medium.com/@future_agi/llm-evaluation-frameworks-metrics-and-best-practices-2026-edition-162790f831f4) · [Top 10 Papers on LLM Evaluation](https://www.reddit.com/r/LangChain/comments/1j529k0/top_10_papers_on_llm_evaluation_benchmarking_and/) | 🟢 2026 | Framework for using LLMs as judges; consistency, correlation with human eval, and best practices for automated quality assessment. |
| **Real-time low-latency serving metrics** | [Human-less LLM Serving](https://arxiv.org/abs/2606.20577v1) · [Stable Token-level Scheduling](https://arxiv.org/abs/2610.17689v1) | 🟢 2026 / 🟠 Oct 2025 | TTFT/TPOT measurement frameworks; feedback-driven token-level scheduling for serverless inference; latency-stable serving under variable load. |

## R. Resource & Energy Modeling

| Sub-component | Paper | When | Relevance |
|---|---|---|---|
| **Energy-proportional inference** | [Green AI energy study](https://arxiv.org/abs/2610.17531) / [ExaGEMM](https://arxiv.org/abs/2607.14622) · [PolyQ](https://arxiv.org/abs/2607.14618) · [Early-Stop Accumulations](https://arxiv.org/abs/2608.06177) | 🟢 2026 | 10 PTQ × 5 models × 4 GPUs: fused-kernel int8 saves up to 4× energy vs FP32; quantized GEMM peaks at long-input/short-output. |
| **Carbon budget per-token** | [A Reference Framework for LLM Inference Carbon](https://arxiv.org/abs/2603.02949v1) · [Carbon-SLO tension](https://arxiv.org/abs/2610.14571) | 🟢 2026 | MILP: −60% carbon at ~3% violations; per-token carbon budget tracking; geo-distributed SLO-vs-carbon scheduler ladder. |
| **Hardware efficiency ratios** | [ExaGEMM autotuning](https://arxiv.org/abs/2607.14622) · [Vendor math libs precision dispatch](https://arxiv.org/abs/2607.14618) · [Early-Stop Accumulations](https://arxiv.org/abs/2608.06177) | 🟢 Jul–Aug 2026 | Exascale GEMM autotuning; multi-precision kernel selection; accumulation early-stopping — the precision-ISA dispatch menu for energy efficiency. |
| **QoS / latency SLO enforcement** | [Stable Token-level Scheduling](https://arxiv.org/abs/2610.17689) · [Priority / SLA manager](https://arxiv.org/abs/2610.16592) (§G) · [Predicted-Latency Based Scheduling](https://llm-d.ai/blog/predicted-latency-based-scheduling-for-llms) | 🟢 2026 | Token-window admission control; oracle-length-knowledge loses to batching/hardware effects; aging-shortest-prompt-first strongest practical baseline. |
| **Model-size vs latency curves** | *(no single 2026 paper owns this)* | ⚪ | Engineering: derive from benchmark suites (Q) + hardware profiles; no canonical 2026 paper defines the curve. |

## P. Extensions

| Extension | Paper | When | Relevance |
|---|---|---|---|
| Vision (VLM serving) | [SP²ec](https://arxiv.org/abs/2610.18735) (§H) | 🟢 | Self-speculative bandits for VLMs — decode acceleration for the vision extension. |
| Vision (video KV) | [CORAL-KV](https://arxiv.org/abs/2610.12514) | 🟢 Oct 2026 | Cross-layer prediction + residual coding for autoregressive video KV (37–48% less cache) — the codec pattern for visual-token span stores. |
| Cross-modal handoff | [Swarm Telepathy](https://arxiv.org/abs/2610.15095) | 🟢 Oct 2026 | A VLM sender's vision ability passed to a text-only receiver via KV transfer — multimodal spans without text round-trips. |
| Embedding head | [Quantize by Drift](https://arxiv.org/abs/2610.09227) | 🟢 Oct 2026 | Label-free mixed-precision PTQ for retrievers (Spearman 0.911) — quantizing the embedding extension. |
| Reranker / reward heads, audio ASR/TTS | *(no strong 2025–2026 engine papers found in this window)* | — | **Gap:** treat as model-side plugs; nothing in the Oct 2026 window constrains the design. |

---

## Coverage Gaps Summary (no strong 2025–2026 paper — build from adjacent systems)

1. **CAS tensor store + provenance/signatures** (A) — engineering: content-addressed storage, sigstore/TUF-style signing.
2. **Compaction/defragmentation + allocation arenas** (C) — borrow from database buffer managers.
3. **Microbatch shape classes** (G) — PRISM novelty; motivated by plan-cache hits (§D).
4. **Sampler theory** (I) — min-p/typical/typical remain community practice; seed manager is determinism work (§N).
5. **Streaming delta assembly / stop trimming** (I) — protocol engineering.
6. **AOT packaging / signed deploy bundles** (M) — reproducible-build practice.
7. **Audio codecs, reranker/reward heads** (P) — model-side plugs.
8. **Net-tier P2P fetch transport** (A/K) — the KV-handoff cluster (§L) covers the *semantic* net tier; byte-level P2P transport remains engineering.

---

## Appendix — Consolidated Index (arXiv ID → component)

**A:** 2605.18475 · 2605.02404 · 2610.18642 · 2610.13328 · 2610.14745 · 2606.12876 · 2510.20984 · 2610.14934 ·
2610.13767 · 2601.07475 · 2610.15092 · 2609.16338 · 2610.16890 · 2608.00860 · 2602.02546 · 2610.09227 ·
2609.11744 · 2609.26828 · 2609.18063 · 2610.08312
**B:** 2610.13325 · 2601.23223 · 2602.13940 · 2607.01239 · 2503.23462 · 2412.18668 · 2410.09303
**C:** 2609.18110 · 2609.12978 · 2609.38090 · 2605.27081 · 2609.04895 · 2607.24787 · 2608.08097 · 2609.28870 ·
2609.35065 · 2609.01338 · 2609.21172 · 2608.07009 · 2610.07593 · 2610.09657 · 2607.22785
**D:** 2609.37916 · 2610.13325 · 2605.13915 · 2610.17531 · 2608.09291 · 2610.11134 · 2610.11201 · 2610.03226 ·
2610.05014 · 2610.05683 · 2610.05123 · 2609.21450 · 2609.15504
**E:** 2605.19049 · 2610.14754 · 2610.11214 · 2610.15715 · 2610.05842 · 2610.13735 · 2610.10114 · 2610.08811 ·
2610.11245 · 2610.17966 · 2610.06927 · 2610.17970 · 2610.06286 · 2610.18642 · 2610.18026 · 2610.05685 ·
2610.15384 · 2610.09827 · 2610.03027 · 2610.10381 · 2610.14475 · 2610.14448
**F:** 2609.34556 · 2605.10414 · 2609.38109 · 2605.09472 · 2605.04217 · 2604.01178 · 2410.07668 · 2610.17959 ·
2610.17202 · 2610.10871 · 2610.09025 · 2608.01662 · 2406.06484 · 2504.14366 · 2608.01651 · 2610.13202
**G:** 2602.14516 · 2604.15039 · 2610.06917 · 2608.16336 · 2602.11530 · 2605.16867 · 2610.14686 · 2610.17689 ·
2610.15899 · 2610.17795 · 2610.18451 · 2610.11158 · 2610.14804
**H:** 2609.36590 · 2602.20217 · 2610.18735 · 2610.02800 · 2606.18394 · 2602.01469 · 2610.12570 · 2602.06036 ·
2607.05147 · 2606.27550 · 2606.03113 · 2610.07659 · 2610.15417 · 2609.33184 · 2610.13708 · 2603.03251 ·
2609.27396 · 2609.24197 · 2609.29364 · 2609.20186 · 2610.07350 · 2610.10411 · 2609.24150 · 2608.30135 ·
2608.09444 · 2302.01318 · 2510.01336 · 2605.08632
**I:** 2601.04426 · 2608.03065 · 2502.05111 · 2312.07104 · 2610.17436
**J:** 2609.37916 · 2609.10632 · 2608.01563 · 2610.17321 · 2608.08730 · 2604.02344 · 2607.00501 · 2610.03394 ·
2610.05123 · 2609.16338 · 2607.28418 · 2609.32114 · 2610.07191 · 2607.17415 · 2607.12839 · 2607.14622 ·
2607.14618 · 2608.06177 · 2610.05305 · 2608.05944
**K:** 2610.07191 · 2610.15366 · 2609.18110 · 2609.12978 · 2609.18063 · 2610.09424 · 2610.08268 · 2609.38697 ·
2610.18451 · 2610.11158 · 2610.03741 · 2604.15039 · 2607.29659 · 2608.02378 · 2608.03436 · 2608.02031
**L:** 2610.12242 · 2603.21354 · 2610.15095 · 2610.15509 · 2610.11358 · 2610.14688 · 2610.15186 · 2610.18014 ·
2610.16165 · 2608.08340 · 2610.17689 · 2610.14571 · 2608.01975
**M:** 2610.15366 · 2610.15899 · 2610.07191 · 2610.17528 · 2607.14622 · 2505.20839 · 2610.03226 · 2610.05683 ·
2610.14475 · 2610.17531
**N:** 2608.01975 · 2607.28097 · 2607.28495 · 2609.15504 · 2608.01676 · 2606.10154 · 2610.14475 · 2610.15366
**O:** 2610.08678 · 2610.10929 · 2610.15207 · 2610.15721 · 2607.01239 · 2609.09794 · 2606.10154
**P:** 2610.18735 · 2610.12514 · 2610.15095 · 2610.09227

**Q:** 2601.03986 · 2511.16682 · 2606.20577 · 2603.02949v1 · 2610.17689 · 2610.14571 · 2510.03243
**R:** 2610.17531 · 2607.14622 · 2607.14618 · 2608.06177 · 2610.14571

---

*Compiled from repo digests (ML-NEW-TECH Parts 8–9, ENGINE_MASTER_SPEC §11–13, data/arxiv_latest_20261015.json)
plus web verification, 15 Oct 2026. All numbers are author-reported. PRISM remains a clean-room design;
the papers above are evidence for or implementations of individual components, not dependencies of the design.*
