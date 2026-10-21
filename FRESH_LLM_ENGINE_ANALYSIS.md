# Deep Analysis: What’s Missing to Create a Fresh LLM Engine (2024–2026)

This analysis synthesizes findings from `ENGINE_MASTER_SPEC.md`, `PRISM_2026_PAPER_MAP.md`, `ML-NEW-TECH.md` (Parts 1–3), the `classified/` corpus, and the full arXiv dataset. The goal is to produce a concrete, actionable blueprint for building a fresh LLM inference engine from scratch, identifying all missing pieces and gaps from 2024–2026.

## 1. Executive Summary

The research corpus gives us an exceptional map of modern LLM inference. The biggest insight from 2026 papers is a shift in priorities:
- **Bandwidth, not FLOPS** dominates optimization (DuoRate, ReadKV, GraphTok)
- **Hybrid architectures** (linear attention + sparse/MLA) are mainstream (Gated DeltaNet, KVBuffer, Orthros)
- **Tiered residency** is now standard practice (SSD-LLaMA, HiSparse, OasisKV)
- **Self-speculation** has overtaken external drafters in many cases (SEED, BitNest)
- **Context handoff** between models is emerging as a new primitive (Swarm Telepathy, CacheLink)
- **Determinism/non-determinism** in MoE is a real concern (expert reduction order causes behavioral divergence)

However, to build a *fresh* engine (not extend existing ones), there are critical missing pieces that span architecture, implementation details, and operational concerns.

## 2. Current State Assessment

### What We Have (Well-Covered)
| Area | Coverage | Notes |
|---|---|---|
| **KV Cache Management** | ★★★★★ | AnchorKV, SAKI, OptR, QEvict, RestoreKV, OasisKV, HiSparse all covered |
| **MoE Routing & Caching** | ★★★★★ | Pre-attention predictors, ExpertFlow, Fate, Speculating Experts well-mapped |
| **Speculative Decoding** | ★★★★★ | Self-speculation (SEED), MTP, adaptive depth, tree speculation all present |
| **Quantization** | ★★★★★ | GAMMA/SLQ allocation, SAGE VQ, mixed-precision, KV quant well-covered |
| **Multi-backend/Portability** | ★★★★☆ | RLX, Meganeura, Vulkan/Metal paths identified but integration details sparse |
| **Tiered Residency/Offload** | ★★★★★ | SSD-LLaMA, SeqMoE, Mira give concrete blueprints |

### What's Missing or Underspecified
| Area | Coverage | Critical Gap |
|---|---|---|
| **End-to-end engine architecture** | ★★☆☆☆ | No single reference showing how all pieces compose into one coherent system |
| **Scheduler design details** | ★★★☆☆ | Vesta gives hints of lane-based design, but full FSM, microbatching, backpressure underspecified |
| **Memory allocator/arena** | ★★☆☆☆ | KV block pools, expert arenas, compaction mentioned but implementation patterns missing |
| **Tokenizer/runtime integration** | ★★★☆☆ | GraphTok shows CUDA-graph tokenization but general integration unclear |
| **Determinism harness** | ★★★☆☆ | Expert aggregation order divergence noted but full testing framework not specified |
| **Model format/loading** | ★★★☆☆ | Tied Trit-Planes, FORGE mentioned but unified loader design missing |
| **Cross-model/context handoff** | ★★★★☆ | Papers exist but API design, security model, and integration underspecified |
| **Observability/Debugging** | ★★★☆☆ | TELLER mentioned but full tracing/metrics design missing |

## 3. Missing Components for a Fresh Engine

### 3.1 Core Runtime Architecture (Critical)

| Component | Priority | What's Missing | References to Pull From | Implementation Notes |
|---|---|---|---|---|
| **Unified IR/Execution Graph** | P0 | Need concrete IR design (not just "use MLIR") - tensor ops, control flow, memory ops, backend lowering | RLX (2609.37916), GraphTok (2610.13325) | Design a minimal IR: SSA form, memory views, async ops, backend dialects. RLX is closest to what we need. |
| **Memory Manager/Arena Allocator** | P0 | Block pools for KV, expert arenas, tensor arenas with compaction/defrag | When Fancy Eviction Fails (2609.28870), OasisKV (2608.08097) | Need explicit arena design - bump allocators, free lists, compaction strategies. Block-based KV allocation is table stakes. |
| **Command Replay/Plan Cache** | P0 | CUDA-graph style replay but cross-backend (Vulkan/Metal) | GraphTok (2610.13325), HiSparse (2608.07009) | Replay tokenization, attention bookkeeping, expert loading plans. Critical for latency. |
| **Backend Abstraction Layer** | P0 | Clean interface for CUDA/ROCm/Vulkan/Metal/WebGPU with capability discovery | Meganeura (2608.01563), RLX (2609.37916) | Registry pattern, kernel dispatch, memory ops, sync primitives. Must handle backend differences (determinism). |

### 3.2 Scheduler & Request Lifecycle (Critical)

| Component | Priority | What's Missing | References to Pull From | Implementation Notes |
|---|---|---|---|---|
| **Lane-based Scheduler** | P0 | FSM for request phases (prefill/reasoning/decoding/idle), per-lane token clock, backpressure | Vesta (2610.14686), Pascal (2602.11530), Stable Token-level Scheduling (2610.17689) | Vesta is closest - session-pinned KV, per-request executor choice. Need concrete FSM states. |
| **Microbatch Shape Classes** | P0 | Packing by shape (seq len, batch composition) for plan-cache hits - PRISM novelty | Follow vLLM BatchDescriptor/ShapeKey + SGLang ShapeKey + FlashInfer plan reuse + BucketServe length-bucketing + TensorRT-LLM captured-size bucketing | Use phase-aware keys (prefill/decode), normalize size meaning (token vs request count), track graph-plan-cache hits separately from KV cache. Consider prefix-cache affinity vs length homogeneity tradeoffs. |
| **P/D Disaggregation Support** | P1 | Prefill/decode separation with KV transfer, role switching | AMPD (2602.14516), Prorsa (2610.16592), FluidPD (2610.06917) | For multi-GPU, but also useful for resource isolation. KV transfer mechanism critical. |
| **SLO-aware Admission Control** | P1 | Token-window pacing, per-request budgets, backpressure | Stable Token-level Scheduling (2610.17689), Carbon-SLO (2610.14571) | Need admission controller with queue depth monitoring. |

### 3.3 Model Loading & Format (High Priority)

| Component | Priority | What's Missing | References to Pull From | Implementation Notes |
|---|---|---|---|---|
| **Unified Model Loader** | P0 | Support GGUF, Safetensors, folded layouts, procedural weights | Tied Trit-Planes (2608.08910), FORGE (2610.16890), Kilobyte Models (2608.00860) | Plugin architecture for formats. Handle memory-mapped loading for disk streaming. |
| **Quantization Pipeline** | P0 | Runtime quant selection, per-tensor codecs, mixed precision allocation | GAMMA (2605.18475), SLQ (2605.02404), SAGE (2610.14745), MultiAlloc (2610.18642) | Need sensitivity store, quality targets, allocation solver. GAMMA's IP approach is good starting point. |
| **Expert Loading/Streaming** | P0 | Lazy loading from disk, prefetch, cache management for MoE | SSD-LLaMA (2609.18110), SeqMoE (2609.12978), AcceptMoE (2608.02989) | Critical for running large MoE on consumer hardware. Three-tier (VRAM/RAM/SSD) needed. |
| **Procedural Weight Regeneration** | P2 | Seed+latent decompression for extreme cases | FORGE (2610.16890), Kilobyte Models (2608.00860) | Nice-to-have but enables very small model files. Can be added later. |

### 3.4 Attention & Context Management (High Priority)

| Component | Priority | What's Missing | References to Pull From | Implementation Notes |
|---|---|---|---|---|
| **Paged KV Cache** | P0 | Block allocation, prefix sharing, eviction | PagedAttention (2309.06180) - foundational, but need modern updates | Table stakes. Need to integrate with eviction policies. |
| **Hybrid Attention Support** | P0 | Gated DeltaNet/linear attention + full attention mixing | KVBuffer (2605.19049), Gated DeltaNet (2406.06484), Orthros (2610.13735) | Qwen3-Next style hybrids are common now. Need recurrent state management. |
| **KV Compression Pipeline** | P0 | Composable compression (AnchorKV style + quantization + eviction) | AnchorKV (2608.02901), SAKI (2608.03228), ReadKV (2610.11245), KVFetch (2610.08811) | ReadKV's "read what matters" is particularly important for cold tiers. |
| **Span Store for Long Context** | P1 | Compressed cold storage with retrieval | KVFetch (2610.08811), BELT (2610.17966), AlignPred (2610.17970) | For very long contexts beyond VRAM. |

### 3.5 Decoding & Speculation (High Priority)

| Component | Priority | What's Missing | References to Pull From | Implementation Notes |
|---|---|---|---|---|
| **Self-Speculation** | P0 | SEED-style or BitNest-style without external draft model | SEED (2609.36590), BitNest (2610.02800), KnapSpec (2602.20217) | Much better for memory-constrained scenarios. SEED is a strong starting point. |
| **MTP Support** | P1 | Multi-token prediction heads | Strata MTP mentioned in ENGINE_MASTER_SPEC, training objectives in papers | If models have MTP heads, need to support them. |
| **Adaptive Speculation** | P1 | Depth controller, tree topology, offload-aware speculation | DSpark (2607.05147), DLoop (2610.07659), SpecFit (2610.15417) | Optimize speculation based on acceptance rate and hardware state. |
| **Grammar-Constrained Decoding** | P1 | XGrammar-style integration | XGrammar-2 (2601.04426), SchemaWeaver (2610.17436) | Essential for tool use/structured output. |
| **Diffusion LLM (dLLM) Support** | P1 | Bidirectional attention breaks AR KV cache; need block diffusion or approximate caching | Fast-dLLM (2505.22618), BD3-LM (2503.09573), dKV-Cache (NeurIPS 2025), dLLM-Cache (ICML 2026), Elastic-Cache, DPad, Sparse-dLLM, EB-Sampler | Support block diffusion architecture or approximate KV reuse with delayed refresh, track cache validity per token/layer, handle non-contiguous positions/RoPE, use confidence/entropy-bounded parallel unmasking. Profile by batch size (gains vary). |

## 4. Critical Gaps Not Well Covered in Literature

These are areas where papers mention the problem but don't give complete solutions, or where we need to invent/integrate.

### 4.1 System Integration Gaps

| Gap | Severity | Description | What We Need to Build | Notes |
|---|---|---|---|---|
| **End-to-end composition** | Critical | No paper shows full integration of all pieces. How do scheduler, memory manager, backends, speculation all interact? | Integration architecture, dependency management, lifecycle | This is the hardest part of building fresh. Need to design interfaces carefully. |
| **Configuration & Tuning** | Critical | TraceDSE/ATTUNE exist but no unified config system | Config framework, hardware detection, auto-tuning hooks, profile system | Need to make it work well across diverse hardware without manual tuning. |
| **Error Handling & Recovery** | High | Papers focus on happy path; crashes, OOM, backend failures underexplored | Graceful degradation, OOM handling, retry logic, checkpointing | Especially important for disk streaming/offload scenarios. |
| **Determinism Testing Framework** | Critical | Expert aggregation order causes divergence (2607.28097) - no full test harness described | Determinism tests across backends, sharding, batch sizes; golden outputs | Must test before shipping MoE sharding. CUDA/HIP not bit-identical per Strata - need tolerance handling. |

### 4.2 Operational/Production Gaps

| Gap | Severity | Description | What We Need to Build | Notes |
|---|---|---|---|---|
| **Logging/Metrics/Tracing** | High | TELLER mentioned but full observability stack missing | Structured logging, Prometheus metrics, OpenTelemetry tracing, request spans | Need to debug cross-layer issues. |
| **Model Registry/Discovery** | High | No mention of how to discover/load models, manage versions | Model registry, local cache, download manager, verification | Similar to Ollama's model management but integrated. |
| **API Server Design** | High | OpenAI-compatible API with streaming, batching, tool use | HTTP server, WebSocket for streaming, request queuing, rate limiting | Need to be compatible with existing tooling. |
| **Security Model** | High | TRACER/MemLineage mention provenance but full security model missing | Sandboxing, input validation, output filtering, secrets management, model signing verification | For production use. |

### 4.3 Algorithmic/Research Gaps

| Gap | Severity | Description | What We Need to Build | Notes |
|---|---|---|---|---|
| **Microbatch Shape Classes** | High | Explicitly called out as PRISM novelty - no paper covers this | Shape classifier, packer optimizer, plan cache keying by shape class | Could give significant throughput gains. |
| **Cross-model Context Handoff Protocol** | High | Papers show it works but no standard protocol/API | Handoff API, KV transformation, compatibility checks, security model | Very promising for multi-model workflows. |
| **Unified KV Format Across Architectures** | High | Hybrid models complicate things (recurrent state + KV) | Abstract KV interface, serialization, transfer format | Needed for P/D disaggregation and handoff. |
| **Weight-Streaming Orchestration** | High | Papers cover pieces but no unified policy | Prefetch policy, eviction policy, admission control for disk loads, bandwidth management | Complex to get right for good UX on slow storage. |

## 5. Missing Items from 2024-2026 (Deeper Scan)

Let me think about what's emerged recently that might not be fully captured:

### 2024 Patterns Still Relevant
- **vLLM PagedAttention** (2309.06180) - foundational, we have it
- **SGLang RadixAttention** (2312.07104) - prefix caching, mentioned
- **GGUF/llama.cpp ecosystem** - quantization, CPU kernels, very relevant for constrained hardware

### Emerging 2026 Trends Not Fully Covered
| Trend | Status in Corpus | Missing Details | Action |
|---|---|---|---|
| **Diffusion LLMs (DLLMs)** | Mentioned as "al dllm inference" gap in PRISM §A | Now researched - block diffusion, approximate KV caching (dKV-Cache, dLLM-Cache, Fast-dLLM), sparse attention (DPad, Sparse-dLLM), entropy-bounded sampling | Implement block-diffusion structure or approximate bidirectional caching with confidence-aware parallel decoding; note bidirectional attention breaks standard AR KV cache
| **Video/Multimodal Serving** | Covered in §P lightly | CORAL-KV for video but full multimodal pipeline unclear | Vision encoders, cross-modal attention, different memory patterns |
| **Agentic Workflows** | Covered in §L, 3 | Long-running agents, tool use, memory systems, context management across turns | Need agent runtime integration, not just inference |
| **Reasoning Models (long CoT)** | Mentioned throughout | Very long contexts, thinking traces, different scheduling needs | May need special handling for long reasoning traces |
| **Test-Time Compute Scaling** | Covered in Part 1 §2 | Integration with engine scheduling/admission control | Need to coordinate with batch scheduler |

### Specific Papers/Techniques Possibly Missing
Based on scanning ML-NEW-TECH more carefully:
- **Speculative Decoding with KV Reuse** - some coverage but interplay complex
- **Prompt Caching at Scale** - RadixAttention covered but distributed prefix cache?
- **Disaggregated Serving** - covered but full implementation details sparse
- **Multi-LoRA Serving** - mentioned (SALT) but scheduling/routing unclear
- **CPU-optimized Kernels** - ggml/llama.cpp style mentioned but not deeply analyzed
- **NUMA Awareness** - HiNa-MoE mentions NUMA but general strategy missing

## 6. Concrete Implementation Plan for a Fresh Engine

### Phase 0: Foundation (Weeks 1-4)
Focus on getting the skeleton right.

| Task | Priority | Dependencies | Details |
|---|---|---|---|
| **Project Structure** | P0 | None | Modular design: core/, backends/, sched/, mem/, models/, api/, utils/ |
| **Backend Abstraction** | P0 | None | Start with CPU reference impl, add CUDA via custom kernels or cutlass bindings later. Or use Meganeura approach. |
| **Tensor/Memory Primitives** | P0 | Backend | Basic tensor ops, memory allocation, device abstraction |
| **Logging/Config** | P0 | None | Structured logging, TOML/YAML config, CLI parsing |

### Phase 1: Basic Inference (Weeks 4-8)
Get a simple model running end-to-end.

| Task | Priority | Dependencies | Details |
|---|---|---|---|
| **Model Loader (GGUF first)** | P0 | Backend, Memory | GGUF is simplest, widely used, good for constrained hardware. Use gguflib or implement minimal parser. |
| **Tokenizer** | P0 | None | Start with HuggingFace tokenizers or implement BPE. GraphTok ideas for optimization later. |
| **Forward Pass** | P0 | Model Loader | Basic transformer forward pass for dense models first (easier than MoE). |
| **KV Cache (basic paged)** | P0 | Memory Manager | Simple block allocation, no fancy compression initially. |
| **Sampling** | P0 | None | Temperature, top-p, top-k - standard. |
| **Simple API** | P0 | Scheduler | Basic OpenAI-compatible completions endpoint |

### Phase 2: Performance Optimizations (Weeks 8-12)
Add the high-impact optimizations from papers.

| Task | Priority | Dependencies | Details |
|---|---|---|---|
| **Prefix Caching (Radix)** | P0 | KV Cache | RadixAttention style prefix reuse. Big win for multi-turn. |
| **Continuous Batching** | P0 | Scheduler | Interleave prefill/decode. Essential for throughput. |
| **Quantization Support** | P0 | Model Loader, Kernels | GGUF quants (GGML type system) - leverage existing kernel knowledge. |
| **PagedAttention** | P0 | Memory Manager | Proper block management, sharing. |
| **Basic Speculative Decoding** | P1 | Scheduler | Start with n-gram or draft model, add self-speculation later. |

### Phase 3: Advanced Features (Weeks 12-16)
Add MoE, tiered residency, advanced optimizations.

| Task | Priority | Dependencies | Details |
|---|---|---|---|
| **MoE Support** | P0 | Router, Expert Cache | Token-choice routing, expert caching, load balancing. |
| **Hot Expert Prediction** | P1 | MoE, Profiler | Start with simple heuristics, add learned predictors (Pre-Attention style). |
| **Tiered Residency** | P1 | Disk I/O, Memory Manager | RAM + disk offload for weights/KV. Use mmap + prefetch. |
| **KV Compression** | P1 | KV Cache | Add AnchorKV-style or simple quantization. |
| **Self-Speculation (SEED)** | P1 | Speculation framework | High value for memory-constrained. |
| **Multi-backend** | P2 | Backend Abstraction | Add Vulkan via wgpu/ash or use existing libraries. |

### Phase 4: Polish & Production Readiness (Weeks 16+)
| Task | Priority | Dependencies | Details |
|---|---|---|---|
| **Observability** | P1 | All | Metrics, tracing, logging. |
| **Determinism Testing** | P0 | MoE | Critical - test expert order sensitivity. |
| **API Completeness** | P1 | API Server | Chat completions, function calling, streaming, etc. |
| **Documentation** | P1 | All | Architecture docs, deployment guides. |
| **Benchmarking** | P1 | All | vs llama.cpp/vllm on same hardware. |

## 7. Technology Stack Recommendations

### Language Choice
- **Core engine**: Rust (memory safety, performance, great for systems programming, good async, growing ML ecosystem)
- **Or C++**: More mature for ML kernels, easier to integrate with CUDA/ROCm
- **Or Zig**: Interesting alternative but less ecosystem

**Recommendation**: Rust for safety and maintainability, with careful FFI for vendor libraries (CUDA, etc.)

### Key Dependencies/Libraries
| Component | Recommendation | Rationale |
|---|---|---|
| **GGUF Parsing** | Use `gguf` crate or implement | Well-defined format, good for starting |
| **Tokenization** | `tokenizers` crate (HuggingFace) | Battle-tested, supports most tokenizers |
| **HTTP Server** | `axum` or `actix-web` | Async, good ecosystem |
| **CUDA Bindings** | `cudarc` or `rust-cuda` | For NVIDIA support |
| **Vulkan** | `ash` or `wgpu` | wgpu is more portable, ash is lower level |
| **Tensor Ops** | Implement minimal or use `candle-core` patterns | Candle has good abstractions to learn from |

### Build System
- Cargo for Rust core
- CMake for C/CUDA kernels
- Or use `cc` crate to compile kernels

## 8. Critical Design Decisions

| Decision | Options | Recommendation | Rationale |
|---|---|---|---|
| **Start Model Support** | Dense only vs MoE from start | Dense first, MoE second | Simpler to get right, MoE has many gotchas (routing, determinism) |
| **Backend Strategy** | Port everything vs use libraries | CPU + CUDA first, add others later | Focus on correctness then portability |
| **Quantization Approach** | GGUF runtime vs custom | Support GGUF quants natively (like llama.cpp) | Leverages existing model ecosystem |
| **Memory Model** | Custom allocator vs system malloc | Custom arena allocators | Critical for performance and control |
| **Async Runtime** | Tokio vs custom | Tokio | Mature, good for API server and async I/O |

## 9. Immediate Action Items

### Research/Reading (Do First)
1. **Read vLLM's PagedAttention paper** (2309.06180) - understand block management
2. **Read GraphTok** (2610.13325) - command replay is crucial
3. **Read Vesta** (2610.14686) - lane scheduler design
4. **Read SSD-LLaMA** (2609.18110) - disk streaming approach
5. **Study llama.cpp source** - reference implementation for GGUF, CPU kernels

### Prototyping (Do Second)
1. **Minimal GGUF loader** - parse and load a small model
2. **CPU forward pass** - basic matmul and attention
3. **Simple API server** - generate text
4. **Add paged KV cache** - enable longer contexts
5. **Add continuous batching** - improve throughput

### Validation (Ongoing)
1. **Compare against llama.cpp** - correctness and performance on same model
2. **Test determinism** - run same prompt multiple times
3. **Memory profiling** - ensure efficient usage
4. **Long context tests** - stress test KV cache

## 10. Final Recommendations

### Must-Haves for v1.0
- GGUF loading, CPU + CUDA backends
- Paged KV cache with prefix caching
- Continuous batching
- Basic speculative decoding
- OpenAI-compatible API
- Quantization support (GGUF quants)

### High-Value Additions
- MoE support with expert caching
- Tiered residency (RAM/disk offload)
- Self-speculation (SEED-style)
- Hot expert prediction
- Hybrid attention support

### Nice-to-Haves (Future)
- Multi-backend (Vulkan/Metal)
- P/D disaggregation
- Cross-model context handoff
- Procedural weight regeneration
- Full observability stack

## Key Insight

The biggest challenge isn't any single component - it's **integration**. The papers give us excellent solutions to individual problems, but composing them correctly is non-trivial. 

**Focus on getting a simple, correct engine working first, then incrementally add optimizations from the papers.** Start with the dense model case on GGUF, get it to match llama.cpp's behavior, then branch out to MoE and advanced features.

The corpus is comprehensive enough to build something competitive. The main work is engineering integration, not research - most of the hard algorithmic problems have been solved in recent papers. The gaps are mostly in how to put everything together cohesively.