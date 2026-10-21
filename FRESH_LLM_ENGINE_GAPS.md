# Detailed Gap Analysis with 2024-2026 Timeline

## 1. Timeline of Missing/Unsolved Items

### 2024 (Recent Foundational Gaps)
| Topic | Gap | Status in 2026 | Action Needed |
|---|---|---|---|
| **PagedAttention integration patterns** | How to integrate with tiered storage | Partially solved (vLLM + LMCache, SSD-LLaMA) | Need clean abstraction that supports arbitrary tiers |
| **Radix tree for prefix caching** | How to scale for long contexts & many requests | Better (HiSparse, KVFetch) | But distributed radix caches unexplored |
| **MoE routing determinism** | Expert aggregation order caused divergence | Discovered (2607.28097) - now identified as problem | Solution not yet proposed - must invent ordering guarantees |

### 2025 (Transitional)
| Topic | Gap | 2026 Progress | Action Needed |
|---|---|---|---|
| **Linear attention serving** | How to batch recurrent states | KVBuffer (2605.19049) addresses | But full hybrid model scheduling still unclear |
| **Speculative decoding memory overhead** | External drafters use extra VRAM | Self-speculation (SEED) reduces | But integration with tiered memory unexplored |
| **Cross-model KV reuse** | Was experimental | Better formalized (Cross-Model KV Transfer 2608.03893) | Still no standard protocol/API |

### 2026 (Emerging, Unsolved)
| Topic | Gap | Evidence | Priority | Notes |
|---|---|---|---|---|
| **Microbatch shape classes** | Now mapped to existing patterns | PRISM explicitly states "Gap/PRISM novelty" but implementations exist in vLLM/SGLang/FlashInfer/TensorRT-LLM/BucketServe | P0 | Shape-class packing for plan-cache hits - implement via phase-aware ShapeKey/BatchDescriptor (separate keys for graph dispatch vs attention plans vs request batching), with captured-size bucketing and length-affinity balancing. Tune bucket boundaries from traces. |
| **CAS tensor store + provenance** | Engineering gap | PRISM notes ahead of literature | P0 | sigstore/TUF style for model artifacts - could be differentiating |
| **Expert aggregation order** | Theoretical problem, no fix | 2607.28097 documents behavioral divergence | P0 | Need to define canonical ordering (e.g., fixed device order, stable sort by expert ID) - must implement defensively |
| **Net-tier P2P KV transport** | Byte-level transport unspecified | Semantic covered by KV handoff cluster | P1 | Cross-cluster KV transfer needs wire protocol |
| **DLLM inference** | No optimization work | "al dllm inference" marked ⚪ gap - now researched | P1 | Bidirectional attention breaks AR KV cache. Solutions: block diffusion (BD3-LM/Fast-dLLM v2), approximate KV caching (dKV-Cache, dLLM-Cache, Fast-dLLM DualCache), adaptive refresh (d²Cache, Elastic-Cache), sparse attention (DPad suffix windows, Sparse-dLLM), entropy/confidence-bounded parallel sampling (EB-Sampler). Need to track cache validity by token/layer and handle non-contiguous positions/RoPE. |
| **Streaming delta assembly** | Protocol engineering gap | I. Logit & Output gap | P1 | How to stream tokens with backpressure, partial tokens, stop conditions |
| **AOT packaging + signed bundles** | No LLM-specific std | M gap, engineering only | P1 | For reproducible, verifiable deployments |

## 2. What's Completely Missing (Not Covered at All)

These are components needed for a production engine that have minimal coverage:

| Component | Why It's Missing | References | Proposed Solution |
|---|---|---|---|
| **Process/Memory Management** | Not a research topic, but critical | None in papers | Design: memory arenas, OOM handling, graceful shutdown, resource limits |
| **Load Balancing/Scaling** | Multi-node serving details sparse | TokenRouter covers routing but not full LB | Need health checks, failover, request spilling |
| **Rate Limiting & Quotas** | Not research focus | None | Token-based rate limits, per-user quotas, priority classes |
| **Caching Beyond KV** | Papers focus on KV/expert caches | None cover prompt template cache, response cache | Add generic cache layer with TTL, LRU |
| **Model Warmup** | Mentioned in gap N but no paper | "Memory snapshots / warmup" gap | Implement model warmup (JIT compile kernels, prefill dummy requests) |
| **Telemetry/Exporters** | TELLER is tracing but not export | Limited | Add OpenTelemetry, Prometheus, JSON logging |
| **CLI/Management Tools** | No mention | N/A | Need CLI for serving, benchmarking, converting models |
| **Plugin System** | For extensibility | None | Allow custom codecs, schedulers, backends |
| **Multi-tenant Isolation** | Security considerations light | Limited | Request isolation, resource quotas, KV cache isolation strategies |

## 3. Critical Engineering Decisions (No Paper Gives Answer)

These are pragmatic choices where research gives tradeoffs but not definitive answers:

| Decision | Research Guidance | Tradeoffs | Recommendation |
|---|---|---|---|
| **Start with GGUF vs Safetensors** | Both used | GGUF has quantization baked in, easier for CPU. Safetensors cleaner. | Start with GGUF (matches llama.cpp, good for constrained hardware) |
| **Rust vs C++** | Papers don't specify | C++ has kernel ecosystem, Rust safer. | Rust core + C FFI for kernels (best of both) |
| **Custom kernels vs reuse** | Mix recommended | Custom gives control, reuse faster. | Start by wrapping ggml-style kernels, write custom later for bottlenecks |
| **Monolithic vs microservices** | Not discussed | Simpler monolithic for single-node, microservices for scale. | Monolithic first, modular enough to split later |
| **Sync vs async runtime** | Implicitly async in serving | Tokio adds overhead but convenient | Tokio for API/networking, thread pools for compute-heavy work |

## 4. Validation Gaps (How to Know It Works)

Testing/validation is underrepresented:

| Test Type | Coverage | Missing | How to Implement |
|---|---|---|---|
| **Determinism Tests** | Expert order mentioned | Full cross-backend determinism | Golden outputs per model, compare across runs/backends with tolerances |
| **Numerical Accuracy** | Mentioned | Systematic regression tests | Perplexity comparison vs reference (llama.cpp), unit tests for kernels |
| **Memory Leak Tests** | None | Critical for long-running | Stress test with many requests, check memory growth |
| **Performance Regression** | Limited | Missing | Microbenchmarks (attention, matmul), e2e benchmarks, CI gates |
| **Stress/Chaos Tests** | None | Important for production | Kill backend mid-request, disk I/O failures, OOM scenarios |
| **Cross-Platform Tests** | Meganeura shows promise | Limited coverage matrix | Test matrix: Linux/Windows/macOS × CUDA/ROCm/Vulkan/CPU |

## 5. Prioritized Missing Implementation List

### Must Implement Before v0.1 (Core)
1. [P0] Memory arena allocator for KV/tensors
2. [P0] Paged KV cache with block management  
3. [P0] GGUF model loader
4. [P0] Tokenizer integration
5. [P0] CPU/CUDA backend abstraction
6. [P0] Basic forward pass for dense transformer
7. [P0] Sampling (temperature/top-p/top-k)
8. [P0] Simple continuous batching scheduler
9. [P0] OpenAI-compatible API server
10. [P0] Determinism harness (at least for single-GPU)

### Must Implement Before v0.5 (Production-Ready)
1. [P0] Expert aggregation order canonicalization (fix 2607.28097)
2. [P0] Prefix caching (Radix tree)
3. [P0] Quantization support (GGUF types + kernels)
4. [P0] MoE support with expert caching
5. [P0] Observability (metrics/tracing/logging)
6. [P0] Error handling & recovery
7. [P1] Self-speculation (SEED-style)
8. [P1] Tiered memory (RAM offload at minimum)
9. [P1] Configuration system
10. [P1] Integration tests

### Nice-to-Have (v1.0+)
1. [P1] Disk streaming/offload (SSD tier)
2. [P1] Hot expert prediction
3. [P1] Hybrid attention (Gated DeltaNet)
4. [P1] Multi-backend (Vulkan/Metal)
5. [P2] P/D disaggregation
6. [P2] Cross-model KV handoff
7. [P2] Procedural weights (FORGE/Kilobyte)
8. [P2] Shape-class optimization (PRISM novelty)

## 6. Bottom Line

**The research is 90% there. The remaining 10% is critical engineering:**

1. **Integration architecture** - no paper shows the full picture, need to design it
2. **Correctness/Determinism** - MoE aggregation order is a real correctness issue we must fix
3. **Production hardening** - error handling, observability, resource management
4. **The shape-class microbatching idea (PRISM novelty)** is unclaimed - could be a meaningful contribution
5. **Cross-model context handoff standardization** is emerging - early implementation could shape the field

**Start small, test often, prioritize correctness over features.** The corpus gives excellent blueprints for individual components - the main work is stitching them together coherently into a fresh engine.