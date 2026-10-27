# Detailed Gap Analysis - Corrected for Mellum2.1 (MoE-first, Vulkan, Windows-first)

## Priority: Expert Disk Paging is v0.1, not v1.0+

The review is correct - this is a local, single-user, one-model engine targeting Mellum2.1 (MoE). The previous analysis inverted priorities badly.

## 1. Critical Gaps for v0.1

| Component | Priority | Status | Notes |
|---|---|---|---|
| **GGUF Loader + mmap** | P0 | Missing | Dump tensor list, verify shapes against llama.cpp. Must match exactly. |
| **Tokenizer (exact match)** | P0 | Missing | **Must** match llama.cpp token IDs on fixed corpus. Test first - tokenizer bugs look like model bugs. |
| **CPU Reference Forward (f32)** | P0 | Missing | Oracle for all validation. Greedy decode. |
| **MoE Router + Top-8 + Expert FFN** | P0 | Missing | Fixed reduction order: **sort selected experts by expert ID before weighted sum**. No float atomics. Argmax ties go to lowest index. |
| **Expert Store: logical cache `L` + physical store `P`** | P0 | Missing | Experts addressed by file offset, read via pread/mmap. **`L`** = deterministic admission/eviction over *predicted* routing (`n` and bias `beta` are model config; contents are a function of the token sequence only). **`P`** = LRU/score cache with byte budget, any prefetch depth. **Routing reads `L`, never residency.** Resident mode = all of `P` populated, same `L`. Same code path, different budget. **This is v0.1** (Mellum2.1 is always disk-paged). Design in ENGINE_MASTER_SPEC.md §1.2. |
| **Contiguous KV Cache** | P0 | Missing | One session, sized to context. No paging for v0.1 (simpler). |
| **Vulkan Backend** | P0 | Missing | RDNA4 first, RDNA2 next, Arc later. Each op diffed against CPU reference alone. |
| **Gates & Validation Harness** | P0 | Missing | Gates 0,A,B,C defined before code. Bit-identical for paged vs resident on same backend. |

## 2. What is OUT for v0.1 (Do not creep in)

| Item | Reason |
|---|---|
| **HTTP/OpenAI API, tokio, async runtime** | CLI only, single user, single session |
| **Continuous batching, multi-session** | Single user, single session |
| **Paged KV (for KV cache)** | Contiguous KV is fine for v0.1. Add only if needed. |
| **Prefix/radix cache** | Not needed for single session |
| **Speculative decoding, MTP, self-speculation** | Add after gates pass |
| **Sampling beyond greedy** | Add after gates pass |
| **Other quant types** | Use whatever Mellum2.1 GGUF ships |
| **OpenVINO, CUDA, ROCm, Metal** | Vulkan only. Drop CUDA/Metal/ROCm from plan. |
| **Prometheus/OpenTelemetry** | Simple per-token timing + physical hit/miss counts, reported separately from logical-cache quality |
| **Plugin system, model registry, CAS/provenance, signing, sandboxing** | Overengineering for local single-model engine |
| **Multi-tenant, P/D disaggregation, load balancing, rate limiting** | Not applicable |
| **Cross-model KV handoff** | Not applicable |
| **dLLM/Diffusion LLM** | Only if Mellum2.1 uses it - check config first |
| **Shape-class microbatching** | Irrelevant for single-user. Drop from roadmap (can't be "novelty" and "already in vLLM/SGLang/TensorRT-LLM" simultaneously) |
| **Hybrid attention** | Only if Mellum2.1 config uses it. Check before implementing. |

## 3. Gates (Defined Before Code)

| Gate | Criterion | Notes |
|---|---|---|
| **Gate 0 (Tokenizer)** | Identical token IDs vs llama.cpp on fixed corpus | Do this FIRST. |
| **Gate A (Resident CPU)** | Per-layer hidden state error under threshold set in advance. Greedy decode: longest common prefix >= N tokens across M prompts vs llama.cpp. | Pick threshold and N,M before running. Kernels won't match bit-identical to llama.cpp. |
| **Gate B (Paged vs Resident, same backend)** | **Bit-identical logits every step**. Any difference = bug. No tolerance. | Only achievable if both runs share `(n, beta, admit rule)` and differ in physical residency alone. A cache that biases the router toward *resident* experts makes logits timing-dependent and can never pass this — the reason for the logical/physical split (PSR, 2610.34721). Run at the `L`-complete milestone, before tuning eviction. |
| **Gate C (Paged vs llama.cpp)** | Paged with default cache budget vs llama.cpp greedy prefix. | Final validation. Run with `beta = 0` (exact routing): llama.cpp has no logical cache, so a biased run compares the bias instead of the kernels. |
| **Vulkan Validation** | Each kernel vs CPU reference alone (matmul, rmsnorm, rope, attention, router, expert FFN, greedy argmax). Then Gate A on Vulkan. | Unit test per op before end-to-end. |

## 4. Bring-up Strategy

1. **Create synthetic GGUF** with Mellum2.1's architecture (few layers, few experts, random weights). Debug entire pipeline without downloading 7GB+ or RAM pressure. Still only one supported model - it's a test fixture.
2. **Confirm llama.cpp runs Mellum2.1** - Gate A depends on it. If not, oracle is gone, plan changes.
3. **Read Mellum2.1's config** - attention type, RoPE variant, norm, expert layout. Don't build features it doesn't use.
4. **Pick language** - C99/C11 matching ninfer and VAiSt. Don't switch to Rust just because generic docs say so.

## 5. Implementation Order

1. **Tokenizer test harness** (Gate 0) - against llama.cpp
2. **GGUF loader** - dump tensors, verify shapes
3. **CPU reference** - f32 forward, greedy decode
4. **MoE core** - router with fixed ordering, experts from file offsets
5. **Expert cache, `L` then `P`** - logical cache first (deterministic admission/eviction; the Gate B harness runs here), then the physical store with LRU byte budget
6. **Vulkan kernels** - one by one vs CPU reference
7. **Gates A,B,C** - validation against oracle
8. **CLI** - prompt in, tokens out, gate report

## Bottom Line

This is a **correctness-first, local MoE engine**. Expert disk paging is not optional - it's core to making Mellum2.1 run. The previous roadmap was generic server engineering; this is purpose-built for constrained local hardware. Gates with bit-identical requirements (for same-backend paging) are non-negotiable.