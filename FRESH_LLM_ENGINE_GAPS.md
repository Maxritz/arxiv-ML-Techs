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

## 6. 2024–2026 Gap Deltas

The gap list above was set from the Aug–Oct 2026 window. The annual snapshots add **367 MoE-tagged papers** for 2024–2026 (50 / 122 /
195), none of the 2024–2025 ones cited in this document. The deltas are small and mostly *confirming*, which is the useful result: the
v0.1 boundary was drawn in about the right place.

**Gaps confirmed as P0 (no change needed, now with three years of support):**

| Gap | Evidence | Note |
|---|---|---|
| Expert store `L` + `P` as the core of v0.1 | Fiddler 2402.07033; MoE-Lightning 2411.11217; HOBBIT 2411.01433; 2502.05370; MoE-CORE 2610.01950; Ostinato 2611.11533 | The 2026 local-agent systems are *only* this problem; the priority call was right |
| Fixed expert ordering before the weighted sum | 2611.15842 (routing-order proxies do not predict transfers) | Ordering is a numerics guarantee, not a performance trick — keep it non-negotiable |
| Contiguous KV, one session | PiKV 2508.06526 (KV stays dense under MoE) | Skip it because there is one session, not because KV is small |

**Gaps to add to v0.1 (both cheap counters, neither a feature):**

| New item | Why | Cost |
|---|---|---|
| **bytes read per token** in telemetry | HOBBIT, SliceMoE 2512.12990, 2611.15842 all argue in bytes moved; hit % without bytes hides prefetch waste | one counter |
| **explicit "decode-only" phase label** in the gate report | PDScope 2610.39216: prefill wants aggressive loading, decode wants conservative transfers; inheriting one policy silently is a later bug | one line of output |

**Gaps that stay out, now with explicit reasons:**

| Item from the literature | Why it stays out of v0.1 |
|---|---|
| Mixed-precision / cross-precision expert slots (HOBBIT 2411.01433, MatMoE 2611.01704, PagedMoE 2611.03173) | Real, but it multiplies slot bookkeeping; record fixed `slot_bytes` as a simplification with a v0.2 path |
| Phase-split scheduling (DuoServe-MoE 2509.07379, layered prefill 2510.08055, MoE-CORE 2610.01950) | Needs a prefill path to be meaningful; v0.1 has none |
| Speculative decoding (2508.21706 vs SP-MoE 2510.10302) | The sign of the offload interaction is unresolved; it is an experiment, not a phase |
| Disaggregation (Janus 2512.13525, MegaScale-Infer 2504.02263, AFD 2605.28302) | Multi-device, multi-tenant assumptions |
| Serverless / multi-tenant experts (MoEless 2603.06350, EaaS 2509.17863, CrossPool 2606.24506) | Fleet-scale systems |
| Resilience (Tarragon 2601.01310, ReviveMoE 2602.21140) | A single workstation does not have fleet failure modes |
| Batch-aware expert selection (Lynx 2411.08982, D2MoE 2504.15299) | Changes outputs, and batch-1 removes the motivation |

**One correction to the reasoning, not the scope:** §1's "SSD offloading is core" remains true for *capacity*, but 2508.06978 shows SSD
read energy per bit is far above DRAM's, so SSD tiering is not an efficiency claim. Say capacity, and leave energy as optional telemetry.

## 7. ESP32 / MCU-Class Gap Register

**No new P0 or P1.** The gap list in §1 is unchanged, and no MCU work belongs in v0.1. The scan does add one explicit non-goal and one
small real gap.

**Explicit non-goal (write it down so it stops being re-litigated):** MCU/ESP32 inference. One Q4 expert is ~100 MB against 8 MB of
PSRAM on an ESP32-S3; a 0.5B 4-bit model is ~250 MB. The tier that does exist above MCU is SBC class, and even there the ceiling is
about 1.5B params with runtime choice moving throughput 4× (2511.07425).

**The one real gap:** §1 never states a minimum hardware contract. Add it: *targets ≥ X GB usable RAM; MCU-class devices are clients.*
Without it, "runs on low-end hardware" is unfalsifiable.

| Item | In scope for v0.1? | Why |
|---|---|---|
| TinyML on MCU (2412.09289: 64 KB transformer, 32 KB Mamba) | No | Below the model class this engine loads by four orders of magnitude |
| Hierarchical inference client (2406.09424) | Only if a board exists | Correct pattern for a sensor client; no host-side code changes |
| On-device LoRA at the extreme edge (2603.09511) | No | −23% memory and 15× fewer trainables is impressive at 0.28M params; irrelevant at 12B |
| Ternary ROM / FPGA accelerators (2602.20662, 2510.15926) | No | Custom silicon; 2510.15926 is a 5 W FPGA at 25 tok/s |
| Design-time energy prediction (2608.03589) | Checklist only | 16.6% WAPE on Cortex-M4; useful before deploying an auxiliary model, not for the engine itself |
| SBC benchmarks and energy studies (2511.07425, 2511.11624) | Reference only | Define the tier above MCU; 1.5B-param ceiling, GPU vs CPU energy ordering |

**Cheap addition:** a `device_class` field in the telemetry header. One string, and the gate report becomes self-describing.

**One correction to a common reading:** "fewer parameters means faster on the target" is false on scalar cores — 2608.00737 measured
13.5×/8.0× latency and 11.3×/5.6× energy penalties for the parameter-efficient architecture, plus up to 43× earlier INT8 divergence.
Parameter count is not a deployment-cost proxy, and this belongs in the caveats, not the roadmap.

## Bottom Line

This is a **correctness-first, local MoE engine**. Expert disk paging is not optional - it's core to making Mellum2.1 run. The previous roadmap was generic server engineering; this is purpose-built for constrained local hardware. Gates with bit-identical requirements (for same-backend paging) are non-negotiable.