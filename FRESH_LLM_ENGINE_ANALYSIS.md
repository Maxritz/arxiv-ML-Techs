# Fresh LLM Engine: Analysis & v0.1 Plan (Corrected)

Review date: October 21, 2026  
Target: Local, Windows-first, Vulkan, single-user, one-model (Mellum2.1 MoE) engine

## Verdict

The previous analysis was a generic, CUDA-first multi-model server plan. This corrected version targets **ninfer/VAiSt style**: C99/C11, custom Vulkan stack, correctness-first, MoE-first with expert disk paging as v0.1.

## 1. Core Principles

- **Local-first**: Single user, single session, CLI only. No server infrastructure.
- **MoE-first**: Mellum2.1 is MoE. Dense-first is a waste of time - build for MoE from the start.
- **Disk-paging is mandatory**: Expert disk paging is v0.1, not v1.0+. Mellum2.1 won't fit fully resident.
- **Correctness over speed**: Bit-identical for paged vs resident on same backend. Determinism is non-negotiable.
- **Vulkan-first**: Covers RDNA4, RDNA2, Arc on Windows. Drop CUDA, ROCm, Metal from v0.1.
- **Match llama.cpp as oracle**: Use llama.cpp to validate tokenization and outputs.
- **Fix the aggregation bug**: MoE expert reduction order must be canonical (sort by expert ID). 2607.28097 shows this causes divergence.

## 2. Hardware Target

| Hardware | Priority | Notes |
|---|---|---|
| **AMD RDNA4** | P0 | Primary target |
| **AMD RDNA2** | P0 | Secondary target |
| **Intel Arc iGPU** | P1 | Later, via Vulkan |
| **Windows** | P0 | Primary OS |
| **Vulkan** | P0 | Only graphics API for v0.1 |

## 3. v0.1 Scope

### Must Have (In)
- [ ] **GGUF loader** with mmap, tensor dump, shape verification
- [ ] **Tokenizer** with exact match to llama.cpp (Gate 0)
- [ ] **CPU reference** (f32) for oracle validation
- [ ] **MoE core**: router, top-8, expert FFN with canonical reduction order
- [ ] **Expert disk store**: file-offset addressed, pread/mmap, LRU hot cache with byte budget
- [ ] **Resident vs paged modes**: same code path, different budget
- [ ] **Contiguous KV cache** (sized to context, one session)
- [ ] **Vulkan backend**: matmul, rmsnorm, rope, attention, router, expert FFN, argmax
- [ ] **Per-op validation** against CPU reference
- [ ] **Gates A, B, C** as defined
- [ ] **CLI**: prompt → tokens, gate report output
- [ ] **Telemetry**: per-token timing, expert cache hit/miss counts

### Explicitly Out (v0.1)
- HTTP API, async runtime (tokio/axum)
- Continuous batching, multi-session
- Paged KV cache
- Prefix/radix cache
- Speculative decoding (any form)
- Sampling beyond greedy
- Other quant types
- CUDA/ROCm/Metal/OpenVINO
- Observability stack (Prometheus/OpenTelemetry)
- Multi-tenant features
- Shape-class batching
- dLLM support (check Mellum2.1 config first)
- Cross-model handoff
- P/D disaggregation
- Plugin system, CAS/provenance, signing

## 4. Architecture Design

### 4.1 Tokenizer First
Tokenizer bugs masquerade as model bugs. Gate 0 is non-negotiable - must produce identical token IDs to llama.cpp on a fixed test corpus before anything else.

### 4.2 CPU Reference is Oracle
All Vulkan kernels validated against CPU reference (f32). End-to-end validated against llama.cpp. This catches bugs early.

### 4.3 MoE with Canonical Ordering
```c
// Critical: sort selected experts by expert ID before weighted sum
// No float atomics
// Ties in argmax go to lowest index
```
Expert aggregation order must be deterministic and canonical. This fixes the divergence issue from 2607.28097.

### 4.4 Expert Paging
- Experts stored in GGUF (or separate file), addressed by byte offset
- On access: check LRU hot cache, if miss read via mmap/pread from disk
- LRU tracks by expert ID, evicts based on byte budget
- Resident mode: budget = total expert size (or large enough) - same code path
- Track hit/miss counts for validation

### 4.5 Validation Strategy
- **Synthetic test model**: Create minimal GGUF matching Mellum2.1 arch (few layers, few experts) with random weights. Debug pipeline without 7GB downloads.
- **Per-op diffs**: Each Vulkan kernel compared elementwise to CPU reference
- **Gate A**: Hidden state tolerance set in advance (not "close enough" - pick threshold before running)
- **Gate B**: Bit-identical logits for paged vs resident on same backend (zero tolerance)
- **Gate C**: Greedy prefix match vs llama.cpp with paged cache

## 5. Implementation Phases

### Phase 0: Prep (Day 1)
1. **Verify llama.cpp runs Mellum2.1** - confirm oracle exists
2. **Inspect Mellum2.1 GGUF config** - arch, attention, RoPE, norm, expert layout
3. **Set up build system** (CMake/Make for C99)
4. **Create synthetic test GGUF** for rapid iteration

### Phase 1: Core Pipeline (Days 2-4)
1. **Gate 0 harness** - tokenizer comparison vs llama.cpp
2. **GGUF loader** - mmap, tensor parsing, shape validation
3. **CPU reference forward** - dense parts + MoE with correct ordering
4. **Basic greedy decode**

### Phase 2: Expert Paging (Days 5-6)
1. **Expert store** - file offset access, pread/mmap
2. **LRU cache** - byte budget, hit/miss tracking
3. **Resident mode** - test with full budget
4. **Gate B** - paged vs resident bit-identical

### Phase 3: Vulkan Backend (Days 7-10)
1. **Backend interface** - device init, buffer management
2. **Kernels one by one**: matmul → rmsnorm → rope → attention → router → expert FFN → argmax
3. **Per-op validation vs CPU**
4. **End-to-end Vulkan vs CPU (Gate A)**

### Phase 4: Validation & Polish (Days 11-12)
1. **Gate C** vs llama.cpp with paged cache
2. **CLI interface** with gate reports
3. **Performance profiling** (timing, hit rates)
4. **Documentation of results**

*Note: Timeline is rough - use gates as milestones, not calendar weeks.*

## 6. Language & Stack

**Recommended:** C99/C11  
**Rationale:** Matches ninfer and VAiSt style, explicit control, easy to debug, no async runtime complexity, portable

**Build:** CMake or plain Make  
**Vulkan:** Direct Vulkan API (via vulkan.h) or minimal wrapper  
**Dependencies:** Keep minimal - only what's needed for GGUF parsing, Vulkan, file I/O

## 7. Research Notes (Citations are leads, not facts)

The corpus has useful ideas, but don't cargo-cult:
- **Expert ordering (2607.28097)** - critical, implement canonical sort by expert ID
- **Tiered residency/SSD offload** - concepts apply (SSD-LLaMA, SeqMoE), but adapt to our simple LRU approach
- **PagedAttention concepts** - not needed for KV in v0.1, but memory management ideas useful
- **Determinism** - must enforce strictly for same backend

Treat all 40+ arXiv IDs as research leads to verify against llama.cpp behavior, not ground truth.

## 8. Immediate Next Steps

1. **[BLOCKER]** Confirm `llama.cpp` can run Mellum2.1 and produce outputs we can compare against
2. **[BLOCKER]** Read Mellum2.1's GGUF metadata - get full config (layers, hidden size, num experts, top-k, RoPE settings, etc.)
3. **[P0]** Create tokenizer comparison test harness
4. **[P0]** Design minimal GGUF loader structure
5. **[P0]** Plan synthetic test model creation

**Do not start writing Vulkan kernels until Gate 0 passes and CPU reference matches llama.cpp on simple cases.**

## 9. 2024–2026 Re-Check (full sweep)

The plan above was written against the Aug–Oct 2026 window. The repo's annual snapshots hold **367 MoE-tagged papers** for 2024–2026
(50 / 122 / 195), none of the 2024–2025 ones cited in this document. Each load-bearing claim of the v0.1 plan is re-checked here; the
scope does not change, but two metrics and one boundary do.

| Claim in this plan | 2024–2026 evidence | Verdict |
|---|---|---|
| Expert disk paging is the core problem, not a v1.0 feature | Fiddler 2402.07033, MoE-Lightning 2411.11217, HOBBIT 2411.01433, fine-grained offloading 2502.05370, MoE-CORE 2610.01950, Ostinato 2611.11533 | **Confirmed, three years running.** If anything the plan understates it: the 2026 local-agent designs are *entirely* about this |
| CPU f32 reference is the oracle | No paper supersedes it; the 2025–26 systems assume a known-good reference for numerics | **Confirmed.** Unchanged |
| Simple LRU is sufficient for v0.1 | When Fancy Eviction Fails 2609.28870 (LRU robust across 14 policies); SliceMoE 2512.12990 frames caching as a miss-rate constraint; 2611.15842 measures real copies | **Confirmed.** This was the plan's most contested choice; the longer record supports it |
| Canonical expert ordering is critical (2607.28097) | 2611.15842 shows routing-order proxies do not predict transfer counts — the ordering guarantee is about *numerics*, and it stays | **Confirmed, with a sharper reason:** determinism of the reduction, not of the I/O |
| Skip PagedAttention / KV paging for v0.1 | PiKV 2508.06526: MoE sparsifies compute but KV stays dense; 2610.39216 and 2610.40305 show KV and weights competing for one budget | **Confirmed for one session**, but the reason changes: KV is skipped because there is one session, not because MoE makes it small |
| Skip speculative decoding | 2508.21706 (helps by hiding transfer latency) vs SP-MoE 2510.10302 (loses to bandwidth contention) | **Confirmed as "out of v0.1"**, and now with a reason: the sign of the effect is unresolved, so it cannot be planned, only tested |
| Treat all 40+ arXiv IDs as leads, not facts | Still true, and more so: several 2025 numbers are batch ≥ 8 with SLO targets | **Confirmed.** Add batch size and residency fraction to the list of things to check before trusting a number |

**Two additions the plan should absorb now (both cheap):**

1. **Report bytes/token next to hit rate.** HOBBIT, SliceMoE and 2611.15842 all argue in bytes moved; a hit percentage without bytes
   hides prefetch waste. This is a counter, not a feature.
2. **Record the phase boundary.** Every v0.1 run is decode-dominated, and PDScope (2610.39216) shows prefill and decode want opposite
   transfer policies. Writing "decode-only" in the gate report prevents a later prefill path from silently inheriting a decode policy.

**One addition that is explicitly deferred:** energy per token (2508.06978). It needs platform counters; note it as optional telemetry
rather than estimating it from byte counts.

**One honest limitation to write into the plan:** a 98%-style hit rate is not a property of the engine — 2024–25 comparables report lower
numbers at similar residency, and Lynx (2411.08982) shows batching changes the denominator entirely. State the trace, the batch size and
the residency fraction with every number.

## 10. ESP32 / MCU Re-Check of This Plan

**No day changes.** The 12-day plan stands. The MCU record adds one scope sentence and one review rule, and it removes a claim the
plan should never have made implicitly.

1. **Write the scope sentence at the top of the plan.** Target hardware is a machine with ≥ X GB usable RAM and a Vulkan GPU or a
   ≥4-core CPU. MCU-class devices (ESP32, Cortex-M, RP2040) are clients, not targets. The arithmetic is one line: one Q4 expert
   ≈100 MB against 8 MB of PSRAM on an ESP32-S3, ~12× over before KV or activations.
2. **If a companion board is in scope, budget half a day for hand-written firmware.** 2506.11003 measured 29.4% best pass@1 for
   ESP-IDF across 10 LLMs on simple cases, and 21.4% for Arduino→ESP32 migration (27.8% with RAG + compiler feedback). That is not
   a codegen tool; it is a draft generator with a mandatory review step.
3. **Do not cite ESP32 inference numbers, because this corpus has none.** The single ESP32 paper is a codegen benchmark. Any
   performance figure for an ESP32 running a language model would be extrapolation from SBC results (2511.07425: reliable to 1.5B
   params; 2512.06443 and 2407.00088 measure LUT kernels on Pi-class hardware) and should be labelled as such or omitted.

**One cheap addition to Day 1:** put a `device_class` string in the run manifest. It costs a field and prevents a Pi-class run and a
desktop run from being compared as if they were the same experiment — the same failure mode Lynx (2411.08982) warns about for batch
size, one tier down.

## Final Word

Build the smallest thing that proves expert disk paging works correctly. Everything else is distraction. Gates enforce correctness. The engine should be boring, debuggable, and bit-correct where it counts (same backend paged vs resident). Speed comes after correctness is proven.