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

## Final Word

Build the smallest thing that proves expert disk paging works correctly. Everything else is distraction. Gates enforce correctness. The engine should be boring, debuggable, and bit-correct where it counts (same backend paged vs resident). Speed comes after correctness is proven.