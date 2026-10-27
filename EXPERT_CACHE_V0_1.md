# Expert Cache v0.1 — Interface & Data Layout

Implements the logical/physical split defined in `ENGINE_MASTER_SPEC.md` §1.2, for the v0.1 engine
described in `FRESH_LLM_ENGINE_GAPS.md` (own C11 engine, GGUF + mmap, Vulkan, one model, one session,
experts always disk-paged).

**Status of this document:** the declarations and invariants below are normative. Prose is rationale. If
an implementation needs to violate an invariant, the invariant changes here first and the gates in
`FRESH_LLM_ENGINE_GAPS.md` are re-read — not the other way round.

**Substrate note:** `IMPLEMENTATION_RECOMMENDATION.md` §1 recommends ggml/llama.cpp as the compute layer.
The interface below is substrate-agnostic: `ec_store` hands out `(uint8_t *base, uint64_t len)` per
expert sub-region, so a ggml-based build would wrap `ggml_backend_alloc` instead of `aligned_alloc` and
keep everything else. The only thing that must not change is that both modes present the **same pointers**.

---

## 1. Invariants

| ID | Invariant | Checked by |
|---|---|---|
| INV-1 | `L` is a pure function of `(weights, token sequence, cfg)`. No clock, no I/O state, no counters, no thread count, no residency may enter `ec_logical_advance` or `ec_router_bias`. | code review + INV-9 assertion |
| INV-2 | Selection is `topk(score + bias)` with fixed order: score descending, ties by lower expert id. No epsilon comparisons, no NaN. | `ec_select` unit test on crafted scores |
| INV-3 | The slot arena has identical base alignment, identical slot stride and identical sub-region offsets in `paged` and `resident` modes. Kernels receive identical pointers and strides. | `ec_store_layout_hash()` compared across modes |
| INV-4 | No allocation, no lock, no syscall on the compute thread's `L`/bias/selection path. Misses block on a per-slot wait, which is a syscall but is *not* on the deterministic path. | `-DEC_DEBUG` asserts |
| INV-5 | A physical miss stalls the compute thread. It never changes the selected expert set, and it never evicts an expert that the current selection needs. | INV-9 assertion |
| INV-6 | Expert bytes are copied verbatim from the GGUF data section. The cache never converts, requantizes or reorders. Quant type stays quant type. | byte-compare against `mmap` view in `-DEC_DEBUG` |
| INV-7 | Every reduction over selected experts runs in ascending expert-id order, single-threaded per reduction. No float atomics. | by construction; reviewed per kernel |
| INV-8 | `L` and the selected set are recorded per token per layer (binary trace) in every build, not just debug builds. | trace file exists for every run |
| INV-9 | `exact` mode (`beta == 0.0f`) produces a selected set bit-identical to a build with the cache disabled entirely. | `-DEC_DEBUG` self-check, then a one-off release check |

INV-1 through INV-3 are what make Gate B (paged vs resident, bit-identical logits) true *by construction*
rather than by tolerance. See §9.

---

## 2. Configuration

```c
/* include/ec/ec.h */
#ifndef EC_H
#define EC_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define EC_EXPERT_NONE 0xFFFFu      /* uint16_t expert-id sentinel */
#define EC_SLOT_NONE   0xFFFFFFFFu  /* slot-index sentinel */
#define EC_SUB_ALIGN   64u          /* per-tensor region alignment inside a slot */
#define EC_SLOT_ALIGN  4096u        /* slot start alignment; keeps O_DIRECT possible later */
#define EC_MAX_LOGICAL 256u         /* cfg.n <= this */

typedef enum { EC_MODE_PAGED = 0, EC_MODE_RESIDENT = 1 } ec_mode;

typedef struct {
    ec_mode  mode;
    uint32_t n_logical;      /* L capacity, per layer per token. MODEL config. */
    float    bias_beta;      /* 0.0f == exact routing (Gate A/C). MODEL config. */
    uint32_t top_k;          /* router top-k from GGUF metadata (Mellum2.1: 8) */
    uint64_t arena_bytes;    /* P budget. MEMORY config — must not affect selection. */
    uint32_t io_threads;
    uint32_t prefetch_layers;/* 1 = next layer only (v0.1) */
    uint32_t n_layers;
    uint32_t n_experts;      /* per layer; uniform in v0.1, else reject at load */
} ec_cfg;
```

`n_logical` and `bias_beta` are printed in every gate report next to every speed number, per
`ENGINE_MASTER_SPEC.md` §1.2 rule 5. Changing either one is a model change: re-run the whole gate set.

---

## 3. Logical cache `L`

```c
typedef struct {
    uint16_t *ids;      /* [n_layers * n_max], ascending expert ids, EC_EXPERT_NONE padded */
    uint16_t *count;    /* [n_layers], <= n_max */
    uint16_t  n_max;    /* cfg.n_logical */
    uint32_t  n_layers;
} ec_logical;

/* Pure: no allocation, no I/O, no state outside the struct and its inputs. */
void ec_logical_reset_token(ec_logical *L);
/* Fills L[p->layer] from the predictor. Target layer is p->layer, not an argument. */
int  ec_logical_advance(ec_logical *L, const ec_pred *p, uint16_t n);
bool ec_logical_contains(const ec_logical *L, uint32_t layer, uint16_t expert_id);
```

`ids` is sorted ascending, so membership is a binary search over at most `n_max` entries and the array is
canonical (two equal sets are byte-identical). `count[layer]` is layer-local: `L` is a per-token rolling
set along layers, reset at token start, which is exactly the "per layer and per token" insertion the
paper describes.

### 3.1 The predictor (v0.1 form)

```c
typedef struct {
    const float *gate_w;   /* [n_layers][n_experts][hidden] adjacent-layer gate, always resident */
    const float *hidden;   /* current layer hidden state, f32, owned by the caller */
    uint32_t     hidden_dim;
    uint32_t     layer;    /* predicts layer+1 */
} ec_pred;
```

v0.1 uses the adjacent-layer gate predictor: score the experts of layer `ℓ+1` with layer `ℓ+1`'s own
gate applied to layer `ℓ`'s output hidden state — `pred(e) = dot(gate_w[ℓ+1][e], hidden_ℓ)`. No extra
parameters, no training, deterministic, and it is the cheapest member of the Fate-family predictors.
The gate weights are ~`n_experts × hidden` floats per layer, which is why they stay resident and are
excluded from the arena budget.

### 3.2 Admission

```text
ec_logical_advance(L, pred, n):          // writes L[pred->layer]
    scores[e] = dot(gate_w[layer][e], hidden)     for e in 0..n_experts-1   // fixed order
    L[p->layer] = top_n(scores, tie = lower id)                          // insertion into n-sized array
```

`top_n` is a straight insertion selection into a fixed array of `n` entries: O(`n_experts · n`), no
allocation, no partial sort of the full vector, no early exit on equal scores. Ties resolve to the lower
expert id, so the result is uniquely determined by the score vector.

Eviction is implicit: an expert not in the new top-`n` leaves `L`. **No retention/hysteresis in v0.1** —
it is a stateful knob that would need its own re-gating. `L` is never seeded from the previous token.

### 3.3 Bias

```c
float ec_router_bias(const ec_logical *L, uint32_t layer, uint16_t expert_id, float beta)
{
    return ec_logical_contains(L, layer, expert_id) ? beta : 0.0f;
}
```

```c
/* Fills out[0..k-1] with selected expert ids, ascending. Returns k. */
uint32_t ec_select(const float *router_logits, const ec_logical *L, uint32_t layer,
                   float beta, uint32_t k, uint16_t *out);
```

Applied to the router logits of the **same** layer before top-k:

```text
score_b[e] = router_logit[e] + (in_L(layer, e) ? beta : 0.0f)
selected   = top_k(score_b)          // k = cfg.top_k, ties by lower id
```

Shared experts are always computed, are never part of `L`, and are not counted against `top_k`.

---

## 4. Physical store `P`

```c
typedef enum { EC_SLOT_EMPTY = 0, EC_SLOT_LOADING = 1, EC_SLOT_READY = 2 } ec_slot_state;

typedef struct {
    uint8_t  *base;        /* slot base; arena_base + slot_index * slot_bytes */
    uint32_t  expert_id;   /* valid when state != EMPTY */
    uint64_t  last_use;    /* compute-access counter only; prefetch does NOT touch this */
    uint32_t  state;       /* ec_slot_state; written with release semantics by readers */
} ec_slot;

typedef struct {
    uint8_t      *arena;        /* aligned_alloc(EC_SLOT_ALIGN, arena_bytes) */
    uint64_t      slot_bytes;   /* fixed, derived from GGUF metadata, mode-independent */
    uint32_t      n_slots;
    ec_slot      *slots;
    uint32_t     *slot_of;      /* [n_layers * n_experts] -> slot index or EC_SLOT_NONE */
    uint64_t      use_clock;    /* incremented once per compute access */
    /* omitted from this header: queue_lock, per-slot wait/cond, reader threads */
} ec_store;

/* Non-blocking. Enqueues every id in L[layer] that has no READY or LOADING slot.
   Never evicts, never blocks, never writes last_use, never touches L. */
void ec_prefetch(ec_store *s, const ec_logical *L, uint32_t layer);
```

`slot_bytes` and `n_slots` are computed once at load from GGUF metadata:

```text
slot_bytes = sum over the layer's expert tensors (gate, up, down)
             of align_up(tensor_bytes_per_expert, EC_SUB_ALIGN)
n_slots    = arena_bytes / slot_bytes        // floor; refuse if 0
```

`resident` mode is the same struct with `arena_bytes = n_slots_needed * slot_bytes`, arena filled at
init. No second code path, no mmap of the weight file for compute.

**`last_use` policy:** only compute access bumps the clock. Prefetch completion does not. This is not
required for Gate B (eviction order cannot change selection) but it keeps two paged runs with the same
`L` trace replayable for debugging a stall pattern.

**Miss path:**

```text
ec_store_get(store, layer, expert_id) -> const uint8_t* base
    slot = slot_of[layer*n_experts + expert_id]
    if slot != EC_SLOT_NONE and slots[slot].state == READY:
        slots[slot].last_use = ++store->use_clock
        return slots[slot].base                       // hot path: load + compare + store
    return ec_store_fill_blocking(store, layer, expert_id)   // enqueue + wait; may evict
```

`ec_store_fill_blocking` picks a victim that is `READY` and not in the current selection or the current
`L` for the layer being computed (INV-5), by lowest `last_use`, ties by slot index. If every slice is
protected, it evicts the LRU `READY` slot anyway and records a `protected_evict` counter — v0.1 logs it,
because it means `n_slots` is too small for `top_k` and the budget needs raising.

---

## 5. On-disk layout and the expert index

GGUF stores each layer's experts as three 3-D tensors, `blk.<L>.ffn_{gate,up,down}_exps.weight`, of
shape `[n_expert, ne0, ne1]`. Consecutive experts are therefore contiguous with a fixed stride, so a
per-expert read is three contiguous ranges — no scatter/gather beyond `preadv` with three iovecs.

```c
typedef struct {
    uint64_t off;   /* absolute file offset of this sub-region */
    uint32_t len;   /* bytes (expert slice only) */
    uint32_t pad;   /* = align_up(len, EC_SUB_ALIGN) - len */
} ec_extent;

typedef struct {
    uint64_t   file_base;                 /* data-section base offset */
    ec_extent *ext;                       /* [n_layers * n_experts * 3] */
    uint64_t   tensor_stride[3];          /* per-expert byte stride per tensor */
    uint64_t   slot_region_off[3];        /* sub-region offsets inside a slot */
    uint64_t  *layer_bytes;               /* [n_layers] expert bytes for one layer */
} ec_index;
```

Slot layout — identical in both modes, which is INV-3:

```text
slot base (EC_SLOT_ALIGN = 4096 aligned)
  +0                      gate  [len_g bytes][pad to 64]
  +align64(len_g)         up    [len_u bytes][pad to 64]
  +align64(len_g+len_u)   down  [len_d bytes][pad to 64]
                          ... rest of slot_bytes unused (slack) ...
slot_bytes = align_up(align64(len_g)+align64(len_u)+align64(len_d), EC_SLOT_ALIGN)
```

Sub-region pointers are `base + slot_region_off[i]`. Kernels take those pointers plus the quant type from
GGUF; they never see slot indices, file offsets or modes. A future repacked single-iovec blob (Tied
Trit-Planes / Strata folded layout) changes `ec_index` only — `slot_region_off` stays, so no kernel or
gate code changes.

Reads use `preadv(fd, iov, 3, ext[e].off)` with one iovec per sub-region. `mmap` is used only for the
oracle path (`llama.cpp` comparison harness) and for the INV-6 byte-compare, never for compute.

---

## 6. Threading and ownership

| State | Compute thread | Reader threads | Prefetcher |
|---|---|---|---|
| `ec_logical` | owns, read/write | never touches | reads a copy of the enqueued ids only |
| `ec_cfg` | read-only | read-only | read-only |
| `ec_slot.state` | reads (acquire) | writes (release) | reads |
| `ec_slot.last_use` | writes | reads for stats only | never writes |
| `ec_store.slot_of` | reads | writes under `queue_lock` | reads under `queue_lock` |
| `ec_index` | read-only | read-only | read-only |

The compute thread never waits while holding `queue_lock`. Publication order per slot:
`preadv` → `__atomic_store_n(&slot->state, READY, release)` → futex wake. Consumer:
`__atomic_load_n(&slot->state, acquire)` → read bytes. v0.1 has exactly one session and one compute
thread, so there is no refcounting and no generation handshake.

---

## 7. Call sequence (per token, per layer)

Order matters more than any single function; this is the sequence the paper's overlap comes from.

```text
token start:
    ec_logical_reset_token(L)
    for layer in 0..n_layers-1:
        if layer == 0:
            ec_logical_advance(L, pred(embedding, 0), n)         // L for layer 0
        h = attention(layer, h)
        // h is now available; predict and prefetch the NEXT layer before compute
        if layer + 1 < n_layers:
            ec_logical_advance(L, pred(h, layer+1), n)           // pure, deterministic
            ec_prefetch(store, L, layer+1)                       // timing only, non-blocking
        scores = router_logits(layer, h)
        sel    = ec_select(scores, L, layer, beta, top_k, sel_buf)  // INV-2
        for e in sel ascending:
            base = ec_store_get(store, layer, e)                 // may stall (INV-5)
            ec_expert_ffn(base, quant_type, x, scratch)
            acc = acc + scale[e] * scratch                       // ascending order (INV-7)
        h = h + acc
```

Two consequences worth stating explicitly, because they are the things implementations usually get
wrong:

1. `ec_logical_advance` for layer `ℓ+1` runs **before** layer `ℓ`'s expert compute, so the prefetch has
   the whole FFN to complete. It does not depend on the experts chosen at layer `ℓ`.
2. `ec_select` reads `L` for the layer being computed, which was set one layer earlier (or at token start
   for layer 0). `L` is therefore always available before it is needed, and never depends on a transfer.

---

## 8. Telemetry

```c
typedef struct {
    uint64_t tokens, layers;
    uint64_t selects;            /* total selected experts across all layers/tokens */
    uint64_t select_in_logical;  /* selected AND was in L at selection time */
    uint64_t phys_access;        /* ec_store_get calls that were not prefetch */
    uint64_t phys_hit;           /* returned without stalling */
    uint64_t phys_miss;
    uint64_t bytes_read;
    uint64_t stall_ns;           /* summed wait time on misses */
    uint64_t pf_issued, pf_used; /* enqueued; served a later access before eviction */
    uint64_t evictions, protected_evict;
} ec_stats;
```

Reported metrics, always together and always with `(n_logical, bias_beta, n_slots/n_experts)`:

```text
logical_agree   = select_in_logical / selects        # does L know what the router picks (with bias)
physical_hit    = phys_hit / phys_access             # does the store serve it without stalling
bytes_per_token = bytes_read / tokens                # the streaming cost model's input
```

`logical_agree` is the number to compare against the predictor-accuracy figures in the literature.
`physical_hit` is the number to compare against hit-rate figures. They are not interchangeable, and
`ENGINE_MASTER_SPEC.md` §4 requires both. One JSON line per run, on stdout, machine-parsable:

```json
{"mode":"paged","n_logical":32,"beta":0.5,"slot_frac":0.25,"tokens":256,
 "logical_agree":0.931,"physical_hit":0.887,"bytes_per_token":19371008,
 "stall_ms":842,"pf_used":0.91,"protected_evict":0}
```

---

## 9. Gate B harness

Three run modes, one binary:

| Flag | Meaning |
|---|---|
| `--ec-mode resident` | arena holds every expert, filled at init |
| `--ec-mode paged --ec-arena BYTES` | normal operation |
| `--ec-beta 0` (with either mode) | exact routing; the Gate A / Gate C arm |

Per run, three artifacts in the run directory:

| Artifact | Content | Purpose |
|---|---|---|
| `run.json` | cfg + `ec_stats` + build id | comparability |
| `logits.bin` | `[n_steps][n_vocab]` raw `float32`, little-endian | the bit-identical comparison |
| `route.bin` | `[n_steps][n_layers]` header (`uint16 n_selected, uint16 n_logical`) then selected ids ascending, then `L` ids ascending | localises a divergence before logits differ |

`ecdiff A B` reports the first divergent `(step, layer)` and then compares logits bytewise:

```text
if route.bin differs        -> deterministic-path bug: L admission, predictor input, bias, or top-k ties
if route.bin same, logits differ
                            -> I/O path touching compute: arena aliasing, alignment, slot reuse while in
                               flight, or a reduction whose order depends on completion order
if both same                -> Gate B passes for this build, prompt set and (n, beta)
```

Why this is bit-identical by construction, not by luck — three lemmas that map to INV-1..INV-3:

1. `L` is a function of weights and the token sequence (INV-1) ⇒ both modes compute the same `L` trace.
2. `ec_select` is a function of router logits, `L`, `beta`, `top_k` (INV-2) ⇒ both modes select the same
   experts in the same order.
3. Slot layout and sub-region offsets are mode-independent (INV-3), expert bytes are verbatim from the
   file (INV-6), and reductions run in ascending id order (INV-7) ⇒ every kernel call sees the same
   inputs at the same addresses in the same order ⇒ identical outputs.

The only thing that differs between the two runs is `stall_ns` and wall time. That is the claim Gate B
tests, and a failure localises to a violation of one of the three lemmas.

Run Gate B at three residency fractions (e.g. all, 50%, 25% of experts) and with `beta` at its shipping
value. Gate A and Gate C additionally run at `beta = 0` against `llama.cpp`, with tolerance, because the
oracle has neither `L` nor a bias.

---

## 10. Sizing

```text
per-expert bytes        = (gate + up + down) quantized weights for one expert
slot_bytes              = align_up(align64(gate)+align64(up)+align64(down), 4096)
n_slots                 = arena_bytes / slot_bytes
experts per layer       = n_experts
worst-case live set     = top_k  (+ shared experts, which are pinned and outside the arena)
```

`n_slots` below `n_experts/2` is the regime where the papers' numbers live (25–50% residency); below
`n_slots < n_experts/8` expect `protected_evict > 0` and report it rather than hiding it. Print the
actual numbers from GGUF metadata at load (`--print-expert-index`) instead of assuming any — the figures
in the papers are for other models.

---

## 11. Assertions (`-DEC_DEBUG`)

Cheap, compiled out in release except INV-6 and the trace:

| Assertion | Cost | Catches |
|---|---|---|
| `L.ids` strictly ascending, all `< n_experts`, `count <= n` | O(n) | admission bugs, stale state across tokens |
| selected ids strictly ascending, `count == top_k` | O(k) | top-k tie handling, duplicate picks |
| `beta == 0.0f` ⇒ selection equals unbiased top-k | O(k) | bias leaking into the exact arm (INV-9) |
| slot state transition legal (`EMPTY→LOADING→READY→EMPTY`) | O(1) | double-free, reuse while in flight |
| compute never reads a slot whose state != READY | O(1) | the classic paging race |
| slot bytes equal `mmap` view of the file | O(slot) once per fill | requantization, offset miscalculation |
| arena base and every sub-region 64-byte aligned; layout hash equal across modes | O(1) | INV-3 violations, which would break Gate B *silently* |

---

## 12. Out of scope for v0.1

Explicitly not in this interface, each with the reason it is deferred rather than forgotten:

| Deferred | Why |
|---|---|
| Retention / hysteresis in `L` | new state in the deterministic path; needs its own gate comparison, not a free win |
| Adaptive `n` or `beta` per token | makes selection depend on a policy state machine; keep both constant and reported |
| CPU fallback execution for misses (adaptive expert assignment) | needs the split-execution scheduler; v0.1 stalls instead |
| `prefetch_layers > 1` | more speculative reads, same Gate B status, needs a bandwidth budget first |
| Repacked expert blobs / folded layout | `ec_index` already isolates it; do it when `preadv` count shows up in `stall_ns` |
| `O_DIRECT` | arena is 4096-aligned to allow it later; buffered `pread` first |
| Second session, KV interaction, multi-request batching | v0.1 is one session, contiguous KV |
| Eviction-aware post-training / cache-aware router adaptation | model-side change; out of engine scope |

---

## 13. Where this touches the plans

| Doc | Section | Relationship |
|---|---|---|
| `ENGINE_MASTER_SPEC.md` | §1.2 | defines the split; this doc is its implementation form |
| `ENGINE_MASTER_SPEC.md` | §4 | the metrics here are the two numbers §4 requires |
| `FRESH_LLM_ENGINE_GAPS.md` | §3 gates | Gate B/C conditions; §5 step 5 is `L` then `P` |
| `ML-NEW-TECH.md` | §32.1, §32.8 | PSR mechanism and the mapping table this design follows |
