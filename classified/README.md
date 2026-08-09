# Classified: all 992 cs.LG papers vs Part-3 mechanism ideas

Each paper is assigned a **primary mechanism category** (category with the most matched keywords; tie-break by inference-goal priority). Papers already detailed in `ML-NEW-TECH.md` Parts 1–3 are marked ✅.

## Category sizes

| Category | Papers | File |
|---|---|---|
| A — Weight Storage as a Recipe (seed/latent, multi-precision, factorization) | 30 | [`A_weight_recipe.md`](A_weight_recipe.md) |
| B — Compute That Skips Work (early-exit, dynamic width, sparsity, hashing, quantization) | 172 | [`B_skip_compute.md`](B_skip_compute.md) |
| C — Portable Execution Across Hardware (kernel/codegen, compilers, accelerators, edge) | 135 | [`C_portable_exec.md`](C_portable_exec.md) |
| D — Growing / Elastic / Modular Models (routing, MoE, PEFT, adapters, continual) | 144 | [`D_grow_modular.md`](D_grow_modular.md) |
| E — Split / Distributed Mechanisms (federated, split learning, offload, scheduling, comms) | 212 | [`E_split_distributed.md`](E_split_distributed.md) |
| F — Low-End Parallelism (many weak cards, heterogeneous clusters, parallelization) | 27 | [`F_lowperf_parallel.md`](F_lowperf_parallel.md) |
| G — Memory & Bandwidth (KV cache, compression, recurrent state, retrieval) | 239 | [`G_mem_bandwidth.md`](G_mem_bandwidth.md) |
| (no mechanism match) | 33 | [`00-other-domain.md`](00-other-domain.md) |

## Total
- Mechanism-classified: 959
- No mechanism match (pure domain papers): 33
- Already covered in Parts 1–3: 148
