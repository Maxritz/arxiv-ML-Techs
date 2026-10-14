# ML New Tech — cs.LG Recent (Aug 2026)

**Source:** arXiv cs.LG recent submissions, Aug 3–7 2026 (992 papers parsed via arXiv API, ranked by LLM-relevance)
**Generated:** 2026-08-09

This digest filters the full cs.LG recent batch down to papers with concrete, actionable value for **improving LLMs** (training, post-training, inference, agents, safety, efficiency). Each entry links to the arXiv abstract. Scores reflect how many high-signal LLM keywords matched title+abstract.

---

## 1. KV Cache & Long-Context Efficiency (highest density of wins)

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.04074](https://arxiv.org/abs/2608.04074) | Spend Bits Where Queries Look: KV Cache Vector Quantization with Attention-Preserving Transforms | KV-cache VQ that derives the orthogonal transform from a distortion criterion instead of Hadamard/random rotations, so attention products survive at 2 bits/element; directly raises decode throughput and serving capacity on bandwidth-bound long-context decoding. |
| 2 | [2608.02691](https://arxiv.org/abs/2608.02691) | Output-Aware Rotation for INT2 KV-Cache Quantization | OptR: minimizes post-`W_O` attention-output error (not proxy cache statistics) with per-head orthogonal corrections and key reparameterization — better accuracy at INT2 KV cache than rotation baselines. |
| 3 | [2608.02901](https://arxiv.org/abs/2608.02901) | AnchorKV: Anchor-Residual KV Cache Compression | 20× KV compression with *no token discarding*: cache expressed as exact anchors + residual codes, keeps ~99% of full-cache score at 70B scale. |
| 4 | [2608.05326](https://arxiv.org/abs/2608.05326) | QEvict: Recoverable Quantized KV Eviction | Shows eviction is irreversible and attention importance *drifts* during decoding; three-tier recoverable scheme with "Future Missed Mass"/Global LIR diagnostics. |
| 5 | [2608.01247](https://arxiv.org/abs/2608.01247) | RestoreKV: Recovering Full-Cache Behavior Under Aggressive Query-Agnostic KV Eviction | Learns a compact context-conditioned "restore cache" via a single LoRA-adapted pass over the full cache; adapters disabled afterward. Restores full-cache behavior under tight budgets. |
| 6 | [2608.03228](https://arxiv.org/abs/2608.03228) | SAKI: Score-Aware Low-Rank Key Indexing | Training-free low-rank KV index that preserves *attention scores* (not key reconstruction); beats key-PCA at every rank on LLaMA-3.1-8B, Qwen-2.5-7B, Mistral-7B. |
| 7 | [2608.03893](https://arxiv.org/abs/2608.03893) | Cross-Model KV Cache Transfer in LLM Families | Closed-form linear map transfers KV cache between sizes in a family (e.g. Qwen3 14B→32B), skipping prefill on model swaps / mid-conversation routing. |
| 8 | [2608.01651](https://arxiv.org/abs/2608.01651) | Bole: Efficient Tree Speculation for Hybrid-Attention LLMs | Kernel–runtime co-design verifying all tree-speculation nodes in parallel for hybrid attention; closes the linear-attention recurrence into a tree closed form. |
| 9 | [2608.02515](https://arxiv.org/abs/2608.02515) | LiveMem: Maintaining Memory State Continuity in Long-Running LLM Inference | "State continuity under context turnover": persistent intrinsic memory state + bounded KV window + memory-oriented post-training + state-aware serving. |
| 10 | [2608.02947](https://arxiv.org/abs/2608.02947) | ATFlash: Per-RoPE-Wavelength Attention Windows | Exploits RoPE wavelength structure to prune QK terms beyond wavelength-proportional distance; closed-form, input-independent reduction, stacks on top of MInference-style dynamic sparse. |
| 11 | [2608.01662](https://arxiv.org/abs/2608.01662) | LongCat Sparse Attention | Hardware–algorithm co-design fixing DeepSeek Sparse Attention's O(L²) Lightning Indexer: streaming-aware contiguous layouts, cross-layer index reuse, coarse-to-fine hierarchical scoring. |
| 12 | [2608.02032](https://arxiv.org/abs/2608.02032) | DART: Decoded Attention over Recurrent States | Extends Mamba-2 SSD: decode *token-conditioned keys* from the recurrent state, giving attention-style retrieval over the compressed state — long-context efficiency for hybrid models. |
| 13 | [2608.01672](https://arxiv.org/abs/2608.01672) | Learning What to Remember: Test-Time Training via Context Distillation | TTCD: long-window teacher supervises short-window student fast weights; hidden-state discrepancy teaches the model *what to memorize* for future use. |
| 14 | [2607.28627](https://arxiv.org/abs/2607.28627) | ReToken: One Token to Improve Vision-Language Models | Single learnable token that retrieves query-relevant visual tokens from a pre-filled visual KV cache; +13.4 points on Visual Haystacks for Qwen3VL-8B, transfers to long video. |
| 15 | [2608.01676](https://arxiv.org/abs/2608.01676) | Understanding Sparse Attention Selectivity | Counterfactual audit framework (Gold/Poison/Benign probe cards) proving sparse-attention route replay *causes* output decisions to change in 13/16 cells — a test harness for any sparse attention deployment. |

---

## 2. Reasoning & Test-Time Scaling

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.04001](https://arxiv.org/abs/2608.04001) | Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility | Formalizes test-time scaling as budgeted inference over the implicit prefix tree; distinguishes single-trajectory, sampling/voting, and search regimes — the missing reproducibility standard for TTS results. |
| 2 | [2608.02585](https://arxiv.org/abs/2608.02585) | GradCuit: Credit-Assigned Gradient Flow | Optimizes continuous latent states at a chosen layer at test time with reward-weighted gradients assigned directly through the transformer (no token-level proxies); robust, interpretable test-time latent reasoning. |
| 3 | [2608.06296](https://arxiv.org/abs/2608.06296) | On-Policy Self-Distillation without Any Supervision | Self-distillation with no labels at all — dense logit supervision from the model itself; relevant to post-training without RLHF data pipelines. |
| 4 | [2608.04962](https://arxiv.org/abs/2608.04962) | SpecRoll: Fast-Slow Verifier-Feedback Adaptation for Speculative RL Rollouts | Speculative rollout engine that keeps target sampling distribution while adapting drafter at two timescales during RL — cuts the autoregressive rollout bottleneck in RL post-training. |
| 5 | [2608.03447](https://arxiv.org/abs/2608.03447) | Approximate Speculative Decoding | Training-free verifier replacing binary first-mismatch truncation with budgeted longest-prefix selection, reusing target-scored suffix tokens — more acceptance per forward pass. |
| 6 | [2608.00220](https://arxiv.org/abs/2608.00220) | Verifier-Induced Support Reshaping in On-Policy Optimization | Warning: RLVR on one objective can *deplete* the rewardable support for later objectives ("successful behaviors too rare to sample") — a training-dynamics hazard to design around. |
| 7 | [2608.04408](https://arxiv.org/abs/2608.04408) | Not Every Divergence Should Be Suppressed | Counterfactual recoverability: replay error states through teacher-continuation vs rollback branches to decide retain/rollback/supervise; divergence alone has AUC 0.392 vs 1.000 for recoverability. |
| 8 | [2608.01285](https://arxiv.org/abs/2608.01285) | Stop When Memory Suffices | Router-Mem: evidence-conditioned progressive execution — low-cost retrieval first, sufficiency router decides early termination to cut agent latency without losing answer quality. |

---

## 3. On-Policy Distillation & Post-Training RL (the hottest cluster)

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2607.29209](https://arxiv.org/abs/2607.29209) | SAF-OPD: Stable Advantage Fusion | Fuses RLVR + OPD advantages with magnitude/temporal calibration — fixes the entropy collapse from fusing raw OPD advantage with bounded RLVR advantage. |
| 2 | [2608.01263](https://arxiv.org/abs/2608.01263) | Distill What the Student Can See | Fisher-Projected OPD: distill only *locally realizable* teacher corrections (via continuous visual perturbations) instead of the full teacher distribution — fixes capacity-gap distillation in VLMs. |
| 3 | [2608.04419](https://arxiv.org/abs/2608.04419) | SPOT: Sparse Probing and Outcome Calibration | Decides *where to probe* (position-level score over teacher entropy + top-k mass + mismatch) and *what to distill* with outcome-calibrated targets. |
| 4 | [2608.01953](https://arxiv.org/abs/2608.01953) | Look Ahead Before You Distill | FutureBridge-OPD: executes a short teacher bridge at high-disagreement states and validates it improves downstream trajectory density before using the guidance (agentic OPD). |
| 5 | [2607.28076](https://arxiv.org/abs/2607.28076) | Group-Reflective Self-Distillation for Agentic RL | Derives capability-aligned, outcome-discriminative natural-language skills from the policy's *own* verified rollouts (group reflection + stop-gradient contrast), for agentic RLVR. |
| 6 | [2607.28026](https://arxiv.org/abs/2607.28026) | Contrastive Reinforced Policy Optimization | Reformulates agentic OPD as contrastive learning: predictive entropy separates reflective-exploration positions from exposure-bias positions; group-wise contrast preserves fine-grained signals. |
| 7 | [2608.01303](https://arxiv.org/abs/2608.01303) | AlphaG-OPD: Reliability-Gated Sibling Counterfactuals | Uses sibling counterfactuals gated by reliability to pick which divergences to distill — companion to the "don't suppress every divergence" line. |
| 8 | [2607.29494](https://arxiv.org/abs/2607.29494) | Adaptive FastOPD: Progress-Aware Rollout Horizon Expansion | Expands rollout horizon adaptively as training progresses — cheaper OPD early, full horizon later. |
| 9 | [2607.29078](https://arxiv.org/abs/2607.29078) | DASH-OPD: Discrepancy-Aware Switching with Hysteresis | Switches distillation targets with hysteresis to prevent oscillation between unstable teacher signals. |
| 10 | [2608.03316](https://arxiv.org/abs/2608.03316) | Any-OPD: Heterogeneous On-Policy Distillation for Flow-Matching Models | Ports OPD to flow-matching (diffusion) models via representation-space alignment. |
| 11 | [2607.28022](https://arxiv.org/abs/2607.28022) | Flux-OPD: On-Policy Distillation with Evolving Contexts | Uses *evolving* contexts (student-performance-adapted) as in-training supervision with reverse-KL stabilization + conflict downweighting — for open-ended domains without verifiable rewards. |
| 12 | [2608.05987](https://arxiv.org/abs/2608.05987) | AgentOPSD: Recursive Self-Distillation for Agentic RL | Turn-level credit assignment: aggregates token-level teacher–student gaps into turn-level evidence, recursively, critic-free. |
| 13 | [2608.04788](https://arxiv.org/abs/2608.04788) | Agentic RL with Observation-Calibrated Self-Distillation | Fixes a confounder in OPSD: the replay scaffold itself shifts scores; calibrates so support is attributed to privileged information, not scaffolding. |
| 14 | [2608.05802](https://arxiv.org/abs/2608.05802) | On-Policy Delta Distillation for Multilingual Math | OPD² (gap between post-trained teacher and base model as signal) outperforms OPD; narrows EN–KR/JA math reasoning gaps. |
| 15 | [2608.01597](https://arxiv.org/abs/2608.01597) | HindSearch: Hindsight Critique for Search-Augmented RL | After failed rollouts, a frozen judge writes a critique from the gold answer; critique is used as auxiliary on-policy distillation on search actions (GRPO) — + on seven-benchmark suite. |
| 16 | [2608.00533](https://arxiv.org/abs/2608.00533) | Native Multilingual CoT in Low-Resource Southeast Asian Languages | Pipeline for native (not translated) multilingual chain-of-thought reasoning in low-resource languages. |
| 17 | [2608.02951](https://arxiv.org/abs/2608.02951) | SP3O: RL from Segment Preferences without Reward Model | Gradient-based preference RL on *segments* (shorter than trajectories) without learning a reward model. |
| 18 | [2608.02087](https://arxiv.org/abs/2608.02087) | Instruction-Conditioned Exploration | Appends a small fixed instruction set to prompts during RL training to widen behavioral coverage; distills back to an unconditioned policy. |
| 19 | [2608.01804](https://arxiv.org/abs/2608.01804) | LEAP: Lean Environment-Feedback via Adaptive Pruning for Code RL | Multi-turn RL for low-level code (CUDA kernel gen): difficulty-conditioned rollout pruning to fight sparse binary rewards + compile-latency reward dilution. |
| 20 | [2608.06246](https://arxiv.org/abs/2608.06246) | A Six-Dimensional Taxonomy of Post-Training Adaptation | Survey + taxonomy (mechanism/goal/data/persistence/structure/…)—useful map before choosing a post-training method. |
| 21 | [2607.28495](https://arxiv.org/abs/2607.28495) | Stage-Replay Divergence Follows the KV Cache | Cautionary empirical study: fresh-prefill continuation vs retained-live-cache replication is *not* exact in BF16 at whole-stage boundaries — matters for evaluation harnesses using stage replay. |

---

## 4. Retrieval / RAG / Memory

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.02560](https://arxiv.org/abs/2608.02560) | Structured Memory for Edge Language Models | PRECOG: pre-encode corpora as SSM hidden states and *inject the matching state* at query time → O(1) prefill per query for RAG (SSMs only). Plus SMC hierarchical persistent memory with cognitive-domain clustering. |
| 2 | [2608.03148](https://arxiv.org/abs/2608.03148) | Lightweight Chunk Selection for Mobile RAG | Evidence-alignment chunk selector using question hidden states + MoE routing expert signals + lexical features; keeps RAG cost mobile-budget friendly. |
| 3 | [2608.02112](https://arxiv.org/abs/2608.02112) | Do Static Embeddings Add Value to Hybrid Retrieval? | Rigorous RRF-fusion study (BM25 + Qwen3-Embedding + static embeddings) with bootstrap CIs — answers *when* cheap retrievers add complementary signal. |
| 4 | [2608.02907](https://arxiv.org/abs/2608.02907) | Bayesian Data Reweighting Improves Multimodal Retrieval | Reweights training data for knowledge-based visual QA retrieval. |
| 5 | [2608.02508](https://arxiv.org/abs/2608.02508) | RoMeRL: Balancing Feedback Coverage and the Memory-Reward Trap | Reduced-order memory RL: fixed-dimensional per-task memory state factorized by outcome polarity/dynamics to concentrate sparse feedback; avoids irrelevant co-retrieved memories getting utility credit. |
| 6 | [2608.00585](https://arxiv.org/abs/2608.00585) | Verification Without Sufficiency: Per-Chunk Filtering Fails on Multi-Hop RAG | Evidence that chunk-level verification is insufficient for multi-hop queries — design guidance for RAG verifiers. |
| 7 | [2608.02678](https://arxiv.org/abs/2608.02678) | DenialRAG: Single-Document RAG Poisoning | Attack that *names and denies* the correct answer in a single poisoned doc; a red-team reference for RAG robustness. |
| 8 | [2608.02880](https://arxiv.org/abs/2608.02880) | Field Aware Agent Skill Retrieval | Retrieval of agent skills conditioned on task fields — improves agent skill reuse. |

---

## 5. MoE (Mixture of Experts)

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.04407](https://arxiv.org/abs/2608.04407) | MESH: Memory-Efficient Sinkhorn Optimization for MoE Training | Fixes a real bug: Sinkhorn/AdamW-free optimizers fail on routed MoE experts (conditional, temporally-varying gradients); hidden-momentum Sinkhorn restores temporal first-moment without storing optimizer state. Cuts optimizer state 0.88GB→0.33GB at 110M. |
| 2 | [2608.02989](https://arxiv.org/abs/2608.02989) | AcceptMoE: Commitment-Weighted Self-Sizing Verifier Expert Sets | For MoE speculative decoding: verifier-side expert selection using router scores × offline commitment probabilities; auto-sizes expert count per block; conditions on cache residency for offloading. |
| 3 | [2608.04401](https://arxiv.org/abs/2608.04401) | Elbow-Based MoE Routing | Training-free per-token dynamic top-k: finds elbow in sorted router probabilities, activates only relevant experts — saves compute on easy tokens. |
| 4 | [2608.05303](https://arxiv.org/abs/2608.05303) | EdgeXpert | HW/SW co-designed accelerator resolving MoE + speculative-decoding incompatibility; prompt-wise expert reuse + shared expert sets for edge LLMs. |
| 5 | [2608.04454](https://arxiv.org/abs/2608.04454) | Beyond Global Routing Aggregation: Phase-Aware Expert Merging for MoE VLMs | Phase-aware expert merging instead of global routing aggregation for MoE vision-language models. |
| 6 | [2607.28097](https://arxiv.org/abs/2607.28097) | From Expert Reduction to Behavioral Divergence | Shock study: mathematically-equivalent expert-reduction orders produce *different continuations* in DeepSeek-V4-Flash (e.g. 202 layoffs vs 113 hiring under one prompt) — nondeterminism/associativity hazard in sparse MoE inference. |
| 7 | [2608.00916](https://arxiv.org/abs/2608.00916) | Tevatron Meets Megatron | Expert-parallel MoE reranker training (Megatron backend into Tevatron) on academic budgets; up to 22% faster, HF-compatible. |

---

## 6. Quantization / Compression

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.01528](https://arxiv.org/abs/2608.01528) | Gram-Space: Structure-Preserving Codebook Compression | Gram-Schmidt orthonormal basis for VSA codebooks with provably preserved inner products — memory cut for neuro-symbolic/attention computations. |
| 2 | [2608.05499](https://arxiv.org/abs/2608.05499) | APQF: Agentic Profiling-Guided Pruning + Mixed-Precision Quantization | Uses agentic profiling (per-layer sensitivity across tasks) to drive structured pruning and mixed-precision assignment. |
| 3 | [2607.28589](https://arxiv.org/abs/2607.28589) | MixFrag: Fragility-Guided Mixed-Precision PTQ | KL-divergence fragility estimates per component → heterogeneous bit-width allocation for ViTs; pattern transfers to LLMs. |
| 4 | [2607.28292](https://arxiv.org/abs/2607.28292) | CACHE-UK: Stability-Aware Memory Editor for Quantized LLMs | Rank-1 LoRA-subspace memory editing + "degradation debt" stability controller — safe sequential fact edits on 4-bit quantized LLMs (finance domain). |
| 5 | [2608.02700](https://arxiv.org/abs/2608.02700) | NANQ: Noise-Floor-Aware Mixed-Precision Quantization | For analog compute-in-memory: converts hardware noise profile into adaptive quantization density. |
| 6 | [2608.03579](https://arxiv.org/abs/2608.03579) | Pin Once, Swap Light: Subspace-Aligned Centroid-Residual Training | Efficient low-precision training with subspace-aligned centroid-residual decomposition (subspace-constrained quantization-aware training). |

---

## 7. Serving & Systems

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2607.28848](https://arxiv.org/abs/2607.28848) | DeltaServe: Host-Agnostic Co-Serving of Inference and Fine-Tuning | Runs LoRA fine-tuning on idle inference GPU capacity via SLO-aware scheduler + shared prefill/fine-tune forward structure. Converts idle compute into training throughput. |
| 2 | [2608.03036](https://arxiv.org/abs/2608.03036) | LLM Serving in the Wild | Empirical study of how vLLM/SGLang/TensorRT-LLM etc. are actually adopted in OSS — the data to base serving-stack choices on. |
| 3 | [2608.06117](https://arxiv.org/abs/2608.06117) | LLM Inference Under Bursty Workload Distribution: Modifying the WAIT Algorithm | Adjusts the WAIT scheduling algorithm for bursty workload distributions. |
| 4 | [2608.01536](https://arxiv.org/abs/2608.01536) | Celty: SpMspV GPU Kernel + SIMT Co-Design | Co-designed sparse format/kernel/microarchitecture for dual-sparse (weight + activation) single-user LLM decoding. |
| 5 | [2608.01263](https://arxiv.org/abs/2608.01263)* | Efficient KD for LLMs (see §9) | Offline top-K logit KD 29% faster / 41% higher throughput; fused chunked KL avoids full-vocab logit tensors. |

---

## 8. Distillation / Model Compression

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.03796](https://arxiv.org/abs/2608.03796) | Efficient Knowledge Distillation for LLMs | Offline caching of teacher top-K logits matches online KD but 29% faster/iter, up to 41% higher throughput, teacher freed from memory; fused chunked KL loss avoids full-vocab materialization. |
| 2 | [2608.00129](https://arxiv.org/abs/2608.00129) | Progressive²: Teacher-Student Progressive Co-Evolving KD | Handles large teacher–student capability gaps via progressive co-evolving distillation. |
| 3 | [2608.04794](https://arxiv.org/abs/2608.04794) | Privileged, but Biased: How PI-Conditioned Teachers Break Self-Distillation | Shows privileged-information-conditioned teachers inject bias into self-distillation — mitigation guidance. |
| 4 | [2608.06137](https://arxiv.org/abs/2608.06137) | SkillTFM: Gated Skill Evolution for Tabular Foundation Models | Training-free adaptation of tabular foundation models via gated skill evolution (relevant if you serve tabular LLM-family models). |
| 5 | [2608.02689](https://arxiv.org/abs/2608.02689) | Stuck on "A": Attention-to-Linear Conversion Diagnostic | Converting 21/28 attention layers to Kimi Delta Attention: perplexity recovers but MCQ accuracy collapses to 25–29%; "interface injury" diagnosed via option-permutation — caution for linear-attention conversion pipelines. |

---

## 9. Interpretability / Steering / Safety

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.05732](https://arxiv.org/abs/2608.05732) | CircuitSteer: Multi-Layer Steering via SAE Circuits | SAE feature-flow circuits + geometric alignment of decoder directions across layers → dense multi-layer steering vectors; beats single-layer CAA-style intervention. |
| 2 | [2608.04893](https://arxiv.org/abs/2608.04893) | When Does Latent Communication Pay? | Causal audit of relayed KV caches between multi-agent LLMs — when sharing hidden state between agents actually helps. |
| 3 | [2608.05783](https://arxiv.org/abs/2608.05783) | GROM: Gradient-Free One-Shot Machine Unlearning | One-shot, gradient-free unlearning; notably addresses that LoRA-finetune unlearning "restores" after quantization (the classic quantize-to-revive attack). |
| 4 | [2608.01085](https://arxiv.org/abs/2608.01085) | Collective Evidence-Threshold Backdoors in Multi-Agent Systems | New backdoor class: malicious behavior triggers only when *peer evidence* crosses a threshold in MAS; plus Boundary-Conditioned Backdoor Injection + mitigation. |
| 5 | [2608.04052](https://arxiv.org/abs/2608.04052) | Conformal Backdoor Detection in Multimodal Contrastive Learning | CLIPScore overlap makes fixed-threshold detection unreliable; conformal guarantees for backdoor detection. |
| 6 | [2608.03711](https://arxiv.org/abs/2608.03711) | Attention is Case-Sensitive | Casing modulates internal attention: uppercase/alternating-case target text concentrates attention on those spans (universal across 13 models) — prompt-engineering lever + interpretability insight. |
| 7 | [2608.05608](https://arxiv.org/abs/2608.05608) | GAUGE: Granularity-Adaptive Counterfactual Gating of Evidence | Counterfactual gating of evidence for incomplete multimodal classification (robustness to missing modalities). |
| 8 | [2608.00200](https://arxiv.org/abs/2608.00200) | TRACE-TS: Attribution-Grounded Sensor-Language Reasoning | Signal-grounded (expert-classifier attribution) natural-language reasoning over time series — reduces unverifiable/fluent-but-wrong explanations. |
| 9 | [2608.01481](https://arxiv.org/abs/2608.01481) | Interpretable MEG Decoding of Perceived Speech | Not LLM-core, but a rigorous "interpretable neural decoding" template (spherical-harmonic spatial attention, source-matched filters) adaptable to interpretable speech/signal layers. |
| 10 | [2607.28374](https://arxiv.org/abs/2607.28374) | LEDGERMIND: Provenance-Constrained Multimodal Agentic Reasoning | Structured Evidence Ledger as trajectory state; reasoning claims must cite active ledger entries → grounded, auditable multimodal agent reasoning. |
| 11 | [2608.03130](https://arxiv.org/abs/2608.03130) | DP-MemView: Memory Interface with Attribute-Level Transcript Privacy | Differential-privacy memory interface for long-term agent transcripts — per-attribute privacy in agent memory. |

---

## 10. Data / Synthetic Data / Curation / Eval

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2607.28801](https://arxiv.org/abs/2607.28801) | Benchmarks Are Not Monolithic | Sample-level meta-evaluation auditing MMLU/ARC/WinoGrande/HellaSwag/TruthfulQA along 5 latent dimensions; heterogeneity hidden by aggregate scores. Use for eval-set curation. |
| 2 | [2608.01004](https://arxiv.org/abs/2608.01004) | Who Belongs in the Eval Set? | Capability-taxonomy-driven pipeline for curating eval sets — avoids sample-selection bias in benchmarks. |
| 3 | [2608.02690](https://arxiv.org/abs/2608.02690) | GLOBE: Trajectory-Aligned Gradient Matching Coreset Selection | Coreset selection aligned with optimization trajectory (multiple snapshots) + structured sparse optimization — data-efficient training. |
| 4 | [2607.29120](https://arxiv.org/abs/2607.29120) | Curriculum Matters: Data-Efficient Relational PFN Pretraining | Shows synthetic-data *ordering* changes downstream PFN performance — curriculum as a first-class lever in synthetic pretraining. |
| 5 | [2608.04056](https://arxiv.org/abs/2608.04056) | Multi-Agent Perspectivist Preference Optimization | Trains one model per annotator-cluster (by labeling behavior), coordinates with preference optimization — preserves label disagreement instead of majority-vote collapse. |
| 6 | [2608.04678](https://arxiv.org/abs/2608.04678) | Kathleen Writes: Autoregressive Generation and Data Scaling Without Attention | Byte-level attention-free (wavetable + multi-scale reverberant state) model beats a parameter-matched transformer at every scale on byte-level LM (1.84 vs 2.04 bits/byte) — attention-free scaling data point. |
| 7 | [2607.28879](https://arxiv.org/abs/2607.28879) | RareSense: Rarity-Aware Similarity Search | Rare-itemset profiles for anomaly retrieval in sparse transactional data. |
| 8 | [2607.28880](https://arxiv.org/abs/2607.28880) | LayoutBench: Storage Layout Benchmark for Multimedia Data | First benchmark evaluating cloud-storage layout strategies for multimedia retrieval workloads. |

---

## 11. RL / Agents (general methodology with LLM transfer)

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.06223](https://arxiv.org/abs/2608.06223) | TS-RAG: RAG for Time Series Forecasting | RAG pipeline adapted to multivariate time series with magnitude-mismatch handling. |
| 2 | [2608.05571](https://arxiv.org/abs/2608.05571) | Align-RAG: Alignment Is All You Need for TSFM In-Context Learning | Context alignment to make in-context learning work for time-series foundation models. |
| 3 | [2608.02391](https://arxiv.org/abs/2608.02391) | CoPES: Cooperative Coevolution for Agentic LLM Post-Training | Memory-efficient full-parameter post-training via evolution strategies + cooperative parameter-subspace decomposition — gradient-free agentic post-training on a few GPUs. |
| 4 | [2608.05111](https://arxiv.org/abs/2608.05111) | Reward Structure × Episodic Exploration × Neural Memory | Controlled study: same exploration bonus gives three different outcomes depending on memory architecture — choose exploration/memory jointly. |
| 5 | [2608.06362](https://arxiv.org/abs/2608.06362) | AV-AIVAT: 74× Cheaper Agent Evaluation | Certified anytime-valid stopping for agent-vs-agent eval (imperfect-information games); variance-reduced AIVAT + valid optional stopping → 74× cheaper evaluations. |
| 6 | [2608.02528](https://arxiv.org/abs/2608.02528) | Uncertainty Is Not Enough: Value-of-Information Routing for LoRA Experts | Routes between LoRA experts by value-of-information, not just uncertainty — cheaper + better routing for adapter mixtures. |
| 7 | [2608.00335](https://arxiv.org/abs/2608.00335) | RMSWeb: Reflection, Failure-Mode Mining, Salvage-DS for Web Agents | Three-part recipe for web agents: reflection data, failure-mode mining, salvage-DS fallback signal for rejected GRPO groups. |
| 8 | [2608.06112](https://arxiv.org/abs/2608.06112) | Compliance-First Agentic Platforms (Healthcare) | Multi-layered agentic architecture blueprint with governance for regulated domains. |
| 9 | [2608.04317](https://arxiv.org/abs/2608.04317) | Trident: Breaking DRL Cyber Defenses | Agentic LLM red-team framework producing adaptive red agents for DRL cyber defense. |
| 10 | [2608.06130](https://arxiv.org/abs/2608.06130) | Hardware Keystores for AI Agent Signing | Zero-trust MCP architecture enforcing hardware-confined keys + content-aware authorization for agent crypto ops. |
| 11 | [2608.01597](https://arxiv.org/abs/2608.01597) | HindSearch (see §3) | Hindsight critique self-distillation for search-augmented RL. |

---

## 12. Cross-Cutting Methodology (adoptable patterns)

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.06352](https://arxiv.org/abs/2608.06352) | CalibForge | Dataset + solver-calibration method for adversarial calibration of learnable terminal tasks (data available on HF/GitHub). |
| 2 | [2608.05859](https://arxiv.org/abs/2608.05859) | Evidential Rule Learning with Abstention | Interpretable rule classifier that can abstain — template for safe LLM-guard/classifier layers. |
| 3 | [2607.28037](https://arxiv.org/abs/2607.28037) | ClawTrack | Trace-level evaluation + improvement framework for real-world autonomous (agent) systems. |
| 4 | [2608.05797](https://arxiv.org/abs/2608.05797) | Predicting Task Difficulty Without Rollouts | Predict difficulty from the model's own weights/activations without running rollouts — budget RL compute by difficulty. |
| 5 | [2608.06216](https://arxiv.org/abs/2608.06216) | Continual Learning in Transition | Survey on continual learning in the LLM/agentic-AI era — memory replay, knowledge consolidation, and forgetting control frameworks. |
| 6 | [2608.03875](https://arxiv.org/abs/2608.03875) | Enhancing VLM Reward Models Through Structure-Aware Fine-Tuning | Structure-aware fine-tuning for VLM reward models — directly improves RM quality for VLM RLHF. |
| 7 | [2607.28627](https://arxiv.org/abs/2607.28627) | ReToken (see §1) | Single-token retrieval for VLM visual KV cache. |

---

## Immediate takeaways (highest ROI for a LLM engineering effort)

1. **KV-cache: the easiest wins.** AnchorKV (20× no-discard compression), QEvict (recoverable eviction), SAKI (score-preserving low-rank index), OptR (INT2 with output-aware rotation), and Cross-Model KV Transfer are all drop-in serving/training-independent techniques — start here if serving long-context.
2. **On-policy distillation is the new RLHF hotness.** The OPD cluster (SAF-OPD, FP-OPD, SPOT, FutureBridge-OPD, GRSD, AgentOPSD, Delta-OPD) gives a coherent, label-light post-training alternative with dense supervision; several explicitly fix known OPD failure modes (magnitude/temporal mismatch, capacity gap, exposure bias).
3. **A real MoE training bug + a real MoE inference hazard.** MESH documents and fixes optimizer-state failure on routed experts; "From Expert Reduction to Behavioral Divergence" documents order-dependent nondeterminism in sparse MoE inference. Both are things you must test for.
4. **Eviction/sparse-attention audits exist now.** Sparse-attention selectivity counterfactual audit and Stage-Replay divergence study give you test harnesses to validate that compression actually preserves behavior before shipping.
5. **Test-time scaling finally has a reproducibility framework** (prefix-tree budget formalism) and cheap test-time reasoning (GradCuit, Router-Mem early-exit).
6. **Agentic RL recipes are converging**: hindsight critiques (HindSearch), group reflection (GRSD), observation-calibrated self-distillation, and CoPES (gradient-free post-training) cover the data/reward/training stack for agents.

---

# Part 2 — Constrained-Hardware Inference

**Analysis target:** papers usable for (1) reducing VRAM usage, (2) streaming models from disk, (3) CPU-based inference, (4) distributed inference over the network, (5) using multiple low-end GPUs in parallel.

## 13. VRAM Reduction (weights + KV cache + activation)

| # | arXiv | Title | What it does for VRAM |
|---|-------|-------|------------------------|
| 1 | [2608.02901](https://arxiv.org/abs/2608.02901) | AnchorKV: Anchor-Residual KV Cache Compression | 20× KV-cache shrink with zero token loss; anchors stored exactly + residuals — keeps ~99% of full-cache score at 70B. The single highest-VRAM-savings-per-effort item here. |
| 2 | [2608.04074](https://arxiv.org/abs/2608.04074) | Spend Bits Where Queries Look (KV VQ) | KV-cache vector quantization at 2 bits/element with attention-preserving transforms — cuts the dominant long-context VRAM/bandwidth term. |
| 3 | [2608.02691](https://arxiv.org/abs/2608.02691) | Output-Aware Rotation for INT2 KV Cache | INT2 KV cache with post-W_O error minimization — deeper quantization without the usual accuracy cliff. |
| 4 | [2608.05326](https://arxiv.org/abs/2608.05326) | QEvict | Recoverable quantized KV eviction; fixes the "evicted-forever" flaw with three-tier cache management. |
| 5 | [2608.01247](https://arxiv.org/abs/2608.01247) | RestoreKV | LoRA-adapted "restore tokens" regenerate full-cache behavior from a small cache under aggressive eviction. |
| 6 | [2608.03228](https://arxiv.org/abs/2608.03228) | SAKI | Training-free low-rank KV index that preserves attention scores (not reconstruction); beats key-PCA at every rank on 8B models. |
| 7 | [2608.01536](https://arxiv.org/abs/2608.01536) | Celty: SpMspV GPU Kernel | Dual-sparse (unstructured weight pruning + activation sparsity) LLM decoding; RLC-CSC format skips memory accesses — lower VRAM traffic and cache pressure for single-user decode. |
| 8 | [2608.06291](https://arxiv.org/abs/2608.06291) | BaKron: Kronecker-Factored Hessian Quantization | GPTQ-class quantization using two-sided Kronecker-Hessian curvature at GPTQ's O(mn(m+n)) cost → better low-bit weight compression. |
| 9 | [2608.05499](https://arxiv.org/abs/2608.05499) | APQF | Agentic profiling-guided structured pruning + mixed-precision quantization + recovery — automated per-layer bit-width/pruning assignment for edge GPUs. |
| 10 | [2608.04405](https://arxiv.org/abs/2608.04405) | BinaryPC: Training-Free Hashing Attention | Binary-PCA hash codes for sparse attention — training-free long-context acceleration, shrinks the effective KV read set. |
| 11 | [2608.04401](https://arxiv.org/abs/2608.04401) | Elbow-Based MoE Routing | Training-free per-token dynamic expert count (5.3% avg latency cut, no accuracy loss) — less expert activation = less weight VRAM traffic. |
| 12 | [2608.02989](https://arxiv.org/abs/2608.02989) | AcceptMoE | MoE speculative decoding that self-sizes verifier expert sets and conditions on cache residency under offloading — directly minimizes expert-weight traffic. |
| 13 | [2608.05303](https://arxiv.org/abs/2608.05303) | EdgeXpert | HW/SW co-design resolving MoE + speculative-decoding incompatibility; prompt-wise expert reuse + depth-aware expert coalescing cut FFN memory access. |
| 14 | [2608.03579](https://arxiv.org/abs/2608.03579) | Pin Once, Swap Light (SALT) | Ultra-LoRA (r≤2) serving: one shared centroid pinned in VRAM + tiny task residuals swapped over PCIe → tens-to-hundreds of adapters without VRAM blowup. **Core idea for multi-tenant VRAM. |
| 15 | [2608.00860](https://arxiv.org/abs/2608.00860) | Kilobyte Models | A trained net = seed + quantized latent (mapping networks); bytes ≈ latent size, not parameter count. Extreme-weight compression for transfer/streaming. |
| 16 | [2608.03796](https://arxiv.org/abs/2608.03796) | Efficient KD for LLMs | Offline top-K logit KD + fused chunked KL: teacher removed from memory during distillation, peak VRAM capped (memory-light way to produce small models). |
| 17 | [2607.28292](https://arxiv.org/abs/2607.28292) | CACHE-UK | Stability-aware memory editing for 4-bit quantized LLMs (rank-1 LoRA subspace) — lets quantized small models stay accurate with cheap fact updates. |
| 18 | [2608.05253](https://arxiv.org/abs/2608.05253) | AuroOFT | Quantized orthogonal fine-tuning + zero-start gated nonlinear residual — adapt low-bit models more expressively than plain qoft. |
| 19 | [2608.02032](https://arxiv.org/abs/2608.02032) | DART | Decoded attention over recurrent (Mamba-2 SSD) states: attention-style retrieval over a fixed-size recurrent state instead of a growing cache. |

## 14. Model Streaming from Disk / Offload / Weight Swapping

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.03579](https://arxiv.org/abs/2608.03579) | Pin Once, Swap Light (SALT) | The explicit pin-and-swap design: centroid resident in VRAM, low-rank residuals swapped host↔device over PCIe. Directly transferable to weight-slicing strategies for disk-backed LLMs. |
| 2 | [2608.00860](https://arxiv.org/abs/2608.00860) | Kilobyte Models | Store only the seed+latent; regenerate weights on the target device — the extreme end of "stream less from disk." |
| 3 | [2608.02989](https://arxiv.org/abs/2608.02989) | AcceptMoE | Explicitly models expert-weight *transfer traffic* and cache residency under offloading; eligibility decided by what's resident — a pattern for SSD-resident MoE weights. |
| 4 | [2608.05303](https://arxiv.org/abs/2608.05303) | EdgeXpert | External-memory-access reduction for FFN layers; prompt-wise expert reuse means you can page-in only a shared expert set per prompt. |
| 5 | [2608.02560](https://arxiv.org/abs/2608.02560) | PRECOG (Structured Memory for Edge LMs) | Pre-encode corpora as fixed SSM hidden states and inject O(1) — kills RAG prefill and KV growth, so edge/CPU devices can run RAG without caching long contexts. |
| 6 | [2608.04428](https://arxiv.org/abs/2608.04428) | Deltoris | Temporal bit-sparsity + speculative inference: computes only deltas between consecutive inputs and amortizes data loading across steps — cuts off-chip traffic (relevant for disk/DRAM-bound inference). |
| 7 | [2608.03893](https://arxiv.org/abs/2608.03893) | Cross-Model KV Cache Transfer | Linear ridge map reuses KV across model sizes → swap models mid-conversation *without* re-paying prefill (relevant for cascade/disk-backed model switching). |
| 8 | [2608.01536](https://arxiv.org/abs/2608.01536) | Celty | Sparse format that vectorizes compressed weight columns and skips zero activations — fewer bytes pulled from HBM/disk per decode step. |

## 15. CPU-Based Inference

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.01563](https://arxiv.org/abs/2608.01563) | Meganeura | Portable GPU inference/training compiled through **Vulkan and Metal** — runs the same compiled graph on NVIDIA, AMD discrete, AMD APU, Apple Silicon, **and Intel iGPUs**, 48/50 cells passing, often within ~1.1–1.8× of native. Practical route to CPU/iGPU-adjacent execution without per-vendor toolchains. |
| 2 | [2608.02560](https://arxiv.org/abs/2608.02560) | PRECOG (Structured Memory for Edge LMs) | SSM-state injection makes long-context/RAG tractable on edge and CPU: O(1) prefill per query, fixed-size recurrent state instead of KV cache. |
| 3 | [2607.29353](https://arxiv.org/abs/2607.29353) | ECL: Versatile On-device Adaptation | Unifies few-shot/zero-shot/continual/in-context learning *on-device* (embedder-centric learning) — for CPU/edge personalization without cloud retraining. |
| 4 | [2608.05064](https://arxiv.org/abs/2608.05064) | Certified Deferral for Small LMs | When small/CPU-hosted models should defer to a human; Clopper-Pearson finite-sample risk certificates for 0.5–14B models on local hardware. |
| 5 | [2608.04980](https://arxiv.org/abs/2608.04980) | Protoreasoning in Tiny Transformers | CoT/protoreasoning demonstrated at ~1M-parameter scale — evidence that reasoning techniques transfer to tiny CPU-runnable models. |
| 6 | [2608.03148](https://arxiv.org/abs/2608.03148) | Lightweight Chunk Selection for Mobile RAG | Evidence-alignment chunk selector for mobile RAG budgets — reduces what must be ingested/generated on-device. |
| 7 | [2608.00720](https://arxiv.org/abs/2608.00720) | CascadeLUT | LUT-based streaming inference on FPGAs (no multipliers, 4–12.5× lower latency, 3–5× throughput on bandwidth-limited hardware). |
| 8 | [2608.06177](https://arxiv.org/abs/2608.06177) | Threshold-Based Early Stopping of Accumulations (binary activation) | Early-stopped MAC accumulation in binary-activation nets — power/latency win pattern for low-end CPUs/iGPUs. |
| 9 | [2608.03711](https://arxiv.org/abs/2608.03711) | Attention is Case-Sensitive | (bonus) Formatting lever to concentrate attention — free accuracy on small models without compute cost. |
| ⚠️ | [2608.00737](https://arxiv.org/abs/2608.00737) | Embedded RISC-V KAN eval | **Caution:** KAN "parameter efficiency" evaporates on real embedded CPUs — 13.5×/8.0× slower and 11.3× more energy than MLP under PTQ on RISC-V. Don't assume small-parameter = fast on CPU. |
| ⚠️ | [2608.03854](https://arxiv.org/abs/2608.03854) | Quantization Effects on Biomedical LLM Reliability | **Caution:** INT4/INT8 classifier calibration depends on the *probability extraction protocol* (summed vs mean log-likelihood reverses calibration ranking). On CPU you must re-verify calibration, not just accuracy. |

## 16. Distributed Inference over Network / Collaborative

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.02031](https://arxiv.org/abs/2608.02031) | Collaborative MEC for LLM Inference | Transformer-enhanced PPO scheduling that distributes LLM inference subtasks across collaborative mobile edge servers under soft deadlines — a ready framework for splitting inference across a network of nodes. |
| 2 | [2607.29659](https://arxiv.org/abs/2607.29659) | GQ-FSL: Green Quantized Federated Split Learning | Split learning: client runs a sub-model, server runs the rest, quantized asymmetric precision on each side; joint split-point+precision optimization with convergence bound → offload compute to a networked server. |
| 3 | [2607.29071](https://arxiv.org/abs/2607.29071) | FedSLM | Heterogeneous compressed clients (SVD low-rank) that stay self-contained and aggregatable — pattern for weak nodes contributing to / serving parts of a foundation model. |
| 4 | [2608.02378](https://arxiv.org/abs/2608.02378) | Gecko | Split public/private inference: run a public encoder client-side (or on a cheap node), evaluate only a compact predictor under crypto — the split-inference security/bandwidth pattern. |
| 5 | [2608.04893](https://arxiv.org/abs/2608.04893) | When Does Latent Communication Pay? | Causal audit of relayed KV caches in multi-agent LLMs: relaying latent state only pays when the receiver *needs the sender's private info* (100% vs 23–25% otherwise). **Design rule for networked/distributed LLM comms: send text unless the peer needs your hidden state. |
| 6 | [2608.05944](https://arxiv.org/abs/2608.05944) | Multi-Node Full Fine-Tuning on B300 (Field Report) | Operational telemetry/triage for multi-node FSDP/ZeRO-3 training: power-draw tables to distinguish compute/comm/data-starvation/deadlock, NFS vs local caching measurements. Useful ops playbook if you run multi-node. |
| 7 | [2608.00916](https://arxiv.org/abs/2608.00916) | Tevatron Meets Megatron | Expert-parallel reranker training on academic budgets (Megatron backend in Tevatron) — distributed-training infra for MoE without H100 clusters. |
| 8 | [2608.01975](https://arxiv.org/abs/2608.01975) | TELLER | Non-intrusive cross-layer root-cause analysis for LLM inference (request spans engine + CUDA + distributed comms) — the observability tool you'll want when debugging a networked inference stack. |
| 9 | [2608.06135](https://arxiv.org/abs/2608.06135) | LLM Inference Under Bursty Workloads | Modified WAIT scheduler with online request-intensity estimation for bursty traffic — throughput/latency control for shared network inference. |

## 17. Parallel / Low-End GPU Utilization (many weak cards as one)

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.01563](https://arxiv.org/abs/2608.01563) | Meganeura | Proven cross-vendor (NVIDIA/AMD/Intel iGPU/APU) portable execution — the enabling layer for "whatever cards you own, run the same model." |
| 2 | [2608.05033](https://arxiv.org/abs/2608.05033) | SparseDitto | LLM-agent system that *generates a GPU kernel per matrix/operator/target GPU* (SpMV/SpMM/SpGEMM) — adapts to whatever GPU each node has, closing the 350× cuSPARSE format gap. Great for heterogenous clusters. |
| 3 | [2608.05499](https://arxiv.org/abs/2608.05499) | APQF | Automated per-layer pruning + bit-widths by measured sensitivity — makes old/low-VRAM cards usable with the least accuracy loss. |
| 4 | [2608.04428](https://arxiv.org/abs/2608.04428) | Deltoris | Bit-level sparsity + speculative inference co-design for edge VLAs (50–200 Hz control loops) — shows how to make real-time inference fit low-end hardware. |
| 5 | [2608.05303](https://arxiv.org/abs/2608.05303) | EdgeXpert | MoE + speculative decoding made compatible → each weak card only needs a subset of experts resident. |
| 6 | [2608.00860](https://arxiv.org/abs/2608.00860) | Kilobyte Models | Regenerable weights mean you can broadcast a model to N weak cards with near-zero transfer cost. |
| 7 | [2608.03695](https://arxiv.org/abs/2608.03695) | Dynamic Graph Clustering on GPU (cuGraph) | Multi-GPU via Dask workload distribution (~1000× vs CPU) — pattern for distributing workloads across low-end cards. |
| ⚠️ | [2607.28097](https://arxiv.org/abs/2607.28097) | From Expert Reduction to Behavioral Divergence | **Critical warning for multi-card MoE:** mathematically-equivalent expert-reduction/aggregation orders produce *different outputs* (720 A-mode orders → 10 continuation basins; one prompt forks into "202 layoffs vs 113 hiring"). If you shard experts across cards, you MUST standardize aggregation order and test determinism, or outputs diverge between machines. |

## 18. Recommended stack for "run a big LLM on weak hardware"

Combining the batch into a concrete pipeline:

1. **Weights → VRAM:** quantize with BaKron (Kron-Hessian-aware) or APQF (sensitivity-guided mixed precision), prune with Celty-style dual sparsity.
2. **KV cache:** AnchorKV (20×) or SAKI/INT2-OptR for the cache; DART/Mamba-2-style recurrent states to remove it entirely where possible.
3. **If VRAM still short → disk streaming:** SALT pin-and-swap pattern for adapters/experts, AcceptMoE cache-residency-aware expert paging, Kilobyte-seed regeneration as the extreme.
4. **If no GPU / CPU-only:** Meganeura (Vulkan/Metal portable kernels) + PRECOG (SSM state injection to kill prefill cost) + ECL for on-device adaptation + Certified Deferral for risk-controlled small-model serving.
5. **If you have multiple weak GPUs:** Meganeura for portability, SparseDitto for per-GPU kernels, split-inference from GQ-FSL/Gecko across nodes, MEC-PPO for scheduling — and **respect the MoE aggregation-order determinism warning (2607.28097)** before sharding experts.
6. **Network/distributed:** relay text (not KV) unless the peer needs private hidden state (2608.04893); use Cross-Model KV Transfer when cascading model sizes; run TELLER-style tracing for debugging.

**Honest gap note:** within this batch there is no paper that directly implements llama.cpp-style CPU weight-streaming, mmap-from-disk weight loading, or a full multi-card tensor-parallel runtime. The closest actionable items are the pin-and-swap (SALT), cache-residency-aware expert offloading (AcceptMoE), split learning (GQ-FSL/Gecko), portable kernel generation (Meganeura/SparseDitto), and the KV-cache compression cluster — plus the determinism warning (2607.28097) that matters before any expert sharding.

---

# Part 3 — Mechanism Menu (domain-agnostic ideas, any paper in the batch)

Part 2 listed papers already framed for LLMs. This part is the **idea toolbox**: mechanisms from *any* paper in the 992-entry batch that could be lifted and re-targeted to your five goals — reducing VRAM, disk streaming, CPU inference, distributed inference, and parallel use of low-end cards. Nothing here requires the source to be an LLM paper.

Legend: 🧠 VRAM · 💾 disk/stream · ⚡ CPU · 🌐 network/distributed · 🎴 low-end parallel

---

## A. Weight storage as a "recipe," not a table

| Mechanism | Source | Idea | Targets |
|-----------|--------|------|---------|
| **Seed + latent regeneration** | [2608.00860](https://arxiv.org/abs/2608.00860) Kilobyte Models | Store only a random seed + small quantized latent; regenerate weights on device via a mapping network. Storage/transfer cost ≈ latent size, not parameter count. → Extreme disk streaming: the model file on disk is kilobytes; the device re-derives weights. | 💾🧠 |
| **Multi-precision from one checkpoint** | [2608.04048](https://arxiv.org/abs/2608.04048) Recurrent Residual Quantization | Weights = 2-bit base + sequence of 2-bit residuals → 2/4/6/8-bit all derivable from ONE stored file, calibration-free. One model on disk serves many VRAM budgets; adapt to device by how many residual layers you load. | 💾🧠⚡ |
| **Compress along 3 axes at once** | [2608.00859](https://arxiv.org/abs/2608.00859) SparseKAN | Learnable gates over basis functions × neurons × bit-width, hardened into dense tensors (not masks). Compression knob for arbitrary parameterized-function layers, incl. KAN-style and polynomial activations. | 🧠💾 |
| **Binary nets with restored expressivity** | [2608.01490](https://arxiv.org/abs/2608.01490) BiKAN | Fixes the "all even powers collapse to 1" failure of 1-bit nets by injecting Walsh parity terms computed with the same XNOR-popcount ops. Idea: don't just binarize — engineer a basis that survives binarization. | 🧠⚡ |
| **Gradient-free, tiny-footprint weight updates** | [2608.01624](https://arxiv.org/abs/2608.01624) Not the Dimension, the Norm | Full-weight random perturbation search is unnecessary; what matters is the *norm/subspace* of the perturbation. Enables adapting a model with a handful of trainable scalars — minimal VRAM for fine-tuning on weak cards. | 🧠🎴 |
| **Low-rank weight modeling without Adam's bias** | [2608.05136](https://arxiv.org/abs/2608.05136) The Loss Does Not See the Basis, but Adam Does | Which optimizers actually drive factored (low-rank) models to low-rank solutions (GD/Muon/Shampoo yes; Adam/RMSProp no). Guidance for choosing optimizers that yield compact factors on disk. | 🧠💾 |

## B. Computation that skips work (CPU / low-end friendly)

| Mechanism | Source | Idea | Targets |
|-----------|--------|------|---------|
| **Early-stop accumulations by sign certainty** | [2608.06177](https://arxiv.org/abs/2608.06177) Threshold-Based Early Stopping | In binary nets the running partial sum's final sign is often predictable early — stop adding. Trivial to port to any dot-product-heavy kernel (int8 SIMD, CPU): skip the tail of the accumulation once the partial sum is "decided." | ⚡🎴🧠 |
| **Early-exit / progressive execution** | [2608.01285](https://arxiv.org/abs/2608.01285) Stop When Memory Suffices | Router decides whether evidence already suffices → terminate early. Idea generalizes: per-request layer/step budgets for weak devices. | ⚡🎴💾 |
| **Dynamic width pruning per token** | [2607.28418](https://arxiv.org/abs/2607.28418) WIDE | Token-level dynamic width pruning, end-to-end differentiable — compute spent proportional to token difficulty, on a per-token basis. | 🧠⚡ |
| **Streaming input, progressive prediction** | [2608.00720](https://arxiv.org/abs/2608.00720) CascadeLUT | Instead of buffering full input, feed feature subsets in order and refine prediction as they arrive. Same trick inverts for generation: emit/score partial outputs while next chunks are still streaming from disk. | 💾⚡ |
| **Dual-sparse (weight + activation) kernels** | [2608.01536](https://arxiv.org/abs/2608.01536) Celty | Co-designed sparse format + SIMT core that skips zero-weight AND zero-activation work with run-length compressed columns. Direct CPU/GPU speedup + less memory traffic. | ⚡🧠🎴 |
| **Data-aware hashing instead of attention scan** | [2608.04405](https://arxiv.org/abs/2608.04405) BinaryPC | Binary PCA hash codes: nearest-KV lookup by hash instead of scanning all KV pairs. Training-free, data-aware — a pattern for fast lookup over large cached state on slow hardware. | 🧠⚡ |
| **Learn when attention should be sparse** | [2608.02938](https://arxiv.org/abs/2608.02938) LTGA | Learn a per-edge Tsallis entropy index interpolating heavy-tail↔softmax↔compact attention. If the model can learn to use compact attention on easy connections, compute collapses on the average case. | 🧠⚡ |
| **Skip quantization work for redundant params** | [2608.06291](https://arxiv.org/abs/2608.06291) BaKron | Kronecker-Hessian-informed adaptive rounding at GPTQ cost — better low-bit quality, i.e. same accuracy at fewer bits → less VRAM/disk. | 🧠💾 |

## C. Portable execution across whatever hardware exists

| Mechanism | Source | Idea | Targets |
|-----------|--------|------|---------|
| **Compile once, run on any GPU/iGPU/APU** | [2608.01563](https://arxiv.org/abs/2608.01563) Meganeura | Vulkan + Metal backend from one typed static graph; autodiff, optimizer, memory planner, runtime all portable. Runs NVIDIA/AMD/Intel/Apple incl. iGPUs and APUs. → Your "low-end cards" (old GPUs, iGPUs, APUs) all become first-class targets. | ⚡🎴 |
| **Generate the kernel per-hardware, not one universal kernel** | [2608.05033](https://arxiv.org/abs/2608.05033) SparseDitto | LLM-agent picks representation/execution strategy per (matrix, operator, target GPU). Closes a 350× gap between cuSPARSE formats. → Every weak card gets a kernel tuned to its own capabilities. | 🎴⚡🧠 |
| **Design-time energy/feasibility planning** | [2608.03589](https://arxiv.org/abs/2608.03589) Design-Time DNN for Intermittent Learning on MCUs | Hardware-aware energy predictor + multi-objective optimization to pick a DNN that fits intermittent, energy-budgeted devices *before* deployment. Idea: size the model to the sustained power of your weakest node, and checkpoint for arbitrary interruption. | ⚡💾🎴 |
| **Contraction-plan ranking before execution** | [2608.05819](https://arxiv.org/abs/2608.05819) Learn-to-Rank Tensor Contraction Plans | GPU performance of equivalent tensor-contraction plans varies wildly; learn-to-rank plans from structural features before running. Applicable to any tensor-op scheduling choice on heterogeneous hardware. | 🎴⚡ |

## D. Growing / elastic models (adapt to available resources over time)

| Mechanism | Source | Idea | Targets |
|-----------|--------|------|---------|
| **Grow hidden units to restore plasticity** | [2608.01475](https://arxiv.org/abs/2608.01475) Plasticity of Growing & Elastic Nets | Incrementally add randomly-initialized units preserves learning ability in online learning. Idea: models that can *grow* — start tiny on the weakest card, add capacity as compute becomes available. | 🎴💾🧠 |
| **Parameter-free routing / modular specialization** | [2608.04084](https://arxiv.org/abs/2608.04084) SpecDrop | Fixed category-conditioned routing beats learned routers at matched budgets. Idea: deterministic module-to-device assignment (no learned router to train/serve) for sharding across cards. | 🎴🌐 |
| **Superposition experts** | [2608.05303](https://arxiv.org/abs/2608.05303) EdgeXpert (idea) | Route by *prompt-level* expert reuse: one shared expert set per prompt instead of per-token — fewer distinct weights needed resident. | 🧠💾🎴 |

## E. Split / distributed mechanisms (any weak node participates)

| Mechanism | Source | Idea | Targets |
|-----------|--------|------|---------|
| **Split learning with asymmetric precision per side** | [2607.29659](https://arxiv.org/abs/2607.29659) GQ-FSL | Cut the model: weak node runs low-precision shallow layers, server runs the rest, cut-layer data quantized for the network. Joint optimization of split point + bit-widths. → Direct blueprint for "distributed inference over a network." | 🌐🎴⚡ |
| **Offload the heavy public encoder** | [2608.02378](https://arxiv.org/abs/2608.02378) Gecko | Run a public encoder outside the trust/protection boundary; only the small private tail is evaluated expensively. Same pattern with perf: run the big general encoder on the server, tiny task-tail on the device. | 🌐🎴 |
| **Heterogeneous clients stay self-contained** | [2607.29071](https://arxiv.org/abs/2607.29071) FedSLM | SVD-decompose each client's model so low-rank subspaces are aggregatable yet each node remains standalone. Idea: shard by SVD subspace — nodes need only their rank slice, not the full matrix. | 🌐🎴🧠 |
| **Sparse incremental aggregation over a ring** | [2608.03436](https://arxiv.org/abs/2608.03436) FedRings | Ring-topology, link-aware scheduling, adaptive sparse incremental aggregation + historical compensation for interruptions. → Distributed inference on intermittent/mobile links (LEO sats, ad-hoc LAN). | 🌐 |
| **Reuse cached updates, send only deltas** | [2608.05358](https://arxiv.org/abs/2608.05358) DG-FedReuse | Gate stale cached updates with a gradient-discrepancy proxy; fresh updates use per-tensor Top-K. 83–85% uplink savings at ~0 accuracy cost. → For a fleet of weak nodes, don't retransmit what didn't change. | 🌐💾 |
| **Communication protocol = the bottleneck, design for it** | [2608.05327](https://arxiv.org/abs/2608.05327) Collaborative Communication via Coarsening | Provably: a short high-utility protocol exists if communication complexity is small — and designing it can be NP-hard. Implication: for distributed inference, *minimize bits per node* (quantize/coarsen the messages), don't just scale the pipe. | 🌐 |
| **Freshness-aware scheduling** | [2608.01128](https://arxiv.org/abs/2608.01128) MA-HEAD-Net | Age-of-Information-optimal scheduling with mini-slot embedding + adaptive checkpoints for heterogeneous traffic. → When nodes produce partial results over a network, schedule which result to wait for vs discard by its "age." | 🌐⚡ |
| **Collaborative MEC scheduling for LLM subtasks** | [2608.02031](https://arxiv.org/abs/2608.02031) Collaborative MEC | Transformer-enhanced PPO splits LLM inference subtasks across edge servers under soft deadlines. → The scheduler layer for a distributed inference pool. | 🌐🎴 |
| **Relay hidden state only when it's needed** | [2608.04893](https://arxiv.org/abs/2608.04893) When Does Latent Communication Pay? | KV-cache relay only pays when the receiver needs the sender's private info (100% vs 23–25%). → Distributed nodes: send text unless the peer genuinely needs your hidden state. | 🌐💾 |

## F. Low-end parallelism (many weak cards as one)

| Mechanism | Source | Idea | Targets |
|-----------|--------|------|---------|
| **Per-GPU specialization via generated kernels** | [2608.05033](https://arxiv.org/abs/2608.05033) SparseDitto (see C) | Each node compiles kernels matched to its own GPU → a heterogenous cluster behaves better than its weakest-link uniform baseline. | 🎴 |
| **Workload distribution across cards** | [2608.03695](https://arxiv.org/abs/2608.03695) cuGraph multi-GPU | Dask-based distribution of graph clustering across multiple GPUs (~1000× vs CPU). The Dask-style scatter/gather pattern ports to model serving across cards. | 🎴🌐 |
| **Growing architecture on each card** | [2608.01475](https://arxiv.org/abs/2608.01475) Growing nets (see D) | Each weak card can run a *smaller slice* that grows; ensemble of growing slices ≈ one big model over time. | 🎴💾 |
| **Determinism guardrail before sharding** | [2607.28097](https://arxiv.org/abs/2607.28097) Expert-Reduction Divergence | Any cross-card MoE/expert sharding MUST fix aggregation order; equivalent orders yield different outputs. Test per-machine reproducibility first. | 🎴🌐 |

## G. Memory & bandwidth, the shared theme

| Mechanism | Source | Idea | Targets |
|-----------|--------|------|---------|
| **Query-aware bit allocation** | [2608.04074](https://arxiv.org/abs/2608.04074) KV VQ | Spend bits where the query looks — transform derived from a distortion criterion, not a fixed rotation. Pattern: allocate precision by what the consumer actually reads. | 🧠💾 |
| **Anchor-residual decomposition** | [2608.02901](https://arxiv.org/abs/2608.02901) AnchorKV | Keep a small exact "anchor" set; express the rest as residuals to their nearest anchor; refine only the impactful ones. Generalizes from KV cache to *any* large tensor/state that must stay resident. | 🧠💾 |
| **Cache-residency-aware eviction/paging** | [2608.02989](https://arxiv.org/abs/2608.02989) AcceptMoE | Decide what stays resident by expected future commitment (offline-estimated), not just current score — the right policy for disk-paged weights. | 💾🎴 |
| **Restore/regenerate rather than keep everything** | [2608.01247](https://arxiv.org/abs/2608.01247) RestoreKV | One cheap adapted pass reconstructs full-cache behavior from a tiny cache. Idea: store a compact "regenerator," derive the big state on demand. | 💾🧠 |
| **Recurrent state instead of growing cache** | [2608.02032](https://arxiv.org/abs/2608.02032) DART | Fixed-size recurrent state + attention-style decoding from it → long context without a growing KV buffer. | 🧠💾 |
| **Query-efficient sparse sampling** | [2607.28047](https://arxiv.org/abs/2607.28047) Query-Efficient Volume Rendering | Sparse query strategy for implicit volumes when each query is a neural inference. Idea: when every read from "memory" is expensive (disk-backed models), read as few entries as possible with adaptive sampling. | 💾⚡ |
| **Lightweight parameter-efficient representations** | [2608.03041](https://arxiv.org/abs/2608.03041) PLAN | Parallelized liquid-neural dynamics: discretized, parallel form of a sequential recurrent model — keeps the parameter efficiency of liquid/SSM-style models while removing the sequential bottleneck. | 🧠⚡ |

---

## Summary: which mechanism to steal for which goal

| Your goal | Steal this mechanism (best matches) |
|-----------|-------------------------------------|
| **Reduce VRAM** | RRQ one-checkpoint-multi-precision (2608.04048), AnchorKV anchor-residual (2608.02901), KV VQ bit allocation (2608.04074), SparseKAN 3-axis compression (2608.00859), BaKron curvature-aware bits (2608.06291) |
| **Stream model from disk** | Seed+latent regeneration (2608.00860), AcceptMoE residency-aware paging (2608.02989), RestoreKV regenerate-on-demand (2608.01247), Kilobyte-broadcast to N nodes |
| **CPU inference** | Early-stop accumulations (2608.06177), Meganeura Vulkan/Metal portability (2608.01563), WIDE token-level dynamic width (2607.28418), CascadeLUT streaming-input (2608.00720), PLAN parallel liquid nets (2608.03041) |
| **Distributed over network** | GQ-FSL split learning + asymmetric precision (2607.29659), FedRings ring aggregation (2608.03436), DG-FedReuse delta-only updates (2608.05358), MEC-PPO scheduler (2608.02031), Gecko encoder-offload split (2608.02378) |
| **Parallel low-end cards** | SparseDitto per-GPU kernels (2608.05033), Meganeura cross-vendor runtime (2608.01563), FedSLM SVD-subspace sharding (2607.29071), SpecDrop parameter-free module routing (2608.04084), growing nets (2608.01475) — with the determinism guardrail (2607.28097) |

**The meta-idea:** the batch's strongest reusable insight isn't any single paper — it's that the five goals are all the same problem seen from five sides: **"make the working set smaller than the available memory, and only ever pay for what the current query needs."** Seed-regeneration and RRQ solve *disk size*; anchor-residual and residency-aware paging solve *resident size*; dynamic width, early-stop, and CascadeLUT solve *compute spent*; split learning and FedRings solve *what must travel*; SparseDitto and Meganeura solve *where it can run*.

---

# Part 4 — Vulkan & Cross-Platform GPU ML/LLM Inference

Online search results on arXiv for Vulkan-based LLM, neural network execution, and GPU isolation:

| # | arXiv | Title | Focus & Key Insights |
|---|-------|-------|----------------------|
| 1 | [2608.01563](https://arxiv.org/abs/2608.01563) | Meganeura: Portable GPU Training and Inference through Vulkan and Metal | Native compiler + runtime lowering compute graphs to **Vulkan & Metal** across NVIDIA, AMD discrete, AMD APU, Apple Silicon, and Intel iGPU. Outperforms PyTorch native on 12/20 minimal-latency cells in f32. |
| 2 | [2604.02344](https://arxiv.org/abs/2604.02344) | Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors | Benchmarks dispatch & validation overhead of cross-platform WebGPU/Vulkan for LLM inference across 4 GPU vendors (NVIDIA, AMD, Intel, Apple). Analyzes small-batch autoregressive LLM decoding bottlenecks in Vulkan/Metal/D3D12. |
| 3 | [2605.01352](https://arxiv.org/abs/2605.01352) | VUDA: Breaking CUDA-Vulkan Isolation for Spatial Sharing of Compute and Graphics on the Same GPU | Zero-copy spatial memory sharing between CUDA (AI compute) and Vulkan (rendering) on a single GPU for embodied AI workloads. |
| 4 | [2605.00219](https://arxiv.org/abs/2605.00219) | VkSplat: High-Performance 3DGS Training in Vulkan Compute | Cross-vendor high-performance compute pipeline in pure Vulkan Compute. |

---

# Part 5 — CPU LLM Execution, Hybrid Offloading & Disk-Streamed MoE

Online search results on arXiv for CPU LLM inference, SIMD kernel co-design, hybrid CPU-GPU offloading, and disk-streamed MoE:

| # | arXiv | Title | Focus & Key Insights |
|---|-------|-------|----------------------|
| 1 | [2607.14618](https://arxiv.org/abs/2607.14618) | PolyQ: Codesigning End-to-End Quantization Framework for Scalable Edge CPU LLM Inference | Channel-wise bit allocation ({2,3,4,8,16}-bit) + compile-time channel clustering into bit-homogeneous blocks. Generates SIMD & LUT CPU kernels and merges activation reordering into compile time (reduces reorder traffic by 70.8%). Proportional speedup on workstation, laptop, and mobile CPUs. |
| 2 | [2607.10183](https://arxiv.org/abs/2607.10183) | ATSInfer: Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices | **Tensor-granularity** (not layer/expert) offload scheduler for hybrid CPU-GPU LLM inference on laptops/desktops when VRAM is tight. Async CPU-GPU coordination delivers 1.94× prefill and 3.29× decode throughput improvement over coarse offload. |
| 3 | [2608.08910](https://arxiv.org/abs/2608.08910) | Tied Trit-Planes: Persistent Folded Format for Disk-Streamed MoE Serving | 1.6-bit (balanced-ternary 9-level) weight quantization format with a folded byte layout designed specifically for **disk-streamed MoE serving** directly from NVMe/SSD. |
| 4 | [2607.14622](https://arxiv.org/abs/2607.14622) | ExaGEMM: Exploration Framework for CPU-Driven ML Inference via Associative In-Register Computing for Low-Bit GEMM | CPU-driven framework for low-bit (1/2/4-bit) GEMM using associative in-register computing on conventional CPUs. |
| 5 | [2607.17415](https://arxiv.org/abs/2607.17415) | Transition-Aware Backend Dispatch for Edge LLM Inference | Dynamic operator dispatch across edge CPU, GPU, and ONNX Runtime CPU backends accounting for shape transitions → 17.4% latency and 14.4% energy cuts on Jetson. |
| 6 | [2607.12839](https://arxiv.org/abs/2607.12839) | HeteroMosaic: Energy-Efficient Edge LLM Inference | Fine-grained scheduler mapping transformer sub-ops across CPU, iGPU, and NPU on edge SoCs. |

---

# Part 6 — GGUF Format, Speculative Decoding & Safety under Quantization

Online search results on arXiv for GGUF model execution, local llama.cpp/Ollama benchmarks, and safety decoupling under low-bit GGUF:

| # | arXiv | Title | Focus & Key Insights |
|---|-------|-------|----------------------|
| 1 | [2607.03876](https://arxiv.org/abs/2607.03876) | AdaptiveSD: A Stability-Aware, Runtime-Adaptive Speculative Decoding Framework for CPU-Constrained LLM Inference | Tailored specifically for **GGUF-quantized models** on CPU-constrained devices. Uses an 11-rule policy hierarchy, dynamic heuristic/RL draft controller, and INT8 shadow buffer KV cache coordination layer to prevent resource exhaustion and bandwidth bottlenecks on pure CPU execution. |
| 2 | [2606.10154](https://arxiv.org/abs/2606.10154) | Quality Is Not a Safety Proxy Under Quantization | Audits **7-level GGUF quantization ladders** (Q2_K through Q8_0) across 6 LLM families. Discovers benchmark quality and refusal/safety behavior decouple under low-bit GGUF (quality stays stable while refusal rate drops 12–68 percentage points). Proposes the Refusal Template Stability Index (RTSI) to catch safety drift. |
| 3 | [2607.14181](https://arxiv.org/abs/2607.14181) | Quantize with Confidence? An Empirical Study of Quantization for Code Generation | Evaluates GGUF (in llama.cpp / Ollama) alongside AWQ, GPTQ, QuIP#, and AQLM for local code models (Qwen2.5-Coder & CodeLlama) across Python/Java benchmarks. |
| 4 | [2607.10137](https://arxiv.org/abs/2607.10137) | RDQ: Residual Distribution Quantization for Large Language Models | Fixes residual stream distributional drift in sub-4-bit GGUF PTQ, correcting accumulated layer noise to preserve sub-4-bit model accuracy. |
| 5 | [2606.12280](https://arxiv.org/abs/2606.12280) | Holding the FP8 Quality Ceiling at 8-Bit Weights and Activations | Applies GGUF INT8 post-training quantization to a 9.3B flow-matching transformer for consumer GPU deployment. |

---

# Part 7 — Advanced LLM Offloading & Scheduling Infrastructure

Online search results on arXiv for KV-cache offloading, host-to-device prefetching, SLO-aware scheduling, and tensor-granularity offload:

| # | arXiv | Title | Focus & Key Insights |
|---|-------|-------|----------------------|
| 1 | [2607.26475](https://arxiv.org/abs/2607.26475) | DualDecoder: Accelerate Long Context LLM Inference by Predictive Prefetch | Eliminates the GPU memory wall by **predictive prefetching** of critical KV entries from host DRAM over PCIe, overlapping host-to-device transfers with previous token decoding. Increases decoding throughput by up to 2.62×. |
| 2 | [2608.06557](https://arxiv.org/abs/2608.06557) | Cascade: Exploiting SLO-Aware Latency Budget for Fair and High Goodput LLM Inference Serving | Continuously updates per-request "latency headroom budget" to jointly govern queue scheduling and multi-tier KV-cache management (retire, restore, prefetch, or recompute). Cuts SLO violations by 40% and boosts goodput by 2.4× over vLLM FCFS. |
| 3 | [2607.10183](https://arxiv.org/abs/2607.10183) | ATSInfer: Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices | **Tensor-granularity** (not layer/expert) offload scheduler for hybrid CPU-GPU LLM inference on laptops/desktops when VRAM is tight. Async CPU-GPU coordination delivers 1.94× prefill and 3.29× decode throughput improvement over coarse offload. |
| 4 | [2607.20481](https://arxiv.org/abs/2607.20481) | Routing Without Training: Controllable-Ratio LLM Offloading via Reliability Gating | Training-free edge-cloud offload router using reliability gating to route hard queries to cloud while processing confident queries locally. |
| 5 | [2608.05926](https://arxiv.org/abs/2608.05926) | BALANCE: Hybrid Autoregressive-Speculative LLM Inference in Wireless Edge Networks | Jointly schedules autoregressive decoding and speculative draft verification offloading across edge nodes. |
| 6 | [2608.06135](https://arxiv.org/abs/2608.06135) | LLM Inference Under Bursty Workload Distribution: Modifying the WAIT Algorithm | Online request-intensity estimation modifying the WAIT scheduling algorithm for bursty LLM traffic. |

---

# Part 8 — Latest System Breakthroughs (Upstream SGLang & vLLM Innovations)

Newly released papers (Aug 8–10 2026):

| # | arXiv | Title | Focus & Key Insights |
|---|-------|-------|----------------------|
| 1 | [2608.07009](https://arxiv.org/abs/2608.07009) | HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management | **Merged into upstream SGLang.** Bounds GPU HBM footprint with a small fixed-size GPU cache while keeping full KV history in host RAM. Fused CUDA graph kernels resolve hit detection, LRU replacement, and layer-wise prefetching. Boosts long-context peak throughput by up to **4.7×** on H200/B200/GH200 platforms. |
| 2 | [2608.08097](https://arxiv.org/abs/2608.08097) | OasisKV: Scaling In-Decode KV Cache Beyond HBM with Lookahead Sparse Prefetching | **Implemented on vLLM.** Uses speculative decoding lookahead tokens to predict future important KV blocks in advance and prefetches them into HBM from host/remote memory during decoding. Delivers **1.69× to 2.1×** throughput gains over dense vLLM with 6.5–9.7× less KV memory footprint. |
| 3 | [2608.09291](https://arxiv.org/abs/2608.09291) | UnionSparse: An Index-Efficient Sparsity Framework for Low-Bit Sparse LLM Inference on Edge | [Open Source Code](https://github.com/Victor-Alen/UnionSparse). Solves SpMM index bottleneck for W4A4 sparse LLMs on edge GPUs using Index-Efficient Bitmap Encoding & Low-Bit Shared-Memory Parallel Decoding. Outperforms FlashLLM/SpInfer by **1.43–2.30×** and Tensor Cores by **1.56–3.46×**. |
| 4 | [2608.09888](https://arxiv.org/abs/2608.09888) | BDH-CQ: In-Context Learning with Recurrent Latent Reasoning | 150M-parameter model performing non-verbalized recurrent latent reasoning in high-dimensional state space, cutting inference cost to $0.0007 per task while scoring 29.5% pass@2 on ARC-AGI-1. |
| 5 | [2608.08340](https://arxiv.org/abs/2608.08340) | OpRAG: A Resource-Deterministic Runtime for GPU-Backed Multi-Stage RAG Workflows | Resource-deterministic execution runtime for multi-stage agentic RAG workflows combining embedding, vector search, and LLM decoding. |





---

# Part 9 — Latest LLM Articles (Oct 6–14, 2026)

**Source:** arXiv fresh submissions, Oct 6–14 2026 — 3,591 unique papers pulled from cs.CL / cs.LG / cs.AI / cs.DC / cs.AR / cs.SE, filtered to LLM-relevant work (122 kept, 91 scoring ≥64), and de-duplicated against every arXiv ID already cited anywhere in this repo. Raw ranked set: `data/arxiv_latest_20261015.json`.

**Windowing:** this extends the Sep–Oct layer in `ENGINE_MASTER_SPEC.md` §12 (which covers ≈7 Oct and earlier) with the newest arrivals. Numbers are author-reported, not independently verified, and several are v1 preprints.

**Headline shifts this window:** (1) cross-model / cross-agent KV reuse graduated from "linear map + caveat" to a real serving paradigm — seven papers, several showing that reconstruction fidelity is the *wrong* objective; (2) KV compression moved to behavior-aligned and query-adaptive objectives, plus codecs designed for *persistent* caches; (3) agentic RL/serving became the dominant workload assumption (PD disaggregation, token-level routing, agent flow decomposition); (4) negative results got louder — an edge audit says no tested KV compressor beats FP16 on energy/token, and a multi-agent memory tier showed no detectable accuracy gain.

## 19. KV Cache — Compression, Quantization & Eviction

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2610.18642](https://arxiv.org/abs/2610.18642) | MultiAlloc: Budget-Aware Precision Allocation for KV Cache Quantization | Profiling-guided allocation of a fixed KV memory budget **between** layer-wise precision and token-wise preservation, solved offline so there is no online allocation search. Beats layer-wise, token-wise and multi-granularity KV-quant baselines at matched budgets on Mistral/Llama/DeepSeek-Qwen/QwQ. |
| 2 | [2610.15384](https://arxiv.org/abs/2610.15384) | E8 Lattice Key Quantization for Low-Bit KV-Cache Compression | Vector-quantizing keys on the **E8 lattice** makes uniform **k2v2** viable: ~6× KV compression (≈20% smaller than scalar k3v2), near-baseline long-context perplexity, and a table-based asymmetric scoring path that keeps decode throughput at scalar parity across 5 model families. |
| 3 | [2610.10381](https://arxiv.org/abs/2610.10381) | ResidualQuant: KV Cache Quantization for Looped Transformers with 2-Bit Residuals | For looped/recurrent-depth transformers: take final-loop KV as reference and code the other loops as 2-bit residuals (+ least-squares scaling/rotations, loop-wise mixed precision). −80.7% theoretical KV storage, up to +13.0% accuracy over rotation baselines at equal budget; RTX 5090: 2.73× fixed-batch decode, 4.15× peak throughput. |
| 4 | [2610.17966](https://arxiv.org/abs/2610.17966) | BELT: BEhavior-Aligned Low-Rank KV Cache Compression with Learned Top-k Rank Allocation | Low-rank KV compression whose key-side objective targets **attention distributions** (not keys or pre-softmax logits), with a closed-form solution and a learned top-k rank allocator under a user budget. 70% compression with up to +23.3 points on LongBench; 17.8× total KV reduction with quantization; Triton kernels give 2.7× attention / 3.3× end-to-end decode. |
| 5 | [2610.18026](https://arxiv.org/abs/2610.18026) | ChampionKV: Head-Wise Adaptive Budget Allocation for KV Cache Compression | Input-adaptive **per-head** budget allocator that plugs into SnapKV/PyramidKV without changing their token selection. Largest and statistically significant gains on Llama-3.1-70B at tight budgets; honest caveat — does not uniformly beat Ada-KV's global cross-head Top-K. |
| 6 | [2610.11245](https://arxiv.org/abs/2610.11245) | Read What Matters: Query-Adaptive Quantization for KV Caches | Splits KV budgeting into **bits retained vs bits fetched per query**: entries are stored in a progressive code, and each query buys key-channel prefixes (then value-token prefixes) from the attention it just computed. Reading 4 bits average from an 8-bit cache costs ≤0.66% C4 perplexity at ~¼ the logical reads; 39% lower latency than a TurboQuant codec. |
| 7 | [2610.17970](https://arxiv.org/abs/2610.17970) | AlignPred: Error-Controlled KV Cache Compression via Alignment and Predictive Coding | Codec for **long-lived agent caches** under a configurable reconstruction-error budget: local orthogonal alignment across heads/layers + predictive coding + lattice-quantized, entropy-coded residuals (no learned predictors or mode search). Its SGLang HiCache integration cuts persistent target-KV storage by 42–72% at comparable latency (128K). |
| 8 | [2610.14754](https://arxiv.org/abs/2610.14754) | GateSight: Recurrent Memory Anticipates Attention Memory for Query-Agnostic KV Eviction in Hybrid LLMs | In hybrid (Gated DeltaNet + attention) models the GDN **write gate, decay and write magnitude during prefill already anticipate future KV utility** — before any query exists. Per-head MLPs predict utility, the native attention output gate sets each layer's keep-budget. With 1/10 the KV it is the most accurate of five query-agnostic methods on 3/4 model–benchmark pairs, leading KVzip by 9.5–47.2 points, at ~prefill-level scoring overhead. |
| 9 | [2610.14475](https://arxiv.org/abs/2610.14475) | EdgeLLM-Nano: Auditing Polar-Guided Key-Cache Compression and Consistency-Anchored Pruning on a 33.6 MB On-Device Language Model | Negative-result audit on a 33.6 MB LM: a polar-guided key-cache quantizer is far worse than plain round-to-nearest, consistency-anchored progressive pruning is no better than one-shot magnitude pruning, and **every tested compressor is less energy-efficient per token than uncompressed FP16 on a Raspberry Pi 5** (KIVI closest to lossless). Contributes Long-Context Decode Fidelity, which separates backends that perplexity scores identically (5.461). |
| 10 | [2610.11214](https://arxiv.org/abs/2610.11214) | Bridging KV-Cache Quantization and Linear Attention: From Theory to Pretrained Weight Migration | Shows RAM-Net (soft assignments over a discrete address space) is the bridge between per-KV discrete compression and multi-KV recurrent aggregation, proving a separable read–write overlap locally approximates full attention; then migrates 9 pretrained transformers (0.3–7B) with 500M tokens, recovering 87.1% of teacher accuracy gains. |
| 11 | [2610.12514](https://arxiv.org/abs/2610.12514) | CORAL-KV: Compact Cross-Layer Prediction and Residual Coding for Autoregressive Video KV Caches | Exploits cross-layer KV dependencies in autoregressive **video** models: predict a layer's keys/values from earlier layers with block-diagonal + low-rank predictors, then quantize the residuals. 37–48% less cache than QVG-4 while improving PSNR, and it targets the error-compounding that hurts clustered video caches. |
| 12 | [2610.15721](https://arxiv.org/abs/2610.15721) | To Cache or Not to Cache: Understanding and Mitigating Safety Incompatibility in Cross-Model KV Cache Reuse | Safety audit of cross-model KV reuse: deviation from the receiver's native safety behavior is pair- and config-dependent, and **value reuse is far worse than key reuse**. VReparo learns per-layer/per-head affine maps on sender values offline: HarmBench ASR 57.28%→39.21%, StrongREJECT 0.4215→0.2163, while keeping 85.62% mean TTFT reduction. |

## 20. Cross-Model / Cross-Agent KV Reuse (zero-prefill handoff) — the breakout cluster

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2610.15095](https://arxiv.org/abs/2610.15095) | Swarm Telepathy: Prefill-Free KV-Cache Transfer Across Families | First **cross-family** prefill-free KV transfer for multi-agent systems: both models stay frozen, <50M tokens and <1 GPU-day of training, and it beats text-to-text communication by a mean 40% accuracy and 4× wall-clock across 17 benchmarks / 12 LLMs / 6 families. Can even pass a VLM sender's vision ability to a text-only receiver. |
| 2 | [2610.15509](https://arxiv.org/abs/2610.15509) | CacheLink: Functional KV Transport for Zero-Prefill Cross-Model Context Handoff | Demonstrates that **KV reconstruction fidelity is a poor proxy for receiver utility** (equal K/V error → different attention, distributions and accuracy), and instead learns a coupled K/V correction against receiver behavior. Best equal-weight mean on every one of 4 sender–receiver pairs; ≤7.27× lower handoff latency vs re-prefill at 32K while moving K/V MSE <0.4%. |
| 3 | [2610.11358](https://arxiv.org/abs/2610.11358) | RaReCache: Bridging the Gap in Cross-Model KV Cache Reuse via Rank disagreement-based Selective Recomputation | Transfer failures concentrate on a few information-dense tokens; a **rank-disagreement** score finds them. On a 23× size gap (Qwen3-0.6B→14B) recomputing 30% of positions retains 95–99% of target accuracy; on 8.8× (Llama3-8B→70B) 40% retains 96.5%. 3.04× prefill speedup; −5.0× median / −6.4× P99 TTFT at the target's saturation load. Small models can prefill on behalf of huge ones. |
| 4 | [2610.14688](https://arxiv.org/abs/2610.14688) | Accelerating Cross-Model LLM Inference via Attention Map Reuse | Reuses **attention maps** rather than KV states for heterogeneous serving (maps capture token-dependency structure and transfer better), combining partial recomputation with lightweight per-layer/head linear translation, block-level sparse compression and a fused retrieval kernel. Code: PKU-DAIR/MapFlow. |
| 5 | [2610.15186](https://arxiv.org/abs/2610.15186) | SAFE-KV: Causal Suffix Repair for Behavior-Preserving Non-Prefix KV Cache Reuse in RAG | Formalizes non-prefix reuse as behavior preservation: a cached document is a **context-conditioned causal state**, so prefix-conditioned composition keeps only 47.2% (HotpotQA-500) / 37.8% (2Wiki-500) of full-prefill answers, while causal-suffix repair restores 99.8–100%. Sober caveat: faithful repair can re-encode most of the context, so its operating rule fell back to full prefill in their single-GPU study. |
| 6 | [2610.18014](https://arxiv.org/abs/2610.18014) | LatentRemedy: Efficient Remedy For Non-Prefix KV Cache Reuse in Multi-Agent Systems | Reusing non-prefix KV across agents with **conflicting tasks** can drop performance up to 90%; a few task-conditioned latent tokens (MLP map of the target prefix's last hidden state, trained by full-prefill distillation) let the target agent interpret the shared block without recomputing it, keeping TTFT near direct reuse. |
| 7 | [2610.16165](https://arxiv.org/abs/2610.16165) | Think Before You Speak: Enabling Training-Free Latent Communication Between Language Models | Training-free latent communication: linear interpolation of an upstream model's KV cache into a downstream model already beats the standalone model (up to +8.6 points) and gives up to 12.2× TTFT speedup vs text-based multi-model communication — no per-pair fusion adapters to train. |

## 21. Sparse & Efficient Attention

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2610.10871](https://arxiv.org/abs/2610.10871) | Sparse Attention Is Matrix Approximation, Not Choosing from a Bag of Values | Argues current sparse attention is mathematically wrong — it treats the attention matrix as a bag of values instead of a structured matrix used in a product. MASA scores each sparse unit by its reduction of matrix-product approximation error and drops into existing sparse kernels/budgets unchanged. |
| 2 | [2610.09025](https://arxiv.org/abs/2610.09025) | SPIN: Shadow Predictive Indexer for Sparse Attention | Indexer-based sparse attention still rescans the whole KV cache every decode step; SPIN uses lightweight history-based prediction to pick KV blocks instead. 30–40% sparsity at preserved quality; in vLLM up to +14.9% output throughput and −13.2% median inter-token latency. |
| 3 | [2610.11134](https://arxiv.org/abs/2610.11134) | QUILT: Rethinking Sparse-Attention Prefill through Shared Query Execution | Neighboring sparse queries pick overlapping KV, so QUILT executes them jointly with Shift-and-Compare Set Decomposition, cascaded sharing and tile-aware execution. −55.1% kernel latency, −55.9% KV data processed, −36.8% TTFT on GLM-5.3 / DeepSeek-3.2. |
| 4 | [2610.11201](https://arxiv.org/abs/2610.11201) | PageWeaver: KV-Guided Query Unions for Sparse Attention | Groups queries by **selected-page affinity** to share page loads and keep Tensor Core tiles full, preserving each query's support and output ownership. H200 Union8: 1.70× geometric-mean complete-call speedup over the tested FlashInfer path; whole-model prefill +7.88–14.36%; gains shrink or reverse where preparation cost dominates. |
| 5 | [2610.13735](https://arxiv.org/abs/2610.13735) | Orthros: Phase-Aware Heterogeneous Attention for Efficient Transformers | One module that switches computation pattern by **phase**: linear-complexity prefill, full-attention decode — lower asymptotic complexity for prefill-heavy/light-decode long-context work. 21.9× prefill speedup with near-perfect 128K needle-in-a-haystack, plus a data-efficient Transformer→Orthros conversion. |
| 6 | [2610.17959](https://arxiv.org/abs/2610.17959) | Virtual Head Attention | Attacks attention's "impossible triangle" with two **linear** ops — Q Premix (recovers virtual head diversity from halved physical heads) and Linear Postmix — both of which fold into existing query/output projections, leaving a standard GQA-2 model that reuses every GQA inference optimization. Beats GQA at all four scales; on a 30B-A3B MoE it beats MLA by +0.70% and GQA-4 by +2.60% on a 12-benchmark suite while using two rather than four KV heads. |
| 7 | [2610.09186](https://arxiv.org/abs/2610.09186) | The Dichotomy Between Pattern Recognition and Step-by-Step Reasoning | Models reasoning traces as paths on a De Bruijn graph: a task's traces form a DAG whose edge count is vanishingly small, and samples needed follow a power law in edges. Real data has this structure — Qwen3-14B/32B keep >75% of GSM8K/MATH-500/GPQA accuracy with attention restricted to a window under 15% of the trace length. |

## 22. Quantization (weights / activations / low-bit recovery)

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2610.14934](https://arxiv.org/abs/2610.14934) | FidelityQ: Preserving Predictive Fidelity in Ultra-Low-Bit Language Model Quantization | Analytic coding toward a centered symmetric source + mirrored four-level minimum-distortion quantizer with lookup-free decoding, plus Knowledge-Intensive Calibration that spends the budget on high teacher–student divergence tails. LLaMA-2-7B: 6.62 WikiText2 perplexity at W2A16; prior W2A4KV4 SOTA 8.31→6.96; 8.0× weight memory reduction vs FP16. |
| 2 | [2610.13767](https://arxiv.org/abs/2610.13767) | DualQuant: Dual-scale Calibration-free Weight-and-Activation Quantization for LLMs | Shows that minimizing weight-reconstruction MSE also produces effective **per-channel activation rescalings** — so W4A4 needs no calibration data. First dual-scaling method to formally support MX and NVFP4 block formats; matches SmoothQuant; real packed W4A4 path on RTX 4090 gives up to 2.61× memory reduction and 2.46× throughput over BF16. |
| 3 | [2610.16368](https://arxiv.org/abs/2610.16368) | LoopQuant: Accurate and Efficient Post-Training Quantization for Looped Transformers | Loop-aware PTQ: cross-loop distribution canonicalization, loop-wise precision allocation under a fixed bit budget, and loop-conditioned task calibration. Ouro-2.6B W4A4 beats all W4A8 baselines; −65.3% model memory and 1.7× decode speedup over BF16. |
| 4 | [2610.15092](https://arxiv.org/abs/2610.15092) | VLCQ: Post-training quantization for deep neural networks using variable length coding | Relaxes the fixed-point constraint of PTQ to **variable-length coding** with no data or fine-tuning. Per-channel it exceeds FP32 and PWLQ on most models (up to 6× vs FP32, 1.7× vs PWLQ) and up to 12.25× compression with <4% degradation. |
| 5 | [2610.17531](https://arxiv.org/abs/2610.17531) | Towards Green AI: Evaluating Energy Efficiency for Quantized LLM Inference | First unified energy study of 10 PTQ strategies × 5 models × 4 GPUs (H100/A6000/A100/L4): quantization peaks in energy efficiency at **long input / short output**, gains grow only modestly with batch size, and fused-kernel int8 (EETQ, Bitsandbytes) saves up to 4× vs FP32 at negligible accuracy loss. Energy tracks runtime, so latency work is a usable carbon proxy. |
| 6 | [2610.09346](https://arxiv.org/abs/2610.09346) | OnlineQAT: On-Policy Distillation for Ultra-Low-Bit Large Language Models | Two-stage recovery: block-wise QAT for a usable low-bit init, then on-policy distillation on **student-generated prefixes** with a frozen FP teacher (sampled reverse-KL). Qwen3-1.7B: 57.28 avg at W3A16 and 32.52 at W2A16 (beats ReasoningQAT by 2.90 / 0.44). Core insight: quantization error pushes the model into states absent from offline recovery data. |
| 7 | [2610.09227](https://arxiv.org/abs/2610.09227) | Quantize by Drift: Label-Free Mixed-Precision Post-Training Quantization for Text Embedders | Uses quantization-induced **output-representation drift** as a label-free sensitivity signal for mixed-precision PTQ of retrievers (macro Spearman 0.911 against held-out retrieval quality). Pre-registered: drift is a robust *coarse* signal, not a universally optimal objective — at the main budget it trails same-budget uniform precision on all three sealed embedders. |

## 23. Speculative Decoding

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2610.13708](https://arxiv.org/abs/2610.13708) | Adaptive DHiSparse: Speculative Decoding over a Load-Adaptive KV Hierarchy | Brings draft-model speculative decoding to hierarchical sparse KV (HiSparse successor): DSpark/DFlash2 rules untouched, draft KV in a window-bounded ring, verification staging reused across layer groups, residency adapted to load. GLM-5.3 at 100K/conc 128: 793 output tok/s per decode GPU (1.45× HiSparse, 1.27× non-speculative adaptive); up to 4× concurrency at matched latency; host fetches drop 14.2%→1.4% at low load. |
| 2 | [2610.12570](https://arxiv.org/abs/2610.12570) | Contextual Correction of Parallel Drafts for Speculative Decoding | Diffusion drafters predict a block in parallel and therefore lose **intra-block** dependencies; DCOCO keeps multiple candidates per position and scores their contextual compatibility across neighboring positions, still in parallel. Highest average throughput in all 4 verifier backbones × 10 benchmarks (up to 6.6× over vanilla), and +6–23% over the strongest competitor at 4–32 concurrency. |
| 3 | [2610.13202](https://arxiv.org/abs/2610.13202) | S3R: Fast and Faithful RL Rollouts via Self-Sparse Speculative Decoding | For RLVR rollouts: sparse attention proposes tokens, then **full-attention verification with exact speculative sampling** preserves the dense-policy response groups that group-relative advantages depend on (proved not recoverable by response reweighting). 1.39×/1.35× training throughput vs autoregressive rollouts on Qwen3; a KV-aware scheduler bounds offload thrash. |
| 4 | [2610.15417](https://arxiv.org/abs/2610.15417) | SpecFit: Not All Draft Tokens Cost the Same in Offloaded MoE Inference | When MoE experts live in host memory, each verified draft token drags in its own experts over H2D — so bigger draft trees can be *slower* than AR. SpecFit maximizes **predicted throughput** instead of acceptance length, using a monotone submodular expert-coverage cost and exact search over ancestor-closed subtrees. 21.6%/7.0% faster than EAGLE-3 (1.97×/1.72× vs vanilla) on Qwen3-30B-A3B. |
| 5 | [2610.18735](https://arxiv.org/abs/2610.18735) | SP2ec: Adaptive Self-Speculative Decoding for Vision-Language Models | Self-speculative decoding for VLMs as a **training-free bandit** over draft configurations, exploiting the single-peak wall-time curve, with wall-time-regret guarantees and negligible exploration cost. Up to +11% over SWIFT/KnapSpec/SpecVLM on Qwen2.5-VL-7B and LLaVA-NeXT-Mistral-7B. |
| 6 | [2610.12939](https://arxiv.org/abs/2610.12939) | Surfing the Activation Wave: Analyzing the Information and Geometry of LLM Hidden States for Speculative Decoding | Replaces fixed layer heuristics (e.g. [2, N//2, N−3]) with principled **multi-layer hidden-state selection**: informativeness via matrix entropy + diversity via CKA, plus token-entropy filtering. Selected sets beat the defaults in EAGLE3-style drafting and explain why some defaults stay competitive. |

## 24. Serving & Scheduling Systems

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2610.13325](https://arxiv.org/abs/2610.13325) | Taking Tokenization off the Host for Agentic LLM Serving | Agentic prefix-cached traffic makes **tokenization the largest part of TTFT** and the frontend the fleet bottleneck. GraphTok runs the whole encode pipeline as one replayable CUDA graph, bit-exact on GPT-2/Llama 3/Qwen 2.5/DeepSeek-V3/Gemma 3, plugging into unmodified vLLM and NVIDIA Dynamo. 1M GPT-2 tokens in 1.7–3.2 ms; −2.1–2.4× mean TTFT with a cached prefix; one frontend sustains 518 req/s of 121K-token traffic vs 296 (gigatoken) and 18.5 (default) within a 2 s P99 target. |
| 2 | [2610.12242](https://arxiv.org/abs/2610.12242) | TokenRouter: Efficient Serving System for Token-Level LLM Routing | Serving system for **token-level** routing: developers write request-centric routing logic, while the runtime runs a per-model subserver with delayed-batching schedulers whose hyperparameters come from a throughput model (fixing step desynchronization and admission delays). 2.01–64.15× higher decode throughput than existing systems. Code: thu-nics/TokenRouter. |
| 3 | [2610.14686](https://arxiv.org/abs/2610.14686) | Vesta: Queueing-Aware LLM Scheduling for Real-World Workloads | For agentic multi-turn traffic: pin a stable holder for each session's authoritative KV cache but choose the executor per request, optimizing backlog + reusable prefix + KV transfer + the queueing externality of extra demand, and migrating prefixes across length buckets. +14.0% throughput and −57.8% mean TTFT over the SOTA scheduler (PD-colocated); up to −29.0% P99 TTFT when disaggregated. |
| 4 | [2610.17689](https://arxiv.org/abs/2610.17689) | Stable Token-level Scheduling for Serverless LLM Inference | Serverless admission control at **token** granularity instead of batch granularity: Token Window Pacing is a closed loop over inter-token latency, queue depth and workload characteristics, plus chunked prefill to cut prefill interference. Stabilizes TTFT, cuts latency spikes and P99 tails, +45% throughput over ServerlessLLM and +58% over conventional schedulers. |
| 5 | [2610.15899](https://arxiv.org/abs/2610.15899) | Lightweight Scheduling Policies for Continuously Batched LLM Serving Across GPU Architectures | A 1,152-run empirical study (8 policies × 6 Qwen/Gemma/Mistral models × L40S/A100/H100 × ShareGPT traffic): scheduler choice alone moves throughput, TTFT, completion latency and tails at fixed model/GPU/batch. **Aging shortest-prompt-first** is the strongest practical baseline; even oracle remaining-length knowledge is insufficient because batching and hardware effects dominate. Dataset on Hugging Face. |
| 6 | [2610.17795](https://arxiv.org/abs/2610.17795) | Quota-Based Aging for Prediction-Guided Scheduling in LLM Serving | Quota-bounded, debt-enforced promotion of aged predicted-long requests on top of predicted-short-first admission — a controllable middle ground that keeps most of the mean-latency gain while cutting long-request tail latency, evaluated under asymmetric prediction errors in a serving-aware simulator. |
| 7 | [2610.14804](https://arxiv.org/abs/2610.14804) | AgentWeave: Efficient Distributed Agent Serving via Flow Decomposition | Argues the disaggregated agent architecture creates pipeline bubbles, and an aggregated one creates request blocking; flow decomposition splits long agent requests into fine-grained units preserving agent semantics, with a scheduler balancing splitting gains vs overhead. +1.81× throughput, −2.45× mean latency, −2.04× P90. |
| 8 | [2610.14571](https://arxiv.org/abs/2610.14571) | Dissecting the Carbon-SLO Tension in Geo-Distributed LLM Serving | A latency-centric geo-distributed simulator calibrated with real latency/power, plus a ladder of SLO- and carbon-aware schedulers (greedy → ILP → load-coupling ILP → facility MILP). The MILP option cuts carbon up to 60% vs round-robin at ~3% average SLO violation; savings grow with datacenter carbon-intensity spread and shrink as load rises. |
| 9 | [2610.15366](https://arxiv.org/abs/2610.15366) | Compositional Performance Modeling of LLM Inference Clusters with Hierarchical KV Caching | Discrete-event cluster simulator that separates local service costs from shared resource/KV availability, so 1P-1D-calibrated compute and transfer models transfer across workers without per-configuration fitting. Validated against SGLang (Qwen3-8B/32B, 1P-1D/3P-1D): 1.34% mean E2E latency error on QA, 6.36% on coding sessions, 15.33× faster replay. |
| 10 | [2610.09657](https://arxiv.org/abs/2610.09657) | Fast and Memory Efficient Offload Training Framework with Hybrid XPU Computation | MemFerry exploits Direct Host Access (compute on data in CPU memory) with an execution scheduler + shadow model and DP gradient offload: 1.68× faster and 1.52× larger model than ZeRO-Offload on one GPU, ≥+28.1% scaling to 8 GPUs; ScaleUp-MemFerry on a CloudMatrix384 node (8 NPUs) cuts iteration time 20.7% vs DeepSpeed. |
| 11 | [2610.07593](https://arxiv.org/abs/2610.07593) | TRANSIT: Transparent Scale-in for Multi-Node LLM Training | Uses CPU DRAM as GPU-memory extension through a **user-space interposition layer** requiring no changes to app, framework, scheduler, driver or OS, with a zero-copy CPU–GPU path. Up to +68%/+59%/+42% per-GPU throughput vs TorchTitan/ZeRO-Offload/ZeRO-Infinity; trains on 50% fewer GPUs at >90% baseline per-GPU throughput; −33% per-node network traffic. |

## 25. MoE Serving & Training

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2610.18451](https://arxiv.org/abs/2610.18451) | FastEEP: Fast Intra-Instance Elastic Scaling for Expert-Parallel MoE Serving | Elastic scaling by resizing the **EP group inside an existing instance** instead of tearing down and rebuilding serving instances: "Partial Reuse" separates reusable execution structures from the resources the current EP config activates, preserving costly-to-reconstruct state. 4.5–11.7× faster scaling than the vLLM-based approach. |
| 2 | [2610.11158](https://arxiv.org/abs/2610.11158) | Zepp: Accelerating Distributed MoE Serving under Relaxed Balance Constraints | Treats balance as a **constraint** rather than the objective and instead optimizes the real bottleneck, inter-node communication, across placement, routing and execution under GPU/NIC limits, jointly coordinating compute, token comm and expert-weight movement each iteration. Up to 6.68× MoE-layer speedup and 1.86× geomean over 7 SOTA systems. |
| 3 | [2610.13328](https://arxiv.org/abs/2610.13328) | DuoRate: Two-Rate Per-Expert Precision Allocation for Mixture-of-Experts Serving | Names the **store-read gap**: BitsMoE pays 2.54 bits/weight of storage but 7.50 bpw of decode traffic, yet published allocators constrain only storage. Store one prefix-readable tier offline, then solve a per-workload Lagrangian allocation online from that workload's own prefill (routing frequency provably cancels). Qwen3-30B-A3B at 2.3 bpw beats 2-bit-class baselines by 6.3–47.0 points; at 2.9 bpw it matches 4-bit formats reading 1.4–1.6 bpw more. |
| 4 | [2610.16890](https://arxiv.org/abs/2610.16890) | FORGE: Procedural Weight Generation for Sparse MoE Inference | **Regenerates** expert FFN weights on-chip from O(log²d) rotor seeds per projection (under 1 MB replacing gigabytes) via a Rotor Generation Unit that doesn't compete with Tensor Core execution. Estimated 1.3–2.8× system throughput, 1.10–1.15× lower short-context decode latency, 1.2–2.6× more KV capacity — but accuracy falls below a rotor-free fine-tune, with the replacement ratio as the knob. RGU verified in SystemVerilog at ~0.9% SM area (7nm). |
| 5 | [2610.11317](https://arxiv.org/abs/2610.11317) | DivMoE: Fine-Grained MoE Upcycling via Cross-Domain Expert Composition | Diagnoses why fine-grained upcycling collapses (Qwen3-1.7B: 23.2% avg vs 50.2% for coarse-grained — near from-scratch), then fixes it with **domain-specialized expert initialization** (experts from domain-adaptive continual pretraining) and diversity-constrained routing that forces each token to activate experts from distinct domain groups. 55.6% vs 51.6% best baseline over 15 benchmarks; 12B DivMoE matches 16B Moonlight-MoE. |
| 6 | [2610.09372](https://arxiv.org/abs/2610.09372) | Expert Coupling in MoE Pretraining: Reducing All-to-All Overhead with Correlated Placement and Token Shuffling | Shows router selections are **correlated within and across layers** early in pretraining (0.8% of expert pairs co-selected by 42% of tokens; next-layer experts predictable). Correlated expert placement + one-dispatch-per-GPU + token shuffling during the post-attention reduce-scatter removes up to 58% of dispatched rows and cuts all-to-all time 1.16–2.63× (up to 1.41× step time) at EP 8–64 — without changing routing decisions or expert parameters. |
| 7 | [2610.14798](https://arxiv.org/abs/2610.14798) | Accelerating MoE Training through Expert Placement and Replica Coordination | Jointly optimizes expert layout and source-aware token-to-copy mapping (they are coupled under hierarchical deduplication + replication, so separate optimization is suboptimal), with co-selection-aware grouping, bounded DP for replica allocation, alternating refinement, and replica-gradient sync overlapped with attention backward. Up to 6.94× faster A2A and 1.74–2.33× E2E vs VeOmni on 32 GPUs / 64 NPUs. |
| 8 | [2610.07767](https://arxiv.org/abs/2610.07767) | TRACE: Rollout-Guided Quantization-Aware Training for FP4 Reinforcement Learning of MoE Language Models | FP4 **RL** post-training for MoE: rollout-side quantization outcomes guide training-side FP4 rounding decisions, directly minimizing train–rollout discrepancy rather than optimizing the two paths separately, with mantissa/scale caching to bound the overhead. Joint FP4 weights/activations + FP4 KV rollout at BF16-comparable RL performance with up to 5.4× rollout speedup across 4 large MoE models. |
| 9 | [2610.07774](https://arxiv.org/abs/2610.07774) | Reading, Not Manipulating: Leveraging Router Logits for Multimodal Safety in MoE Vision-Language Models | Instead of steering MoE VLM routing or editing internal states, **read** router logits during prefill as a safety detector — no parameter or routing changes, no over-refusal trade-off. Strong HoliSafe gains on Qwen3-VL/Kimi-VL plus OOD generalization to MISHard and MM-SafetyBench. |

## 26. On-Policy Distillation, Knowledge Distillation & RL Post-Training

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2610.13238](https://arxiv.org/abs/2610.13238) | State-OPD: State-Grounded On-Policy Distillation for Multi-Turn Agents | Analyzes agentic OPD through **interaction states**: prior student reasoning distorts teacher supervision (removing it improves teacher continuation, more so at later turns), while current reasoning shapes the sampled action set. Reallocates sampling to high-response-entropy states. Qwen3-4B macro-average success 26.07%→31.74% on ALFWorld/ScienceWorld/WebShop, beating its Qwen3-32B teacher on ALFWorld-Seen with fewer supervised turns. |
| 2 | [2610.11659](https://arxiv.org/abs/2610.11659) | DIAL-OPD: Learning More from Fewer Tokens in On-Policy Distillation | Shows **training on fewer tokens can beat full-token OPD**: disagreement criteria that ignore probability scale over-weight "low-low" tokens; DIAL-OPD weights reward magnitude by the logarithmic mean of teacher/student probabilities. Retaining 40% of tokens beats Vanilla OPD by up to +5.25 points and doubles AIME25 Pass@16 (13.33%→26.67%); a 4B teacher then beats a full-token 8B-teacher baseline. |
| 3 | [2610.08448](https://arxiv.org/abs/2610.08448) | Rethinking Cross-Tokenizer On-Policy Distillation: From Alignment Coverage to Supervision Reliability | Across 3 heterogeneous teacher–student pairs, strict 1:1 alignment already covers most student tokens, and a student-selected top-16 shared-vocabulary subset matches full shared-vocab OPD. **Adding span-MSE supervision for complete coverage reduces accuracy**, with span gradients weakly/negatively aligned to the strict ones — prioritize supervision reliability over coverage. |
| 4 | [2610.09639](https://arxiv.org/abs/2610.09639) | On-Policy Distillation Teaches New Skills but Not New Knowledge | Controlled synthetic separation of facts vs compositional skill across 4 models / 3 families: reverse-KL OPD reliably transfers **composable skill** to unseen reasoning structures but minimal factual knowledge; swapping in forward KL restores factual transfer, and student rollouts specifically improve multi-step execution. Confirmed on factual QA and competition math. |
| 5 | [2610.08959](https://arxiv.org/abs/2610.08959) | GraphOPD: Graph-Augmented On-Policy Distillation for LLM Agents | Highest-divergence-step selection misfires in multi-turn agents because early drift is absorbed into every later context. GraphOPD reads which steps enabled which from the environment's own state-change record, builds a dependency graph, scores steps by random-walk stationary distribution, and fuses that structural credit with divergence. Up to +5.8 points over the strongest baseline on ALFWorld/WebShop/SearchQA; replay audit shows the score tracks true causal impact. |
| 6 | [2610.17397](https://arxiv.org/abs/2610.17397) | JustOPD: On-Policy Distillation from Autoregressive to Diffusion Language Models | Distilling a strong AR teacher into a weaker **diffusion** LM by using autoregressive rollouts during training to align the student with the teacher's left-to-right decoding, while preserving the DLM's flexible generation order at inference. Qwen3-14B → Dream-7B, consistent math gains across divergences. |
| 7 | [2610.07706](https://arxiv.org/abs/2610.07706) | WASD: Wasserstein-based Knowledge Distillation for Large Language Models | Wasserstein distance with a **token-embedding-derived cost matrix** (Sinkhorn divergence, gradient-equivalent objective, no extra networks) uses token-space semantics that index-wise divergences ignore. Consistent gains on instruction following, math and code across families/scales. Code: aailab-kaist/WASD. |
| 8 | [2610.10854](https://arxiv.org/abs/2610.10854) | KDFP: A first-principles approach to knowledge distillation in large language models | White-box **general-knowledge** distillation — the gap left by the community's focus on post-training abilities. +1.6–4.9% over existing methods across 9 benchmarks with up to 99.1% better training efficiency via ephemeral parameter reduction; aimed at edge/private deployment. |
| 9 | [2610.14939](https://arxiv.org/abs/2610.14939) | VC-OPD: Visual Counterfactual On-Policy Distillation for Grounded Vision-Language Reasoning | Grounding-aware OPD: evaluate the teacher twice per student trajectory (real image vs counterfactual with corrupted instance evidence); the gap estimates token-level **visual dependence** and softly reweights the loss, with mass-preserving normalization to keep update scales stable. |
| 10 | [2610.16592](https://arxiv.org/abs/2610.16592) | Prorsa: Unleashing the Potential of Prefill-Decode Disaggregation in Agentic Reinforcement Learning | Multi-turn agentic RL keeps prefill and decode coexisting throughout rollout, which makes **PD disaggregation** valuable (less DP-attention interference, stage-specific parallelism, shorter trajectories) — but a fixed P/D split decays. Prorsa forecasts short-horizon P/D demand per trajectory and switches rollout-engine roles. 1.11–4.05× E2E rollout speedup over PD colocation, sync and async. |
| 11 | [2610.16352](https://arxiv.org/abs/2610.16352) | A Production-Grade, High-Efficiency Agentic RL Framework for Frontier LLMs with Validated Recipes | LoongSage: agent-native fully asynchronous RL in ~14K LOC on Megatron + SGLang (Ray actors, one controller), with AgentFlow for white-/black-box agents, sliding-window staleness bounds, TransferMesh topology-aware weight sync, and a memory-efficient Teacher Manager natively supporting **multi-teacher OPD**. Four complete training runs with steady reward gains. |
| 12 | [2610.16866](https://arxiv.org/abs/2610.16866) | Efficient Resource-Shiftable RL Post-training with Generative Rewards | GenRM-based RL post-training where rollout, reward evaluation and policy training **shift resources dynamically** instead of using static allocation: Global Parameter Storage, a unified role-context shift mechanism, sequence migration on shift, and a planner trading throughput benefit against shift overhead. 1.81–2.05× throughput over colocated and 1.50–1.95× over disaggregated on 128 GPUs, preserving convergence. |
| 13 | [2610.16533](https://arxiv.org/abs/2610.16533) | Spectral-Driven Adaptive-Rank Optimizer for LLM Reinforcement Learning with Verifiable Rewards | Finds that RLVR gradients progressively concentrate into a **shrinking effective subspace** (absent under SFT, largely removed by random rewards), so Ares estimates the effective rank from the spectral energy distribution and confines each update to it. Improves accuracy over widely used optimizers across RLVR algorithms, tasks and scales. Code: RUCAIBox/Ares. |
| 14 | [2610.09804](https://arxiv.org/abs/2610.09804) | BoT-GRPO: Efficient Process-Reward RL for Reasoning via Bag-of-Token Aggregation | Extends GRPO to token-level reward models with length-invariant bag-of-tokens aggregation (weight each token-reward by inverse sequence length, then compute per-token advantages against weighted group statistics) — critic-free and a drop-in wherever GRPO is used. 80% compile rate up to 1.9× faster than GRPO on React code-gen, beating GSPO/DAPO/PURE; up to +8.1% Pass@k on AIME in half the steps. Practical finding: **stable, bounded, fine-grained rewards beat richer noisier ones**. |
| 15 | [2610.15840](https://arxiv.org/abs/2610.15840) | LLM Value Functions for Credit Assignment in Agentic RL | LVRL uses a frozen LLM as a **policy-aware value function** (conditioned on group context, calibrated at initial/terminal states) instead of training a critic, giving step-level credit from value differences; LVOPD then focuses teacher supervision on value-decreasing actions. Up to +13.6 points on AppWorld and +8.6 on SWE-bench Verified over GRPO; BabyAI goes from total failure to 75.0% success. |
| 16 | [2610.14448](https://arxiv.org/abs/2610.14448) | KV4-RL: Unlocking Faster Asynchronous Rollouts for RLVR with INT4 KV Cache Quantization | Characterizes "KV concentration" in async RLVR and finds **KV-cache quantization beats weight-only and weight-activation quantization** there; INT4 KV with selective INT8 protection of attention sinks and recent context keeps sampler mismatch bounded. 1.80× cumulative rollout and 1.27× full-iteration speedup over BF16 in 400 GRPO iterations of Qwen3-8B-Base, final accuracy within 0.5 points. |
| 17 | [2610.11502](https://arxiv.org/abs/2610.11502) | Fed-GRPO: Reward-Signal-Driven Federated Group Relative Policy Optimization | Federated GRPO that reuses reward statistics as **zero-cost** signals: signal-weighted aggregation (by reward std), global reward calibration against the local–global reward gap, and adaptive sparse communication. Best federated result on math reasoning, approaching centralized, with 32× lossless communication reduction and up to 621× compression under tight bandwidth. |
| 18 | [2610.18572](https://arxiv.org/abs/2610.18572) | LLM Agents for Automated Post-Training via Structured Recipe Optimization | Automated post-training as optimization over a **task-adapted recipe space** (an LLM Recipe-Space Designer plus a multi-domain optimizer using structured trial evidence and multi-objective Bayesian selection), rather than a fixed recipe space or implicit trial-level edits. 41.1 vs 15.4 for the strongest baseline on Qwen3-1.7B, 53.8 vs 40.8 on Qwen3-4B, in joint 3-domain post-training. |
| 19 | [2610.08108](https://arxiv.org/abs/2610.08108) | Enhancing Diffusion Language Models with Autoregressive Post-Training Weights | Training-free A2D: recycle an AR model's post-training weight update into a diffusion LM. AR and diffusion updates are nearly **orthogonal in weight space** yet induce aligned representation changes, and composing them keeps both sets of gains. Improves instruction following, math and coding across Dream/DiffuCoder/DiffusionGemma with SFT and RL updates, no extra training or inference cost. |
| 20 | [2610.10304](https://arxiv.org/abs/2610.10304) | SemanticFold: Latent Sequence Compression Separates Language Modeling, Decodability, and Reasoning | Shows latent prefix compression moves LM fit, decodability and reasoning **non-monotonically and at different thresholds**, with probe accuracy/AUC changing <0.03 while NLL shifts are model-specific and mostly attributable to residual adaptation rather than shortening. A caution against any single scalar certificate for latent compression. |
| 21 | [2610.11111](https://arxiv.org/abs/2610.11111) | Lapras: Latent Reasoning for Time Series Language Models | Latent (continuous-thought) post-training for time-series LMs: a CoT-trained teacher supervises a student that reasons in the joint series–language space and emits text only for the final answer. +10.79% average F1 with **23.9× fewer generated tokens** across 4 backbones / 5 benchmarks, with thoughts decodable into readable traces. |
| 22 | [2610.15620](https://arxiv.org/abs/2610.15620) | From Stages to States: Unified Iterative Reasoning for MLLMs | Reframes multimodal CoT from a stage-driven pipeline to a **state-driven** process: unified reasoning-state modeling plus closed-loop reasoning-state learning with self-generated historical states, so correction needs no extra critic or reflection stage. Ships VersaCoT-10K (8 visual reasoning domains). |

## 27. Agents, Latent Communication & Evaluation

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2610.18861](https://arxiv.org/abs/2610.18861) | Hybrid Reasoning Agents | Agents that dynamically navigate between **latent reasoning and text interaction** in one generation process: a lightweight HybridLink bridges modes and a continuous working memory preserves lossless hybrid context across turns, keeping tool use while making latent thoughts inspectable. HybridSFT + HybridRL post-training; +10.4% over frontier LLM agents across 3 scales / 11 benchmarks with 35.5–54.7% token reduction, 1.8–2.6× faster inference and 2.7–3.7× lower cost. Code: togethercomputer/HybridAgent. |
| 2 | [2610.17953](https://arxiv.org/abs/2610.17953) | Agent-Reflex: Teaching Language Agents to Act Reflectively | Three-stage training for **proactive reflection** under noisy tool feedback: reflection skill discovery from injected tool-feedback perturbations, reflective SFT (self-reflection, skill recovery, user-centric correction), then ORPO combining agentic RL with reflection-guided OPD. Qwen3-8B reaches 58.6 on five long-horizon tool-use benchmarks and 29.1 on AgentNoiseBench. |
| 3 | [2610.17659](https://arxiv.org/abs/2610.17659) | Can Agents Truly Forget? Explore Machine Unlearning for LLM Agents | Shows parameter-level unlearning is insufficient for agents — forgotten data persists in **persistent memory** or is reacquired through **tool calls** — and formalizes agent unlearning across three channels. AUBench extends the fictitious-author setting into an agent environment with memory, an offline web snapshot, tool-accessible documents and provenance-tracked trajectories, exposing the model-level vs system-level gap. |
| 4 | [2610.17816](https://arxiv.org/abs/2610.17816) | Empowering Edge Agents: A Systematic Analysis of Edge-Cloud Collaboration in LLM Agent Execution | EC2-Bench evaluates **edge–cloud collaboration in long-horizon agent execution**, where cloud participation is interleaved through the trajectory rather than a per-query decision. Collaboration balances task performance, resource use and privacy exposure; stronger edge models shift the preferred strategy toward edge-controlled collaboration; cloud-invocation decision-making and instruction following are the key edge abilities. |
| 5 | [2610.12183](https://arxiv.org/abs/2610.12183) | A Closer Look at Agentic BBO: Benchmarking LLM Agents for Black-Box Optimization | AgenticBBO-Bench spans synthetic functions, hyperparameter optimization, database tuning, chip design and molecular design under a unified finite budget. Agentic BBO beats direct LLM methods in all 5 domains and the best numerical optimizers in 4; extra numerical tools do not consistently help, task semantics help broadly while specific priors are unreliable, and numerical optimizers can absorb the gains the agent's search trajectory creates. |
| 6 | [2610.15207](https://arxiv.org/abs/2610.15207) | Exploring Jailbreak Vulnerability Across Depth-Based Dynamic Inference in Large Language Models | Safety analysis of early-exit inference: **shallow exits are consistently more robust** to jailbreaks than deep ones, adversarial suffixes optimize more easily against deep exits, and suffix effects on utility are uniform across exit depths. Directly relevant to anyone exposing layer-skipping as a speed feature. Code: LassNajmuldeen/balcony-jailbreaking. |
| 7 | [2610.08647](https://arxiv.org/abs/2610.08647) | SquidAgent: Parallelize Wisely, Coordinate Efficiently | Explains why parallel multi-agent systems often lose to a single agent: **re-exploration** of context the orchestrator already has, and **alignment** of independently produced outputs. Proposes a token-based (not wall-clock — LLMs are poorly calibrated on time) parallelization criterion, forks workers from the orchestrator session, and pre-generates a shared convention block. 2.2× throughput and 2.6× wall-time speedup over Claude Code. |
| 8 | [2610.11899](https://arxiv.org/abs/2610.11899) | Forms of LLM-Integrated Applications from LLM-Chats to Autonomous AI Agent System | Survey taxonomy of 7 recurring architectural forms (chat, custom agents, RAG, AI-enhanced workflows, copilots, coding agents, agentic RAG) along four structural dimensions over 22 grounded research/vendor systems. Argues the labels carry real architectural content — e.g. copilot = router–worker under step-by-step user confirmation, and four major providers' coding agents share one reason-and-act loop delegating to subagents. |

## 28. Long-Context, RAG & Memory

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2610.08463](https://arxiv.org/abs/2610.08463) | UNREAL: Unifying Retrieval and Long-Context with a Single Model | One **model-internal** evidence-selection mechanism spanning corpus retrieval and long-context inference, deriving retrieval queries from the frozen LLM's representations with <500K trainable parameters and an unchanged backbone. On a 3B-token / 21M-chunk Wikipedia index: recall 49.1→73.2% (HotpotQA), 31.7→60.1% (2Wiki), 8.8→14.4% (MuSiQue); on long-context, NoLiMa 1.0→24.83% at 128K, fewer FLOPs from ~32K onward. |
| 2 | [2610.08452](https://arxiv.org/abs/2610.08452) | Agentic AutoRAG: RAG Pipeline Optimization through Reasoning-Driven Agents | Multi-objective RAG hyperparameter optimization with **retrieval-vs-generation failure attribution**: a Diagnoser assigns each failed exam question to a stage and a Proposer selects the next configuration from a model-ranking/pricing knowledge base, tracing a Pareto frontier. Beats all baselines on 3 multi-hop QA benchmarks and matches their full 30-trial accuracy within 10 trials; on a healthcare corpus, 77% median accuracy (best baseline 71.5%) at ~58% of its cost per query. |
| 3 | [2610.07782](https://arxiv.org/abs/2610.07782) | Persistent Memory in Multi-Agent LLM Inference: What It Costs, What It Buys, and When You Can Tell | Hard-nosed negative result: agent decomposition does shrink the KV working set (14.3 MiB/query peak vs 35.5/35.3 for single-pass and retrieval-augmented), but a persistent reasoning-trace tier buys **no detectable accuracy** (+0.015, 95% CI [−0.011, +0.046]) while costing +0.368 MiB peak cache, and argues the null is structural for single-question benchmarks. Documents four measurement corrections needed to even see the effect. |
| 4 | [2610.10533](https://arxiv.org/abs/2610.10533) | EngramEdit: Decoupled Knowledge Updates in LLMs through Conditional Memory | Turns conditional-memory n-gram embeddings into an **editable factual interface**: jointly update shared embeddings to hit multi-expression targets while penalizing edits to frequently reused embeddings. Near-perfect editing success, revised knowledge works across unseen expressions and multi-hop CoT (~3× the strongest baseline's accuracy), with unrelated knowledge and general ability largely preserved as edits accumulate. |
| 5 | [2610.15921](https://arxiv.org/abs/2610.15921) | Fast and Accurate: Improving Personalized LLM Performance by Semantic Concentration Inspired Workload Split between RAG and Fine-Tuning | Splits a user's history by **semantic concentration**: concentrated "core" knowledge goes to offline parametric fine-tuning while the dispersed tail stays retrieval, and online inference routes core queries to the tuned region and only calls RAG for peripheral ones. Best generation quality across users and tasks plus 14.4–46.6% lower inference latency than the best-performing baseline. |
| 6 | [2610.08312](https://arxiv.org/abs/2610.08312) | CoDe-LoRA: Mitigating the Orthogonality Dilemma in Continual Learning of LLMs via Knowledge Consolidation and Decoupling | Identifies an **"orthogonality dilemma"**: strict orthogonal-projection LoRA isolates tasks so hard that shared representations stop transferring across related tasks. CoDe-LoRA instead consolidates universal knowledge and decouples task-specific knowledge with adaptive null-space projection + semantic routing (replay-free); best average accuracy across 4 backbones / 3 continual-learning benchmarks. |
| 7 | [2610.18051](https://arxiv.org/abs/2610.18051) | inLoRA-MINT: Internal Representations for Auditing Training Data in Domain-adapted LLMs | Membership inference for LoRA-adapted LLMs using **internal representations** (activations, parameter-dependent signals) rather than perplexity alone, without full parameter access. 95–98% across Qwen/Phi-4/DeepSeek-V3/LLaMA-3.2-3B on medical, math and academic reasoning data — a practical tool for data provenance and privacy compliance. |

## Part 9 takeaways

- **Cross-model KV is now a serving paradigm, and fidelity is the wrong loss.** If you transport or reuse KV across models, measure the *receiver's* computation (CacheLink), not reconstruction error; repair causal suffixes rather than stitching document caches (SAFE-KV), and expect safety drift specifically from reused **values** (VReparo, 2610.15721).
- **Quantize and store KV differently than you read it.** ReadKV (per-query read budget), DuoRate (store-rate vs read-rate for MoE), and AlignPred (error-bounded persistent caches) all separate storage from access — the highest-leverage new axis in this window.
- **Sparse attention's scoring, not its selection, is the cost.** SPIN and QUILT both attack indexer/execution redundancy for 30–55% gains; MASA says rebuild the ranking around matrix-approximation error, not raw attention mass.
- **For agentic workloads, schedule the trajectory, not the request.** Vesta (session cache holder ≠ executor), GraphTok (tokenization inside the P99 budget), AgentWeave (flow decomposition), Prorsa (dynamic P/D for RL rollouts), PivotRL (shiftable roles).
- **OPD's frontier is *where to supervise*, not how much.** DIAL-OPD (40% of tokens wins), State-OPD (states, not reasoning prefixes), GraphOPD (dependency-graph credit), plus the knowledge-vs-skill asymmetry (2610.09639) — and train QAT recovery on student-visited states (OnlineQAT).
- **Report negative results.** EdgeLLM-Nano (no compressor beat FP16 on energy/token; perplexity hid the differences), 2610.07782 (persistent agent memory bought nothing detectable), and 2610.14475's LCDF metric are the antidotes to the positive-result bias in the rest of this batch.
