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
