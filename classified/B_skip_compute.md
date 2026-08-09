# B — Compute That Skips Work (early-exit, dynamic width, sparsity, hashing, quantization)

**172 papers** (20 already covered in Parts 1–3).

Mechanism idea: see `ML-NEW-TECH.md` Part 3.

| arXiv | Title | Domain | Matched mechanism keywords |
|---|---|---|---|
| [2607.28047](https://arxiv.org/abs/2607.28047) | A Query-Efficient Stochastic Volume Rendering Framework for Time-Varying Implicit Neural Volumes ✅ | Theory_opt, RL, Graphs | `approximate, caching, prun, pruning` |
| [2607.28418](https://arxiv.org/abs/2607.28418) | WIDE: Boosting Adaptive LLM Inference via Token-level Dynamic Width Pruning ✅ | Vision_media, NLP_lang, Science_eng | `accelerat, acceleration, dynamic pruning, dynamic width, prun, pruning, skip, sparse` |
| [2607.29120](https://arxiv.org/abs/2607.29120) | Curriculum Matters: Data-Efficient Relational PFN Pretraining with Synthetic Data ✅ | Theory_opt, Vision_media, Bio_chem | `approximate, exit, progressive` |
| [2608.00859](https://arxiv.org/abs/2608.00859) | SparseKAN: Compressing Kolmogorov--Arnold Networks Across Basis Functions, Neurons, and Bits ✅ | Theory_opt, Graphs, Vision_media | `bit, low precision, quantiz, sparse` |
| [2608.01536](https://arxiv.org/abs/2608.01536) | Celty: SpMspV GPU Kernel and SIMT Co-Design for Efficient Dual-Sparse LLM Inference ✅ | Vision_media, NLP_lang | `prun, pruning, skip, sparse` |
| [2608.01804](https://arxiv.org/abs/2608.01804) | LEAP: Lean Environment-Feedback via Adaptive Pruning for Code RL in GPU Kernel Generation ✅ | Theory_opt, RL, Vision_media | `accelerat, binary, bit, prun, pruning` |
| [2608.02700](https://arxiv.org/abs/2608.02700) | NANQ: Noise-Floor-Aware Mixed-Precision Non-Uniform Quantization for Analog Compute-in-Memory ✅ | Vision_media, NLP_lang, Science_eng | `bit, quantiz` |
| [2608.02951](https://arxiv.org/abs/2608.02951) | SP3O: Reinforcement Learning from Segment Preferences without Reward Modeling ✅ | Theory_opt, RL, Vision_media | `bit` |
| [2608.03036](https://arxiv.org/abs/2608.03036) | LLM Serving in the Wild: An Empirical Study of Frameworks, Methods, and System Designs ✅ | Theory_opt, RL, Vision_media | `prun, pruning` |
| [2608.03447](https://arxiv.org/abs/2608.03447) | Approximate Speculative Decoding ✅ | Theory_opt, Vision_media, NLP_lang | `accelerat, approximate, binary, speculative` |
| [2608.04001](https://arxiv.org/abs/2608.04001) | Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility ✅ | Theory_opt, RL, Graphs | `inference-time, progressive` |
| [2608.04052](https://arxiv.org/abs/2608.04052) | When Modalities Fail to Tango: Conformal Backdoor Detection in Multimodal Contrastive Learning ✅ | Theory_opt, Vision_media, NLP_lang | `bit, cascade` |
| [2608.04962](https://arxiv.org/abs/2608.04962) | SpecRoll: Fast-Slow Verifier-Feedback Adaptation for Speculative Reinforcement Learning Rollouts ✅ | Theory_opt, RL, Vision_media | `accelerat, sparse, speculative` |
| [2608.04980](https://arxiv.org/abs/2608.04980) | Protoreasoning in Tiny Transformers ✅ | RL, NLP_lang, Bio_chem | `bit` |
| [2608.05111](https://arxiv.org/abs/2608.05111) | Reward Structure Shapes the Interaction Between Episodic Exploration and Neural Memory in Reinforcement Learning ✅ | Theory_opt, RL, Vision_media | `sparse` |
| [2608.05499](https://arxiv.org/abs/2608.05499) | APQF: Agentic Profiling-Guided Structured Pruning and Mixed-Precision Quantization with Adaptive Fine-Tuning ✅ | Graphs, Vision_media, Bio_chem | `bit, prun, pruning, quantiz` |
| [2608.05797](https://arxiv.org/abs/2608.05797) | Predicting Task Difficulty Without Rollouts ✅ | Vision_media, NLP_lang, TimeSeries_signal | `progressive` |
| [2608.05819](https://arxiv.org/abs/2608.05819) | Learning to Rank Tensor Network Contraction Plans for GPU-Accelerated Quantum Circuit Simulation ✅ | Theory_opt, RL, Vision_media | `accelerat, exit` |
| [2608.06177](https://arxiv.org/abs/2608.06177) | Threshold-Based Early Stopping of Accumulations in Neural Networks with Binary Activation ✅ | Vision_media, Science_eng, Eval_bench | `binary, early stopping, early-stopping, threshold-based` |
| [2608.06291](https://arxiv.org/abs/2608.06291) | BaKron: Efficient Quantization with Kronecker-Factored Hessians ✅ | Vision_media, TimeSeries_signal, Eval_bench | `accelerat, quantiz` |
| [2607.28036](https://arxiv.org/abs/2607.28036) | Learning features from Newton's algorithm: a way to accelerate nonlinear parametrized PDE solvers | Theory_opt, Vision_media, Bio_chem | `accelerat, reduced` |
| [2607.28073](https://arxiv.org/abs/2607.28073) | GVR-Coder: A Visual-Feedback Framework for Structured SVG Generation in Complex Document and Meeting Scenarios | Theory_opt, RL, Graphs | `exit, progressive` |
| [2607.28124](https://arxiv.org/abs/2607.28124) | Information Bottleneck Learning for Faithful Time Series Forecasting Explanations | Theory_opt, Vision_media, NLP_lang | `bit` |
| [2607.28135](https://arxiv.org/abs/2607.28135) | LM-GRASP: Instance-Specific Language Models for Combinatorial Construction via Online Imitation Learning | Theory_opt, RL, Vision_media | `accelerat, acceleration` |
| [2607.28220](https://arxiv.org/abs/2607.28220) | Weather Emulators at the Frontier of Heat Extremes Predictability | Theory_opt, Graphs, Vision_media | `reduced` |
| [2607.28248](https://arxiv.org/abs/2607.28248) | Uncertainty quantification for trustworthy deep learning: Methods and measures | Theory_opt, Vision_media, NLP_lang | `approximate` |
| [2607.28282](https://arxiv.org/abs/2607.28282) | (Towards) Scalable Reliable Automated Evaluation with Large Language Models | Theory_opt, RL, Vision_media | `approximate, exit` |
| [2607.28301](https://arxiv.org/abs/2607.28301) | HARGO: Heterogeneity-Aware Reward-Guided Optimization for RL Post-Training of LLMs on HPC Tasks | Theory_opt, RL, Graphs | `binary, bit` |
| [2607.28319](https://arxiv.org/abs/2607.28319) | Fairness Pruning: Locating Demographic Bias in GLU-MLP Layers via Differential Activations | Graphs, Vision_media, NLP_lang | `inference-time, prun, pruning` |
| [2607.28399](https://arxiv.org/abs/2607.28399) | Why Are GUI Agents Correct but Late? Decode on the Decision-Time Critical Path, Tested with Pre-Compiled Policy Trees | Theory_opt, Vision_media, NLP_lang | `bit` |
| [2607.28413](https://arxiv.org/abs/2607.28413) | Windowed thinning and query complexity for the bouncy particle and Zigzag samplers | Theory_opt, Vision_media, Bio_chem | `exit` |
| [2607.28428](https://arxiv.org/abs/2607.28428) | Kohn-Sham Spectral Embedding on Sparse Graphs at the Nishimori Temperature for Image Classification | Theory_opt, RL, Graphs | `exit, sparse` |
| [2607.28456](https://arxiv.org/abs/2607.28456) | Graph Neural Multilevel Preconditioners for Iterative Solvers | Theory_opt, Graphs, Vision_media | `sparse` |
| [2607.28553](https://arxiv.org/abs/2607.28553) | APO: Unsupervised Atomic Policy Optimization for 3D Structure Prediction of Atomic Systems | Theory_opt, RL, Vision_media | `bit` |
| [2607.28628](https://arxiv.org/abs/2607.28628) | Learning to Trace Seiberg Dualities | Theory_opt, Graphs, Vision_media | `exit` |
| [2607.28698](https://arxiv.org/abs/2607.28698) | Flow Matching with Missing Data | Theory_opt, Vision_media, Bio_chem | `approximate` |
| [2607.28706](https://arxiv.org/abs/2607.28706) | Accelerated Random-Sweep Gibbs Sampling for Gaussian Graphical Models via Dual Normal Factor Graphs | Theory_opt, Graphs, Vision_media | `accelerat, acceleration, exit` |
| [2607.28849](https://arxiv.org/abs/2607.28849) | Hypergradient-based Bilevel Reinforcement Learning with Improved Sample Complexity | Theory_opt, RL, Vision_media | `exit` |
| [2607.28916](https://arxiv.org/abs/2607.28916) | Gated Q-learning: Add Off-Policy Bias to Taste | RL, Vision_media, Science_eng | `binary` |
| [2607.28925](https://arxiv.org/abs/2607.28925) | Learning Optimal Dynamic Matching via Graph Neural Networks | Theory_opt, RL, Graphs | `approximate, binary, exit` |
| [2607.28945](https://arxiv.org/abs/2607.28945) | FairDiffuseVQVAE: Sampling-Time Fairness in Tabular Diffusion via Conditional Refinement of Vector-Quantized Latents | Graphs, Vision_media, Bio_chem | `inference time, quantiz` |
| [2607.29038](https://arxiv.org/abs/2607.29038) | DFSC: Error-Controlled Differentiable Mittag-Leffler Propagation for Fractional Scientific Machine Learning | Theory_opt, RL, Vision_media | `accelerat, sparse` |
| [2607.29095](https://arxiv.org/abs/2607.29095) | PiDDM: Physics-Informed Differentiable Degradation Modeling for Lithium-Ion Battery State-of-Health Prediction | Theory_opt, Vision_media, Bio_chem | `accelerat, reduced` |
| [2607.29129](https://arxiv.org/abs/2607.29129) | PluRel-to-RDB-PFN: Schema-Guided Synthetic Relational Pretraining | Vision_media, NLP_lang, TimeSeries_signal | `approximate, binary` |
| [2607.29135](https://arxiv.org/abs/2607.29135) | HERO: History-Enriched Rollout Training for Long-Horizon Autoregressive Neural Operators | Theory_opt, Vision_media, Bio_chem | `bit, inference-time` |
| [2607.29245](https://arxiv.org/abs/2607.29245) | Simple-regret rates and minimax optimality of fixed-prior expected improvement in Matérn and squared-exponential RKHSs | Theory_opt, Vision_media, TimeSeries_signal | `bit` |
| [2607.29294](https://arxiv.org/abs/2607.29294) | Sample Efficient Hierarchical Reinforcement Learning via Best Policy Identification | Theory_opt, RL, Vision_media | `exit, sparse` |
| [2607.29378](https://arxiv.org/abs/2607.29378) | PTP: Previous-Token Prediction based LLM Inversion for Near-Exact Prompt Reconstruction | RL, Vision_media, NLP_lang | `bit` |
| [2607.29398](https://arxiv.org/abs/2607.29398) | OnlineCache: Learning Dynamic Caching Policies with Error Correction for Efficient Diffusion Inference | Theory_opt, RL, Vision_media | `accelerat, acceleration, caching` |
| [2607.29503](https://arxiv.org/abs/2607.29503) | The Grokked Illusion: True Equilibrium Mitigates Catastrophic Forgetting | Theory_opt, Graphs, Vision_media | `approximate, bit` |
| [2607.29593](https://arxiv.org/abs/2607.29593) | Convergence and Regret of the Policy Gradient for Multi-Armed Bandits in Diffusion Environment | Theory_opt, RL, Vision_media | `bit` |
| [2608.00175](https://arxiv.org/abs/2608.00175) | Inference-Time Policy Alignment for Fair Reinforcement Learning | RL, Vision_media, NLP_lang | `inference time, inference-time` |
| [2608.00195](https://arxiv.org/abs/2608.00195) | MedSAM2-Anatomy: Training-Free Inference-Time Optimization for Musculoskeletal Segmentation | Theory_opt, Vision_media, Bio_chem | `inference-time` |
| [2608.00212](https://arxiv.org/abs/2608.00212) | A Physics-Chemistry-Informed Neural Network (PCINN) for Real-Time Spatial-ALD Coverage Prediction and Reliable Kinetics Inversion | Theory_opt, Vision_media, Bio_chem | `sparse` |
| [2608.00296](https://arxiv.org/abs/2608.00296) | Stabilized Best-of-$K$ Training for Neural Combinatorial Optimization | Theory_opt, RL, TimeSeries_signal | `binary` |
| [2608.00315](https://arxiv.org/abs/2608.00315) | Towards General Language-Conditioned Latent Safety Filters | Vision_media, NLP_lang, Bio_chem | `bit` |
| [2608.00484](https://arxiv.org/abs/2608.00484) | From Digital to Physical Reservoir Computing: Co-Optimizing Soft Robotic Reservoirs via Dynamics Matching | Theory_opt, Vision_media, TimeSeries_signal | `accelerat, acceleration` |
| [2608.00508](https://arxiv.org/abs/2608.00508) | RadYOLO: Computationally Efficient 3D Object Detection and Segmentation in CT and MRI | Graphs, Vision_media, Bio_chem | `inference time` |
| [2608.00657](https://arxiv.org/abs/2608.00657) | Causal Inference with Unstructured Treatments | Vision_media, NLP_lang, Bio_chem | `binary` |
| [2608.00667](https://arxiv.org/abs/2608.00667) | Band-Count Dense Modal Estimation with Fixed-Frequency Differentiable Resonator Refinement | RL, Vision_media, Bio_chem | `sparse` |
| [2608.00697](https://arxiv.org/abs/2608.00697) | Evolutionary Curriculum Learning Improves Biological Sequence Modeling | Vision_media, Bio_chem, Generative | `bit, progressive` |
| [2608.00716](https://arxiv.org/abs/2608.00716) | Generated Images Are Easier to Forget: A Machine Unlearning Perspective for Synthetic Image Detection | Theory_opt, Graphs, Vision_media | `bit, prun` |
| [2608.00815](https://arxiv.org/abs/2608.00815) | Paris as a 15-Minute City: An Explainable AI Perspective | Graphs, Vision_media, NLP_lang | `approximate, sparse` |
| [2608.00837](https://arxiv.org/abs/2608.00837) | Pruned BPE: Post-training Visibility Pruning and Token Reallocation for Byte Pair Encoding | Graphs, Vision_media, NLP_lang | `approximate, prun, pruning` |
| [2608.00852](https://arxiv.org/abs/2608.00852) | HyperODE: Zero-Shot Surrogate for Simulation and Inference of Dynamical Systems | Graphs, Vision_media, TimeSeries_signal | `accelerat, approximate, bit` |
| [2608.01005](https://arxiv.org/abs/2608.01005) | Hierarchical Solomonoff Induction: An Unbounded Machine Learning Model | Theory_opt, NLP_lang, Bio_chem | `exit` |
| [2608.01032](https://arxiv.org/abs/2608.01032) | The Fourth Quadrant: A Stylized View of Benign Misfitting | Theory_opt, Vision_media, Bio_chem | `bit` |
| [2608.01091](https://arxiv.org/abs/2608.01091) | Factorized AdaBoost.MH Achieves the Same Convergence Rate as AdaBoost.MH | Theory_opt, Graphs | `binary` |
| [2608.01151](https://arxiv.org/abs/2608.01151) | Learning-Based Stochastic Optimal Control with Infinite-Horizon Probabilistic Constraints | Theory_opt, RL, Vision_media | `approximate, bit, exit` |
| [2608.01217](https://arxiv.org/abs/2608.01217) | Amortizing the Calibration Triple: A Projection-Consistent Neural Operator for Local-Stochastic Volatility | Theory_opt, RL, Vision_media | `bit` |
| [2608.01320](https://arxiv.org/abs/2608.01320) | Dense Language Generation Made Simple: Deterministic, Randomized, and Multi-Order Algorithms | Theory_opt, Vision_media, NLP_lang | `bit` |
| [2608.01357](https://arxiv.org/abs/2608.01357) | Do Neural Networks Really Beat the Curse of Dimensionality? A Bit-Complexity View | Theory_opt, Bio_chem, Eval_bench | `binary, bit, exit, sparse` |
| [2608.01378](https://arxiv.org/abs/2608.01378) | When May a Model Replace the Experiment? Audits, Licenses, and the Price of Trust in Surrogate-Driven Design | Theory_opt, Vision_media, Bio_chem | `exit` |
| [2608.01388](https://arxiv.org/abs/2608.01388) | Why Formal Monitors Fail: Attack Distribution Entropy as a Coverage Bound for LTL-Based LLM Agent Safety | Theory_opt, Vision_media, TimeSeries_signal | `bit` |
| [2608.01428](https://arxiv.org/abs/2608.01428) | When Replanning Becomes the Bottleneck: Budgeted Replanning for Embodied Agents | Vision_media, NLP_lang, Science_eng | `bit, progressive, prun, pruning, token prun` |
| [2608.01460](https://arxiv.org/abs/2608.01460) | Conformalized Large Language Models under Configuration Shift | Theory_opt, Vision_media, NLP_lang | `quantiz` |
| [2608.01545](https://arxiv.org/abs/2608.01545) | Dominant Arm Identification with Mixing and Recycling Observed Samples | Theory_opt, RL, Vision_media | `exit` |
| [2608.01552](https://arxiv.org/abs/2608.01552) | Generalized Quadratic Gradient: A New Direction in Optimization via the Fusion of Positive-Definite Curvature Matrices and Gradients into A Unified Framework | Theory_opt, Vision_media, Bio_chem | `exit` |
| [2608.01586](https://arxiv.org/abs/2608.01586) | Statistical comparisons of time-series feature sets on classification tasks | Theory_opt, TimeSeries_signal, Eval_bench | `bit` |
| [2608.01599](https://arxiv.org/abs/2608.01599) | Latent-Regime Bias Auditing for Volatility Forecasting | Vision_media, TimeSeries_signal, Eval_bench | `bit` |
| [2608.01616](https://arxiv.org/abs/2608.01616) | Online Algorithms via Minimax and Posterior Matching | Theory_opt, Vision_media, Bio_chem | `bit` |
| [2608.01648](https://arxiv.org/abs/2608.01648) | Evaluating Forecasting Techniques for Hardware Errors on a Large-scale HPC System | Theory_opt, TimeSeries_signal, Science_eng | `sparse` |
| [2608.01740](https://arxiv.org/abs/2608.01740) | Disagree to Accelerate: Closing the Loop on Diffusion Feature Forecasts | Theory_opt, Vision_media, TimeSeries_signal | `accelerat, acceleration, skip` |
| [2608.01821](https://arxiv.org/abs/2608.01821) | DAVET: Denoising-Aware Visual Evidence Trajectory Allocation for Diffusion Vision-Language Models | Vision_media, NLP_lang, Bio_chem | `accelerat, acceleration, prun, pruning, token compress` |
| [2608.01845](https://arxiv.org/abs/2608.01845) | WorldDynCache: Risk-Controlled Latent Dynamics Approximation for Diffusion World Model | Theory_opt, Vision_media, NLP_lang | `approximate, bit, caching, skip` |
| [2608.02034](https://arxiv.org/abs/2608.02034) | Upper-Expectile Multi-Step Q-Learning for Off-Policy Reinforcement Learning | Theory_opt, RL, Vision_media | `accelerat` |
| [2608.02052](https://arxiv.org/abs/2608.02052) | Secrets Everywhere: Auditing Memorization in Mobility Prediction Models | Theory_opt, Vision_media, NLP_lang | `inference time` |
| [2608.02083](https://arxiv.org/abs/2608.02083) | A 2-Block Architecture for Real-Time EEG Gait Decoding: A Pilot Study | Theory_opt, Graphs, Vision_media | `binary, exit` |
| [2608.02176](https://arxiv.org/abs/2608.02176) | Randomized Algorithms for Learning Partitions with Near Optimal Query Complexity in Constant Rounds | Theory_opt, TimeSeries_signal, Bio_chem | `exit` |
| [2608.02181](https://arxiv.org/abs/2608.02181) | Start Classifying: Categorical Critics for LLM Reinforcement Learning | Theory_opt, RL, Vision_media | `binary, sparse` |
| [2608.02213](https://arxiv.org/abs/2608.02213) | Fast Discovery of Inclusion Dependencies with Desbordante | Theory_opt, Vision_media | `approximate, hash` |
| [2608.02289](https://arxiv.org/abs/2608.02289) | Extended Field of View Analysis for VideoGAN-based Trajectory Generation | Theory_opt, Graphs, Vision_media | `exit, inference time` |
| [2608.02352](https://arxiv.org/abs/2608.02352) | Qwen-CUA: Native Computer Use for (almost) Everything | RL, Vision_media, Bio_chem | `approximate, binary, sparse` |
| [2608.02507](https://arxiv.org/abs/2608.02507) | Beyond Modern Asymptotics for Log-Likelihood Ratios in Logistic Regression | Theory_opt, Science_eng | `binary, bit` |
| [2608.02533](https://arxiv.org/abs/2608.02533) | Optimal Unambiguous DNFs and Alon-Saks-Seymour | Theory_opt, Vision_media | `approximate, bit, exit` |
| [2608.02538](https://arxiv.org/abs/2608.02538) | Interaction Is Not Necessary for Order-Optimal 1-Bit Mean Estimation | Theory_opt, Bio_chem | `binary, bit, exit` |
| [2608.02575](https://arxiv.org/abs/2608.02575) | Pseudorandom Streams within Diffusion Models Act as Learnable Inputs That Affect Generation Quality | Theory_opt, Vision_media, Bio_chem | `approximate, bit` |
| [2608.02588](https://arxiv.org/abs/2608.02588) | The Condition-Number Barrier in Sparse Least Squares | Theory_opt, Graphs, Vision_media | `sparse` |
| [2608.02662](https://arxiv.org/abs/2608.02662) | Verifier-Guided Model Discovery for Physical Dynamical Systems with Pretrained Symbolic Transformers | Graphs, Vision_media, TimeSeries_signal | `bit, reduced` |
| [2608.02703](https://arxiv.org/abs/2608.02703) | ARCHead: Activation-Metric Residual Correction for Large Language Model Output Heads | Theory_opt, NLP_lang, Eval_bench | `bit, exit, quantiz` |
| [2608.02786](https://arxiv.org/abs/2608.02786) | Evaluation Blindness: How Silent Measurement Failures Corrupt AI Systems from Training to Deployment | RL, Vision_media, NLP_lang | `bit` |
| [2608.02804](https://arxiv.org/abs/2608.02804) | Detecting high-frequency brain disorder signals using dynamic mode decomposition from EEG | Vision_media, TimeSeries_signal, Bio_chem | `approximate, bit` |
| [2608.02826](https://arxiv.org/abs/2608.02826) | Improved Quantum Algorithms for Reinforcement Learning Under a Generative Model | Theory_opt, RL, Vision_media | `approximate, exit` |
| [2608.02845](https://arxiv.org/abs/2608.02845) | NOMADD: Numerical Optimization of Models Adapting to Data Drift | Theory_opt, TimeSeries_signal, Generative | `approximate, inference time` |
| [2608.02957](https://arxiv.org/abs/2608.02957) | Inverted Detection and Control in Steering Vectors | RL, Vision_media, NLP_lang | `bit, inference time` |
| [2608.02961](https://arxiv.org/abs/2608.02961) | Scaling an Autoregressive Transformer for Single-Cell Generation | Graphs, Vision_media, NLP_lang | `quantiz` |
| [2608.03017](https://arxiv.org/abs/2608.03017) | Paired Recipient-based Evaluation of Survival Prediction for Deceased Donor Kidney Transplants | Theory_opt, Vision_media, Bio_chem | `exit` |
| [2608.03086](https://arxiv.org/abs/2608.03086) | Automatic Patient-Specific Microwave Ablation Planning Accelerated by a Physics-Guided Deep Learning Model | Theory_opt, Vision_media, Bio_chem | `accelerat, approximate, reduced` |
| [2608.03095](https://arxiv.org/abs/2608.03095) | VIVID: A Culturally Grounded Benchmark Exposing the Figurative Language Gap in Vietnamese NLP | Theory_opt, Graphs, Vision_media | `bit, exit` |
| [2608.03108](https://arxiv.org/abs/2608.03108) | Convex-Hull-Neighborhood Smooth Dual Generalization: Controlling Local Correction Propagation in Offline RL | Theory_opt, RL, Vision_media | `approximate` |
| [2608.03123](https://arxiv.org/abs/2608.03123) | Trajectory-Guided Forget-Recover Network for Continual LLM Unlearning | Theory_opt, Graphs, Vision_media | `progressive` |
| [2608.03142](https://arxiv.org/abs/2608.03142) | Minimax-Optimal Semiparametric Contextual Dynamic Pricing with Multimodal Revenue | Theory_opt, Vision_media, NLP_lang | `binary, bit` |
| [2608.03223](https://arxiv.org/abs/2608.03223) | Agentic Reinforcement Learning with Self-Distilled Reward Shaping | RL, Vision_media, NLP_lang | `reduced, sparse` |
| [2608.03291](https://arxiv.org/abs/2608.03291) | The Tell-Tale Trace: Detecting Reasoning Failures in LLMs Using Chain-of-Thought Dynamics | Theory_opt, Vision_media, NLP_lang | `exit` |
| [2608.03294](https://arxiv.org/abs/2608.03294) | Provably Learning Multi-Head Attention with Queries | Theory_opt, NLP_lang | `approximate, bit` |
| [2608.03360](https://arxiv.org/abs/2608.03360) | Conformal risk control for model-form uncertainty in parametric non-intrusive reduced-order models | Theory_opt, Science_eng, Eval_bench | `reduced` |
| [2608.03382](https://arxiv.org/abs/2608.03382) | LLM-Derived Priors for Thompson Sampling in Cold-Start Comment Recommendation | Theory_opt, RL, Graphs | `sparse` |
| [2608.03437](https://arxiv.org/abs/2608.03437) | Dynamically Allocating Evaluation Effort for Model Ranking | Theory_opt, RL, Vision_media | `bit` |
| [2608.03439](https://arxiv.org/abs/2608.03439) | Quality Control Algorithms for Pattern Counting | Graphs, Science_eng | `approximate` |
| [2608.03562](https://arxiv.org/abs/2608.03562) | Robust General Utility for Reinforcement Learning | Theory_opt, RL, Vision_media | `approximate, bit` |
| [2608.03573](https://arxiv.org/abs/2608.03573) | SFT Conflicts, RL Coexists: A Theoretical and Empirical Analysis of Multi-Task Learning for LLMs | Theory_opt, RL, Vision_media | `approximate, bit, sparse` |
| [2608.03868](https://arxiv.org/abs/2608.03868) | GENESIS: Towards Explainable Causal Discovery | Theory_opt, RL, Graphs | `progressive` |
| [2608.03916](https://arxiv.org/abs/2608.03916) | Trajectory inference via Acceleration Matching | Theory_opt, Vision_media, NLP_lang | `accelerat, acceleration, bit` |
| [2608.03928](https://arxiv.org/abs/2608.03928) | Robust Low-Tubal-Rank Tensor Completion under Cross-Concentrated Sampling | Vision_media | `bit, sparse` |
| [2608.03930](https://arxiv.org/abs/2608.03930) | Logic Before Language: Pre-pretraining on Formal Derivations Fosters Skill Acquisition and Compressibility | Theory_opt, Vision_media, NLP_lang | `accelerat, prun, pruning` |
| [2608.04047](https://arxiv.org/abs/2608.04047) | Beyond the QBER Threshold: A Temporal QBER Based Machine Learning Framework for Multi Attack Detection in BB84 QKD | RL, Vision_media, TimeSeries_signal | `binary, bit, threshold-based` |
| [2608.04054](https://arxiv.org/abs/2608.04054) | Modality Agreement- and Conflict-Aware Prototype Hypergraph Learning for Multimodal Intent Understanding | Theory_opt, Graphs, Vision_media | `bit, progressive, sparse` |
| [2608.04057](https://arxiv.org/abs/2608.04057) | LaPrune: Controllable Differentiable Sparsity at Million Scale | Science_eng | `binary, prun, sparse` |
| [2608.04113](https://arxiv.org/abs/2608.04113) | Out-Of-The-Loop Multi-Fidelity Bayesian Optimization | Theory_opt, Vision_media, Bio_chem | `bit` |
| [2608.04149](https://arxiv.org/abs/2608.04149) | Sublogarithmic Swap Regret in Multiplayer General-Sum Games via Hybrid Regularization | Theory_opt, Graphs, Vision_media | `approximate, bit` |
| [2608.04180](https://arxiv.org/abs/2608.04180) | A Comparative Study of Feature Selection Methods for EHR Diagnosis Codes in Opioid Use Disorder Prediction | NLP_lang, TimeSeries_signal, Bio_chem | `sparse` |
| [2608.04227](https://arxiv.org/abs/2608.04227) | Random features for Grassmannian kernel approximation with bounded rank-one projections | Theory_opt, Vision_media, NLP_lang | `approximate, binary, bit` |
| [2608.04252](https://arxiv.org/abs/2608.04252) | Dynamical Lie Algebras Cannot Describe Shallow QAOA: Cragged Terrains, Barren Plateaus, and Empirical Hardness Models | Theory_opt, Graphs, Vision_media | `approximate` |
| [2608.04288](https://arxiv.org/abs/2608.04288) | Sample Complexity of Multicalibration for Multilevel Properties | Theory_opt | `binary, exit` |
| [2608.04305](https://arxiv.org/abs/2608.04305) | Adaptive Finite-Budget Training for CVaR Risk-Aware Q-Learning | RL, Vision_media, Bio_chem | `approximate, bit, progressive` |
| [2608.04310](https://arxiv.org/abs/2608.04310) | ArborEnum: Decision Tree Rashomon Sets over Continuous Features | Theory_opt, Vision_media | `approximate, exit, progressive` |
| [2608.04324](https://arxiv.org/abs/2608.04324) | Efficient Online Lexicographic Generalized Low-Rank Matrix Bandits | Theory_opt, RL, Graphs | `exit, reduced` |
| [2608.04333](https://arxiv.org/abs/2608.04333) | Cost-Aware Multi-Objective Bandits: Theory and Application to Budgeted LLM Configuration Evaluation | Theory_opt, Vision_media, NLP_lang | `exit` |
| [2608.04496](https://arxiv.org/abs/2608.04496) | DIVE: Dynamic Iterative Visual Evidence Construction for Efficient Vision-Language Models | Vision_media, NLP_lang, Eval_bench | `prun, pruning, token prun` |
| [2608.04552](https://arxiv.org/abs/2608.04552) | Relational Response Fields: A General Theory of Black-Box LLM Response Consistency and Recovery | Theory_opt, RL, Graphs | `sparse` |
| [2608.04593](https://arxiv.org/abs/2608.04593) | Rethinking Reservoir Pruning: A Dynamical Perspective for Echo State Networks | Theory_opt, Vision_media, TimeSeries_signal | `prun, pruning` |
| [2608.04607](https://arxiv.org/abs/2608.04607) | On MUON optimization: From non-convergence to an error analysis with Polar Express and the Newton-Schulz polynomial from implementations | Theory_opt, Vision_media, NLP_lang | `accelerat, binary, bit` |
| [2608.04651](https://arxiv.org/abs/2608.04651) | Active Learning Guided Design Space Refinement for Scalable Multi-Objective Bayesian Optimization in Materials Discovery | Theory_opt, Vision_media, Bio_chem | `accelerat, approximate, reduced` |
| [2608.04686](https://arxiv.org/abs/2608.04686) | The Sample Complexity of Distributionally Robust PAC Learning under Cressie--Read Divergences | Theory_opt, NLP_lang | `exit` |
| [2608.04777](https://arxiv.org/abs/2608.04777) | IMFACT: Counterfactual Explanations for Time Series via Intrinsic Mode Function Substitution | Theory_opt, Vision_media, TimeSeries_signal | `progressive` |
| [2608.04827](https://arxiv.org/abs/2608.04827) | Intrinsic-Hybrid Latent Diffusion Models for Generative Modeling on Unknown Manifolds | Theory_opt, Vision_media, Bio_chem | `approximate, sparse` |
| [2608.04964](https://arxiv.org/abs/2608.04964) | WorldCycle: Self-Verifiable Reinforcement Learning for Long-Horizon Video World Models | RL, Vision_media, TimeSeries_signal | `bit` |
| [2608.05084](https://arxiv.org/abs/2608.05084) | Learning When to Stop: Prefix-Optimal Dynamic Diffusion Policies for Continuous Control | RL, Vision_media, Bio_chem | `approximate, early stopping` |
| [2608.05103](https://arxiv.org/abs/2608.05103) | Multimodal Spatiotemporal Atmospheric Data Assimilation with Latent Video Flow-matching | Theory_opt, Vision_media, TimeSeries_signal | `sparse` |
| [2608.05110](https://arxiv.org/abs/2608.05110) | Representational separation between unitary and channel quantum generative models via shared classical randomness at shallow depth | Theory_opt, RL, Vision_media | `bit` |
| [2608.05115](https://arxiv.org/abs/2608.05115) | Robust and Efficient Motion Reasoning for Privacy-Aware Classroom Incident Recognition | RL, Vision_media, Bio_chem | `accelerat, acceleration` |
| [2608.05127](https://arxiv.org/abs/2608.05127) | SSTQ:Privacy-Preserving Vector Quantization via Subsampled Stochastic TurboQuant | Theory_opt, Vision_media, Eval_bench | `bit, quantiz` |
| [2608.05240](https://arxiv.org/abs/2608.05240) | One Qubit Can Beat One Bit: Quantum Advantage for Post-Training Quantization | NLP_lang, Science_eng | `binary, bit, quantiz` |
| [2608.05419](https://arxiv.org/abs/2608.05419) | Perturbation Sensitivity at Convergence: A Simple Signal for Identifying Spuriously Correlated Samples | Theory_opt, TimeSeries_signal | `early-stopping` |
| [2608.05454](https://arxiv.org/abs/2608.05454) | Hybrid Probabilistic Zonotopes for Identifiable and Refinable Predictive Uncertainty | Theory_opt, RL, Vision_media | `binary, inference time` |
| [2608.05464](https://arxiv.org/abs/2608.05464) | Effective pruning of task-trained recurrent neural networks using noisy fluctuations and connection rescaling | Bio_chem | `prun, pruning` |
| [2608.05502](https://arxiv.org/abs/2608.05502) | An Inertial Block Proximal Linearized Method with Adaptive Momentum for Nonconvex and Nonsmooth Optimization | Theory_opt, Vision_media, Bio_chem | `accelerat, sparse` |
| [2608.05646](https://arxiv.org/abs/2608.05646) | RASP-QAOA: Resource-Aware Per-Instance Selection for Exact QAOA Simulation | Theory_opt, RL, Graphs | `exit` |
| [2608.05724](https://arxiv.org/abs/2608.05724) | Sparse Mutual Information Graph Averaging for Improving Random Indexing Embeddings | Graphs, Vision_media, NLP_lang | `binary, prun, pruning, skip, sparse` |
| [2608.05727](https://arxiv.org/abs/2608.05727) | LILAC: An Idempotent Neural Speech Codec | Vision_media, NLP_lang, TimeSeries_signal | `bit` |
| [2608.05728](https://arxiv.org/abs/2608.05728) | Engram-E2VID: Reference-Based Event-to-Video Reconstruction via Generative Activation of Appearance Engrams | Theory_opt, Vision_media, NLP_lang | `progressive, sparse` |
| [2608.05782](https://arxiv.org/abs/2608.05782) | VSMP-IMU: Video-Grounded Semantic Motion Programs for Sensor-Aware Synthetic IMU Generation | Vision_media, NLP_lang, TimeSeries_signal | `reduced` |
| [2608.05954](https://arxiv.org/abs/2608.05954) | Training a Conditioned Video Game Agent on a VLM Annotated Dataset | RL, Vision_media, NLP_lang | `sparse` |
| [2608.06004](https://arxiv.org/abs/2608.06004) | Do Tabular Foundation Models Agree with Themselves? | Theory_opt, Vision_media, Eval_bench | `approximate` |
| [2608.06023](https://arxiv.org/abs/2608.06023) | BioKD: Selective Physiology-to-Video Knowledge Distillation via Reliability Gate for Emotion Recognition | Graphs, Vision_media, TimeSeries_signal | `inference time, inference-time, progressive` |
| [2608.06107](https://arxiv.org/abs/2608.06107) | Kastor: An efficient fine-tuning strategy for generative emulation of PDE simulations | Theory_opt, Vision_media, TimeSeries_signal | `accelerat` |
| [2608.06195](https://arxiv.org/abs/2608.06195) | Handling Missing Data in Probabilistic Regression Trees | Vision_media, Eval_bench | `bit, reduced` |
| [2608.06250](https://arxiv.org/abs/2608.06250) | Minimax Optimal Early-Stopped Gradient Descent for Gaussian Mixture Classification | Theory_opt, Science_eng | `bit, early stopping` |
| [2608.06262](https://arxiv.org/abs/2608.06262) | Hypothesis Testing with Conditional Queries: Learnability and the Value of Interaction | Theory_opt, Vision_media, Eval_bench | `exit` |
| [2608.06283](https://arxiv.org/abs/2608.06283) | The Tamed Subgradient Unadjusted Langevin Algorithm beyond Convexity | Theory_opt, Vision_media, Generative | `exit` |
| [2608.06340](https://arxiv.org/abs/2608.06340) | Scalable estimation of VARMA models | Theory_opt, TimeSeries_signal, Eval_bench | `sparse` |
| [2608.06363](https://arxiv.org/abs/2608.06363) | An Optimal Agnostic PAC Algorithm | Theory_opt, Vision_media | `binary, exit` |
