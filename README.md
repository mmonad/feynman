# Feynman Lectures on LLM Adaptation

A series of private 1-on-1 tutoring sessions in the style of Richard Feynman, covering how large language models adapt their behavior — from prompting to fine-tuning to cutting-edge parameter-efficient methods, composable AI review systems, cryptographic protocols, and the statistical foundations of machine learning.

## Teaching Approach

- First-principles reasoning with vivid, mechanism-based analogies
- One concept at a time, with conceptual questions to verify understanding
- Socratic method: wrong answers are met with new analogies, not corrections
- Analogies adjusted to the student's learning style (mechanical/engineering-oriented)

## Shared Vocabulary (Analogies Used Throughout)

| Analogy | Concept |
|---|---|
| **The Sculpture** | Pre-trained model weights — carved during training, frozen afterward |
| **The Stone** | Parametric memory (weights) — permanent knowledge |
| **The Water** | Contextual memory (prompts) — temporary, flows through the sculpture |
| **The Chisel** | Training / gradient descent — reshapes the stone |
| **The Ballroom Musician** | The model's ability to read context and adapt output |
| **The Piano** | Pre-trained model; tuning = fine-tuning specific strings |
| **The Bottleneck** | LoRA's low-rank constraint — compressing adaptation through r dimensions |
| **The Attachment** | LoRA adapter — bolted onto the sculpture, removable |
| **The TV Signal** | Pre-trained weights as a complex broadcast; fine-tuning correction as a few knobs |
| **The Dart Board** | Bias-variance tradeoff — cluster center (bias) vs spread (variance) vs shaking wall (noise) |
| **Eigenvectors / PCA** | Orthogonal adapter training — decomposing useful adaptation into independent components |
| **The Diagonal Bug** | Interaction effects invisible to orthogonal specialists looking along single axes |
| **The Ballroom Crowd vs Committee** | Unstructured multi-reviewer output vs architected review pipeline |
| **The Diamond vs Circle** | L1 vs L2 constraint geometry — corners induce sparsity, smooth surfaces don't |
| **The Orange Peel** | Curse of dimensionality — in high dimensions, all volume is in the skin |

## Course Structure

### Course 1: In-Context Learning vs Fine-Tuning
*5 lessons — What are the two fundamental ways to change an LLM's behavior?*

1. [What Does an LLM Actually "Know"?](course-1-icl-vs-finetuning/lesson-1-what-does-an-llm-know.md) — Parametric vs contextual memory; the sculpture analogy
2. [In-Context Learning: The Art of the Reminder](course-1-icl-vs-finetuning/lesson-2-in-context-learning.md) — Pattern activation, not learning; the ballroom musician
3. [Fine-Tuning: Rewiring the Brain](course-1-icl-vs-finetuning/lesson-3-fine-tuning.md) — Picking up the chisel; catastrophic forgetting; the tug-of-war
4. [The Great Trade-Off](course-1-icl-vs-finetuning/lesson-4-the-great-tradeoff.md) — When to pour water vs carve stone; includes interlude on overfitting & dataset guidelines
5. [The Frontier: Where the Line Gets Blurry](course-1-icl-vs-finetuning/lesson-5-the-frontier.md) — The spectrum between ICL and fine-tuning; includes interlude on transformer weight anatomy (MLP vs attention)

### Course 2: LoRA Deep Dive
*5 lessons — How do you efficiently adapt a frozen model?*

1. [The Core Intuition: Why Low-Rank?](course-2-lora/lesson-1-why-low-rank.md) — Low-rank weight updates; A x B decomposition; includes bias-variance refresher
2. [The Mechanics: How LoRA Actually Works](course-2-lora/lesson-2-the-mechanics.md) — Parallel path, bottleneck, initialization, scaling factor alpha, merge vs swap
3. [The Hyperparameters That Matter](course-2-lora/lesson-3-hyperparameters.md) — Rank, alpha, layer targeting, learning rate; the practical starting recipe
4. [LoRA in Practice](course-2-lora/lesson-4-lora-in-practice.md) — QLoRA, adapter merging, multi-adapter serving, production decision tree
5. [The Frontier of Parameter-Efficient Methods](course-2-lora/lesson-5-parameter-efficient-methods.md) — Prompt tuning, prefix tuning, adapter layers, DoRA, MoLoRA; why LoRA won

### Course 3: Orthogonal Adapters & Composable Code Review Committees
*5 lessons — How do you build diverse AI review systems with guaranteed unique perspectives?*

1. [The Blind Spot Problem](course-3-orthogonal-adapters-code-review/lesson-1-the-blind-spot-problem.md) — Why LLMs reviewing LLM code has inherent limits; the shared manifold problem; diversity hypothesis
2. [Orthogonality: What It Means and Why It Guarantees Diversity](course-3-orthogonal-adapters-code-review/lesson-2-orthogonality.md) — Projection constraints; the diagonal bug problem; includes interlude on training data & eigenvector analogy
3. [Building the Committee](course-3-orthogonal-adapters-code-review/lesson-3-building-the-committee.md) — Three-tier architecture (specialists, composition, prioritization); bootstrapping via mutation testing
4. [When Reviewers Disagree](course-3-orthogonal-adapters-code-review/lesson-4-when-reviewers-disagree.md) — Three categories of disagreement; confidence calibration; the disagreement matrix; composition model bias
5. [Quis Custodiet Ipsos Custodes?](course-3-orthogonal-adapters-code-review/lesson-5-quis-custodiet-ipsos-custodes.md) — Third-order blind spots; defense in depth; the Kegan developmental parallel

### Course 4: From Architecture to Enterprise
*5 lessons — How do you build a defensible AI startup on the verification committee thesis?*

1. [The Generalization: From Code Review to Universal Verification Committees](course-4-from-architecture-to-enterprise/lesson-1-the-generalization.md) — Domain suitability framework; scoring domains; why crypto is the best first vertical
2. [The TAM: Mapping the Opportunity Space](course-4-from-architecture-to-enterprise/lesson-2-the-tam.md) — Bottom-up TAM, continuous monitoring market creation, wedge strategy, data flywheel, pricing paradox
3. [Formal Verification of Cryptographic Protocols with Lean 4](course-4-from-architecture-to-enterprise/lesson-3-formal-verification-lean4.md) — The four levels of correctness; why LLMs fail at Level 4; orthogonal adapters for verification gaps; mutation of specs and proofs
4. [The Moat Question: What Frontier Labs Can't Copy](course-4-from-architecture-to-enterprise/lesson-4-the-moat-question.md) — Five layers of moat; horizontal tax; reputation ratchet; winner-take-most dynamics
5. [Building the Company: Architecture as Strategy](course-4-from-architecture-to-enterprise/lesson-5-building-the-company.md) — Architecture-to-strategy mapping; go-to-market; team; fundraising; 18-month roadmap; the deepest risk

### Course 5: The Self-Improving Harness
*6 lessons — How do you build a system that bootstraps from cloud dependence to local autonomy?*

1. [The Harness: Orchestrating Cloud and Local](course-5-self-improving-harness/lesson-1-the-harness.md) — Cascade routing, cost-quality curve, sending failures to the teacher
2. [Distillation from First Principles](course-5-self-improving-harness/lesson-2-distillation.md) — Dark knowledge, soft targets, temperature; selective distillation into LoRA specialists; sequence-level distillation from cloud APIs
3. [Distillation into LoRA: Merging Teacher Knowledge with Task Adaptation](course-5-self-improving-harness/lesson-3-distillation-into-lora.md) — Rank expansion for dual signals; four combination strategies; LR/epoch/sampling balance
4. [Micro Fine-Tuning: Learning While Serving](course-5-self-improving-harness/lesson-4-micro-fine-tuning.md) — Quality filter, replay buffer, micro learning rate, EWC anchoring, validation gate, three firewalls against model collapse
5. [The Full Architecture](course-5-self-improving-harness/lesson-5-full-architecture.md) — Seven data flows, graceful degradation, build sequence, composition model co-evolution
6. [The Bootstrap Paradox and the Economic Inflection](course-5-self-improving-harness/lesson-6-bootstrap-paradox.md) — Learning starvation, sawtooth improvement, proactive red-teaming, when student surpasses teacher, AGI as civilization

### Course 6: Cryptography from First Principles
*5 lessons — Math foundations through core crypto primitives (for understanding the BABE protocol)*

1. [The Lock and Key Universe](course-6-crypto-foundations/lesson-1-lock-and-key-universe.md) — Security parameter, negligible functions, PPT adversaries, security games, why negligible must be exponential
2. [Finite Fields and Modular Arithmetic](course-6-crypto-foundations/lesson-2-finite-fields.md) — Clock arithmetic, why primes give division, generators, discrete log problem
3. [Hash Functions and the Random Oracle](course-6-crypto-foundations/lesson-3-hash-functions-random-oracle.md) — Hash properties, ROM, notation boot camp (sampling, probability, oracle access)
4. [Digital Signatures and Security Games](course-6-crypto-foundations/lesson-4-digital-signatures.md) — EUF-CMA, Lamport one-time signatures, the Lamport=GC coincidence, dot notation for oracles
5. [Polynomials and Arithmetic Circuits](course-6-crypto-foundations/lesson-5-polynomials-arithmetic-circuits.md) — Schwartz-Zippel, R1CS, the polynomial bridge to Groth16, CRS and trusted setup

### Course 7: The Cryptographic Toolkit
*5 lessons — The algebraic machinery BABE is built from*

1. [Elliptic Curves](course-7-crypto-toolkit/lesson-1-elliptic-curves.md) — Point addition, scalar multiplication, ECDLP, BN254, implicit notation [x]_s
2. [Bilinear Pairings](course-7-crypto-toolkit/lesson-2-bilinear-pairings.md) — The pairing map, bilinearity, Groth16 verification equation, one-shot limitation
3. [SNARKs and Groth16](course-7-crypto-toolkit/lesson-3-snarks-groth16.md) — Complete Groth16 flow (Gen/Prove/Verify), knowledge soundness, extractors, role in BABE
4. [Garbled Circuits](course-7-crypto-toolkit/lesson-4-garbled-circuits.md) — Wire labels, gate garbling, evaluation, free-XOR, half-gates, adaptive privacy
5. [Witness Encryption](course-7-crypto-toolkit/lesson-5-witness-encryption.md) — Encrypt under NP statement, BABE's pairing-based WE, extractable security, the Lemma 10 fix, WE+GC=BABE
### Course 8: BABE — The Protocol and Its Security
*5 lessons — The complete protocol, its security proof, and the Lean mechanization*

1. [Bitcoin as a Cryptographic Platform](course-8-babe-protocol/lesson-1-bitcoin-platform.md) — UTXO model, locking scripts, 6-transaction graph, unstoppable transactions
2. [The BABE Construction](course-8-babe-protocol/lesson-2-babe-construction.md) — WE+GC split, randomized encodings, DRE, linearization by lifting, 1000x size reduction
3. [The Security Proof](course-8-babe-protocol/lesson-3-security-proof.md) — Robustness, knowledge soundness, hybrid arguments, reduction chain, why four proof assistants
4. [The Mechanization](course-8-babe-protocol/lesson-4-mechanization.md) — Four proof assistants, axiom boundaries, audit findings, trust surface, stopping criterion
5. [The Full Circle](course-8-babe-protocol/lesson-5-full-circle.md) — Recursive verification, the complete map, biological parallels, "billions of years of pre-training followed by a lifetime of LoRAs"

### Course 9: Remedial — Strengthening the Weak Spots
*5 lessons — Focused drilling on reductions, precise definitions, and component boundaries*

1. [Reductions: The Technique](course-9-remedial/lesson-1-reductions.md) — Four worked examples, the three-step recipe, student constructs a case-split reduction
2. Reduction Drills *(upcoming)*
3. [The Confusion Matrix](course-9-remedial/lesson-3-confusion-matrix.md) — Six confused concept pairs with precise distinctions and tests
4. [The BABE Component Map](course-9-remedial/lesson-4-component-map.md) — Which operation belongs to which system, the math type test
5. [Remedial Exam](course-9-remedial/lesson-5-remedial-exam.md) — 15 focused questions, 12/15 clean, all major gaps closed

### Course 10: Statistical Learning Theory
*5 lessons — The mathematical foundations of generalization, estimation, and high-dimensional learning*

1. [Bias, Variance, and Empirical Risk Minimization](course-10-statistical-learning-theory/lesson-1-bias-variance-and-erm.md) — Bias-variance decomposition derivation, dart board analogy, ERM pathologies (overfitting, distribution shift)
2. [Regularization — Ridge Regression and Sparsity](course-10-statistical-learning-theory/lesson-2-regularization-ridge-and-sparsity.md) — Ridge derivation with eigenvalue analysis, Bayesian interpretation, L1 sparsity (geometric and subdifferential arguments)
3. [VC Dimension and PAC Learning](course-10-statistical-learning-theory/lesson-3-vc-dimension-and-pac-learning.md) — Shattering, VC dimension examples, generalization bounds, PAC framework, sample complexity
4. [MLE, MAP, and Consistency](course-10-statistical-learning-theory/lesson-4-mle-map-and-consistency.md) — MLE consistency conditions, failure cases (mixtures, Neyman-Scott), MAP as regularized MLE, prior-penalty duality
5. [Dimensionality and KL Divergence](course-10-statistical-learning-theory/lesson-5-dimensionality-and-kl-divergence.md) — Curse of dimensionality (orange peel, distance concentration), KL asymmetry, forward vs reverse KL, JS divergence

### Course 11: Optimization in ML
*5 lessons — How do optimizers navigate loss landscapes, and why does the "wrong" method often win?*

1. [SGD, Generalization, and Saddle Points](course-11-optimization/lesson-1-sgd-generalization-and-saddle-points.md) — SGD noise as implicit regularization, flat minima, SDE approximation, saddle points vs local minima in high dimensions
2. [Adam vs SGD, and Convergence Guarantees](course-11-optimization/lesson-2-adam-vs-sgd-and-convergence.md) — Adam update rules, sharp minima with adaptive methods, AMSGrad, convergence conditions (L-smoothness, strong convexity, PL condition)
3. [Gradient Pathologies](course-11-optimization/lesson-3-gradient-pathologies.md) — Vanishing gradients (chain rule, sigmoid, ReLU, residuals, init), exploding gradients (clipping, batch norm, LSTM gating, transformers)
4. [Second-Order Methods](course-11-optimization/lesson-4-second-order-methods.md) — Hessian and curvature, Newton's method, why O(n²) is infeasible, L-BFGS, Hessian-free, natural gradient, Fisher information, KFAC
5. [Loss Landscape Geometry](course-11-optimization/lesson-5-loss-landscape-geometry.md) — Sharp vs flat minima, PAC-Bayes, Dinh et al. controversy, SAM, line search vs schedules, warmup mystery for transformers

### Course 12: Probabilistic ML & Inference
*5 lessons — EM, MCMC, variational inference, Bayesian foundations, and when the machinery breaks*

1. [The EM Algorithm](course-12-probabilistic-ml/lesson-1-em-algorithm.md) — MLE with latent variables, Jensen's inequality, ELBO, E-step/M-step, monotonic improvement, GMM example, failure modes (local optima, singular covariances, intractable E-step)
2. [MCMC Methods](course-12-probabilistic-ml/lesson-2-mcmc-methods.md) — Metropolis-Hastings (propose-accept/reject, detailed balance, proposal tuning), Gibbs sampling (full conditionals, MH with acceptance=1, correlated variables problem), VI overview (optimization vs sampling, ELBO, mean-field, speed vs accuracy tradeoffs)
3. [The ELBO and VI Bias](course-12-probabilistic-ml/lesson-3-elbo-and-vi-bias.md) — Full ELBO derivation, log P(x) = ELBO + KL(Q||P), reverse KL is mode-seeking, mean-field misses correlations, normalizing flows as fix
4. [Bayesian Foundations](course-12-probabilistic-ml/lesson-4-bayesian-foundations.md) — Exchangeability, de Finetti's theorem, Bayesian justification for parametric models, Dirichlet Process (CRP, stick-breaking), Gaussian Process (kernel as function prior)
5. [Priors and Posterior Collapse](course-12-probabilistic-ml/lesson-5-priors-and-posterior-collapse.md) — Prior sensitivity, Bernstein-von Mises, Jeffreys prior, posterior collapse in VAEs (powerful decoder problem, KL annealing, free bits)

### Course 13: Deep Learning Theory
*5 lessons — Transformers, residual connections, overparameterization, scaling laws, and the phenomena that defy classical intuition (Q31–Q40)*

1. [Transformers vs RNNs](course-13-deep-learning-theory/lesson-1-transformers-vs-rnns.md) — Sequential bottleneck, hidden state compression, parallel processing, attention O(n²d) complexity, sparse/linear/FlashAttention solutions
2. [Positional Encoding and Normalization](course-13-deep-learning-theory/lesson-2-positional-encoding-and-normalization.md) — Permutation equivariance without PE, sinusoidal/learned/RoPE/ALiBi, layer norm vs batch norm, pre-norm vs post-norm, RMSNorm
3. [Residual Connections and Overparameterization](course-13-deep-learning-theory/lesson-3-residuals-and-overparameterization.md) — y=F(x)+x gradient math, 2^L paths, ensemble interpretation, ODE connection, interpolation threshold, NTK, implicit regularization
4. [Lottery Tickets and Double Descent](course-13-deep-learning-theory/lesson-4-lottery-tickets-and-double-descent.md) — Sparse subnetworks matching full performance, iterative magnitude pruning, supermasks, three regimes of double descent, epoch-wise double descent
5. [Scaling Laws and Why ReLU Dominates](course-13-deep-learning-theory/lesson-5-scaling-laws-and-relu.md) — Kaplan/Chinchilla power laws, compute-optimal frontier, sigmoid→tanh→ReLU→GELU arc, dying ReLU, why GELU won in transformers

### Course 14: Computer Vision & NLP
*5 lessons — From convolution priors to tokenization failures and hallucination (Q41–Q50)*

1. [Why Convolution Works](course-14-vision-and-nlp/lesson-1-why-convolution-works.md) — Locality and stationarity priors, weight sharing (FC N⁴ vs conv k²), inductive bias, translation equivariance vs invariance, proof sketch
2. [Pooling and Dilated Convolutions](course-14-vision-and-nlp/lesson-2-pooling-and-dilated-convolutions.md) — Receptive field growth, max vs avg pooling, strided conv, global average pooling, dilated convolutions, WaveNet, gridding artifact
3. [Feature Hierarchies and Word2Vec](course-14-vision-and-nlp/lesson-3-feature-hierarchies-and-word2vec.md) — Edges→textures→parts→objects, GradCAM, transfer learning, Skip-gram/CBOW, softmax bottleneck, king−man+woman≈queen
4. [Negative Sampling and Pretraining](course-14-vision-and-nlp/lesson-4-negative-sampling-and-pretraining.md) — Binary classification reframing, noise distribution P(w)∝freq^(3/4), NCE connection, BERT vs GPT (bidirectional vs causal), convergence toward autoregressive scale
5. [Tokenization and Hallucination](course-14-vision-and-nlp/lesson-5-tokenization-and-hallucination.md) — Character/word/subword (BPE), arithmetic and multilingual impact, byte-level BPE, five causes of hallucination, mitigation strategies (RAG, CoT, calibration)

### Course 15: Reinforcement Learning
*5 lessons — Bellman equations to deep RL instability, exploration, and sample efficiency (Q51–Q60)*

1. [Bellman Equations and Dynamic Programming](course-15-reinforcement-learning/lesson-1-bellman-and-dynamic-programming.md) — Value function derivation, Bellman recursion, optimality equation, Q-function, policy iteration vs value iteration, contraction mapping convergence
2. [Q-Learning Instability and Exploration](course-15-reinforcement-learning/lesson-2-q-learning-and-exploration.md) — Tabular convergence, three sources of DQN instability (correlated samples, non-stationary targets, maximization bias), the deadly triad, ε-greedy vs UCB vs Thompson Sampling
3. [Actor-Critic and Credit Assignment](course-15-reinforcement-learning/lesson-3-actor-critic-and-credit-assignment.md) — REINFORCE variance, advantage function, TD error as advantage estimate, A2C/PPO/SAC, temporal credit assignment, eligibility traces, λ-return, HER
4. [On-Policy vs Off-Policy and Function Approximation Issues](course-15-reinforcement-learning/lesson-4-on-off-policy-and-function-approximation.md) — Behavioral vs target policy, importance sampling variance, PPO clipping, SAC entropy, convergence hierarchy (tabular → linear → neural), Baird's counterexample
5. [Reward Shaping and Sample Efficiency](course-15-reinforcement-learning/lesson-5-reward-shaping-and-sample-efficiency.md) — Potential-based shaping (Ng et al. theorem), telescoping argument, reward hacking, five root causes of sample inefficiency, model-based RL, offline RL

### Course 16: Generalization Theory & Graph Neural Networks
*5 lessons — Why deep nets generalize, compression and stability views, information bottleneck, and the expressivity limits of GNNs (Q61–Q70)*

1. [Why Deep Nets Generalize](course-16-generalization-and-gnns/lesson-1-why-deep-nets-generalize.md) — Zhang et al. random labels puzzle, SGD's simplicity bias, function space perspective, implicit regularization (min-norm, min nuclear norm, LR as regularizer, early stopping)
2. [Compression and Stability](course-16-generalization-and-gnns/lesson-2-compression-and-stability.md) — MDL principle, compression bounds, noise stability, PAC-Bayes, algorithmic stability, Hardt et al. SGD stability proof, connection to differential privacy
3. [Information Bottleneck and GNN Oversmoothing](course-16-generalization-and-gnns/lesson-3-information-bottleneck-and-oversmoothing.md) — Tishby's IB framework, two-phase training, Saxe et al. criticism, oversmoothing as low-pass filtering, eigenvalue convergence, DropEdge/PairNorm fixes
4. [Message Passing and Spectral vs Spatial GNNs](course-16-generalization-and-gnns/lesson-4-message-passing-and-spectral-gnns.md) — MPNN framework, 1-WL expressivity ceiling, graph Laplacian Fourier domain, ChebNet, GCN as 1st-order Chebyshev, spectral vs spatial comparison
5. [Graph Isomorphism and GNN Expressivity](course-16-generalization-and-gnns/lesson-5-graph-isomorphism-and-expressivity.md) — GI complexity status, Babai's algorithm, GIN (injective aggregation = 1-WL), beyond-1-WL strategies (higher-order, random features, positional encodings)

### Course 17: Causality & Practical ML
*5 lessons — Causal inference, Simpson's paradox, class imbalance, data leakage, and the engineering of reliable ML systems (Q71–Q80)*

1. [Causation Formalized](course-17-causality-and-practical-ml/lesson-1-causation-formalized.md) — Pearl's SCMs, structural equations, DAGs, do-operator, adjustment formula, backdoor criterion, frontdoor criterion
2. [Instrumental Variables and Counterfactuals](course-17-causality-and-practical-ml/lesson-2-instruments-and-counterfactuals.md) — Unmeasured confounders, IV conditions, 2SLS, weak instruments, potential outcomes Y(1)/Y(0), fundamental problem of causal inference, Pearl's three-level ladder
3. [Simpson's Paradox and Class Imbalance](course-17-causality-and-practical-ml/lesson-3-simpsons-paradox-and-class-imbalance.md) — Berkeley admissions, aggregate vs stratify (causal structure decides), SMOTE, focal loss, AUPRC vs accuracy, threshold tuning
4. [Data Leakage and Feature Engineering](course-17-causality-and-practical-ml/lesson-4-data-leakage-and-feature-engineering.md) — Target/temporal/train-test leakage, detection and prevention, feature engineering techniques, feature selection methods
5. [Cross-Validation and Hyperparameter Tuning](course-17-causality-and-practical-ml/lesson-5-cross-validation-and-hyperparameter-tuning.md) — K-fold, stratified, time series split, nested CV, grid vs random (Bergstra & Bengio), Bayesian optimization, learning rate as most important hyperparameter

### Course 18: Systems, Robustness & the Frontier
*6 lessons — Distributed training, memory/precision, adversarial robustness, fairness, privacy, generative models, RLHF, alignment, and open questions (Q81–Q100)*

1. [Distributed Training](course-18-systems-robustness-frontier/lesson-1-distributed-training.md) — AllReduce communication overhead, sync vs async SGD, linear scaling rule, LARS/LAMB, data/tensor/pipeline parallelism, ZeRO, 3D parallelism
2. [Memory, Precision & Latency](course-18-systems-robustness-frontier/lesson-2-memory-precision-latency.md) — Memory budget (params, gradients, optimizer, activations), gradient checkpointing, FP16/BF16 mixed precision, GPTQ/AWQ quantization, continuous batching, speculative decoding, paged attention, GQA
3. [Adversarial Robustness](course-18-systems-robustness-frontier/lesson-3-adversarial-robustness.md) — Goodfellow linearity hypothesis, FGSM/PGD attacks, features-vs-bugs debate, adversarial training min-max, accuracy-robustness tradeoff, certified defenses, randomized smoothing
4. [Fairness and Bias](course-18-systems-robustness-frontier/lesson-4-fairness-and-bias.md) — Demographic parity, equalized odds, calibration, Chouldechova impossibility theorem, bias taxonomy, disaggregated metrics, counterfactual fairness, proxy variables
5. [Privacy and Generative Models](course-18-systems-robustness-frontier/lesson-5-privacy-and-generative-models.md) — (ε,δ)-differential privacy, DP-SGD, composition theorem, federated learning, GANs vs diffusion (stability, mode coverage, conditioning), why diffusion training is stable
6. [RLHF, Alignment & the Frontier](course-18-systems-robustness-frontier/lesson-6-rlhf-alignment-and-frontier.md) — RLHF pipeline (SFT→reward model→PPO), DPO, Goodhart's law, mesa-optimization, Constitutional AI, CLIP, test-time compute, mechanistic interpretability, scaling laws, open questions

### Course 19: LoRA Training in Practice
*5 lessons — Hands-on adapter training, serving, and deployment*

1. [The Toolchain](course-19-lora-practice/lesson-1-toolchain.md) — Qwen3.5, Unsloth, PEFT, vLLM multi-LoRA serving, 2026 best practices (rsLoRA, EVA, all-linear targeting)
2. Training Your First Adapter *(upcoming)*
3. Multi-Adapter Training *(upcoming)*
4. Serving *(upcoming)*
5. The Full Pipeline *(upcoming)*

### Course 20: Beyond the Bitter Lesson — A Mathematical Theory of Intelligence (Survey)
*5 lessons — The math we'd need to escape brute-force scaling: manifolds, persistent homology, statistical mechanics, with a hands-on TDA finale*

1. [Empirical Recipes vs Mathematical Laws](course-20-mathematical-theory-of-intelligence/lesson-1-empirical-vs-theoretical.md) — Kepler vs Newton; Chinchilla scaling laws as Kepler; Shannon's channel capacity theorem as the bar; what "abstraction" means operationally; Kolmogorov complexity as the formal candidate for the "bit of abstraction"
2. [Manifolds — Where an LLM's Knowledge Actually Lives](course-20-mathematical-theory-of-intelligence/lesson-2-manifolds.md) — Intrinsic vs ambient dimension; manifolds as locally-Euclidean / globally-curved spaces; the Manifold Hypothesis; PCA vs nonlinear methods (UMAP); hallucinations as off-manifold drift; multi-component manifolds and sparse bridges
3. [Holes, Voids & Persistent Homology](course-20-mathematical-theory-of-intelligence/lesson-3-persistent-homology.md) — Simplicial complexes; Vietoris-Rips construction; Betti numbers `b_0`, `b_1`, `b_2`; persistence diagrams as deformation-invariant fingerprints; the *conditional* nature of "voids = hallucinations"; sampling bias and the methodological pushback that motivates a theoretical foundation
4. [Statistical Mechanics, Phase Transitions & Grokking](course-20-mathematical-theory-of-intelligence/lesson-4-statistical-mechanics-and-grokking.md) — Boltzmann distribution as softmax-with-temperature; energy = loss; spin-glass loss landscapes; phase transitions and order parameters; grokking as a phase transition; RG flow as abstraction; the implied research program through universality classes
5. [Hands-On — Extracting the Topology of a Real LLM](course-20-mathematical-theory-of-intelligence/lesson-5-hands-on-tda.md) — Full Python pipeline for Qwen3.5: hidden-state extraction, PCA dimensionality test, UMAP visualization, Vietoris-Rips persistent homology, differential persistence; practical engineering Q&A on model size, layer choice, EOS handling, and prompt-vs-generation; recommended run order from 0.8B prototype to 9B headline plus cross-scale comparison

## Key Takeaways Across All Courses

1. An LLM's knowledge lives in its weights (stone) — permanent patterns carved during training
2. In-context learning activates existing capabilities via the prompt (water) — nothing changes, nothing persists
3. Fine-tuning modifies weights (picks up the chisel) — powerful but expensive and risky
4. **"Activating, not carving"** — the student's own summary of ICL vs fine-tuning
5. LoRA represents weight changes as low-rank matrices — because fine-tuning updates are empirically low-rank
6. Orthogonal adapter training guarantees diverse perspectives — the constraint is in weight space, not data space
7. **Orthogonality is like PCA/eigenvectors** — each adapter captures the next most important independent direction of useful adaptation
8. A review committee needs **both specialists (orthogonal) AND generalists (composition layer)** to catch diagonal bugs
9. The distinction between "activation" and "learning" is a matter of where you draw the line — it's learning all the way down
10. The holy grail is **continual learning** — if LoRA adapters can be continually learned, merged, and composed, the line between training and adaptation dissolves
11. Every prediction error decomposes into **bias² + variance + noise** — the pattern of failure tells you what to fix
12. **Regularization is prior knowledge in disguise** — L2 = Gaussian prior, L1 = Laplace prior, MAP = MLE + regularization
13. Generalization depends on the ratio of **model capacity to data**, not capacity alone — VC theory and PAC learning formalize this
14. KL divergence is asymmetric by design — **forward KL covers modes (blurry), reverse KL seeks modes (sharp)** — this explains the VAE vs GAN divide
15. EM is the gateway to variational methods — **bound what you can't compute, optimize the bound** — but it only works when the E-step is tractable
16. **Exchangeability justifies Bayesian modeling** — de Finetti proves a parameter and prior must exist if data order is irrelevant
17. Posterior collapse in VAEs is a degenerate equilibrium, not a bug — **the ELBO is doing what it's told**, but a powerful decoder makes z unnecessary
18. Mean-field VI underestimates uncertainty — **factorization + reverse KL = mode-seeking with no correlations** — calibrated uncertainty requires richer families or MCMC

## Student Profile

- Experience level: understands neural networks and transformer architecture
- Thinks mechanically — reasons about tokens, weights, and machinery rather than anthropomorphizing
- Prefers practical, engineering-grounded explanations over abstract theory
- Strong at synthesis — naturally connects ideas across lessons and extrapolates to frontier implications
- Pushes back on imprecise claims, leading to productive deeper discussions
