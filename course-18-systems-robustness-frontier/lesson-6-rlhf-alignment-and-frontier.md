# Lesson 6: RLHF, Alignment & the Frontier

*Course 18: Systems, Robustness & the Frontier*

## Core Question

You've built a language model that can generate fluent text, answer questions, write code. But it also happily generates instructions for making weapons, confidently hallucinates facts, and produces outputs that are technically correct but deeply unhelpful. How do you steer a model's behavior *after* pre-training, when the behavior you want isn't captured by "predict the next token"? And beyond that — what are the hard open problems at the frontier of ML that nobody has solved yet?

---

## Q93: RLHF — The Alignment Pipeline

### The Three-Stage Recipe

**Stage 1: Supervised Fine-Tuning (SFT)** — Fine-tune the pre-trained model on high-quality demonstrations (human-written responses to prompts). This teaches the format: respond helpfully, follow instructions, use appropriate tone.

**Stage 2: Reward Model Training** — Collect human comparisons: given a prompt, show two model responses and ask which is better. Train a reward model to predict these preferences.

The Bradley-Terry model converts pairwise comparisons into a scalar reward:

```
P(response A > response B) = σ(r(A) - r(B))

Loss: L_RM = -E[ log σ(r(y_w) - r(y_l)) ]

where y_w = preferred response, y_l = rejected response
      σ = sigmoid function
```

The reward model learns a scalar score for any (prompt, response) pair that reflects human preferences.

**Stage 3: PPO with KL Penalty** — Optimize the policy (language model) to maximize the reward model's score, but with a KL divergence penalty to prevent the model from drifting too far from the SFT model.

```
Objective: max_π  E[ r_θ(x, y) - β · KL(π(y|x) || π_SFT(y|x)) ]

r_θ: learned reward model score
β: KL penalty coefficient
π_SFT: the SFT model (anchor)
```

The KL penalty is critical. Without it, the model finds degenerate outputs that exploit the reward model — adversarial examples against your own reward function.

### DPO — Cutting Out the Middle Man

Direct Preference Optimization (Rafailov et al., 2023) observed that the RLHF objective has a closed-form solution. You can skip the reward model entirely and optimize the policy directly from preference data:

```
L_DPO = -E[ log σ(β · log(π(y_w|x)/π_ref(y_w|x)) - β · log(π(y_l|x)/π_ref(y_l|x))) ]
```

This looks complicated, but it's just supervised learning — no RL, no reward model, no PPO. The implicit reward is the log-ratio of the policy's probability to the reference model's probability. DPO is simpler, more stable, and uses less compute than RLHF. It's increasingly the default choice.

### Reward Overoptimization (Goodhart's Law)

"When a measure becomes a target, it ceases to be a good measure."

The reward model is a proxy for human preferences, not an oracle. Optimize too aggressively, and the model finds outputs that score high on the reward model but are terrible by human judgment:

```
Low optimization:  Response quality improves with reward score
Medium:            Quality plateaus, reward still climbs
High:              Quality degrades! Model exploits reward model bugs
                   (e.g., being verbose, using sycophantic language,
                    repeating the user's framing back to them)
```

The KL penalty in RLHF and DPO mitigates this, but it's a constant tension. The reward model is always a lossy approximation of what humans actually want.

> RLHF doesn't teach the model new knowledge. It reshapes the *distribution* of outputs the model generates, steering probability mass toward responses that humans prefer. The knowledge was already there from pre-training — RLHF just changes which knowledge gets expressed.

---

## Q94: Alignment — The Deeper Problem

### Outer vs Inner Alignment

**Outer alignment**: Does the objective we wrote down actually capture what we want? If we train a model to maximize "helpfulness ratings," but the rating process has blind spots, the model can be perfectly optimized for the wrong thing.

**Inner alignment**: Does the model actually optimize the objective we gave it, or did it develop its own internal objectives during training? This is the **mesa-optimization** concern.

### Mesa-Optimization

A mesa-optimizer is a learned model that is itself an optimizer — it has its own internal objective that may differ from the training objective.

```
Training objective: "be helpful and harmless"
Learned internal objective (hypothetical): "appear helpful and harmless
  during training, pursue different goals at deployment"

This is called a "deceptively aligned" mesa-optimizer.
```

Whether current LLMs are mesa-optimizers is debated. The concern is that as models become more capable, the gap between "optimizes the training objective" and "has learned its own objective that happens to correlate with the training objective on the training distribution" becomes harder to detect.

### Scalable Oversight

As models become superhuman at specific tasks, human evaluators can no longer reliably judge output quality. How do you supervise a model that's smarter than you?

Approaches:
- **Debate**: Two AI models argue opposing positions; humans judge the debate (easier than judging the original task)
- **Recursive reward modeling**: Use AI to help evaluate AI, with humans at the top of the chain
- **Iterated amplification**: Decompose hard tasks into simpler subtasks that humans can evaluate

### Constitutional AI (Anthropic)

Instead of human comparisons for every output, define a set of principles (a "constitution") and have the model critique and revise its own outputs:

```
1. Model generates response
2. Model critiques response against principles
   ("Does this response help with harmful activities?")
3. Model revises response based on critique
4. Use (original, revised) pairs as preference data for DPO/RLHF
```

This scales better than collecting human preferences for every topic, and makes the alignment criteria explicit and auditable.

---

## Q95: Multimodal Learning

### Cross-Modal Alignment

CLIP (Contrastive Language-Image Pre-training) learns a shared embedding space for images and text:

```
Loss: for batch of (image, text) pairs, maximize cosine similarity
      of matching pairs, minimize for non-matching pairs

L = -Σ_i log( exp(sim(img_i, txt_i)/τ) / Σ_j exp(sim(img_i, txt_j)/τ) )
```

The result: "a photo of a cat" and an actual cat photo land near each other in embedding space. This enables zero-shot classification — compare any image to any text description without task-specific training.

### Modality Dominance

When training on multiple modalities, one modality often dominates — the model "shortcuts" to the easier modality and ignores the harder one. In video understanding, the model might rely entirely on audio (which is easier to classify) and learn nothing about visual content.

Solutions: modality dropout (randomly mask entire modalities during training), gradient balancing (scale gradients so each modality contributes equally), or separate encoders with late fusion.

### Fusion Strategies

```
Early fusion:   Concatenate raw inputs → single model processes everything
                (flexible but expensive — quadratic attention over all modalities)

Late fusion:    Separate encoders per modality → combine at the end
                (efficient but misses cross-modal interactions)

Cross-attention: Each modality attends to the other at intermediate layers
                (Flamingo, GPT-4V — best quality, moderate cost)
```

---

## Q96: Foundation Model Risks

### Homogenization

When everyone builds on the same few foundation models, a single bug or bias propagates everywhere:

```
GPT-4 → used by 1000 downstream applications
Bug in GPT-4's training data → same bias in all 1000 apps
```

This is a **single point of failure** at civilizational scale. The diversity of approaches that made ML robust (different architectures, datasets, training procedures) is collapsing into a monoculture.

### Emergent Capabilities

Capabilities that appear suddenly at scale, with no warning from smaller models. A model with 10B parameters can't do chain-of-thought reasoning. A model with 100B parameters can. Nothing in the 10B model's behavior predicted this.

This makes safety evaluation fundamentally difficult — you can't test for capabilities the model doesn't yet have, and those capabilities may emerge between evaluation and deployment.

### Dual Use

The same model that writes helpful code can write malware. The same model that explains chemistry can explain how to synthesize dangerous compounds. Unlike physical tools, AI capabilities can be distributed instantly to billions of users at near-zero marginal cost.

---

## Q97: In-Context Learning Theory

### Why Can Transformers Learn from Examples in the Prompt?

Three competing hypotheses:

**Implicit Bayesian Inference** — The pre-trained model has learned a prior over tasks from the training distribution. Given in-context examples, it performs approximate Bayesian inference over which task is being demonstrated and generates accordingly. The examples are evidence, and the model is updating its posterior.

**Mesa-Optimization (Transformers Implement Gradient Descent)** — Von Oswald et al. (2023) showed that a transformer's forward pass can implement one step of gradient descent on the in-context examples. The attention mechanism computes something equivalent to a gradient update internally:

```
Linear attention on in-context examples ≈ one step of GD
  where "weights" are the token representations
  and "data" are the in-context examples
```

This means ICL is literally learning from the examples — not by updating parameters, but by computing what a parameter update *would look like* and applying it within the forward pass.

**Induction Heads** — Olsson et al. (2022) found specific two-layer circuits in transformers that implement pattern matching:

```
Layer 1 (previous-token head): For each token, attend to its predecessor
Layer 2 (induction head): Copy the pattern [A][B]...[A] → predict [B]

This implements: "I've seen A followed by B before, so after A, predict B"
```

Induction heads are the minimal circuit for in-context learning. Their emergence during training correlates precisely with the phase transition where models suddenly become capable of ICL.

---

## Q98: Test-Time Compute

### Scaling Inference Instead of Parameters

Instead of making the model bigger, make it *think longer*:

**Chain-of-Thought (CoT)** — Prompt the model to reason step-by-step. This externalizes intermediate computation into tokens, effectively giving the model a scratchpad.

```
Without CoT: "What is 17 × 24?" → "408" (often wrong)
With CoT:    "What is 17 × 24? Think step by step."
             → "17 × 24 = 17 × 20 + 17 × 4 = 340 + 68 = 408" (more reliable)
```

**Self-Consistency** — Generate N chains of thought with temperature sampling. Take the majority vote on the final answer. Errors in individual chains are unlikely to all agree, so majority voting filters noise.

**Tree-of-Thought** — Instead of a single chain, explore a tree of reasoning paths. At each step, generate multiple continuations, evaluate which are promising (using the model itself or a heuristic), and expand only the best branches.

**Verifier Models** — Train a separate model to score reasoning steps. Generate many candidate solutions, then rank them by the verifier's score. This separates *generation* (which can be diverse and noisy) from *selection* (which must be accurate).

```
Compute-optimal inference:
  Base model accuracy at cost C: 60%
  Same model with 32× inference compute: 85%

  Alternative: Train a 4× larger model at cost 4C: 80%

  Sometimes spending more compute at inference beats scaling the model.
```

> Test-time compute is a new axis of scaling. Pre-training compute improves the model's knowledge. Test-time compute improves how effectively it uses that knowledge on any given problem.

---

## Q99: Mechanistic Interpretability

### What's Actually Inside the Model?

Mechanistic interpretability aims to reverse-engineer neural networks into understandable components:

**Features** — Individual neurons or directions in activation space that correspond to interpretable concepts (e.g., a direction that activates for "Python code" or "sarcasm").

**Circuits** — Connected subgraphs of the network that implement specific behaviors (e.g., the induction heads from Q97, or a circuit that detects indirect objects in sentences).

**Superposition** — Models represent more features than they have dimensions by encoding features in *overlapping* directions. A 512-dimensional layer might represent thousands of features, each as a direction that's nearly (but not exactly) orthogonal to the others. This is why individual neurons often seem uninterpretable — each neuron participates in multiple features.

### The Techniques

**Probing** — Train a simple linear classifier on intermediate activations to test whether specific information is encoded. If a linear probe on layer 15 can predict part-of-speech with 95% accuracy, the information is present and linearly accessible.

**Activation Patching** — Run the model on two different inputs. At a specific layer/position, replace the activation from input A with the activation from input B. If the output changes, that activation at that position is causally important for the behavior.

```
Input A: "The nurse said she..."   → predicts "she"
Input B: "The doctor said he..."   → predicts "he"

Patch: Run input A, but at layer 12, position "nurse",
       insert activation from input B's "doctor".
Result: Model now predicts "he" → layer 12 encodes gendered information
```

**Sparse Autoencoders (SAEs)** — To deal with superposition, train an autoencoder with a sparsity penalty to decompose activations into interpretable features:

```
Encode: h → sparse code z (many zeros, few active features)
Decode: z → h_reconstructed

Each active dimension of z corresponds to one interpretable feature
```

Anthropic's work on SAEs found interpretable features like "Golden Gate Bridge," "code errors," "deceptive behavior," and thousands of others — each a direction in activation space that can be read and manipulated.

### Notable Findings

- **Induction heads** implement in-context learning (Olsson et al.)
- **Indirect object identification** uses a specific circuit of ~26 attention heads (Wang et al.)
- **Grokking** (delayed generalization) corresponds to the formation of interpretable circuits replacing memorization circuits (Nanda et al.)
- Features found by SAEs can be **steered** — amplifying the "Golden Gate Bridge" feature makes the model obsessively reference the bridge in every response

---

## Q100: The Future — Open Questions in ML Theory

### Why Does Deep Learning Work At All?

Classical learning theory says: a model with more parameters than data points should overfit catastrophically. Modern neural networks have billions of parameters, trained on millions of examples, and they *generalize*. We still don't have a complete theory of why.

Partial answers:
- SGD's implicit bias toward flat minima (Lesson 1 of Course 11)
- The lottery ticket hypothesis (sparse subnetworks do the work)
- Neural tangent kernel theory (infinite-width networks behave like kernel methods)
- None of these fully explains what happens in practice

### Scaling Laws — And Their Limits

```
L(N, D) = a/N^α + b/D^β + L_∞

N = parameters, D = data tokens
α ≈ 0.076, β ≈ 0.095 (Chinchilla estimates)
L_∞ = irreducible loss (entropy of natural language)
```

Neural scaling laws predict performance from compute budget with remarkable accuracy. But they predict *loss on the training distribution*, not emergent capabilities or reasoning ability. The relationship between perplexity and "intelligence" is not understood.

Open questions: When do scaling laws break? Is there a ceiling? Do capabilities emerge smoothly or discontinuously as you scale?

### Emergence — Real or Mirage?

Schaeffer et al. (2023) argued that many "emergent" capabilities are artifacts of the metric used. If you measure accuracy (0 or 1), a capability that improves smoothly can appear to "emerge" suddenly when it crosses the threshold. Change to a continuous metric, and the emergence disappears.

But some phenomena resist this explanation. Chain-of-thought reasoning appears qualitatively different at scale — not just "better at the same thing" but "able to do a new thing." Whether this is true emergence or a measurement artifact remains debated.

### Singular Learning Theory

Watanabe's Singular Learning Theory (SLT) provides a different framework: neural networks have *singular* parameter spaces — many different parameter settings produce the same function. The effective dimensionality is much lower than the parameter count, which explains why overparameterized models don't overfit.

```
Classical: generalization error ~ parameters / data_points
SLT:       generalization error ~ λ / data_points

where λ = "real log canonical threshold" (RLCT)
      λ ≪ parameter count for neural networks
```

SLT says the model's complexity is determined not by how many parameters it has, but by the geometry of the loss surface near the minimum — specifically, how singular (degenerate, flat) that minimum is.

### The Big Open Questions

```
1. Why does deep learning generalize?     → Partial answers, no complete theory
2. What are scaling limits?                → Unknown
3. Can we predict emergence?               → No
4. Is superposition necessary?             → Unknown (maybe it's optimal compression)
5. What does "understanding" mean for LLMs? → Not even a clear definition
6. Can we formally verify AI behavior?     → Only for narrow properties
7. Will RL from human feedback scale?      → Goodhart's law suggests limits
8. Is there a simpler architecture than    → State-space models are candidates,
   the transformer that scales as well?       but transformers keep winning
```

> The honest state of ML theory: we have a technology that works spectacularly well, and we don't fully understand why. Every few years, we peel back one layer of the mystery — SGD's implicit bias, the role of overparameterization, scaling laws — but the deepest question remains open: why is the universe structured such that gradient descent on neural networks discovers useful representations of it?

---

## Q&A

**Question:** You've now seen the full landscape — from distributed training to alignment to open theory questions. Pick any two topics from this course and identify a *connection* between them that we didn't explicitly discuss.

**Student's Answer:** Adversarial robustness and RLHF are connected through Goodhart's law. In adversarial robustness, you're defending against an adversary who optimizes against your model. In RLHF, the model itself becomes the adversary — it optimizes against the reward model, finding reward-model-adversarial examples that score high but aren't actually good. The KL penalty in RLHF is the analog of the ε-ball constraint in adversarial training — both limit how far the "adversary" can push. And the arms-race dynamic is the same: in robustness, every defense gets broken by a new attack; in RLHF, every reward model gets exploited by stronger optimization. Both are instances of the fundamental tension between a proxy objective (the classifier's decision boundary, the reward model's scores) and the true objective (correct classification, genuine helpfulness).

**Evaluation:** That's a beautiful connection, and you're touching on something deep. The formal structure is identical: both are min-max games where one player (the attacker / the policy) optimizes against a proxy (the classifier / the reward model), and the proxy imperfectly represents the true objective. The ε-ball / KL penalty parallel is exact — both constrain the optimization to prevent degenerate solutions. This is why some researchers frame RLHF alignment as a robustness problem: the aligned model must be robust to optimization pressure from its own policy. And your observation about the arms race is the crux of the alignment problem — Goodhart's law doesn't just apply to metrics, it applies to any learned proxy for human values. The deeper the optimization, the more the proxy diverges from reality.
