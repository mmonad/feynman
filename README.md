# Feynman Lectures on LLM Adaptation

A series of private 1-on-1 tutoring sessions in the style of Richard Feynman, covering how large language models adapt their behavior — from prompting to fine-tuning to cutting-edge parameter-efficient methods and composable AI review systems.

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
| **The Dart Board** | Bias-variance tradeoff — clustered but off-center vs scattered but centered |
| **Eigenvectors / PCA** | Orthogonal adapter training — decomposing useful adaptation into independent components |
| **The Diagonal Bug** | Interaction effects invisible to orthogonal specialists looking along single axes |
| **The Ballroom Crowd vs Committee** | Unstructured multi-reviewer output vs architected review pipeline |

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

## Student Profile

- Experience level: understands neural networks and transformer architecture
- Thinks mechanically — reasons about tokens, weights, and machinery rather than anthropomorphizing
- Prefers practical, engineering-grounded explanations over abstract theory
- Strong at synthesis — naturally connects ideas across lessons and extrapolates to frontier implications
- Pushes back on imprecise claims, leading to productive deeper discussions
