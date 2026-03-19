# Lesson 5: The Frontier — Where the Line Gets Blurry

*Course 1: In-Context Learning vs Fine-Tuning*

## Setup

For four lessons, a clean separation: the stone versus the water. Permanent weights versus temporary context. Two different things. Nice and tidy.

Now it all gets messy.

## The Uncomfortable Question

During pre-training, the model saw *trillions* of tokens. Many of those tokens were sequences that looked exactly like few-shot prompting — examples followed by a new instance. The model got *trained on* in-context learning patterns.

**In-context learning is itself a behavior that was carved into the stone.**

The water only works because the sculpture was specifically shaped to respond to water. The ability to "learn from the prompt" isn't a free bonus — it was baked in during training. It's a *weight-level skill* that enables *context-level adaptation*.

Is in-context learning really "not learning"? Or is it a learned meta-skill executing at inference time?

## Blurry From the Other Direction Too

Modern fine-tuning with LoRA adds a tiny number of parameters (~0.1% of the model) that can be **swapped in and out at will**:

- Plug in adapter A → medical expert
- Plug in adapter B → legal expert
- Unplug everything → back to base model, completely unharmed

Does that sound permanent? It sounds more like changing what's in the context. The stone is untouched — you're deciding which attachment to bolt on for this session.

Is LoRA really "carving the stone"? Or is it **removable stone** — a third category?

## The Spectrum

It was never a binary. It's a **spectrum**.

```
Pure prompting <------------------------------> Full fine-tuning
    |                    |                          |
 zero changes      LoRA adapters             all weights change
 fully temporary   swappable, small          permanent, deep
 no risk           low risk                  catastrophic forgetting risk
 limited by        limited by adapter        limited by data + compute
 context window    rank/size
```

Researchers keep inventing things in the middle: prompt tuning (learned soft prompt tokens), prefix tuning, adapter layers. Each is a different trade-off point.

The clean story from Lessons 1–4 is **true and useful** — the right mental model for practical decisions. But underneath, these aren't two different things. They're two ends of a continuum.

## The Deepest Blurring: What Even Is Learning?

Some recent research shows that **models can update their effective behavior within a single forward pass** in ways that look like gradient descent. The attention mechanism, mathematically, can implement something *functionally equivalent* to a learning algorithm — constructing and applying a temporary "model" from the examples in context.

If true, then few-shot examples in a prompt don't just trigger pattern-matching. The model runs something like a tiny training loop *inside the forward pass*, building a temporary mini-model that lives only in the activations.

Water that briefly becomes ice, then melts.

Doesn't change practical advice. But changes how you think about the boundary between "already knew" and "just learned."

---

## Q&A

**Question:** If someone told you "in-context learning and fine-tuning are fundamentally different mechanisms," would you agree or disagree? Make your case.

**Student's Answer:** There are a few fundamental differences, first and foremost, in-context learning doesn't change weights but fine-tuning does. But in other aspects, the lines are much more blurred. Fine-tuning is clearly a type of learning, and in-context learning looks like activation but when activation becomes sophisticated enough, it gets closer to learning too, or it may even be activating a meta-learning skill that's already learnt! The practical differences are still pretty big: it's much cheaper and more flexible to change LLM output via prompts than with fine-tuning, the deployment is very different.

**Evaluation:** Outstanding. The student held two truths simultaneously — yes, there is a fundamental mechanical difference (weights change vs. don't), and yes, at a deeper level the boundary dissolves. Didn't pick a side; described the landscape accurately.

**Key student insight:** *"It may even be activating a meta-learning skill that's already learnt"* — the punchline of the entire course, stated more cleanly than most research papers.

## Course 1 Summary

1. An LLM's knowledge lives in its weights — permanent patterns carved during training
2. In-context learning activates existing capabilities by providing signals in the prompt — nothing changes, nothing persists
3. Fine-tuning modifies the weights themselves — powerful but expensive, risky, and hard to undo
4. Default to prompting; fine-tune only when the water provably can't do the job
5. The clean separation is a useful lie — underneath, it's a spectrum, and the ability to "learn from context" is itself a learned skill baked into the weights

---

## Interlude: Transformer Architecture — Where Knowledge Lives in the Weights

*Student asked: Is there a fundamental difference between different parts of weights in an LLM? Which weights are "learning"?*

### Clarification on "Activation"

When the teacher said in-context learning "activates" capabilities, it was **colloquial** — "triggers" or "wakes up." Not a reference to any architectural component. In neural network terminology:

1. **Activations** — intermediate values (tensors) flowing through the network during forward pass
2. **Activation functions** — nonlinear functions (GELU, ReLU) inside MLP layers

Neither is what was meant by "activating capabilities."

### Anatomy of a Transformer Block

Each transformer block has **two main components**:

```
Input
  |
  v
+---------------------------+
|  Multi-Head Attention      |
|                            |
|  W_Q  (query projection)  |  <- linear layers, no activation function
|  W_K  (key projection)    |
|  W_V  (value projection)  |
|  W_O  (output projection) |
|                            |
|  softmax(QK^T / sqrt(d)) * V  <- the attention computation
+---------------------------+
  |
  v
+---------------------------+
|  Feed-Forward Network      |
|  (the "MLP")               |
|                            |
|  Linear -> GELU -> Linear  |  <- THIS has the activation function
+---------------------------+
  |
  v
Output (to next layer)
```

**Correction to student's mental model:** Attention heads don't contain MLP layers. They contain **linear projections** — plain matrix multiplications with no nonlinearity. The MLP block is separate, downstream of attention.

### Which Weights Store What?

**Attention heads** learn *relationships and routing*:
- "The word 'it' refers back to 'the cat'"
- "This adjective modifies that noun"
- "These few-shot examples share the same structure"

Think of attention as the **postal system** — deciding where information gets sent.

**MLP layers** act like **key-value memories** storing *factual associations*:
- "Eiffel Tower → Paris"
- "Water → H₂O"

Research by Meng et al. (ROME and MEMIT) showed factual knowledge is predominantly stored in MLP layers in the middle of the network.

**Important caveat:** It's not a clean separation. Both components contribute to both functions. It's a *tendency*, not a wall.

### What LoRA Targets

Original LoRA paper applied low-rank adapters to **W_Q and W_V** — query and value projections. But LoRA can be applied to *any* weight matrix:

| Target | What you're changing |
|---|---|
| W_Q, W_K, W_V, W_O | How the model routes information between tokens |
| MLP layers | What factual associations and transformations the model applies |
| All of the above | Both routing and knowledge (most common modern practice) |

**Learning happens everywhere.** During training, gradients flow through the entire network. The question isn't where learning *happens* but where different *types of knowledge end up stored*.
