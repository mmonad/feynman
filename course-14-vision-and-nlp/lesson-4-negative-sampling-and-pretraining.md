# Lesson 4: Negative Sampling and Pretraining

*Course 14: Computer Vision & NLP*

## Core Question

Last lesson we hit a wall: the softmax denominator in Word2Vec requires summing over the *entire vocabulary* for every training step. That's O(V) per example, with V in the hundreds of thousands. How do you train when the normalization constant alone takes longer than the rest of the computation combined?

And then: BERT and GPT both learn language representations from raw text. They're both transformers. They're both pre-trained. So why do they exist as separate architectures? What fundamental choice splits them into two lineages?

---

## Q47: Negative Sampling — Dodging the Denominator

### The Problem, Precisely

The Skip-gram objective for a center word `w_i` and context word `w_o` is:

```
maximize log P(w_o | w_i) = log [ exp(u_o · e_i) / Σ_j exp(u_j · e_i) ]

The gradient of the numerator: O(d)  — one dot product, cheap
The gradient of the denominator: O(V × d) — one dot product per vocab word

V ≈ 300,000.  Training examples ≈ billions.
Total cost: O(billions × 300,000 × d).  Unacceptable.
```

### The Key Reframing

Here's the insight: stop trying to solve a V-way classification problem. Instead, solve a *binary* classification problem.

The original question was: "Given center word `w_i`, which of the 300,000 vocabulary words is the correct context word?" That requires a softmax over all 300,000.

The new question: "Given a pair `(w_i, w_o)`, is this a *real* pair from the corpus, or a *fake* pair I just made up?" That's a yes/no question. One sigmoid. O(d).

```
Original: P(w_o | w_i) = softmax over V classes    → O(V)
Reframed: P(real | w_i, w_o) = σ(u_o · e_i)       → O(1)
```

### The Negative Sampling Objective

For each real (center, context) pair from the corpus, sample `k` "negative" pairs by pairing the center word with random words from the vocabulary. Then maximize:

```
L = log σ(u_o · e_i) + Σ_{j=1}^{k} E_{w_j ~ P_n} [log σ(-u_j · e_i)]

Translation:
  - Push the real context word's vector CLOSER to the center word
  - Push k random words' vectors AWAY from the center word

k is typically 5-20.
Cost per example: O(k × d) instead of O(V × d).
Speedup: ~15,000x for V=300,000, k=20.
```

The training loop becomes:

1. Take a real pair ("cat", "sat") from the corpus
2. Sample k=5 random "negative" words: "refrigerator", "democracy", "purple", "seventeen", "aardvark"
3. Train: make the dot product of "cat" and "sat" embeddings large (positive)
4. Train: make the dot products of "cat" with each negative word small (negative)

### The Noise Distribution

Which distribution do you sample negatives from? Not uniform — that would oversample rare words that the model already handles well. Not proportional to frequency — that would oversample common words ("the", "a", "of") that carry little information.

The answer Mikolov found empirically is the **3/4 power of frequency**:

```
P_n(w) ∝ freq(w)^(3/4)

This is between uniform (freq^0) and proportional (freq^1).

Effect: boosts rare words relative to proportional sampling,
        suppresses common words relative to uniform.

    Word        freq    freq^1     freq^(3/4)
    "the"       0.07    0.070      0.037
    "dog"       0.001   0.001      0.003
    "aardvark"  0.00001 0.00001    0.0001

"aardvark" gets 10x more representation than proportional would give.
```

The intuition: you want negatives to be challenging but not impossibly so. Extremely rare words are easy negatives (the model already knows they don't belong). Very common words are uninformative negatives (everything co-occurs with "the"). The 3/4 power sits in the sweet spot.

### Connection to Noise Contrastive Estimation (NCE)

Negative sampling is a simplification of **NCE** (Gutmann & Hyvärinen, 2010). NCE is a general technique for training models with intractable normalizing constants. The full NCE objective learns both the unnormalized model *and* the normalizing constant. Negative sampling simplifies this by fixing the normalizing constant to 1 and only learning the unnormalized scores.

```
NCE: learn log P(w) = s(w) - log Z,  where Z is learned
Negative sampling: assume log Z = 0, just learn s(w)

NCE is theoretically cleaner (converges to the true distribution).
Negative sampling is simpler and works just as well for embeddings.
```

> **Key insight:** Negative sampling transforms an intractable V-way classification into a tractable binary classification by asking "is this pair real or fake?" instead of "which word is correct?" The trick works because you don't actually need a properly normalized probability distribution to learn good embeddings — you just need the *ranking* to be right.

---

## Q48: BERT vs. GPT — Two Philosophies of Pre-training

### The Fork in the Road

Both BERT and GPT are transformers pre-trained on enormous text corpora. The fundamental difference is *what they predict* and *what they're allowed to see while predicting*.

| Property | BERT | GPT |
|---|---|---|
| Full name | Bidirectional Encoder Representations from Transformers | Generative Pre-trained Transformer |
| Objective | Masked Language Modeling (MLM) | Causal Language Modeling (CLM) |
| Attention | Bidirectional — sees entire sequence | Unidirectional — sees only past tokens |
| Predicts | Randomly masked tokens | Next token |
| Strength | Understanding / classification | Generation |

### BERT: The Fill-in-the-Blank Machine

BERT's training is simple. Take a sentence, randomly mask 15% of the tokens, and train the model to predict the masked tokens from the remaining context.

```
Input:  "The cat [MASK] on the [MASK]"
Target: "The cat  sat  on the  mat"

The model sees BOTH sides of the mask:
  "The cat" ← left context
  "on the [MASK]" ← right context

This is BIDIRECTIONAL attention.
```

The 15% masking rate is a deliberate tradeoff: too high and there's not enough context to predict; too low and training is too slow (each example provides little signal). BERT actually uses a mixed strategy: of the 15% selected tokens, 80% are replaced with `[MASK]`, 10% with a random word, and 10% left unchanged. This prevents the model from learning that `[MASK]` is a special signal that only appears in training.

Because BERT sees both left and right context, it can build representations that capture the full meaning of a word in its sentence. For tasks like sentiment classification, question answering, and named entity recognition — tasks that require *understanding* the complete input — bidirectional context is a massive advantage.

### GPT: The Continuation Machine

GPT is trained autoregressively: predict the next token given all previous tokens.

```
Input sequence: "The cat sat on the mat"

Training signal at each position:
  P("cat"  | "The")
  P("sat"  | "The cat")
  P("on"   | "The cat sat")
  P("the"  | "The cat sat on")
  P("mat"  | "The cat sat on the")

Each token can ONLY attend to previous tokens.
This is CAUSAL (unidirectional) attention.
```

The causal mask means each token's representation is built from left context only. This is a *weaker* representation for understanding — "bank" in "I went to the bank to deposit money" would be ambiguous if you could only see "I went to the bank" — but it's the *only* way to do generation. You can't look at the future when the future doesn't exist yet.

### Why This Distinction Matters Mechanically

The attention mask is the key mechanical difference:

```
BERT attention mask (bidirectional):
  Token:  The  cat  sat  on  the  mat
  The      1    1    1    1    1    1
  cat      1    1    1    1    1    1
  sat      1    1    1    1    1    1
  on       1    1    1    1    1    1
  the      1    1    1    1    1    1
  mat      1    1    1    1    1    1

GPT attention mask (causal):
  Token:  The  cat  sat  on  the  mat
  The      1    0    0    0    0    0
  cat      1    1    0    0    0    0
  sat      1    1    1    0    0    0
  on       1    1    1    1    0    0
  the      1    1    1    1    1    0
  mat      1    1    1    1    1    1
```

Same architecture. Same parameters. The only difference is which entries in the attention matrix are zeroed out. But this mechanical difference has profound consequences for what the model can learn and what it can do.

### The Convergence Toward GPT-Scale

Here's the punchline of the last five years: GPT won.

Not because causal LM is theoretically superior — BERT-style models genuinely produce better representations for understanding tasks. But because of three practical forces:

**1. Generation is the killer app.** The thing users actually want — chatbots, code completion, creative writing — requires generation. BERT can't generate fluently because it was never trained to produce text left-to-right.

**2. Scale favors simplicity.** The GPT objective (predict next token) scales trivially. Every token in the corpus is a training example. BERT's masking adds complexity and wastes 85% of tokens (the unmasked ones provide no gradient signal). At the scale of trillions of tokens, this efficiency gap compounds.

**3. Emergent capabilities appear at scale.** Few-shot learning, chain-of-thought reasoning, instruction following — these capabilities emerged in large autoregressive models. Whether they'd emerge equally in scaled BERT-style models is untested because nobody invested the compute to find out.

```
Timeline:
  2018: BERT (340M params) dominates NLU benchmarks
  2019: GPT-2 (1.5B) shows generation quality scales with size
  2020: GPT-3 (175B) demonstrates few-shot learning
  2022: ChatGPT — generation-first LLMs become the dominant paradigm
  2024+: All frontier models (GPT-4, Claude, Gemini) are autoregressive
```

> **Key insight:** BERT and GPT make different bets about what matters. BERT bets on rich, bidirectional representations — better understanding at the cost of generation ability. GPT bets on left-to-right prediction — weaker representations but native generation. The market decided that generation matters more, and scale effects compounded GPT's advantage. But the underlying architectural difference is just one bit: the attention mask.

---

## Q&A

**Question:** You said BERT wastes 85% of tokens because unmasked tokens provide no gradient signal. But unmasked tokens still participate in the *forward pass* — they provide context for predicting the masked tokens. So isn't the "waste" only in the backward pass, not the forward pass?

**Student's Answer:** Right — the forward pass processes all tokens, and unmasked tokens do useful work by providing context. The waste is specifically in gradient signal: only 15% of positions generate a loss term, so only 15% contribute to parameter updates. GPT gets a loss at *every* position. So for the same corpus and the same compute, GPT extracts roughly 6-7x more training signal. The forward compute is similar, but the backward signal density is lopsided.

**Evaluation:** Precisely right. The forward pass is comparably expensive for both architectures. The asymmetry is in *training signal density*: GPT gets a supervised signal at every token position, while BERT only gets one at the 15% that are masked. This means GPT effectively sees 6-7x more "labeled examples" per pass through the same data. At trillion-token scale, that difference is enormous. It's one of the underappreciated reasons why autoregressive models scaled more gracefully — not just the ability to generate, but the raw data efficiency of the training objective.
