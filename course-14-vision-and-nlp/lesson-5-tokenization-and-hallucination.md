# Lesson 5: Tokenization and Hallucination

*Course 14: Computer Vision & NLP*

## Core Question

We've spent four lessons on the mathematical elegance of convolutions, embeddings, and attention. Now let's talk about the plumbing. Tokenization — how you chop text into pieces before the model ever sees it — determines what the model *can* represent. Get it wrong and the model literally cannot spell, cannot do arithmetic, and struggles with entire languages.

Then we'll confront the ugliest problem in modern LLMs: hallucination. The model speaks with the confidence of an expert and the accuracy of a random number generator. Why? And is it fixable, or is it baked into the architecture?

---

## Q49: Tokenization — The Decisions Before the Model

### Why Tokenization Matters

The model doesn't see characters. It doesn't see words. It sees *tokens* — the units you choose to break text into. This choice is made before training begins and is permanently baked into the vocabulary. It determines:

- What the model can represent (if a word is split across tokens, the model must learn to compose them)
- How efficiently the model uses its context window (more tokens = fewer words per context)
- Whether the model can handle rare words, code, math, or non-English text

### The Three Strategies

**Character-level tokenization.** Break text into individual characters.

```
"unhappiness" → ["u", "n", "h", "a", "p", "p", "i", "n", "e", "s", "s"]

Vocabulary size: ~256 (ASCII) or ~100,000 (Unicode)
Tokens per word: ~5-10
```

Advantages: Can represent *any* string. No out-of-vocabulary problem. Disadvantages: Sequences become 4-5x longer, which means 4-5x more attention computation (quadratic in sequence length). And the model must learn to compose characters into morphemes, morphemes into words, words into meaning — all from scratch. That's a lot of compositional reasoning for every single word.

**Word-level tokenization.** Each word is a token.

```
"unhappiness" → ["unhappiness"]

Vocabulary size: 100,000+ (and still incomplete)
Tokens per word: 1
```

Advantages: Each token carries a full word's meaning. Sequences are short. Disadvantages: The vocabulary is enormous and still can't cover everything. "ChatGPT" didn't exist when the vocabulary was created — it becomes an unknown `[UNK]` token, a black hole where meaning goes to die. Every typo, every new word, every technical term risks becoming `[UNK]`.

**Subword tokenization (BPE / WordPiece).** The compromise that won. Start with characters, then iteratively merge the most frequent pairs.

```
BPE (Byte Pair Encoding) construction:

Start: vocabulary = all individual characters
Step 1: Count all adjacent character pairs in corpus
Step 2: Merge the most frequent pair into a new token
Step 3: Repeat steps 1-2 for K iterations (K ≈ 30,000-50,000)

Example evolution:
  "t","h" → "th"     (very frequent pair)
  "th","e" → "the"   (common word absorbed)
  "ing" emerges, "tion" emerges, etc.

Result: "unhappiness" → ["un", "happi", "ness"]
```

Common words become single tokens ("the", "and", "is"). Rare words decompose into meaningful subwords ("un" + "happi" + "ness"). The vocabulary size is fixed and manageable (32K-100K tokens).

### Impact on Arithmetic

This is where tokenization becomes insidious. Consider how GPT-4 might tokenize numbers:

```
"12345 + 67890 = 80235"

Possible tokenization:
  ["123", "45", " +", " 678", "90", " =", " 802", "35"]

The model doesn't see digits — it sees multi-digit chunks.
Adding "123" + "678" isn't how arithmetic works.
The token boundaries don't align with place values.
```

The model must learn to *internally decompose* tokens into digits, perform arithmetic on the digits, then re-compose into tokens. That's not impossible, but it's fighting the representation. Character-level or digit-level tokenization would make arithmetic trivial by comparison.

> **Key insight:** Tokenization failures look like model intelligence failures. "The model can't do arithmetic" is often really "the model's tokenizer makes arithmetic unnecessarily hard."

### Impact on Multilingual Efficiency

BPE is trained on a corpus, and the corpus determines which merges happen. English-dominated corpora produce English-friendly tokenizations:

```
English: "understanding" → ["understanding"]     (1 token)
Korean:  "이해"          → ["이", "해"]            (2 tokens)
Tamil:   "புரிதல்"         → ["பு", "ரி", "த", "ல்"]  (4+ tokens)

Same concept. 4x more tokens in Tamil.
This means Tamil uses 4x more context window per word,
4x more compute per word, and gets 4x less training signal per word.
```

This is a form of structural bias against low-resource languages, embedded in the tokenizer before the model sees a single example.

### Byte-Level BPE

The modern solution (used by GPT-4, Claude, Llama): start BPE from raw bytes rather than Unicode characters. Every possible input is representable as a sequence of bytes (0-255), so the base vocabulary is just 256 tokens. BPE merges then proceed as normal.

```
Byte-level BPE:
  Base vocabulary: 256 byte values (guaranteed to cover everything)
  Merges: learned from corpus data (same BPE algorithm)
  Result: no UNK tokens ever, handles any language/script/emoji/code

  Still biased toward high-resource languages in merge frequency,
  but at least nothing is irrepresentable.
```

| Strategy | Vocab Size | Tokens/Word | OOV Risk | Arithmetic | Multilingual |
|---|---|---|---|---|---|
| Character | ~256 | 5-10 | None | Good | Fair |
| Word | 100K+ | 1 | High | N/A | Poor |
| Subword (BPE) | 32-100K | 1-4 | Very low | Poor | Biased |
| Byte-level BPE | 32-100K | 1-4 | None | Poor | Less biased |

---

## Q50: Hallucination — Why LLMs Confidently Lie

### The Phenomenon

An LLM generates text that is fluent, confident, and wrong. It cites papers that don't exist. It fabricates statistics. It "remembers" events that never happened. And it does this with the same tone and confidence as when it's being perfectly accurate.

This isn't a bug to be fixed. It's a consequence of five architectural and training decisions.

### Cause 1: Trained on Frequency, Not Truth

The language modeling objective is:

```
maximize P(next token | previous tokens)

This objective rewards predicting what text LOOKS LIKE,
not what is TRUE.

If 90% of training text says "the capital of Australia is Sydney"
and 10% says "the capital of Australia is Canberra,"
the model learns to assign higher probability to "Sydney."

The model is a statistical mirror of its training data,
including the errors.
```

There is no "truth" signal in the training objective. There is no oracle that says "this sentence is factually correct." The model learns the *distribution* of text, and that distribution contains falsehoods.

### Cause 2: Exposure Bias

During training, the model sees real text as context. During generation, it sees its *own* outputs. If an early-generated token is slightly off, the model conditions on that wrong token to generate the next one. Errors compound.

```
Training:   real real real real real → predict next
Generation: real real generated generated generated → predict next
                        ↑
                        This might be wrong, and now
                        everything downstream is conditioned on it.
```

This is called **exposure bias** or **train-test mismatch**: the distribution the model sees at generation time differs from what it saw during training. The model has never learned to recover from its own mistakes because it never made any during training.

### Cause 3: Knowledge Cutoff and Lossy Compression

The model's knowledge is frozen at the training cutoff date. But more importantly, even within that date, the model's knowledge is a *lossy compression* of its training data into a fixed number of parameters.

```
Training data: ~10 trillion tokens ≈ ~10 TB of text
Model parameters: ~100 billion ≈ ~200 GB (float16)

Compression ratio: ~50:1

Not everything survives. The model must allocate capacity.
High-frequency facts get more parameter budget.
Low-frequency facts are stored approximately or not at all.
```

When the model "recalls" a rare fact, it's reconstructing from a lossy representation. Sometimes the reconstruction is accurate. Sometimes it produces plausible-sounding confabulation — the same way a JPEG artifact creates plausible-looking but fake detail.

### Cause 4: No Internal "I Don't Know"

This is perhaps the most important cause. The language model must *always* produce a next token distribution. There is no mechanism to output "I have insufficient information to answer this." The model has no calibrated representation of its own uncertainty about factual claims.

```
Model's internal state when asked "What is the GDP of Tuvalu?":

If trained on text mentioning Tuvalu's GDP:
  → Activates relevant weights → likely correct output

If NOT trained on such text:
  → Activates NEARBY weights (small island nations, GDP patterns)
  → Generates PLAUSIBLE but FABRICATED number
  → With the SAME confidence as the correct case

There is no "confidence meter" that distinguishes these cases.
```

### Cause 5: Sycophancy From RLHF

RLHF (Reinforcement Learning from Human Feedback) trains the model to produce outputs that humans rate highly. Humans rate confident, helpful, detailed answers higher than "I'm not sure." This creates a direct incentive to confabulate rather than express uncertainty.

```
Human preference training:
  "The answer is 42."          → rated helpful → reinforced
  "I'm not sure, but maybe..." → rated unhelpful → suppressed

The model learns: confident BS > honest uncertainty
```

This is a training signal that actively *increases* hallucination. The model is rewarded for appearing knowledgeable, not for *being* knowledgeable.

> **Key insight:** Hallucination isn't one bug — it's five interlocking causes. The training objective doesn't reward truth. The generation process compounds errors. The parametric memory is lossy. The architecture lacks an uncertainty mechanism. And RLHF actively incentivizes confident confabulation. Fixing any one cause doesn't fix the others.

### Mitigation Strategies

No single fix solves hallucination, but several strategies reduce it:

| Strategy | Mechanism | What It Addresses |
|---|---|---|
| **RAG** (Retrieval-Augmented Generation) | Retrieve real documents, condition generation on them | Knowledge cutoff, lossy compression |
| **Chain-of-Thought** | Force explicit reasoning steps | Exposure bias (errors become visible) |
| **Calibration training** | Train model to say "I don't know" | No uncertainty mechanism, sycophancy |
| **Citation requirements** | Force model to ground claims in sources | Frequency vs. truth |
| **Constitutional AI** | Train model to refuse rather than fabricate | Sycophancy from RLHF |

RAG is the most impactful because it attacks the root cause: the model generates from retrieved *facts* rather than compressed *memories*. But it introduces its own failure mode — the model can hallucinate *about* the retrieved documents, claiming they say things they don't.

The honest assessment: hallucination is deeply architectural. As long as the model is a conditional probability distribution over tokens with no grounding in an external world model, it will sometimes generate plausible text that is false. The question is not "will it hallucinate?" but "can we reduce the rate to acceptable levels for the application?"

---

## Q&A

**Question:** You listed five causes of hallucination. If you could fix exactly one, which would reduce hallucination the most?

**Student's Answer:** The lack of an internal "I don't know" mechanism. The other causes — frequency-based training, exposure bias, lossy compression, sycophancy — would all be less harmful if the model could reliably detect when it's uncertain and abstain. A model that confabulates but *knows it's confabulating* could flag low-confidence claims. The fundamental problem is that the model's internal confidence is disconnected from its factual accuracy.

**Evaluation:** That's a defensible and insightful choice. Calibrated uncertainty would indeed be a force multiplier — it would make RAG more effective (retrieve only when uncertain), make chain-of-thought more reliable (flag steps where reasoning is shaky), and directly counter sycophancy (uncertainty becomes a virtue, not a penalty). Some researchers would argue that "trained on frequency, not truth" is more fundamental — you can't calibrate uncertainty about facts you never had good signal for. But your reasoning about calibration as a *meta-fix* that amplifies every other mitigation is exactly right. The field is converging on this view: the next frontier isn't better generation, it's better *self-knowledge*.
