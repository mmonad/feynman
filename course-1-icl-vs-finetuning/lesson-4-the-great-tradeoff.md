# Lesson 4: The Great Trade-Off

*Course 1: In-Context Learning vs Fine-Tuning*

## Core Question

Theory is nice, but someone walks into your office and says "we need the model to do X" — do you pour the water, or pick up the chisel?

## The Restaurant Analogy

You own a restaurant with a brilliant chef who knows every cuisine. Two ways to influence what the chef cooks:

**Option A: Hand them a note every morning.** "Today, focus on Italian. Here are five example dishes I like." This is in-context learning. Flexible — tomorrow you hand a different note. But the chef reads the note every morning, and if it gets too long, they skip parts.

**Option B: Send them to a month-long intensive in Tuscany.** They come back fundamentally changed. They *dream* in Italian food. They default to it. You never have to ask. But good luck getting them to pivot to sushi next month. And the trip wasn't cheap.

The question is never "which is better." The question is: **what's your situation?**

## The Decision Framework

### Pour the water (in-context learning) when:

- **The behavior needs to change frequently.** Different tasks week to week — you don't want to re-carve every time.
- **You don't have much training data.** Fine-tuning on 50 examples is risky. But 50 examples in a prompt is a strong signal.
- **You need to experiment quickly.** Changing a prompt takes seconds. Fine-tuning takes hours/days plus evaluation and deployment.
- **The task is already within the model's strong capabilities.** Translation, summarization, code generation — the sculpture already has deep grooves. Just aim.

### Pick up the chisel (fine-tune) when:

- **The behavior must be consistent across all interactions without prompt overhead.** Same style, format, or personality every time — bake it in.
- **You need to teach genuinely new domain knowledge.** Proprietary ontologies, internal taxonomies — things not in training data that won't fit in context.
- **Latency and cost matter at scale.** A 2,000-token system prompt on every request across millions of users adds up. Fine-tuning amortizes cost into a one-time run.
- **You want to change deep defaults** — personality, refusal behavior, formatting instincts. Surface-level prompting nudges, but the stone wins the tug-of-war for habitual behaviors.

## The Trap Most People Fall Into

> *"The prompt isn't working perfectly, so we should fine-tune."*

No. **Before picking up the chisel, exhaust the water.** Better instructions? More examples? Structured prompts? Chain-of-thought? Different formatting?

Fine-tuning is not a fix for lazy prompting. It's expensive, slow, risks catastrophic forgetting, and worst of all — once behavior is in the stone, it's **hard to undo**. You can always delete a prompt. You can't easily un-fine-tune a model.

> **Stay in the prompt as long as you can. Pick up the chisel only when the water provably can't do the job.**

---

## Q&A

**Question:** You work at a company that receives customer emails in 14 different languages. You need the model to (a) detect the language, (b) respond in that same language, and (c) always follow a specific 5-step response template that changes roughly every quarter. Budget is limited. Fine-tune, in-context learning, or combination? Why?

**Student's Answer:**
1. Easy and cheap to change the template in prompt every quarter
2. Even if 1 example per language, that's 14 good examples, also leveraging LLM's already-learned language skills
3. We are activating, not carving

**Evaluation:** All three points correct.

- Point 1 is the killer argument — re-fine-tuning four times a year vs editing a prompt.
- Point 2 shows internalization of Lessons 1 and 2 — the model already knows 14 languages. Aiming, not carving.
- Point 3 — "activating, not carving" — a better summary of the course than anything the teacher had said. Adopted as shared vocabulary going forward.

**Additional point:** The budget angle — fine-tuning has upfront compute costs, risky investment for something you'd redo quarterly. Prompt approach has near-zero upfront cost.

---

## Interlude: Clarification on Overfitting and Catastrophic Forgetting

*Student pushed back on the claim that "fine-tuning on 50 examples is usually a bad idea — you'll overfit."*

**Student's challenge:** On one hand, 50 examples is nothing compared to pre-training data size — how can it overfit if LR is low enough? And what are the guidelines on fine-tuning dataset size and mix to avoid catastrophic forgetting?

**Response:**

### On the "50 Examples" Claim

The problem isn't one pass through 50 examples. The problem is what people *do* with 50 examples in practice: train for **multiple epochs** — the model sees the same 50 examples over and over. The model doesn't learn the *general pattern*. It **memorizes specific surface features** of those 50 examples.

With 5,000 examples, coincidences wash out. With 50, they don't.

Low LR *reduces* the risk but doesn't eliminate memorization across multiple epochs. Revised claim: **50 examples *can* work with careful hyperparameter discipline, but the margin for error is razor thin.**

### Fine-Tuning Data Guidelines

**Size:**

| Goal | Rough Minimum |
|---|---|
| Adjust tone/style/format | 500–1,000 examples |
| Teach a new task structure | 1,000–5,000 examples |
| Inject real domain knowledge | 10,000+ examples |

Rule that overrides all: **quality crushes quantity.** 1,000 expertly curated examples beat 50,000 noisy ones.

**Mix — The Key to Avoiding Catastrophic Forgetting:**

Mix in general-purpose data alongside specialized data:
- **80–90% general-purpose data** (conversations, instructions, Q&A)
- **10–20% specialized data**

Sounds counterintuitive, but that 80% is the immune system against catastrophic forgetting. Keeps existing grooves intact while carving new ones.

**Technique — LoRA:**

Methods like LoRA freeze most of the model and only train a small set of additional parameters (~0.1% to 1%). Instead of re-carving the sculpture, you're **bolting a small attachment onto it**. Original stone untouched. Catastrophic forgetting nearly impossible.

| Risk | How to Mitigate |
|---|---|
| Overfitting on small data | More data, fewer epochs, low LR, use LoRA |
| Catastrophic forgetting | Mix 80-90% general data with 10-20% specialty data |
| Inconsistent behavior | Curate examples ruthlessly for consistency |
| Overkill | Ask first: can the prompt handle this? |
