# Lesson 3: Fine-Tuning — Rewiring the Brain

*Course 1: In-Context Learning vs Fine-Tuning*

## Core Concept

Everything so far has left the sculpture untouched. Now we pick the chisel back up.

## What Fine-Tuning Actually Is

Take a model that's already been trained — the sculpture is done — and run *additional training steps* on a new, usually much smaller, dataset. This means:

**The weights actually change.** The stone gets re-carved. Not from scratch — you're making small, precise modifications to an existing sculpture. Sharpening a cheekbone here, adjusting an angle there.

Mechanically, it's the same process as the original training — forward pass, compute loss, backpropagate, update weights — but with three important differences:

1. **Much less data.** Maybe thousands or tens of thousands of examples instead of trillions of tokens.
2. **Much smaller updates.** Tiny learning rate — fine chisel and gentle tap, not a sledgehammer.
3. **Targeted behavior.** Not teaching "everything about language" — teaching something specific: "respond in this style," "follow this format," "excel at this domain."

## The Piano Tuning Analogy

A grand piano already built and tuned at the factory — it can play any piece of music. That's the pre-trained model.

Fine-tuning is like bringing in a specialist who *slightly* adjusts the tension of certain strings. Maybe you're putting this piano in a jazz club, so certain notes ring warmer, a little looser. The piano is still a piano. It can still play classical. But there's a subtle bias — a *tendency* — toward jazz.

Key part: **once the strings are adjusted, they stay adjusted for every single person who plays the piano.** It doesn't matter what song you request. The tuning affects everything. This is fundamentally different from in-context learning, where you're just choosing what song to request and the piano itself never changes.

## The Danger: Catastrophic Forgetting

Re-carving an existing sculpture is dangerous. Swing the chisel too hard or too many times, and you **destroy features that were already there**.

**Catastrophic forgetting:** Fine-tune a model to be amazing at medical diagnosis, and suddenly it can't write poetry anymore. You carved the doctor's face into the stone, but accidentally chiseled off the poet's hands.

This is why fine-tuning is delicate work. The learning rate has to be low. The dataset has to be curated carefully. You're always making a trade-off — every tap that makes one thing better *risks* making something else worse.

With in-context learning, this risk is literally zero. You can't damage the sculpture by pouring water on it.

## What Fine-Tuning Is Good At (That In-Context Learning Can't Do)

1. **Changing the model's deep default behavior.** Want the model to *always* respond in JSON, in every conversation, without being asked? That's a new default. That's stone-level.

2. **Teaching genuinely new knowledge.** A specialized medical ontology, an internal company taxonomy — things not in the training data and too large for a context window.

3. **Efficiency at inference time.** Fifty examples in the prompt = fifty examples processed every single call. That costs computation and money. Fine-tuning bakes those examples *into the weights* — free at runtime.

---

## Q&A

**Question:** A company fine-tunes a model on 10,000 customer support conversations in a very specific style — short, friendly, always ending with "Is there anything else I can help with?" After fine-tuning, an employee puts a system prompt saying: "Never end your messages with a question. Use a formal, authoritative tone." What happens? Which force wins — the stone or the water? And why?

**Student's Answer:** The stone wins because the employee didn't provide any examples to counter the fine-tuning data sets, therefore there is no familiar pattern to activate.

**Evaluation:** Right neighborhood — the fine-tuned behavior does have an advantage — but the reasoning had a gap. The model can follow direct instructions without examples (the instruction-following machinery is in the weights). The prompt *isn't* powerless.

**Corrected explanation via the Habit Analogy:** Imagine a person who has developed a deep habit over months of practice — always ending with that polite question. Their boss hands them a memo saying "stop doing that." They *read* the memo. They *understand* it. But in the heat of conversation, on autopilot, **the habit leaks through**.

It's not a clean victory for either side. It's a **tug-of-war**, often resulting in a weird blend — tone shifts *somewhat* toward formal, but the closing question keeps sneaking back in. The stone wins not because the water is inert, but because **habits carved into weights operate at every layer of processing**, while the prompt is just one input signal competing against those habits.

**Follow-up Question:** Can you make the water win against the stone, without re-fine-tuning?

**Student's Answer:** Provide a lot of examples of authoritative ending in the prompt.

**Evaluation:** Exactly right. Loading the prompt with concrete examples creates a strong pattern that competes with the habit in the weights. Turning up the volume of the water until it drowns out the muscle memory of the stone.

**Ironic observation:** This creates a situation where a company *paid* to fine-tune specific behavior, and now someone has to *stuff the prompt* to fight against that fine-tuning. Raises the question: was fine-tuning the right choice to begin with?
