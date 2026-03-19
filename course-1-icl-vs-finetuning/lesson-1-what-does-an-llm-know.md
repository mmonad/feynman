# Lesson 1: What Does an LLM Actually "Know"?

*Course 1: In-Context Learning vs Fine-Tuning*

## Core Question

Before we can talk about two different ways of teaching an LLM something, we need to agree on what it means for an LLM to "know" anything in the first place.

## The Sculpture Analogy

Imagine a gigantic block of marble — billions of pounds of stone. Training an LLM is like spending months with a chisel, carving that block into an incredibly detailed sculpture. Every tap of the chisel is one training step. Every little curve and ridge in the final sculpture encodes some pattern the model learned from the training data.

When training is done, you put down the chisel. The sculpture is fixed. It's solid stone. It doesn't change when someone walks up and looks at it.

**The "knowledge" of the model lives in the shape of the stone itself** — the values of its billions of parameters. It learned that "Paris" relates to "France" the way "Tokyo" relates to "Japan" not because it *remembers reading that*, but because the chisel carved a particular geometry into the stone that makes those relationships fall out naturally.

The critical thing to internalize:

> **After training, the weights are frozen. The sculpture is done. When you send a prompt to the model, you are not changing the sculpture. Not even a little. Not even temporarily.**

The model you're talking to has the exact same weights whether you ask it about cooking or quantum physics. The same sculpture, viewed from different angles.

## So Where Does the "Smartness" Come From?

If the weights don't change, how can the model seem to *adapt* to what you say? How can it follow instructions it's never seen before?

During training, the chisel didn't just carve *facts* into the stone. It carved **a machine for processing sequences of text**. The sculpture isn't an encyclopedia — it's more like a *very sophisticated water slide*. You pour your prompt in at the top, and the shape of the slide determines what comes out at the bottom.

Same slide. Different water. Different output.

## The Key Takeaway: Two Kinds of Memory

The model has two totally separate kinds of "memory":

| | **Parametric Memory** | **Contextual Memory** |
|---|---|---|
| **Where it lives** | In the weights (the stone) | In the prompt (the water) |
| **When it's set** | During training | Every time you send a message |
| **How long it lasts** | Permanent (until retrained) | Gone after the conversation ends |
| **Can you change it?** | Not without retraining | Yes, just change your prompt |

Everything in the next four lessons comes down to this distinction. In-context learning manipulates the *water*. Fine-tuning picks the chisel back up and re-carves the *stone*.

---

## Q&A

**Question:** Suppose you prompt a model with ten examples of translating English to French, and then it successfully translates an 11th sentence. Has the model "learned" French from your prompt? Or did it already "know" French? What's actually happening?

**Student's Answer:** The model already knows French and also learns from the examples that the task is to translate English into French. The in-context learning here is about learning what the prompt (task) is about.

**Evaluation:** Correct. The model already has French carved into its stone — it learned the structure of French from its training data. The ten examples signal to the machinery what you want it to do right now. You're not adding knowledge. You're aiming the water slide.

**Nuance added:** The examples also establish a **format** and a **style** — not just *what* to do, but *how you want it done*. If all examples use a specific tone or formatting convention, the model mimics that too.

**Key student insight:** The student immediately distinguished between *having capability* and *activating capability* — the backbone distinction for the entire course.
