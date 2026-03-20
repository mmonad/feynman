# Lesson 2: Distillation from First Principles

*Course 5: The Self-Improving Harness*

## Important Reframe

The goal is NOT to replace cloud LLMs with local ones. Cloud LLMs are permanent committee members bringing inter-lab diversity. Distillation makes local LoRA specialists **better at their specific jobs** while the cloud model continues contributing its own perspective.

## What Distillation Actually Is

Normal training uses **hard labels** (ground truth: 1 or 0). Distillation trains on the **teacher model's full probability distribution** — soft predictions across all possible outputs.

### The Dog Show Analogy

Hard label: "Best in Show: Golden Retriever." One fact.

Full scorecard from expert judge:
```
Golden Retriever:  0.45    German Shepherd: 0.30
Border Collie:     0.15    Poodle: 0.08    Chihuahua: 0.02
```

The relative rankings and distances between alternatives = **dark knowledge** (Hinton, 2015). Invisible in the hard label, rich in the distribution. Student learns the *landscape of relationships*, not just the winner.

## Dark Knowledge in Formal Verification

Cloud model's probability distribution over Lean proof steps:
```
"Apply simp [decryption_correct]"              0.40
"Apply rw [gc_scalar_mult_correct] first"       0.25
"Unfold WitnessEncryption.decrypt"              0.18
"The proof needs a stronger hypothesis"          0.12
"This theorem statement is wrong"                0.05
```

Hard output: "use simp." Soft distribution reveals: GC correctness might come first, the theorem might need strengthening, there's a 5% chance the statement is *wrong*. Dark knowledge transfers Level 3-4 awareness to a Level 1-2 model.

## The Distillation Loss Function

```
L_distill = α · CrossEntropy(student_soft, teacher_soft)
          + (1-α) · CrossEntropy(student_output, correct_answer)
```

α controls blend: high α = trust teacher's dark knowledge more.

**Temperature T** amplifies dark knowledge by flattening distributions:
- T=1: highest option dominates (95% one choice)
- T=5-10: alternatives become visible (40/25/15/10/10)

Student learns much more from high-temperature distributions.

## Distillation in Our Architecture: Selective, Not Cloning

```
TRADITIONAL:  Teacher → distill everything → Student ≈ Teacher
OUR DESIGN:   Teacher → selective distill → each LoRA specialist at its job
              Teacher remains in committee as its own perspective
```

Distillation is **directional and selective**: filter traces to relevant adapters.
```
Cloud response about threat model gap:
  → trains Adapter 2 (threat model) with HIGH weight
  → trains Adapter 1 (property completeness) with LOW weight
  → does NOT train Adapter 3 (abstraction fidelity)
```

Routing traces to adapters is itself a design decision.

---

## Q&A

**Question:** Cloud APIs typically DON'T give full probability distributions. You get hard outputs, maybe top-5 logprobs. How do you distill without soft targets?

**Student's Answer:** 1) Prepare teacher traces in SFT format for the student. 2) Use cloud LLM to rewrite each trace in different versions to introduce more training materials.

**Evaluation:** Both instincts correct.

### Instinct 1: Sequence-Level Distillation
Fine-tune student on teacher outputs as ground truth. Lose dark knowledge (uncertainty, rankings) but gain: **teacher's reasoning patterns, formatting, chain-of-thought, domain vocabulary** — encoded in text, not probabilities. For formal verification, when the cloud explains WHY it chose a proof approach, the reasoning pattern IS dark knowledge in text form.

### Instinct 2: Data Augmentation Through Paraphrase
Generate 3-5 rewrites of each trace. Each rewrite samples a different point on the teacher's output manifold. Example:
```
Original:    "Apply simp [decryption_correct]"
Variation 1: "First unfold the definition, then apply simp"
Variation 2: "Use decryption_correct directly via exact"
Variation 3: "Rewrite using gc_scalar_mult_correct, then simp"
Variation 4: "Note: hypothesis needs strengthening first"
Variation 5: "Consider whether theorem statement is too weak"
```

This **reconstructs dark knowledge through text sampling** — variations 1-3 approximate the top of the distribution, variations 4-5 capture the tail where doubt lives. Student trained on all five learns "multiple valid approaches, some question the premise."

### Combined Strategy
```
Step 1: Cloud solves problem → save trace
Step 2: Cloud rewrites in 3-5 variations → save all
Step 3: Cloud critiques its own trace → save (generates Level 3-4 thinking)
Step 4: Route ALL to relevant LoRA specialist as SFT data
```

Gets ~70-80% of full-distribution distillation benefit with zero special API requirements.
