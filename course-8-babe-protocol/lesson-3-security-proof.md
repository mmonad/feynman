# Lesson 3: The Security Proof

*Course 8: BABE — The Protocol and Its Security*

## Two Security Properties

### Robustness: Honest Prover Wins
If Prover has valid witness and follows protocol, they win — even against malicious Verifier. Pr[withdraw] ≥ 1 - 2^(-κ) - negl(λ).

### Knowledge Soundness: Cheating Prover Loses
If Prover lacks valid witness, Verifier wins — even against malicious Prover. Uses extractor: if Prover managed to withdraw, extractor recovers valid witness (meaning Prover actually had valid claim).

## Honest-Setup vs Malicious-Setup

Stage 1 (warm-up): both parties run setup honestly. Stage 2 (full): adversary can cheat during setup. Malicious-setup adds validation checks — each party verifies what it receives, aborts if invalid.

## The Hybrid Argument

Core proof technique. Prove "Game 0 indistinguishable from Game N" via intermediate games:

```
G₀ → G₁ → G₂ → ... → Gₙ
real   slightly    ...     trivially
       modified            secure
```

Each consecutive pair indistinguishable (by reduction to a cryptographic assumption). By triangle inequality: |Pr[win G₀] - Pr[win Gₙ]| ≤ N · negl = negl.

**Color-changing room analogy:** Red → orange-red → orange → ... → blue. Each step imperceptible. After enough steps, red becomes blue without anyone noticing.

## BABE Knowledge Soundness Proof Structure

**Game 0:** Real protocol. Malicious Prover tries to withdraw without witness.

**Game 1:** Replace GC with simulation (by GC adaptive privacy).

**Game 2:** Replace WE encryption of real msg with random message (by WE extractable security — this is where the extractor kicks in).

**Game 3:** Reprogram random oracle (standard ROM technique).

**Game N:** Trivially secure — hash lock preimage unknown to everyone.

```
Pr[win real] ≤ negl(GC) + negl(WE) + negl(ROM) + negl(hash) = negl(λ)
```

## The Reduction Chain

```
BABE knowledge soundness
  ├── GC adaptive privacy → random oracle
  ├── WE extractable security → GGM + random oracle → Master Theorem
  ├── Lamport EUF-1-CMA → hash preimage resistance
  └── Bitcoin: safety, chain growth, liveness + unstoppability
```

Bottom = axioms. Everything above = proved by reduction.

## Why Four Proof Assistants

```
Algebraic correctness:     Lean 4 (Mathlib)
Game-based security:       EasyCrypt (pRHL)
Asymptotic composition:    Rocq
Game reductions in Rocq:   SSProve
```

Each handles a different logical foundation. Boundary axioms are what the axiom audit checks.

---

## Q&A

**Question:** In Game 2, WE ciphertext is replaced. But the adversary gets partial information from Lamport labels on π₁. Does this help distinguish?

**Student's Answer:** "I don't know — I'd be guessing."

**Evaluation:** The right answer when you don't know. Guessing in cryptography introduces proof bugs.

### The Answer: Independence by Construction

The two-phase WE split handles this:

```
ct_setup = (r·[δ]₂, RO(r·Y) ⊕ msg)
Y = e([α]₁,[β]₂) · e(X,[γ]₂)     ← no π₁ in Y

Timeline:
  1. ct_setup created (uses Y, independent of π₁)
  2. π₁ chosen by Prover (AFTER ct_setup)
  3. Lamport labels revealed (for this π₁)
```

Labels leak info about π₁, but ct_setup doesn't depend on π₁. Labels tell you nothing about whether ct_setup encrypts msg₀ or msg₁.

**This IS the Lemma 10 fix.** Original construction didn't have two-phase split. Axiom audit found the ordering gap. Definition 5' (adaptive-statement extractable WE) explicitly models the correct ordering.

Perfect example of a Level 4 gap: proof type-checks, but ordering subtlety in the security argument was wrong. Required multi-perspective verification (abstraction fidelity + threat model adequacy) to catch.
