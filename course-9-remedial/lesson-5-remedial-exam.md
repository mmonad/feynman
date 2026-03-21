# Remedial Lesson 5: Remedial Exam

*Course 9: Remedial — Strengthening the Weak Spots*

## Results: 12/15 clean, 3/15 partial

### All Questions and Answers

R1: Finite order vs generator → ✅ FIXED (order 8 in group of 16: not generator, has finite order)
R2: Forge signature after 50 queries → ✅ EUF-CMA
R3: Find hash preimage for hash lock → ✅ preimage resistance
R4: Why CheckSig tx is unstoppable → ✅ EUF-CMA (cause, not effect) — FIXED
R5: Extractor in security proof → ✅ knowledge soundness
R6: Blocks keep coming → ✅ liveness
R7: Can't create conflicting tx → ✅ unstoppability
R8: Groth16 proving time → ✅ O(n log n), FFTs for polynomial multiplication
R9: Component identification (5 ops) → ✅ 5/5 all correct — FIXED
R10: Reduction for Lamport pk collision → ✅ correct identity reduction
R11: RO(x)=0 consequence → ✅ msg exposed in plaintext
R12: ROM defense → ⚠️ one argument (modular reasoning), missed second (empirical track record)
R13: Protocol flow → ⚠️ mostly correct, missing WE/hash lock/Lamport setup steps
R14: BABE reduction fill-in-blanks → ✅ correct structure
R15: Axiom boundary analysis → ⚠️ correctly identified the problem, reversed the direction (EasyCrypt axiom is STRONGER not weaker — claims more than Lean proved)

### Key Correction: "Stronger = Weaker Foundations"

```
Stronger statement = claims more = requires more trust = DANGEROUS
Weaker statement = claims less = only what's proved = SAFE

EasyCrypt axiom: Dec works for ALL (x,w)       → STRONGER claim, unjustified
Lean theorem:    Dec works for (x,w) ∈ R only   → WEAKER claim, proved
```

Cross-boundary translation silently strengthened the claim. This is what Adapter 5 (cross-cutting) should catch.

### Gap Closure Summary

```
ORIGINAL EXAM:    104.5/135 (77%)
REMEDIAL EXAM:    12/15 (80%) on previously-weakest questions

CLOSED: reductions, finite order/generator, preimage/collision/EUF-CMA,
        liveness/unstoppability, Groth16/GC/WE components, soundness/knowledge

REMAINING (minor): setup phase details, two-argument ROM defense,
                   stronger-statement-weaker-foundations intuition
```

Remaining gaps are detail-level, not conceptual. Framework of thinking is now sound.
