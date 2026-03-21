# Lesson 4: The Mechanization

*Course 8: BABE — The Protocol and Its Security*

## The Proof Decomposition Problem

Three types of reasoning, no single tool handles all:

```
TYPE 1: Algebraic correctness    (deterministic, equational)
TYPE 2: Game-based security      (probabilistic, adversarial)
TYPE 3: Asymptotic composition   (analytic, inductive)
```

## What Each Tool Handles

### Lean 4 (Mathlib) — Algebraic Engine
L1-L8, T1, T3, T4 deterministic parts. 101 theorems, 0 sorry, 0 custom axioms. Proves: "decrypt(encrypt(msg)) = msg." Pure algebra, no probability.

### EasyCrypt — Game Engine
T2-T4 game components, L9-L10. Uses pRHL (probabilistic Relational Hoare Logic) for hybrid arguments. Imports Lean's algebra as axioms via `Lean_Algebra.ec`.

### Rocq — Composition Engine
T2, L9, L10 composition. 254-step hybrid induction, negligible function arithmetic. Strong induction + real analysis.

### SSProve — Bridge
Cross-proves 3 axioms that EasyCrypt and Rocq could only axiomatize. Package algebra + coupling arguments. Eliminates trust boundaries.

## Axiom Boundaries

```
Lean ──axioms──► EasyCrypt ◄──proven lemmas── Rocq / SSProve
```

Each boundary = potential semantic gap. Statement might look same in English but differ formally across type systems.

## Axiom Audit Findings

### Lemma 10 Ordering Gap (CRITICAL)
WE security reduction assumed π₁ fixed before ciphertext. Protocol reveals ciphertext first. Level 4 gap — code compiled, proofs type-checked, security argument had subtle ordering error. Fixed with Definition 5' (adaptive-statement extractable WE).

### Two Unsound Axioms (CRITICAL)
EasyCrypt axioms D1, D2 didn't follow from intended assumptions. Corrected.

### Seven Axioms Eliminated
Rocq axioms that were actually provable. Replaced with proofs → reduced trust surface.

### Three Axioms Cross-Proved
SSProve justified claims both EasyCrypt and Rocq axiomatized → trust boundaries eliminated.

## Trust Surface After Audit

```
All axioms classified:
  SAFE: standard (timing, group laws)
  EXTERNAL: cited literature (GGM, Schwartz-Zippel, Groth16)
  PROVEN: converted to lemmas (7 eliminated)
  CROSS-PROVED: SSProve (3 eliminated)
  IRREDUCIBLE: genuine assumptions (bottom of reduction chain)
```

## Connection to Multi-Perspective AGENTS.md

```
Adapter 1 (Property Completeness)  → caught missing adaptive WE definition
Adapter 2 (Threat Model)           → would catch Lemma 10 ordering gap
Adapter 3 (Abstraction Fidelity)   → catches paper-to-code divergence
Adapter 4 (Assumption Audit)       → found unsound D1/D2, 7 provable axioms
Adapter 5 (Cross-Cutting)          → found boundary mismatches, SSProve opportunities
```

Axiom audit = manual run of the Course 3 committee architecture.

---

## Q&A

**Question:** How do you know when to stop auditing? More provable axioms might exist. More subtle gaps might hide.

**Student's Answer:**
1. The mechanization's trust surface should be ≤ the paper's trust surface (code axioms ≤ paper assumptions)
2. Once code matches paper in axioms and proof is correct, security reduces to axiom security; stop at commonly accepted axioms as the practical boundary

**Evaluation:** Both points exactly right. Together they define a precise stopping criterion.

### The Combined Stopping Rule

```
STOP when BOTH hold:
  1. Every code axiom maps to a paper assumption or cited external result
     (mechanization ≤ paper in trust)
  2. Every paper assumption is either proved in code OR traces to
     widely-accepted cryptographic assumption
     (chain bottoms out at accepted foundations)
```

Remaining irreducible axioms: GGM (1997), ROM (1993), hash preimage resistance (1970s), Groth16 soundness in GGM (2016), Bitcoin safety/liveness (2015). Studied for decades. Going further requires solving P vs NP. The community agrees: these are the reasonable foundations.

Current lean-babe status satisfies both conditions. Audit is thorough enough — not because everything is checked (impossible), but because trust is reduced to foundations the entire field accepts.
