# Lesson 5: Polynomials and Arithmetic Circuits

*Course 6: Cryptography from First Principles*

## Core Question

How can you verify a massive computation by checking a single number? The answer is polynomials.

## Polynomials Over Finite Fields

Same as school polynomials but coefficients and variables live in F_p. Key property: a nonzero polynomial of degree d has **at most d roots**.

Consequence: two degree-d polynomials agreeing at more than d points must be identical.

## Schwartz-Zippel Lemma

If f(X) is nonzero of degree d over F_p, and r ←$ F_p:

```
Pr[f(r) = 0] ≤ d/p
```

For BABE: d ≈ millions, p ≈ 10^76, so d/p ≈ 10^(-70). **One random check is as good as checking every point.**

## Arithmetic Circuits

Any computation expressed as + and × gates over F_p. Universal language for SNARKs.

## R1CS (Rank-1 Constraint Satisfiability)

Each multiplication gate → one constraint: (left) × (right) = (output).

Three matrices A, B, C ∈ F_p^{n×m} (n constraints, m variables):

```
(A × z^T) ∘ (B × z^T) = C × z^T

z = (x ‖ w)     extended witness (statement ‖ secret witness)
∘               Hadamard product (element-wise multiplication)
```

### Notation reference:
```
z = (x ‖ w)       x concatenated with w
∘                  element-wise multiplication (NOT matrix mult)
A ∈ F_p^{n×m}     n-by-m matrix over field F_p
Σᵢ zᵢ · aᵢ(X)    sum over variables: z_i times polynomial a_i(X)
V(X) = ∏ᵢ(X-ωⁱ)  vanishing polynomial (zero at all constraint points)
q(X)               quotient polynomial (Prover computes this)
```

## From R1CS to Polynomials

Columns of A, B, C interpolated via Lagrange into polynomials a_i(X), b_i(X), c_i(X). R1CS becomes:

```
(Σᵢ zᵢ·aᵢ(X))·(Σᵢ zᵢ·bᵢ(X)) - Σᵢ zᵢ·cᵢ(X) = q(X)·V(X)
```

Computation correct ↔ left side divisible by V(X) ↔ valid q(X) exists.

## Why SNARKs Are Possible

1. Computation → circuit → R1CS → polynomial equation
2. Polynomial correct ↔ computation correct
3. Schwartz-Zippel: check at single random point, catch cheaters with prob ≥ 1-d/p

Millions of gates → handful of group elements. That's "succinct."

## CRS (Common Reference String)

```
crs ← Gen(R)    generate shared rulebook for relation R
```

Contains encoded powers of secret τ: {[τⁱ]₁, [τⁱ]₂}. Both Prover and Verifier use it. Neither knows τ.

**Trusted setup:** whoever runs Gen knows τ momentarily, must delete it ("toxic waste"). If kept, can forge proofs.

---

## Q&A

**Question:** τ is fixed in CRS, not chosen fresh per proof. Doesn't this break Schwartz-Zippel?

**Student's clarifying question:** "What's CRS?" → Common Reference String. Public string from setup, shared rulebook. Contains group-encoded secret values.

**Student's Answer:** Fixed beforehand doesn't mean it's not random — the Prover never learns τ.

**Evaluation:** Exactly right. From the Prover's perspective, τ is random because it's hidden behind the discrete log problem:

```
Prover sees:  [1]₁, [τ]₁, [τ²]₁, ...  (group elements)
Prover needs: τ                          (the number)
Gap:          discrete log (infeasible)
```

Prover can evaluate polynomials at τ (linear combinations of [τⁱ]₁) but can't learn τ itself. Can't craft a polynomial to cheat at a specific τ they don't know.

Schwartz-Zippel requires: polynomial determined before adversary learns evaluation point. Groth16 provides: adversary never learns τ (hidden behind DLP). Condition satisfied.

CRS gives **just enough** to compute with τ (evaluate in the exponent) and **nothing** to cheat with.

## Course 6 Summary

Five lessons building the cryptographic foundation:
1. Security parameters, negligible functions, PPT adversaries, security games
2. Finite fields, modular arithmetic, groups, generators, discrete log
3. Hash functions, random oracle model, notation boot camp
4. Digital signatures, EUF-CMA, Lamport, the Lamport=GC coincidence
5. Polynomials, Schwartz-Zippel, R1CS, why SNARKs work, CRS
