# Lesson 2: The BABE Construction

*Course 8: BABE — The Protocol and Its Security*

## The Size Problem

Full Groth16 verifier as Boolean GC: pairings = ~10B gates → 42 GB. Impractical.

## BABE's Split

```
Pairings:           WE (algebraic, free)
Scalar multiplication: GC (Boolean, focused)
```

Even scalar mult alone: 254 iterations × field mults × ~64K AND gates = still GB-scale.

## Randomized Encodings (RE)

f̂(x; r) is an RE of f(x) if: correctness (can recover f(x)), privacy (reveals nothing else), simplicity (structurally cheaper to compute).

### Key Insight: Linear Is Free

Free-XOR: XOR gates cost 0 in garbled circuits. If computation becomes LINEAR (additions/XOR), it's free. Only AND gates cost.

**Point addition with one fixed point → linear function of the other point's coordinates.**

But: raw coordinates (x,y) don't make it linear (division by variable term).

**Solution: extended representation.** Represent point as:
```
ū(P) = (1, x, y, x², y², xy)     6 components
```

Including quadratic monomials as separate variables. Addition formula becomes:
```
ū_output = M × ū_input    (M is 6×6 matrix, depends only on fixed point Q)
```

Linear in ū — nonlinearity absorbed into the representation ("linearization by lifting to monomial space").

## Doubly Randomized Encodings (DRE)

"Doubly" = randomized at both input and output of each step.

At each step: garbler precomputes quadratic terms (x₃², y₃², x₃y₃) and bakes them into garbled tables with fresh randomness. Evaluator receives randomized extended vector for next step without computing quadratics themselves.

```
Matrix multiplication: FREE (XOR)
Re-encoding to extended form: small number of AND gates (the only cost)
```

Privacy: fresh randomness per step masks all intermediate values. Evaluator learns only final result.

## Complete BABE Construction

```
SETUP (off-chain):
  1. Groth16 CRS generation
  2. Prover: msg ←$ {0,1}^λ, creates WE ciphertext + DRE-based GC
  3. Prover publishes H(msg) hash lock on Bitcoin
  4. Prover sends (ct_setup, GC, Lamport pk) to Verifier

PROVING (on Bitcoin):
  5. Assert: Prover posts Lamport-signed π₁ (= GC input labels)
  6. Verifier evaluates GC (DRE scalar mult, efficient!)
  7. WE + GC combined: reveals msg iff proof invalid
  8. Valid → Prover withdraws after timeout
     Invalid → Verifier opens hash lock, blocks Prover
```

## Size Comparison

```
                    On-chain       Off-chain GC
BitVM2:             ~$15,000       N/A
BitVM3 (full GC):   ~$40           ~42 GB
BABE (WE + DRE):   ~$40           ~tens of MB
```

---

## Q&A

**Question:** Addition formula has division λ = (y₂-y₁)/(x₂-x₁). Even with Q fixed, (x_Q - x_P)⁻¹ depends on variable x_P. How can the operation be linear?

**Student's Answer:** In F_p, division is multiplication by the multiplicative inverse.

**Evaluation:** Correct about F_p arithmetic, but doesn't resolve linearity — the inverse still depends on the variable. The actual solution:

1. **Extended representation** ū = (1, x, y, x², y², xy) — includes quadratic monomials
2. Addition formula becomes M × ū — **linear in ū** even though ū contains nonlinear terms
3. Quadratic terms for next step precomputed by garbler, baked into garbled tables with DRE randomization
4. AND gates only for re-encoding; matrix multiplication is XOR (free)
