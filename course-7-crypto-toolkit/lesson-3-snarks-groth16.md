# Lesson 3: SNARKs and Groth16

*Course 7: The Cryptographic Toolkit*

## SNARK = Succinct Non-interactive Argument of Knowledge

```
Succinct:        constant-size proof, constant-time verification
Non-interactive: one message from Prover to Verifier
Argument:        secure against computationally bounded adversaries
Knowledge:       Prover must actually KNOW the witness
```

Three algorithms: Gen(R)→crs, Prove(crs,x,w)→π, Verify(crs,x,π)→0/1

Three properties: correctness (honest proofs verify), succinctness (proof size independent of witness), knowledge soundness (extractors can recover witness from any successful prover).

## The Groth16 Flow

### Step 1: Setup — Gen(R) → crs

Sample trapdoor: τ, α, β, γ, δ ←$ F_p (toxic waste — must be destroyed).

CRS contains group elements encoding:
- Powers of τ: {[τⁱ]₁, [τⁱ]₂} — so Prover can evaluate polynomials at τ
- Polynomial evals: {[aᵢ(τ)]₁, [bᵢ(τ)]₂} — R1CS polynomials at τ
- Terms/δ: quotient polynomial lands correctly for verification
- Terms/γ: separates public from private parts

CRS = precomputed toolkit for working with τ without knowing it.

### Step 2: Prove — Prove(crs, x, w) → π

```
2a: z = (x ‖ w)                           form extended witness
2b: q(X) = [left·right - output] / V(X)   compute quotient (exact IFF R1CS satisfied)
2c: construct proof elements from CRS:
    π₁ = [α]₁ + Σᵢ zᵢ·[aᵢ(τ)]₁           ∈ G₁
    π₂ = [β]₂ + Σᵢ zᵢ·[bᵢ(τ)]₂           ∈ G₂
    π₃ = witness terms + quotient terms     ∈ G₁
```

Proof = three group elements (π₁, π₂, π₃). Constant size regardless of computation.

### Step 3: Verify — Verify(crs, x, π) → 0/1

```
3a: X = Σᵢ xᵢ · [public_terms/γ]₁        compute from public statement
3b: check: e(π₁, π₂) = e([α]₁,[β]₂) · e(X,[γ]₂) · e(π₃,[δ]₂)
```

Four pairings. Done.

## Why the Verification Equation Works

Honest π₁ encodes (α + left(τ)), honest π₂ encodes (β + right(τ)). Their pairing expands to αβ + cross terms + left(τ)·right(τ). The R1CS product equals output(τ) + q(τ)V(τ) when computation is correct. Everything balances.

Dishonest Prover: q(X) doesn't exist as proper polynomial → can't construct valid π₃ → would need τ or δ (discrete log) to fake.

## Knowledge Soundness

Stronger than plain soundness. Uses an **extractor** E:

```
Pr[Verify(crs,x,π)=1 ∧ (x,w)∉R : crs←Gen(R), (x,π)←A(crs,aux), w←E(crs,aux)] ≤ negl(λ)
```

"If A produces valid proof, extractor E can extract valid witness. If extracted witness invalid, proof must have been invalid."

Holds in the Generic Group Model for Groth16.

## Groth16 in BABE

- Prover computes Groth16 proof for application's relation
- π₁ posted on Bitcoin via Lamport signature
- Bitcoin can't verify Groth16 (no pairing support)
- BABE garbles the Groth16 verifier into a garbled circuit
- GC's key operation: scalar multiplication α·π₁ on BN254
- BABE's innovation: reducing GC size by 1000×

---

## Q&A

**Question:** Why does BABE need knowledge soundness rather than plain soundness?

**Student's Answer:** If the proof is correct, the extractor is guaranteed to extract the witness which activates certain action as a consequence. Without knowledge soundness, a fake proof considered correct but with unextractable witness means the subsequent action can't take place.

**Evaluation:** Right neighborhood — extractor enabling subsequent action is the key. Sharpened: the extractor is used in the SECURITY PROOF, not the running protocol.

BABE's knowledge soundness theorem (T4):
```
1. Assume malicious Prover produces valid proof π
2. By Groth16 knowledge soundness: extractor E extracts witness w
3. Check (x,w) ∈ R → Prover had valid claim → no security violation
4. Therefore: only Provers with valid witnesses can withdraw
```

Step 2 requires knowledge soundness. Plain soundness gives "false statements can't be proved" but no extractor to connect "valid proof" to "Prover knew the answer."

**General principle:** Knowledge soundness needed whenever one protocol's security proof needs to reach inside another protocol's Prover and extract information. BABE composes Groth16 + WE + GC and chains extractors through the composition.

```
Plain soundness:      fact about the proof ("it's correct")
Knowledge soundness:  fact about the Prover ("they know why")
```

Extractor is a proof tool, not a runtime tool.
