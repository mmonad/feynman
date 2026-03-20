# Lesson 5: Witness Encryption

*Course 7: The Cryptographic Toolkit*

## Concept: Encrypt Under a Statement

Normal encryption: encrypt under a KEY, decrypt with the key.
Witness encryption: encrypt under a STATEMENT, decrypt with the WITNESS.

```
Enc(crs, statement, msg) → ct
Dec(crs, ct, witness) → msg     only if (statement, witness) ∈ R
```

**Lockbox analogy:** Box that opens only if you solve a specific puzzle. Don't need to know who solves it or share keys in advance. For BABE: puzzle = "produce a valid Groth16 proof."

## BABE's WE Construction

Not general-purpose WE (expensive, exotic assumptions). WE for a **specific relation** — Groth16 verification — using pairings.

### The Relation R'

```
R' = { ((crs, x, π₁); w) : ∃ π₂,π₃ s.t. Groth16.Verify(crs, x, (π₁,π₂,π₃)) = 1 ∧ (x,w) ∈ R }
```

Statement = (crs, x, π₁) — public. Witness = w (implies π₂, π₃). π₁ is public because it's posted on Bitcoin via Lamport signature.

### Construction

Exploits Groth16 verification equation: e(π₁,π₂) = e([α]₁,[β]₂)·e(X,[γ]₂)·e(π₃,[δ]₂)

```
Enc: r ←$ F_p*, Y ← e([α]₁,[β]₂)·e(X,[γ]₂), ct ← (r·[δ]₂, RO(r·Y) ⊕ msg)
Dec: compute π₂,π₃ from w → complete pairing → recover r·Y → RO(r·Y) ⊕ ct₂ = msg
```

```
WITH witness:    complete pairing → recover r·Y → decrypt ✓
WITHOUT witness: can't complete pairing → RO(r·Y) random → msg hidden ✓
```

## Extractable Security

```
If Pr[A distinguishes encryptions] = 1/2 + ε
Then Pr[E extracts valid witness] ≥ ε - negl(λ)
```

Only way to distinguish = know the witness. Needed because BABE's security proof chains extractors through multiple primitives.

## Two-Phase Construction (Lemma 10 Fix)

Axiom audit found gap: adversary sees ciphertext then chooses π₁, but original definition assumed π₁ fixed first.

Fix — adaptive-statement extractable WE:
```
Phase 1 (setup, before π₁): ct_setup = (r·[δ]₂, RO(r·Y) ⊕ msg)  — independent of π₁
Phase 2 (proving, after π₁): ct_prove = r·π₁                      — uses same r
```

ct_setup independent of π₁ → adaptive ordering safe. Exactly the kind of Level 4 gap our orthogonal adapter committee should catch.

## How WE + GC Combine: The BABE Core

```
SETUP:
  1. Prover generates secret msg
  2. WE ciphertext: ct_setup (msg encrypted under "valid Groth16 proof exists")
  3. Garbled circuit: implements Groth16 verification
  4. Prover publishes H(msg) as hash lock on Bitcoin
  5. Prover sends (ct_setup, GC) to Verifier

PROVING:
  6. Prover posts Lamport-signed π₁ on Bitcoin
  7. Verifier takes Lamport signature from Bitcoin = GC input labels
  8. Verifier evaluates GC:
     Invalid proof → GC reveals msg → Verifier opens hash lock → Verifier wins
     Valid proof → msg hidden → timeout → Prover withdraws BTC → Prover wins
```

**Security:**
- Robustness: valid proof → GC doesn't reveal msg → Prover wins
- Knowledge soundness: invalid proof → GC reveals msg → extractable WE ensures any successful Prover must know witness

---

## Q&A

**Question:** Looking at the full BABE construction, identify the single most elegant design choice that minimizes on-chain cost.

**Student's Answer:** The only on-chain cost is the Lamport signature; the fact that it's also the input to an offline GC is where the dramatic cost reduction happens. It serves as both a commitment from the Prover and enables offline computation.

**Evaluation:** Perfect — and actually three purposes in one action:

```
One Lamport signature on Bitcoin (~$40):
  1. COMMITMENT:      Prover bound to π₁ on blockchain
  2. GC INPUT:        Verifier uses signature as GC input labels (off-chain)
  3. WE DECRYPTION:   GC evaluation reveals/hides msg → hash lock resolution
```

BitVM2: post intermediate states on-chain → megabytes → $15,000
BABE: post one Lamport signature → kilobytes → $40

The Lamport-GC coincidence is the **entire economic thesis** of the protocol.

## Course 7 Summary

Five primitives in the toolkit:
1. Elliptic curves (the group structure)
2. Bilinear pairings (multiplication of hidden values)
3. Groth16 (succinct proof with 4-pairing verification)
4. Garbled circuits (encrypted computation, one-time evaluation)
5. Witness encryption (encrypt under a statement, decrypt with witness)

All combine in BABE through the Lamport-GC coincidence.
