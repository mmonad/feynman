# Lesson 4: Digital Signatures and Security Games

*Course 6: Cryptography from First Principles*

## What Signatures Do

Prove authenticity — anyone can verify, nobody can forge.

```
Gen(1^λ) → (sk, pk)          generate keys (1^λ = "security parameter λ")
Sign(sk, m) → σ              sign message with secret key
Verify(pk, m, σ) → 0/1       check signature validity
```

**Notation: 1^λ** = unary encoding of λ (string of λ ones). Just read as "the security parameter."

## EUF-CMA Security Game

**Existential Unforgeability under Chosen Message Attack** — gold standard:
- Existential: forge on ANY message (not a specific one)
- Chosen Message Attack: adversary can request signatures on chosen messages before forging

```
Pr[
    Verify(pk, m*, σ*) = 1 ∧ m* ∉ Q
    :
    (sk, pk) ← Gen(1^λ)
    (m*, σ*) ← A^{Sign(sk,·)}(pk)
] ≤ negl(λ)
```

### Notation: A^{Sign(sk,·)}(pk)

```
A              adversary algorithm
^{Sign(sk,·)}  oracle access to signing (superscript = access)
               dot (·) = "adversary fills in this argument"
               adversary calls Sign(sk, m) for any m, but sk is hidden
(pk)           actual input to A
```

**Dot notation** means "this argument supplied by caller, others fixed by game."

## Lamport One-Time Signatures

Built entirely from hash functions. Used in BABE because verifiable in Bitcoin script.

### Key Generation (for ℓ-bit messages)

```
sk = { L_i^b }    2×ℓ matrix of random λ-bit values (two per bit position)
pk = { H(L_i^b) } hashes of those values

         bit 0    bit 1   ...  bit ℓ-1
    0:   L_0^0    L_1^0        L_{ℓ-1}^0       sk (secret: random values)
    1:   L_0^1    L_1^1        L_{ℓ-1}^1

         H(L_0^0) H(L_1^0)    H(L_{ℓ-1}^0)    pk (public: their hashes)
         H(L_0^1) H(L_1^1)    H(L_{ℓ-1}^1)
```

### Signing m = (m_0, ..., m_{ℓ-1})
Reveal L_i^{m_i} for each bit — the secret matching the actual bit value.

### Verification
Hash each revealed value, compare to corresponding public key entry.

### Concrete Example (ℓ=3, message "101")
```
sk row 0: "aaa" "bbb" "ccc"    pk row 0: H("aaa") H("bbb") H("ccc")
sk row 1: "ddd" "eee" "fff"    pk row 1: H("ddd") H("eee") H("fff")

Sign "101": σ = ("ddd", "bbb", "fff")
Verify: H("ddd")=pk_0^1 ✓  H("bbb")=pk_1^0 ✓  H("fff")=pk_2^1 ✓
```

### Why One-Time
After signing "101": adversary knows L_0^1, L_1^0, L_2^1 (one per position).
After signing "101" AND "110": adversary knows BOTH values for bits 1 and 2. Can forge. Security broken after two signatures.

For BABE: fine. Prover signs exactly once per protocol instance.

## The Magical Coincidence: Lamport = Garbled Circuit Inputs

```
Lamport secret key  =  GC encoding key       { L_i^0, L_i^1 }
Lamport signature   =  GC input encoding      (L_0^{m_0}, ..., L_{ℓ-1}^{m_{ℓ-1}})
Signing a message   =  providing GC input labels
```

BABE exploits this: Prover posts Lamport-signed π₁ on Bitcoin. That signature simultaneously:
1. **Commits** Prover to π₁ on blockchain (Lamport function)
2. **Provides input labels** for Verifier to evaluate garbled circuit (GC function)

One action, two purposes. "An uncanny coincidence" — Yao (1982) and Lamport (1979) independently created identical structures.

---

## Q&A

**Question:** EUF-CMA allows many signature queries. Lamport breaks after two. Is Lamport EUF-CMA secure? What weaker notion does it satisfy? Why sufficient for BABE?

**Student's Answer:** Lamport is not EUF-CMA secure but still secure as one-use only. For BABE, the secret key is never reused.

**Evaluation:** Exactly right.

```
EUF-CMA:    q signatures allowed (polynomial)  → Lamport FAILS
EUF-1-CMA:  exactly 1 signature allowed        → Lamport SECURE
```

Sufficient because BABE generates fresh keys per instance, uses each exactly once.

**General principle:** Don't need strongest security notion — need one matching usage pattern. Using one-time primitive exactly once is perfectly secure.

**Deeper connection:** One-time restriction on Lamport AND garbled circuits is the SAME restriction, same structure:
```
Two Lamport sigs  → learn both L_i^0 and L_i^1 → forge anything
Two GC evals      → learn both L_i^0 and L_i^1 → evaluate anything
Same structure, same vulnerability, same fix: use exactly once.
```
