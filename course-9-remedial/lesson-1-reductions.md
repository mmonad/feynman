# Remedial Lesson 1: Reductions — The Technique

*Course 9: Remedial — Strengthening the Weak Spots*

## The Recipe (Three Steps, Every Time)

```
STEP 1: STATE THE ASSUMPTION     "Problem X is hard"
STEP 2: ASSUME THE OPPOSITE      "Suppose A breaks my scheme S"
STEP 3: BUILD THE MACHINE        "Construct B that uses A to break X"
                                  → X is hard → A can't exist → S is secure
```

## Worked Examples

### Example 1: Enc' = Enc ‖ 0 (trivial)
B receives Enc challenge ct, appends 0, forwards to A. A's advantage against Enc' = B's advantage against Enc. Enc secure → Enc' secure.

### Example 2: Commitment = H(m) (identity reduction)
A breaks binding → produces m₁≠m₂ with H(m₁)=H(m₂) → this IS a collision → breaks H collision resistance.

### Example 3: Lamport EUF-1-CMA (embedding + guessing)
B embeds preimage challenge y at a random bit position i. When A forges, A reveals preimage at position i. Tightness loss: 1/ℓ (probability of guessing correct position).

Key techniques: embedding (plant challenge in scheme), simulation (A can't tell it's in a reduction), extraction (convert A's forgery to solution), guessing (lose a polynomial factor).

### Example 4: BABE → Groth16 (sketch)
B simulates BABE using real Groth16 CRS. If A cheats BABE, either A's proof was valid (no break), GC failed (eliminated by separate reduction), or WE failed (eliminated). Remaining: A produced valid Groth16 proof → extractor recovers witness → contradiction or A had valid claim.

## The Pattern

```
1. Assume X is hard
2. Suppose A breaks my scheme
3. Build B that simulates my scheme using X's challenge
   → A can't tell it's in a reduction
   → A's success translates to breaking X
4. X is hard → contradiction → scheme is secure
```

---

## Q&A: Student Constructs a Reduction

**Scheme:** Sig'.Sign(sk,m) = Sig.Sign(sk, H(m)). Sign the hash, not the message.
**Claim:** Sig EUF-CMA + H collision-resistant → Sig' EUF-CMA.

**Student's Reduction (case analysis):**

Case 1: H(m*) ≠ any H(m_i) previously signed → σ* is a Sig forgery on new message H(m*) → breaks Sig EUF-CMA

Case 2: H(m*) = H(m_j) for some queried m_j, but m*≠m_j → (m*, m_j) is a collision → breaks H collision resistance

Both cases → contradiction → Sig' is secure ∎

**Evaluation:** Correct. The student independently discovered **case-split reduction** — reducing to the conjunction of two assumptions. More advanced than any worked example. Previous "pass" on E1 was confidence, not capability. Worked examples provided the template; reasoning was already there.
