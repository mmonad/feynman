# Lesson 4: Garbled Circuits

*Course 7: The Cryptographic Toolkit*

## Wire Labels

Each wire gets two random labels: L^0 (for bit 0), L^1 (for bit 1). Labels look like random noise. Evaluator holds exactly one label per wire — knows A label but not WHICH label.

## Garbling One Gate

AND gate, wires a,b → c. Build 4 encrypted entries:

```
Row i,j:  Enc(L_a^i, L_b^j, L_c^{AND(i,j)}) = H(L_a^i ‖ L_b^j) ⊕ L_c^{AND(i,j)}
```

Each entry encrypts the correct output label under the corresponding input labels. **Rows randomly shuffled** — position must not leak input bits.

## Evaluating One Gate

Evaluator has one label per input wire. Tries decrypting each row — only one succeeds (the one matching their actual input labels). Gets output label. Doesn't know which bit it represents.

## Chaining Gates

Output label → input to next gate → cascade through entire circuit → final output decoded via decoding table.

## One-Time Security

Evaluator has one label per wire. Finding the other requires breaking hash (preimage resistance). One set of labels → one evaluation. Same one-time constraint as Lamport signatures — same structure.

## Free-XOR Optimization

Choose all labels so L_w^1 = L_w^0 ⊕ Δ (global secret). XOR gate: evaluator just XORs input labels. **No table, no encryption, no communication. Free.**

## Half-Gates Optimization

AND gates: 2 ciphertext entries instead of 4. Combined with free-XOR:

```
XOR: 0 ciphertexts    AND: 2 ciphertexts
Total GC ≈ 2 × (AND gates) × (label size)
```

For BN254 scalar mult: ~3B AND gates → ~42 GB. Why BABE needs randomized encodings to reduce by 1000×.

## Adaptive Privacy

Adversary chooses input AFTER seeing garbled circuit. Real world indistinguishable from simulation that only knows output C(x). Important for BABE: Verifier gets GC at setup, Prover posts π₁ later (adaptive input choice).

Privacy-free variant: only correctness, not privacy. Used in parts of BABE for authenticated computation.

---

## Q&A

**Question:** What goes wrong if garbled table rows aren't shuffled?

**Student's Answer:** Without shuffling, once a row is successfully decrypted, you also learn which bits the two input wires are. That breaks security.

**Evaluation:** Exactly right. Position = information leakage. Unshuffled: row number reveals input combination. Shuffled: evaluator learns output label and nothing else.

Point-and-permute optimization: each label carries a random pointer bit directing evaluator to correct row — efficiency of direct lookup without leakage of fixed ordering.
