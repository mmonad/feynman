# Lesson 1: Bitcoin as a Cryptographic Platform

*Course 8: BABE — The Protocol and Its Security*

## UTXO Model

Bitcoin tracks unspent transaction outputs, not account balances.

```
UTXO = (amount, locking_script)
Transaction = (inputs, witnesses, outputs)
Rule: total outputs ≤ total inputs (difference = miner fee)
```

## Locking Scripts

1. **CheckSig(pk):** require valid signature under pk
2. **HashLock(h):** require preimage w s.t. H(w)=h — where BABE's secret msg lives
3. **RelativeTimelock(τ):** require τ blocks elapsed — gives Verifier challenge window
4. **Taproot Trees:** multiple spending paths ⟨Leaf₀,...,Leafₖ₋₁⟩, satisfy any one leaf

## The Six BABE Transactions

```
Pegin → Assert → ChallengeAssert → WronglyChallenged → Payout (Prover wins)
                                  → NoPayout (Verifier wins)
```

- Assert: Prover posts Lamport-signed π₁ (= GC input labels)
- ChallengeAssert: Verifier posts matching labels, evaluates GC off-chain
- Valid proof → WronglyChallenged → Payout
- Invalid proof → GC reveals msg → NoPayout (Verifier opens hash lock)

## Timing Parameters

Δ₁: Verifier's challenge window after Assert. Δ₂: Prover's response window after ChallengeAssert. Miss your window → lose.

## Chain Growth (τ-rate)

Blockchain grows at rate τ. Guarantees timing windows are meaningful.

## Unstoppable Transactions

tx is u-unstoppable if adversary can't prevent its inclusion in any of the next u blocks. Transactions requiring honest party's signature are unstoppable (adversary can't forge → can't create conflicts).

```
Assert:            CheckSig(Prover)     → unstoppable
ChallengeAssert:   CheckSig(Verifier)   → unstoppable
WronglyChallenged: HashLock(msg)        → unstoppable IF Prover knows msg
NoPayout:          HashLock(msg)        → unstoppable IF Verifier knows msg
```

## Security Proof Assumptions from Bitcoin

```
Ledger safety:       consistent history across honest parties
Chain growth:        blocks keep coming (timing meaningful)
Liveness:            unstoppable transactions get included
EUF-CMA signatures:  can't forge → can't create conflicts
```

Axiomatized in EasyCrypt/definitions/Ledger.ec.

---

## Q&A

**Question:** What prevents a malicious Prover from front-running the Verifier's ChallengeAssert (seeing it in the mempool and posting a competing transaction)?

**Student's Answer:** It can't, due to unstoppable transactions.

**Evaluation:** Exactly right. ChallengeAssert requires Verifier's signature (CheckSig). Prover can't forge it (EUF-CMA). Can't create a competing transaction spending the same UTXO. ChallengeAssert is unstoppable.

Every critical transaction in the protocol is designed to be unstoppable for the honest party. This is not accidental — the locking scripts are carefully crafted so unstoppability holds.
