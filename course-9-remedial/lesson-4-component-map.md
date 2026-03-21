# Remedial Lesson 4: The BABE Component Map

*Course 9: Remedial — Strengthening the Weak Spots*

## The Four Systems

```
Groth16:  prove computation correct    polynomials + FFTs    GGM + ROM
GC:       verify proof off-chain       Boolean gates         ROM
WE:       bridge algebra ↔ Bitcoin     pairings              GGM + ROM
Bitcoin:  enforce outcomes on-chain    hash + signatures     EUF-CMA + preimage + liveness
```

## Complete Setup and Proving Phase Map

Detailed step-by-step showing who does what, when, where, with which math, for every operation in the protocol. See lesson file for full map.

## The Math Type Test

```
Polynomial evaluation, FFTs, R1CS?           → GROTH16
Boolean gates, wire labels, garbled tables?  → GC
Pairings for encrypt/decrypt a secret?       → WE
Hash locks, signatures, timelocks, UTXOs?    → BITCOIN
Labels that are also signing keys?           → LAMPORT
```

When group elements [x]₁ appear, distinguish by PURPOSE:
```
Construct a PROOF element (π₁,π₂,π₃)   → Groth16
ENCRYPT/DECRYPT a secret (r·Y, msg)     → WE
WIRE LABELS in garbled tables            → GC/Lamport
```

The Groth16 verification equation e(π₁,π₂) = e([α]₁,[β]₂)·e(X,[γ]₂)·e(π₃,[δ]₂) belongs to GROTH16 (defines proof validity) even though WE exploits its algebraic structure.

## Drill Results
```
Drill 1: 3/6 (Groth16/GC/WE confused)
Drill 2: 5/6 (only verification equation misattributed)
```
