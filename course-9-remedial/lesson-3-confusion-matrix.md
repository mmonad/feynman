# Remedial Lesson 3: The Confusion Matrix

*Course 9: Remedial — Strengthening the Weak Spots*

## Six Confused Concept Pairs

### 1. Finite Order vs Generator
- Finite order: g^k = 1 for some k. TRUE for ALL elements in a finite group.
- Generator: g^k visits ALL elements. TRUE for SOME elements only.
- Test: "does it visit ALL elements or just SOME?"

### 2. Preimage vs Second Preimage vs Collision
- Preimage: given h, find x with H(x)=h (given target)
- Second preimage: given x, find x'≠x with H(x')=H(x)
- Collision: find any x,x' with H(x)=H(x') (free choice)
- Test: "was A given a target?"

### 3. EUF-CMA vs Preimage Resistance
- EUF-CMA: can't forge signatures → protects CheckSig transactions
- Preimage: can't invert hash → protects HashLock transactions
- Test: "does A need to forge a sig or find a preimage?"

### 4. Soundness vs Knowledge Soundness
- Plain soundness: false statements can't be proved (fact about statements)
- Knowledge soundness: if you can prove, you KNOW witness (fact about prover, provides extractor)
- Test: "do I need an extractor?"

### 5. Liveness vs Unstoppability
- Liveness: blockchain keeps making progress (property of ledger)
- Unstoppability: adversary can't invalidate YOUR transaction (property of specific tx)
- Test: "about the chain or about MY transaction?"

### 6. Groth16 vs GC
- Groth16: polynomials, FFTs, R1CS constraints, O(n log n)
- GC: Boolean gates, wire labels, AND vs XOR, garbled tables
- Test: "polynomial math or Boolean gates?"

## Drill Results
Quick drill: 4/6 (generator confusion persisted, EUF-CMA cause vs effect)
