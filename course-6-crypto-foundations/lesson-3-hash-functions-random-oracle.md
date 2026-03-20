# Lesson 3: Hash Functions and the Random Oracle

*Course 6: Cryptography from First Principles*

*This lesson doubles as a notation boot camp — every concept gets its notation immediately.*

## Notation Boot Camp, Part 1: Sets and Strings

```
{0,1}^n         all n-bit strings
{0,1}*          all strings of any length
x ← H(m)       deterministic assignment (compute H(m), store in x)
x ←$ {0,1}^n   random sampling (flip n coins)
r ←$ F_p        sample random field element
```

## What a Hash Function Is

```
H : {0,1}* → {0,1}^λ
```
"H maps arbitrary-length strings to λ-bit strings." Any-size input, fixed-size output. Avalanche effect: one bit change → completely different output.

## Three Security Properties

1. **Preimage resistance (one-way):** Given y=H(x), finding x' with H(x')=y is hard
2. **Second preimage resistance:** Given x, finding x'≠x with H(x')=H(x) is hard
3. **Collision resistance:** Finding ANY (x,x') with x≠x' and H(x)=H(x') is hard

### Notation for "Hard"

```
Pr[A(y) = x : x ←$ {0,1}*, y ← H(x)] ≤ negl(λ)
```

Reading: "The probability that adversary A recovers x from y, where x was randomly sampled and y is its hash, is negligible."

**General template:**
```
Pr[ WIN CONDITION : SETUP STEPS ] ≤ negl(λ)
```

### Game Format (used in BABE paper)
Same content, stacked vertically:
```
Pr[
    H(x') = y               ← win condition
    :
    x ←$ {0,1}*             ← setup (read top to bottom like a program)
    y ← H(x)
    x' ← A(y)
] ≤ negl(λ)
```

## The Random Oracle (RO)

An idealized hash function that, first time it sees an input, generates a random output and remembers the mapping forever. Perfect security properties — no exploitable structure.

"...in the Random Oracle Model (ROM)..." means the security proof treats H as a random oracle.

### Oracle Access Notation

```
A^RO          "adversary A with oracle access to RO"
```

Superscript = "can call this function as many times as it wants." CRITICAL parsing:

```
b' ← A^RO(ct_b, aux)

  A         = the adversary algorithm
  ^RO       = has oracle access to RO (superscript = access, NOT input)
  (ct_b, aux) = inputs to A (parenthesized = actual inputs)
```

More examples:
```
A^H(crs)              adversary gets crs, can query H
A^{O₁,O₂}(x)         adversary gets x, can query two oracles
Sim^A(1^λ, C(x))      simulator gets (1^λ, C(x)), queries adversary as oracle
```

## Notation Boot Camp, Part 2: Algorithm Specification

```
Gen(R) → crs              "Gen takes R, outputs crs"
Enc(crs, x, msg) → ct     "Enc takes crs, statement, message; outputs ciphertext"
```

Arrow zoo:
```
→     function type / output
←     assignment
←$    random sampling
```

## How BABE Uses Hashing

1. **Lamport signatures:** sk = random values, pk = their hashes. Sign by revealing preimages. Preimage resistance = unforgeability.
2. **Hash locks:** publish H(msg), later reveal msg to unlock. Preimage resistance = nobody else can find msg.
3. **Random oracle in WE:** ct = (r·[δ]₂, RO(r·Y) + msg). Security proof treats hash as RO.

---

## Q&A

### Notation Drill

**Expression 1:** `Pr[A(y) = x : x ←$ {0,1}^λ, y ← H(x)] ≤ negl(λ)`

**Student:** "x is a random lambda-bit string, y is the output of hash function H on x, given these, the probability of adversary using y to recover x is negligible." ✓ PERFECT.

**Expression 2:** `H : {0,1}* → {0,1}^256`

**Student:** "H is a function that takes arbitrary string as input and outputs 256-bit string." ✓ PERFECT.

**Expression 3:** `b' ← A^RO(ct_b, aux)`

**Student:** "Adversary has access to RO that takes as input (ct_b and aux) and outputs b'."

**Correction:** The superscript RO is oracle ACCESS, not the function taking (ct_b, aux). Correct reading: "b' is the output of adversary A, which receives (ct_b, aux) as inputs, and has oracle access to RO that it can call internally as many times as it wants."

Score: 2/3 perfect, 1/3 parsing error (superscript = access, not input). Conceptual understanding solid.

### Conceptual Question

**Question:** Why prove security in the ROM if it's an idealization?

**Student's Answer:** If you can prove security under RO, you translate the burden to whether the specific hash function is exploitable as a de facto RO, which is a well-studied area.

**Evaluation:** Exactly right. ROM creates modular reasoning — prove protocol correct assuming ideal interface, separately verify implementation meets interface. Same as testing against a mock. 30+ years of practice: no real scheme proven secure in ROM has been broken due to hash function failure.
