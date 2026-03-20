# Lesson 2: Finite Fields and Modular Arithmetic

*Course 6: Cryptography from First Principles*

## Core Concept

All cryptographic operations in BABE happen inside a finite, looping number system where addition, subtraction, multiplication, AND division always work and always stay in the system.

## Clock Arithmetic (Modular Arithmetic)

Numbers wrap around at a modulus. Z₇ = {0, 1, 2, 3, 4, 5, 6}.

```
3 + 5 = 8 ≡ 1 (mod 7)
2 - 6 = -4 ≡ 3 (mod 7)
4 × 5 = 20 ≡ 6 (mod 7)
```

## Why Primes Are Magic

Division in Z₇: 3 ÷ 5 means "find x where 5x ≡ 3 (mod 7)." Answer: x=2 (5×2=10≡3). Division works and the answer is an integer.

**When modulus is prime, every nonzero element has a multiplicative inverse.**

```
In Z₇: 1⁻¹=1, 2⁻¹=4, 3⁻¹=5, 4⁻¹=2, 5⁻¹=3, 6⁻¹=6
```

In Z₆ (not prime): 2 has NO inverse. Division breaks. Universe is broken.

## Finite Fields (F_p)

Z_p with prime p, with all four operations working = a **finite field** (F_p or GF(p)).

In BABE: p is the BN254 prime (254-bit, 77-digit number). Every field element in Groth16 lives in this F_p.

## Groups

A set with ONE operation satisfying: closure, associativity, identity, inverses.

F_p* = nonzero elements of F_p under multiplication. Has p-1 elements.

### Generators and Cyclic Groups

Pick element 3 in F₇*. Compute powers:
```
3¹=3, 3²=2, 3³=6, 3⁴=4, 3⁵=5, 3⁶=1 (wraps back!)
```

Visited every element. 3 is a **generator**. The group is **cyclic**.

> Every nonzero element in F_p can be written as g^x for some generator g and exponent x.

The exponent x = the **discrete logarithm**.

## The Discrete Logarithm Problem

```
EASY:  given g and x, compute g^x mod p    (fast via repeated squaring)
HARD:  given g and g^x mod p, find x        (no efficient algorithm known)
```

This asymmetry makes cryptography possible. Publish g^x (public key), keep x secret (private key).

## Connecting to BABE

- F_q: scalar field (where exponents live)
- [x]_s := x · g_s: implicit notation (scalar times generator)
- G₁, G₂, G_T: groups built on elliptic curves
- Groth16: equations over these fields and groups

---

## Q&A

**Question:** WHY does every nonzero element have an inverse when p is prime? Consider multiplying a fixed nonzero a by each of 1, 2, ..., p-1.

**Student's Answer:** Noticed the cyclic pattern — outputs wrap around and visit everything — but couldn't formalize the proof.

**The Proof:**

**Claim:** {a·1, a·2, ..., a·(p-1)} mod p are all distinct.

**Proof by contradiction:** Suppose a·i ≡ a·j (mod p) for i ≠ j. Then p divides a·(i-j). Since p is prime and divides a product, it must divide one factor. But p∤a (a<p, a≠0) and p∤(i-j) (|i-j|<p). Contradiction.

**Consequence:** p-1 distinct values in {1,...,p-1} = a permutation. Since 1 is in the range, some k has a·k ≡ 1. That k is a⁻¹.

```
Z₇, a=5: {5,3,1,6,4,2} = permutation of {1..6}. 1 at k=3, so 5⁻¹=3 ✓
Z₆, a=2: {2,4,0,2,4} = collisions! 1 never appears. No inverse. ✗
```

**Key property of primes:** a prime can't divide a product without dividing a factor. Non-primes can (6|2×3 without 6|2 or 6|3).

**Learning note:** Student sensed the cyclic structure but needed the contradiction technique. This proof pattern (assume collision, use primality for contradiction) recurs throughout cryptography.
