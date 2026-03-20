# Lesson 1: Elliptic Curves

*Course 7: The Cryptographic Toolkit*

## Why Elliptic Curves?

F_p* has sub-exponential DLP algorithms (number field sieve). Need 3,072-bit p for 128-bit security. Elliptic curves: best attack is fully exponential. 256-bit prime suffices. 12× smaller keys and faster operations.

## The Curve Equation

```
y² = x³ + ax + b    (mod p)
```

Plus point at infinity O (identity element). BN254 (used in BABE): y² = x³ + 3, p is 254-bit prime.

## Point Addition (Geometric)

**P + Q:** Draw line through P and Q → hits curve at R' → reflect across x-axis → R = P + Q.

**2P (doubling):** Draw tangent at P → hits curve at R' → reflect → 2P.

## Point Addition (Algebraic)

P = (x₁,y₁), Q = (x₂,y₂), P ≠ Q:
```
λ = (y₂ - y₁)/(x₂ - x₁)       slope (division in F_p)
x₃ = λ² - x₁ - x₂
y₃ = λ(x₁ - x₃) - y₁
```

Doubling (P = Q):
```
λ = (3x₁² + a)/(2y₁)           tangent slope
x₃ = λ² - 2x₁
y₃ = λ(x₁ - x₃) - y₁
```

Division in F_p is why we need a field (prime modulus).

## Scalar Multiplication

x·P = add P to itself x times. Efficient via **double-and-add** (O(log x)):

```
13·P:  13 = 1101₂
  P → 3P → 6P → 13P   (4 steps, not 13)
```

**In BABE:** the garbled circuit computes exactly this — scalar multiplication on BN254.

## ECDLP

```
EASY:  given P, x → compute x·P    (double-and-add)
HARD:  given P, Q=x·P → find x     (no efficient algorithm)
```

## Implicit Notation [x]_s

```
[x]₁ := x · g₁       (scalar x times generator of G₁)
[x]₂ := x · g₂       (scalar x times generator of G₂)

Rules:
[x]₁ + [y]₁ = [x+y]₁     addition works within a group
c · [x]₁ = [cx]₁          scalar multiplication works
[x]₁ + [y]₂ = UNDEFINED   can't add across groups

Matrices: [A]₁ = each entry aᵢⱼ replaced by [aᵢⱼ]₁
```

Reading Groth16 proof: `π₁ = [α]₁ + Σᵢ zᵢ·[aᵢ(τ)]₁` = "group element encoding α plus sum of zᵢ times encoding of aᵢ(τ), all in G₁."

## Connection to BABE

GC computes scalar multiplication α·π₁ on BN254. The Lean files in `Babe/Crypto/EllipticCurve.lean` formalize select matrix, addition matrix, conditional add — the gate-by-gate computation inside the GC.

254-bit scalar mult = billions of Boolean gates → original BitVM3 GC was 42 GB. BABE's randomized encodings bring this down 1000×.

---

## Q&A

**Question:** In doubling, λ = (3x₁²+a)/(2y₁). What happens when y₁ = 0?

**Student's Answer:** The tangent is infinite — the tangent line is vertical.

**Evaluation:** Exactly right. Vertical tangent → line "intersects at infinity" → 2P = O (identity). Point is its own inverse: P = −P since (x,y) reflects to (x,−y), and when y=0 the point equals its reflection. Called **points of order 2** — rare (at most 3), handled as special case.

Good geometric instinct — went to "vertical tangent" rather than algebraic resolution.
