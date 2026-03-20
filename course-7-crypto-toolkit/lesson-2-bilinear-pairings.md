# Lesson 2: Bilinear Pairings

*Course 7: The Cryptographic Toolkit*

## The Problem

In elliptic curve groups, you can ADD ([x]₁+[y]₁=[x+y]₁) and do scalar mult (c·[x]₁=[cx]₁), but CANNOT multiply hidden scalars ([x]₁·[y]₁ is undefined). Groth16 verification needs to check products of hidden values.

## The Pairing

```
e : G₁ × G₂ → G_T

e([x]₁, [y]₂) = [xy]_T
```

Takes one element from G₁, one from G₂, produces element in G_T. **Multiplies hidden scalars** across groups.

## Bilinearity

```
e([a]₁, [b]₂) = [ab]_T

Scalars move freely between inputs:
e([a]₁, [b]₂) = e([1]₁, [ab]₂) = e([ab]₁, [1]₂) = e([b]₁, [a]₂)
```

**Paint mixing analogy:** 3 blue + 5 yellow → 15 green. Can verify the green is "15 units." Can't unmix green. Can't mix blue with blue — only blue with yellow.

## Groth16 Verification Equation

```
e(π₁, π₂) = e([α]₁, [β]₂) + e(X, [γ]₂) + e(π₃, [δ]₂)
```

- π₁, π₃, X, [α]₁ ∈ G₁
- π₂, [β]₂, [γ]₂, [δ]₂ ∈ G₂
- All pairing outputs ∈ G_T

**Four pairings and a comparison = entire verification.** Constant time regardless of computation size.

## The One-Shot Limitation

```
e: G₁ × G₂ → G_T     ✓ can pair once
e: G_T × anything → ?  ✗ cannot pair again
```

Exactly one level of multiplication on hidden values. Groth16 designed to need exactly one level. All pairing-based crypto is shaped by this constraint.

## The Three Groups in BABE

```
G₁:  first source group    most proof elements     [x]₁
G₂:  second source group   CRS verification keys   [x]₂
G_T: target group          pairing outputs          [x]_T
```

---

## Q&A

**Question:** Four pairings are computationally expensive. Why is the verifier still "efficient"?

**Student's Answer:** Verification is constant size in computation complexity instead of polynomial complexity.

**Evaluation:** Exactly right.

```
Without SNARK: re-execute ~10B gates: O(n)    minutes/hours
With SNARK:    4 pairings: O(1)               ~10-15ms
Same cost whether 1,000 gates or 10,000,000,000 gates.
```

For BABE on Bitcoin: script capacity ~thousands of ops. Direct computation (10B ops) impossible. Groth16 verification (4 pairings, constant) feasible. That's why BABE exists.
