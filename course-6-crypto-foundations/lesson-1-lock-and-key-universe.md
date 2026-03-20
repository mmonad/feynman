# Lesson 1: The Lock and Key Universe

*Course 6: Cryptography from First Principles*

## What Cryptography Actually Does

Not just hiding secrets (encryption). Four capabilities:

```
1. HIDING:         encrypt secrets
2. BINDING:        commit to a value, reveal later, prove it wasn't changed
3. PROVING:        convince someone you know something without revealing it
4. AUTHENTICATING: prove a message came from you
```

BABE uses all four: witness encryption (hiding), Lamport signatures (binding), SNARKs (proving), digital signatures (authenticating).

## Security Is Not About Impossibility

Nothing is impossible to break. Everything can be brute-forced. The question: **how long would it take?** Like a house lock — doesn't make break-in impossible, makes it expensive enough that most won't bother.

## The Security Parameter λ

Single number controlling how hard everything is to break. Measured in bits:

```
λ = 40:    2^40 ops    ≈ trivial (seconds)
λ = 80:    2^80 ops    ≈ infeasible for individuals
λ = 128:   2^128 ops   ≈ infeasible for nation-states
λ = 256:   2^256 ops   ≈ infeasible for the observable universe
```

Every construction in BABE is parametrized by λ.

## Negligible Functions

f(λ) is **negligible** if it shrinks faster than any polynomial grows:

```
1/λ²      NOT negligible (polynomial — too slow)
1/λ^100   NOT negligible (still polynomial)
1/2^λ     NEGLIGIBLE (exponential — beats any polynomial)
```

"negl(λ)" in the paper = "a number so small it might as well be zero."

## PPT Adversary

The attacker is a **Probabilistic Polynomial-Time** machine:
- Can do at most poly(λ) operations (can't brute-force 2^λ)
- Can use randomized algorithms
- Incredibly smart but not infinitely powerful

## Security as a Game

```
CHALLENGER (system) vs ADVERSARY (attacker)
```

Scheme is secure if no PPT adversary wins with probability significantly better than random guessing. "Significantly" = advantage is negligible.

Example — encryption security game:
```
1. Challenger generates key K
2. Adversary picks m₀, m₁
3. Challenger flips coin b, encrypts m_b
4. Adversary guesses b'
5. Secure if P(b'=b) ≤ 1/2 + negl(λ)
```

Every BABE security definition (knowledge soundness, extractable WE, adaptive GC privacy) follows this same game structure.

---

## Q&A

**Question:** Why must security decline be negligible (exponential) rather than polynomial (e.g., 1/λ²)?

**Student's Initial Answer:** The security parameter is in bits and exponential by nature. Cheap to double λ. If we set the bar as polynomial decline, the bar is too high — theoretically insecure when practically secure.

**Evaluation:** Practical cost intuition correct (increasing λ is cheap), but conclusion backwards. Polynomial decline isn't too strong a requirement — it's too WEAK.

**Redirect:** What happens when a protocol is used not once but many times?

**Student's Corrected Answer:** The probability in combinatorial math of running N trials with at least 1 success in breaking is polynomial... so if decline is polynomial, we push breaking probability arbitrarily close to 1 when used many times. Therefore it has to be exponential.

**Evaluation:** Exactly right. The critical insight: security must survive composition and repetition.

```
Polynomial decline × polynomial uses = constant failure probability
  1/λ² × λ² uses ≈ 63% chance of at least one break

Negligible decline × polynomial uses = still negligible
  1/2^λ × λ² uses ≈ 10^(-34) at λ=128
```

For BABE: thousands of instances on Bitcoin, each using same crypto. Per-instance guarantee must survive the union bound across all instances.

**Learning pattern noted:** Student thinks about systems under load (engineering instinct). Got practical cost right immediately. Needed redirect to see the composition/repetition argument. Future lessons should always frame crypto properties in terms of "what happens at scale."
