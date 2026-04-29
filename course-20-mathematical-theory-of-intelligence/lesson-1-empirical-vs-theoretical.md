# Lesson 1: Empirical Recipes vs Mathematical Laws

*Course 20: Beyond the Bitter Lesson — A Mathematical Theory of Intelligence (Survey)*

## The Setup

The entire AI industry runs on a recipe:

> Make models bigger. Feed them more data. Loss goes down. Capabilities emerge. We don't entirely know why. Don't ask.

This is what *The Bitter Lesson* names. The Shannon parallel — that we are pre-Shannon
telegraph engineers, pumping more watts into a wire we don't mathematically
understand — is correct. But to understand what kind of paradigm shift would
end this regime, you first have to nail down a distinction in science that
sounds pedantic but is in fact load-bearing:

**An empirical regularity is not a law. A law is not a recipe.**

The difference between these is what separates an industry from a science, and
almost every paradigm shift in physics is the moment one becomes the other.

---

## Kepler vs Newton — the cleanest example

In 1609, **Johannes Kepler** stared at decades of Tycho Brahe's planetary
observations and noticed: *planets travel in ellipses, with the Sun at one
focus.* He published this. It was true. It fit the data exquisitely.

But Kepler **could not tell you**:

- Why an ellipse and not, say, an egg-shape or a figure-eight?
- What does a comet do, if it's not bound to the Sun? Also an ellipse?
- What if I throw an apple? Does Kepler's law apply?
- If the Sun were twice as massive, what shape would the orbits become?

Kepler had a beautifully accurate **description**. There was no machinery
underneath. It would happily extrapolate to nonsense regimes and you'd have no
way to know.

Then in 1687, **Newton** wrote down:

```
F = G·M·m / r²
```

This equation says nothing about ellipses. It is a statement about *why two
masses pull on each other.* And then — through several pages of calculus —
Newton **derives** that bound orbits in such a force must be ellipses. As a
corollary. He also derives:

- Comets trace parabolas or hyperbolas (depending on energy)
- Apples fall at 9.8 m/s²
- The Moon stays in orbit for the exact same reason apples fall
- If the Sun were twice as massive, orbital periods shrink by √2

One equation, an entire universe of predictions. **Including predictions in
regimes nobody had measured yet.** That is the Newton move: the theory makes
confident statements about regions you have not checked, and you trust the math
because the math itself is internally consistent.

### Key Insight

The signature of a real theory is not that it fits data better. Kepler's
ellipses fit *exactly*. The signature is that the theory makes confident
statements in regimes you didn't fit. It generalizes by mathematical necessity,
not by curve-matching.

Newton's gravity didn't *kill* Kepler's law — it **absorbed** Kepler as a
special case. This is how paradigm shifts almost always work: the new theory
contains the old recipe as a derivable corollary in a restricted regime. So
when the AI paradigm shift comes, today's "scaling laws" probably won't be
*wrong* — they'll be the limit of some deeper equation in a narrow regime.

---

## Modern "scaling laws" — Kepler in disguise

The Chinchilla paper (Hoffmann et al., 2022, DeepMind) fit:

```
L(N, D) = E + A/N^α + B/D^β
```

where:
- `L` is cross-entropy loss
- `N` is parameter count
- `D` is training tokens
- `E ≈ 1.69`, `A ≈ 406.4`, `B ≈ 410.7`, `α ≈ 0.34`, `β ≈ 0.28`

Those constants — `1.69`, `0.34`, `0.28` — are **not derived from anything.**
They are fit numbers from a curve-matching exercise across a few hundred
training runs. Nobody on Earth knows why α is approximately 0.34 instead of,
say, 0.5 or 1.0. Nobody knows what α becomes if you double the tokenizer's
vocabulary, or trade the transformer for a state-space model. The exponents
might shift, and the equation gives you no internal mechanism that would
predict the shift.

Compare to Newton: `9.8 m/s²` is not a fit constant. It's `G·M_earth /
R_earth²`, derived. That's what it means to *have* a theory.

**Modern scaling laws are Kepler. We are still waiting for our Newton.**

---

## What Shannon actually did

Shannon proved a **theorem**, not a fit:

```
C = B · log₂(1 + S/N)
```

Channel capacity = bandwidth × log of (1 + signal-to-noise).

That `log₂`, that `+ 1`, that exact functional form — none of it is fit. It
falls out of an information-theoretic derivation that starts from the
definition of a *bit* of entropy and ends as a forced consequence.

And the theorem makes a violent prediction: **there is a hard ceiling.** No
matter how much power you pump in, capacity grows only logarithmically with
S/N. You will hit a wall. The wall has an exact mathematical address.

This is what flipped the field from engineering to science.

---

## The Bar for "A Mathematical Theory of Intelligence"

A genuine theory must:

1. **Predict regimes that haven't been measured.** Derive what α should be for
   a Mamba-style architecture from architecture properties, not measure it
   after the fact.
2. **Establish hard ceilings**, like Shannon's channel capacity. *"No
   architecture of geometry G can generalize past Z logical steps, period."*
3. **Reduce to current scaling laws as a special case** in their fit regime.

We are not within an order of magnitude of having this. The Chinchilla
equation is not on the path to it — it is an *observation* about a path.

---

## Conceptual Question (Answered)

> Imagine that next year, OpenAI releases a model trained at 10× the compute
> Chinchilla extrapolated to. They report the loss exponent has shifted: from
> α ≈ 0.34 to α ≈ 0.51, fitting cleanly to the new data.
>
> **Q:** Under the current scaling-laws framework, what does this finding
> *mean*? What can be predicted from it? How would a Newton-style theory of
> intelligence change the answer?

### Student's answer

> Since we don't know why alpha shifted, we can't make predictions about the
> "scaling law" as the "law" itself keeps changing without our explanation.

### Evaluation: correct

Tight, accurate, captures the negative side of the answer (what we *lose* when
α drifts). Two extensions to lock in:

1. **The "law" framing was always wrong.** A constant that drifts under regime
   change was never a law. It was an empirical fit pretending to be one.
2. **We don't know if 0.51 is stable either.** It could shift again at 100×
   compute. Empirical fits always live one regime away from being wrong.

### What a Newton-style theory would do (positive side)

1. **Predict the shift before the experiment.** Whatever caused α to move
   (architecture detail, data composition, scale-induced regime change) would
   be an *input* to the theory, and the new α would be an *output* of the
   equations.
2. **Explain why α was 0.34 in the old regime.** The new theory absorbs the
   old recipe as a special case, the way Newton derived Kepler's ellipses.
3. **Tell you the next regime where α breaks.** Not just "we observe 0.51
   now," but "at compute level X with data property Y, the functional form
   changes again — here's the new equation."

### Follow-up question (also answered correctly)

> A mathematician claims a theory of intelligence. Her equation perfectly
> reproduces Chinchilla's α ≈ 0.34. She publishes. Is that, by itself,
> sufficient evidence for a real theory?

**Student's answer:**

> Insufficient; a theory has to generalize and make predictions of unobserved
> phenomenon and regimes.

**Evaluation: correct.** The minimal falsification test is generalization to
unmeasured regimes. One match is a coincidence with good PR.

---

## Interlude — What Does "Abstraction" Mean?

The student asked a clarifying question about the discussion's mention of a
hypothetical "Bit of Abstraction." This is the right question to ask, and
abstraction is the through-line of the rest of the course.

### Operational definition

> **Abstraction is a shorter description of the world that still works.**

Three properties must hold:

1. **Compression** — the description uses less information than the data it
   replaces.
2. **Preservation** — the description still reproduces the data it was
   abstracted from.
3. **Generalization** — the description correctly predicts cases that were
   *not* in the original data.

Compression alone gives you a **zip file**. Generalization without
preservation is a **wild guess**. Abstraction requires all three.

### Examples (in increasing depth)

| Data | Abstraction | Compresses | Preserves | Generalizes |
|---|---|---|---|---|
| `1, 2, 3, 4, 5, ...` | "add 1" | Yes (5 chars vs ∞) | Yes | Yes |
| Tycho's tables of Mars | Kepler's ellipse | Yes | Yes | Partially |
| Tycho + apples + tides | Newton's `F = GMm/r²` | Yes | Yes | Yes |
| 10TB of training text | A trained LLM | ~10× | Approximately | Sometimes |

Look at the last row. An LLM compresses ~10× and reproduces approximately
while generalizing sometimes. That's why we currently can't tell, mechanically,
whether the model **abstracted** or merely **memorized**. An LLM is somewhere
in the messy middle and we have no good measuring stick.

### The Kolmogorov candidate

The deepest formal version we have is **Kolmogorov complexity**: the length,
in bits, of the shortest computer program that outputs a given dataset. If
data needs N bits to specify but a program of length n bits can generate it,
the data has been *abstracted by N − n bits*.

That `N − n` is the closest candidate for the "Bit of Abstraction" — the unit
of how much truth has been compressed.

But Kolmogorov complexity is **uncomputable**. There's no algorithm that finds
the shortest program for arbitrary data. We have a *definition* without a
*measuring instrument*.

### Memorization vs Compression vs Abstraction

| Concept | Storage cost | Reproduces training data | Generalizes |
|---|---|---|---|
| Memorization | High (≥ data size) | Yes (exact) | No |
| Compression | Lower | Yes (lossy or lossless) | No |
| Abstraction | Lower | Yes | Yes |

LLMs do all three simultaneously, in different parts of their parameters.
Decoupling them — making memorization and abstraction live in *measurable,
separate* parts of a model — is one of the architectural goals discussed in
the framing material.

### Why "abstraction" is the through-line

- **Manifold (Lesson 2):** the *geometric* form of abstraction — high-dim
  appearance, low-dim reality.
- **Persistent Homology (Lesson 3):** *topological* abstraction — invariants
  preserved under deformation.
- **Renormalization Group (Lesson 4):** *physical* abstraction — coarse-graining
  to extract macroscopic truths from microscopic noise.

A future "theory of intelligence" will be an abstraction of the trillion-
parameter chaos of an LLM into a small, structured object that predicts how
the model behaves at scales nobody has run yet.

---

## Vocabulary Established in This Lesson

| Term | Meaning |
|---|---|
| **Empirical regularity** | A pattern observed in measured data, with no derived mechanism |
| **Theoretical law** | A statement about why a pattern holds, derived from foundational definitions |
| **Scaling law (modern)** | Empirical regularity of loss vs N and D in LLM training |
| **Chinchilla equation** | `L(N,D) = E + A/N^α + B/D^β` with fit constants |
| **Channel capacity theorem** | Shannon's `C = B·log₂(1 + S/N)`; canonical example of a derived theorem with a hard ceiling |
| **Abstraction** | Shorter description that compresses, preserves, and generalizes |
| **Kolmogorov complexity** | Length of shortest program producing given data; uncomputable; the formal candidate for "bit of abstraction" |

## Key Takeaways

1. The Chinchilla equation is **Kepler**, not Newton. Its constants are fit,
   not derived.
2. A real theory is one that **predicts unmeasured regimes**, **establishes
   hard ceilings**, and **absorbs prior empirical recipes as special cases**.
3. **Abstraction = compression + preservation + generalization.** Two out of
   three is not enough.
4. Kolmogorov complexity is the formal definition of abstraction, but it is
   uncomputable, so we have a target without a ruler.
5. The rest of this course installs three candidate rulers: **manifolds**,
   **persistent homology**, and **statistical-mechanical free energy**.
