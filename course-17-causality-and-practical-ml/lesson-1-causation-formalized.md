# Lesson 1: Causation, Formalized

*Course 17: Causality & Practical ML*

## Core Question

Everyone and their dog knows "correlation is not causation." They say it at dinner parties. They say it on Twitter. They say it and then go right back to drawing causal conclusions from observational data. The question is: can we make causation *precise*? Can we write down equations that distinguish "X causes Y" from "X and Y happen to co-occur"? Judea Pearl did exactly that, and the machinery he built is one of the most beautiful things in modern statistics.

---

## Q71: Structural Causal Models and the do-Operator

### From Curves to Arrows

A correlation is a pattern in data. It tells you: when X goes up, Y tends to go up too. But there are three very different mechanisms that could produce that pattern:

1. **X causes Y** — turning up the thermostat makes the room hotter.
2. **Y causes X** — the room getting hotter triggers the thermostat to read higher. (Reverse causation.)
3. **Z causes both X and Y** — it's winter, so both the thermostat reading and the number of hot cocoas sold go up. (Confounding.)

A correlation coefficient cannot distinguish between these. It sees all three as "X and Y are associated." To actually reason about causation, you need a *model of the mechanism* — a story about what causes what.

### Structural Causal Models (SCMs)

Pearl's Structural Causal Model is a triple (U, V, F):

- **U** = exogenous variables (stuff determined outside the model — noise, unobserved causes)
- **V** = endogenous variables (the things you're modeling)
- **F** = a set of structural equations, one per endogenous variable

Each equation tells you how a variable is *generated* from its direct causes:

```
X = f_X(Pa_X, U_X)
Y = f_Y(Pa_Y, U_Y)
```

where `Pa_X` means "parents of X" — the direct causes of X. The structural equation is an *assignment*, not an algebraic equality. It says: "X is *determined by* its parents plus noise." This is a claim about the physical mechanism, not a statistical statement.

### The DAG

Every SCM implies a Directed Acyclic Graph (DAG). Each variable is a node. Each structural equation creates edges from parents to children. The "acyclic" part means no variable can cause itself through a chain of effects — no time loops.

```
    Z
   / \
  v   v
  X   Y
```

This DAG says: Z causes X and Z causes Y, but X does not cause Y. If you compute the correlation between X and Y, it will be nonzero — because they share the common cause Z. But intervening on X won't change Y at all.

### Observing vs. Intervening: The do-Operator

Here's the crucial distinction. When you *observe* that X = x, you're filtering. You look at the world and select cases where X happened to equal x. But those cases come with all the baggage — the confounders that made X take that value are still active.

When you *intervene* and set X = x, you physically reach in and force X to equal x, regardless of what would have caused it. You sever the connection between X and its parents. In the DAG, intervention means deleting all arrows into X.

Pearl writes this as `do(X = x)`. The interventional distribution is:

```
P(Y | do(X = x))    ← "what happens to Y when we force X to x"
P(Y | X = x)        ← "what Y looks like when we observe X = x"
```

These are generally **not equal**. The whole field of causal inference exists because of this inequality.

### A Concrete Example

Suppose a hospital gives Drug X to sicker patients (because they need it more). You observe that patients who received Drug X die at higher rates. Is the drug harmful?

```
Severity → Drug
Severity → Death
Drug → Death
```

If you compute `P(Death | Drug = 1)`, it's high — but that's because severity is a confounder. Sick people get the drug AND sick people die. When you compute `P(Death | do(Drug = 1))` — what happens when you randomly force some patients to take the drug — you remove severity's influence on drug assignment. The drug might actually help.

### The Adjustment Formula

When you can't run an experiment (can't physically `do(X = x)`), you can sometimes *compute* the interventional distribution from observational data using the adjustment formula:

```
P(Y | do(X = x)) = Σ_z P(Y | X = x, Z = z) · P(Z = z)
```

This works when Z is a sufficient set of variables to adjust for — which brings us to the next question: how do you know *which* variables to adjust for?

> The do-operator is the formal dividing line between statistics and causal inference. Statistics asks "what is associated with what?" Causal inference asks "what would happen if we intervened?" The SCM + do-operator gives you the machinery to answer the second question from observational data — when conditions are right.

---

## Q72: The Backdoor Criterion

### Confounders Create Backdoor Paths

In a DAG, a "path" between X and Y is any sequence of connected nodes, regardless of arrow direction. A **backdoor path** is a path from X to Y that starts with an arrow *into* X — it goes "in the back door" of X.

Why does this matter? A backdoor path creates a spurious association between X and Y that has nothing to do with X's causal effect. It's the mechanism behind confounding.

```
    Z
   / \
  v   v
  X → Y
```

There are two paths from X to Y here:
1. `X → Y` (the causal path, going forward through the front door)
2. `X ← Z → Y` (the backdoor path, going backward through Z)

The backdoor path makes `P(Y | X = x) ≠ P(Y | do(X = x))` because observing X tells you something about Z (since Z causes X), which tells you something about Y through a non-causal route.

### The Criterion

Pearl's **backdoor criterion** states: a set of variables Z satisfies the backdoor criterion relative to (X, Y) if:

1. **No node in Z is a descendant of X** — you don't want to condition on things that X causes, because that can open new spurious paths (collider bias).
2. **Z blocks every backdoor path from X to Y** — every non-causal path is "plugged."

When Z satisfies the backdoor criterion, you can use the adjustment formula:

```
P(Y | do(X = x)) = Σ_z P(Y | X = x, Z = z) · P(Z = z)
```

This formula "simulates" an experiment by adjusting for confounders. You stratify by Z, compute the effect within each stratum, and then average over the strata weighted by their prevalence.

### What Does "Blocking" Mean?

A path is blocked by a set Z if it contains:

- A **chain** `A → B → C` where B is in Z (you condition on the mediator, blocking the flow).
- A **fork** `A ← B → C` where B is in Z (you condition on the common cause, removing the spurious association).
- A **collider** `A → B ← C` where B is *not* in Z and no descendant of B is in Z (colliders naturally block a path — conditioning on them *opens* it).

That last point is the trap. Conditioning on a collider — or a descendant of a collider — creates a spurious association where none existed. It's called **collider bias** or Berkson's paradox, and it's the reason you can't just "control for everything."

| Path structure | Blocked by conditioning on B? |
|---|---|
| A → B → C (chain) | Yes |
| A ← B → C (fork) | Yes |
| A → B ← C (collider) | No — conditioning *opens* it |

### When Backdoor Fails: The Frontdoor Criterion

Sometimes there's an unmeasured confounder U between X and Y, and you can't block the backdoor path because U isn't in your data. The **frontdoor criterion** provides an alternative: if there's a mediator M such that X → M → Y, with no unblocked backdoor path from X to M, and the backdoor paths from M to Y can be blocked by conditioning on X:

```
P(Y | do(X = x)) = Σ_m P(M = m | X = x) · Σ_{x'} P(Y | M = m, X = x') · P(X = x')
```

The classic example: you can't randomly assign people to smoke, and the confounder (genetic predisposition) is unmeasured. But if smoking → tar deposits → cancer, and you can measure tar, the frontdoor criterion lets you estimate the causal effect of smoking on cancer through the mediator.

> The backdoor criterion is the workhorse of observational causal inference. But its power comes with a warning: it requires you to know the DAG. If your causal graph is wrong — if you've missed a confounder or drawn an arrow backward — your adjustment will produce the wrong answer with complete confidence.

---

## Q&A

**Question:** You're building a model to predict whether a marketing email leads to a purchase. You have data on: (1) email sent (yes/no), (2) customer's prior purchase history, (3) purchase outcome. Draw the likely DAG, identify the backdoor path, and explain why a naive model trained on P(Purchase | Email) would give a misleading estimate of the email's causal effect.

**Student's Answer:** The DAG is: Prior History → Email Sent (because the marketing team targets customers based on purchase history), Prior History → Purchase (because past buyers are more likely to buy again), and Email → Purchase (the actual causal effect we want). The backdoor path is Email ← Prior History → Purchase. A naive model computing P(Purchase | Email = 1) vs P(Purchase | Email = 0) conflates the causal effect of the email with the selection bias — frequent buyers were more likely to receive emails AND more likely to purchase regardless. The model would overestimate the email's effectiveness. To get the causal effect, you need to adjust for Prior History: P(Purchase | do(Email)) = Σ_h P(Purchase | Email, History = h) · P(History = h).

**Evaluation:** That's exactly right. You've identified the confounder (prior purchase history), drawn the correct DAG, traced the backdoor path, and applied the adjustment formula correctly. The key insight is that the marketing team's targeting strategy *is* the confounding mechanism — they're selecting based on a variable that independently predicts the outcome. One extension worth noting: in practice, most companies solve this with A/B testing (which is literally the do-operator — randomly assigning emails regardless of history). The adjustment formula is what you use when A/B testing isn't feasible or when you need to analyze historical data retroactively.
