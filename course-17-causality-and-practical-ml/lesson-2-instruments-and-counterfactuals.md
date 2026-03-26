# Lesson 2: Instruments and Counterfactuals

*Course 17: Causality & Practical ML*

## Core Question

Last lesson, we learned the backdoor criterion: if you can measure and adjust for all confounders, you can estimate causal effects from observational data. But what happens when the confounder is *unmeasurable*? Maybe it's genetic predisposition, or motivation, or some latent market force you can't put a number on. Are you stuck? Not necessarily — if you can find an **instrument**. And to understand why instruments work, we need to think about the deepest idea in causal inference: counterfactuals.

---

## Q73: Instrumental Variables

### The Unmeasured Confounder Problem

Suppose you want to know: does education cause higher earnings? The naive regression `Earnings ~ Years_of_Education` gives a positive coefficient. But there's a massive unmeasured confounder: **ability**. People with higher innate ability tend to get more education AND tend to earn more, regardless of education. You can't measure "ability" directly. The backdoor path `Education ← Ability → Earnings` is open, and you can't block it.

```
Ability (unmeasured)
   / \
  v   v
Education → Earnings
```

The adjustment formula requires conditioning on Ability, but Ability isn't in your dataset. You're stuck — unless you find an instrument.

### What Is an Instrument?

An **instrumental variable** (IV) is a variable Z that affects Y *only through X*. Think of it as a lever that pushes on X but has no other connection to Y.

```
Z → Education → Earnings
         ↑          ↑
         Ability----/  (unmeasured)
```

For Z to be a valid instrument, it must satisfy three conditions:

| Condition | Meaning | Why it matters |
|---|---|---|
| **Relevance** | Z is correlated with X | The lever must actually move X |
| **Exclusion restriction** | Z affects Y only through X | No direct path Z → Y |
| **Independence** | Z is independent of unmeasured confounders U | Z must not share causes with Ability |

The classic instrument for education's effect on earnings is **quarter of birth** (Angrist & Krueger, 1991). Due to compulsory schooling laws, people born in Q1 start school slightly older and tend to get slightly less education. Quarter of birth is plausibly random with respect to ability, affects earnings only through education, and is correlated with years of schooling. It satisfies all three conditions.

### Two-Stage Least Squares (2SLS)

The estimation procedure has two stages — and the name is wonderfully literal:

**Stage 1:** Regress X on Z. Get the predicted values X-hat — the part of education that's driven by the instrument (quarter of birth), stripped of the ability confounding.

```
Stage 1:  X̂ = α + β·Z        (predict education from the instrument)
```

**Stage 2:** Regress Y on X-hat. Since X-hat contains only the variation in education caused by Z, and Z is independent of Ability, the coefficient gives you the causal effect.

```
Stage 2:  Y = γ + δ·X̂ + ε    (δ is the causal estimate)
```

The intuition: you're using Z as a filter. You extract only the "exogenous" variation in X — the part that moves because Z pushed it, not because Ability pulled it — and ask how Y responds to *that* variation specifically.

### The Weak Instruments Problem

Here's where it gets dangerous. If the instrument is only weakly correlated with X (the relevance condition barely holds), 2SLS produces estimates that are *biased toward the OLS estimate* — exactly the biased answer you were trying to avoid. Worse, the standard errors become unreliable.

The rule of thumb: the F-statistic from Stage 1 should exceed 10 (Staiger & Stock, 1997). Below that, you have a weak instrument, and your "causal" estimate might be worse than the naive correlation.

```
Strong instrument:  F > 10      → 2SLS works well
Weak instrument:    F ≈ 1-5     → biased, unreliable
Irrelevant:         F ≈ 0       → you're dividing by noise
```

### Natural Experiments

The gold standard for instruments comes from **natural experiments** — situations where nature or policy creates quasi-random variation. Examples:

- **Draft lottery numbers** → military service → lifetime earnings (Angrist, 1990)
- **Distance to college** → years of education → earnings (Card, 1993)
- **Rainfall** → economic activity → conflict (Miguel et al., 2004)

Each of these exploits a source of variation that's plausibly random and affects the outcome only through the treatment. The art of instrumental variables is *finding* these natural experiments — and that's as much creativity as statistics.

> An instrument is a causal backdoor hack: when you can't block the confounding path directly, you find an external source of variation that moves only your treatment variable. You estimate the causal effect using only that exogenous push. The danger is that bad instruments — ones that violate exclusion or are too weak — give you false confidence in a wrong answer.

---

## Q74: Counterfactual Reasoning and the Fundamental Problem

### The Deepest Question

Here's the question that launched an entire field: for a specific person who took the drug and recovered, *would they have recovered without it?*

This isn't a statistical question. It's asking about a specific individual in a specific scenario that didn't happen. It's a question about a **counterfactual** — a world that could have been but wasn't.

### Potential Outcomes Framework

Neyman and Rubin formalized this with **potential outcomes**. For each individual i, define:

```
Y_i(1) = outcome if individual i receives treatment
Y_i(0) = outcome if individual i does not receive treatment
```

The **Individual Treatment Effect** (ITE) is the difference:

```
ITE_i = Y_i(1) - Y_i(0)
```

This is exactly what we want. Did the drug help *this person*? But here's the fundamental problem: **you can only observe one of the two potential outcomes**. If the patient took the drug, you see Y_i(1) but Y_i(0) is forever unknowable. If they didn't take it, you see Y_i(0) but Y_i(1) is missing.

This is called the **Fundamental Problem of Causal Inference** (Holland, 1986). The individual causal effect is *in principle unobservable*. You cannot simultaneously give and not give someone a drug. You cannot clone a person and run both versions.

### What We Can Estimate: ATE

Since we can't compute individual effects, we retreat to averages. The **Average Treatment Effect** (ATE) is:

```
ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]
```

This is the average causal effect across the population. We don't know it for any single person, but we can estimate it across groups — *if we handle the assignment mechanism correctly*.

### Why Naive Comparison Fails

Suppose you compare the average outcome of treated people vs. untreated people:

```
E[Y | Treatment = 1] - E[Y | Treatment = 0]
```

This equals the ATE only if treatment assignment is independent of potential outcomes — i.e., the people who got treated aren't systematically different from those who didn't. In observational data, this almost never holds. Sicker patients get treatment. Motivated students seek tutoring. The comparison confounds the causal effect with selection bias.

### RCTs: The Nuclear Option

A **Randomized Controlled Trial** solves the fundamental problem by force. If you randomly assign treatment, then:

```
Treatment ⊥ (Y(0), Y(1))
```

Treatment is independent of potential outcomes. The types of people in the treatment group are, on average, the same as those in the control group. So the naive comparison *does* equal the ATE:

```
E[Y | T = 1] - E[Y | T = 0] = ATE     (under randomization)
```

This is why RCTs are the gold standard. They don't eliminate confounders by adjusting for them — they prevent confounding by design. The randomization severs every backdoor path at once, without needing to know what the confounders are.

### Pearl's Three-Level Ladder

Pearl organizes causal reasoning into three levels, each strictly more powerful than the last:

| Level | Name | Question | Typical tool |
|---|---|---|---|
| 1 | **Association** | What is? P(Y\|X) | Observational data, statistics |
| 2 | **Intervention** | What if I do? P(Y\|do(X)) | Experiments, adjustment formulas |
| 3 | **Counterfactual** | What if I had done differently? P(Y_x\|X = x', Y = y') | SCMs with specific U values |

Level 1 is pure statistics — correlations, regressions, prediction. Most ML lives here. Level 2 is causal inference — estimating the effects of interventions. The do-operator, backdoor criterion, and instrumental variables all operate here. Level 3 is counterfactual reasoning — asking about specific individuals and alternative histories. It requires the full SCM machinery because you need to fix the exogenous noise variables to their actual values and then "replay" the model under a different intervention.

The critical insight: **you cannot answer a higher-level question from lower-level data alone.** No amount of observational data (Level 1) can definitively answer an interventional question (Level 2) without additional assumptions (like the DAG structure). And no amount of interventional data can answer a counterfactual question (Level 3) without a fully specified structural model.

> The fundamental problem of causal inference isn't a technical limitation — it's a logical impossibility. You cannot observe the road not taken. Everything in causal inference, from RCTs to instrumental variables to the do-calculus, is a strategy for working around this impossibility by averaging over individuals, exploiting randomization, or leveraging structural assumptions.

---

## Q&A

**Question:** A tech company runs an A/B test: half of users see a new feature, half don't. They measure engagement. The A/B test shows the feature increases engagement by 5%. A product manager asks: "But does it help *power users* specifically, or is it diluted by casual users who don't care?" Map this to the potential outcomes framework. What can and can't the A/B test answer?

**Student's Answer:** In the potential outcomes framework, each user i has Y_i(1) (engagement with feature) and Y_i(0) (engagement without). The A/B test estimates the ATE — the average effect across all users. The PM's question is about the Conditional ATE (CATE) for the subgroup "power users": E[Y(1) - Y(0) | power user]. Since randomization was done across all users, within the power user subgroup treatment is also random (assuming the randomization didn't stratify by user type, it's still valid by independence). So you CAN estimate the CATE for power users by just filtering to that subgroup and comparing treated vs. control within it. What the A/B test CANNOT answer is the ITE for any specific user — did this particular power user benefit? That's a Level 3 counterfactual question, and you'd need a structural model of how individual users respond to the feature.

**Evaluation:** Very precise. You correctly identified that the CATE for a pre-defined subgroup is estimable from an RCT (since randomization ensures balance within subgroups, at least in expectation), and that individual-level effects remain fundamentally unobservable. One important caveat: if "power user" is defined *post-treatment* (e.g., "users who engaged a lot"), then conditioning on it can introduce collider bias — the feature itself might have changed who counts as a power user. The subgroup must be defined by a pre-treatment variable to avoid this trap.
