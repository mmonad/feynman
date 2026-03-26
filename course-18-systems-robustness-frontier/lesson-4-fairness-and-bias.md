# Lesson 4: Fairness and Bias

*Course 18: Systems, Robustness & the Frontier*

## Core Question

Suppose you build a loan approval model. It achieves 90% accuracy overall. But when you disaggregate by race, accuracy for one group is 95% and for another it's 78%. Is the model fair? Before you answer — *what does fair even mean?* Because there are at least three reasonable definitions, and here's the kicker: you provably cannot satisfy all of them at the same time. This isn't a limitation of our algorithms. It's a mathematical impossibility theorem.

---

## Q88: Fairness Definitions and the Impossibility Result

### Three Intuitions, Three Definitions

**Demographic Parity** — The model's positive prediction rate should be the same across groups.

```
P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)

where A is the protected attribute (e.g., race, gender)
      Ŷ is the model's prediction
```

Intuition: "Equal opportunity to be selected." If 30% of Group A gets approved for loans, 30% of Group B should too. This is the most visible form of fairness — an auditor looking at acceptance rates sees parity.

Problem: If the base rates actually differ between groups (Group A has 50% repayment and Group B has 30%), demographic parity forces the model to either approve too many bad loans in Group B or reject too many good ones in Group A.

**Equalized Odds** — The model's error rates should be the same across groups.

```
P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)   (equal TPR)
P(Ŷ = 1 | Y = 0, A = 0) = P(Ŷ = 1 | Y = 0, A = 1)   (equal FPR)
```

Intuition: "Equal accuracy for equal merit." Among people who *would* repay the loan, approval rates should be equal across groups. Among people who *wouldn't*, rejection rates should be equal. This separates the model's behavior from the base rate.

Problem: Requires access to ground-truth labels Y, which may themselves be biased (more on this in Q89).

**Calibration** — Among people the model gives a score of p, the actual positive rate should be p, regardless of group.

```
P(Y = 1 | Ŷ = p, A = 0) = P(Y = 1 | Ŷ = p, A = 1) = p
```

Intuition: "A score of 0.7 means 70% for everyone." If the model says there's a 70% chance a Group A applicant will repay, that should be true for Group A. And if it says 70% for Group B, that should also be true.

Problem: This is the weakest form — a perfectly calibrated model can still have very different error rates across groups if the score distributions differ.

### The Impossibility Theorem (Chouldechova, 2017)

Here's where the math gets ruthless. Chouldechova proved:

```
If the base rates differ between groups:
  P(Y = 1 | A = 0) ≠ P(Y = 1 | A = 1)

Then you CANNOT simultaneously achieve:
  1. Calibration across groups
  2. Equal false positive rates across groups
  3. Equal false negative rates across groups

...unless the model is perfect (zero error).
```

This isn't a technical limitation. It's a theorem. No amount of engineering, data collection, or clever algorithms can overcome it. When base rates differ, you must *choose* which notion of fairness matters most, and that choice is a values decision, not a technical one.

Think of it mechanically. If Group A has a 50% base rate and Group B has a 20% base rate, a calibrated model will predict higher scores for Group A on average. To equalize false positive rates, you'd need different thresholds for each group — but then the model is no longer calibrated (a score of 0.5 means different things for different groups).

| Fairness Criterion | Requires | Allows | Fails When |
|---|---|---|---|
| Demographic parity | Equal selection rates | Different error rates | Base rates differ |
| Equalized odds | Equal TPR and FPR | Different selection rates | Labels are biased |
| Calibration | Scores mean same thing | Different error rates | Combined with equalized odds |

### Intervention Strategies

**Pre-processing** — Fix the data before training. Reweight or resample training examples to balance representation. Remove or transform features correlated with the protected attribute. Risk: may remove legitimate signal.

**In-processing** — Modify the training objective. Add fairness constraints as regularization terms:

```
L_total = L_task + λ · L_fairness

Example: L_fairness = |TPR_A - TPR_B| + |FPR_A - FPR_B|   (equalized odds penalty)
```

This gives you a knob (λ) to trade off accuracy against fairness — making the impossibility theorem's tradeoff explicit.

**Post-processing** — Adjust predictions after training. Apply group-specific thresholds to equalize error rates. This is the most transparent approach — the model itself is unchanged, and the fairness adjustment is visible and auditable.

> The impossibility theorem isn't a reason to give up on fairness. It's a reason to be honest about tradeoffs. Every deployed model makes a choice about which errors are acceptable. The theorem forces you to make that choice explicitly rather than pretending you can have it all.

---

## Q89: Bias Detection and Mitigation

### A Taxonomy of Bias

Bias doesn't appear from nowhere. It enters the pipeline at specific, identifiable points:

**Historical bias** — The world itself is biased, and the data reflects it. If historical loan data shows lower approval rates for minorities *because of past discrimination*, a model trained on that data will learn to discriminate — not because it's "racist" but because the labels encode historical injustice.

**Representation bias** — Some groups are underrepresented in training data. A face recognition system trained predominantly on light-skinned faces will have higher error rates on dark-skinned faces — not because of algorithmic bias, but because the model simply hasn't seen enough examples.

**Measurement bias** — The features themselves measure different things for different groups. Using "number of prior arrests" as a feature for recidivism prediction is biased because policing intensity differs by neighborhood. The feature measures policing patterns, not criminal behavior.

**Aggregation bias** — A single model is forced to fit all subgroups. If the relationship between features and outcome is genuinely different across groups (e.g., different medical symptoms predict the same disease differently in men and women), a one-size-fits-all model will be worse for everyone.

**Evaluation bias** — The benchmark itself is biased. If you evaluate a model's accuracy on a test set that overrepresents one demographic, the overall accuracy number hides poor performance on underrepresented groups.

### Disaggregated Metrics

The single most important practice: **never report only aggregate metrics.** Always break down performance by relevant subgroups.

```
Overall accuracy: 92%

Disaggregated:
  Group A accuracy: 95%
  Group B accuracy: 82%
  Group C accuracy: 91%

  Male accuracy:    93%
  Female accuracy:  88%

  Intersection (Group B × Female): 76%  ← hidden by aggregates
```

Intersectional analysis is critical — combining multiple axes of identity can reveal failures invisible at any single axis.

### Counterfactual Fairness

A model is counterfactually fair if: for any individual, changing their protected attribute (while keeping everything else consistent) would not change the prediction.

```
P(Ŷ_A←0 | X=x, A=1) = P(Ŷ_A←1 | X=x, A=1)

"If this person had been in Group 0 instead of Group 1,
 with everything else the same, would the model's
 prediction change?"
```

This requires a causal model of how the protected attribute affects other features — which features are *causally downstream* of race or gender? Implementing this is hard, but it makes the fairness question precise.

### The Proxy Variable Problem

Removing the protected attribute from features doesn't ensure fairness. Other features often serve as **proxies**:

```
Removed: race
Still present: zip code (95% correlated with race)
Still present: name (predictive of ethnicity)
Still present: school attended (correlated with socioeconomic status)
```

The model can reconstruct the protected attribute from the remaining features, often with high accuracy. Blindness (removing the variable) is not fairness. Sometimes you need to *include* the protected attribute so you can explicitly control for it.

### When Fairness Matters Most

Not all ML applications carry equal fairness stakes. A movie recommender with biased suggestions is annoying. A criminal sentencing model with biased predictions ruins lives.

```
High stakes: Criminal justice, hiring, lending, healthcare, education
Medium stakes: Content moderation, ad targeting, insurance
Lower stakes: Recommendation, search ranking, spam detection
```

The required level of fairness scrutiny should scale with the consequences of the decision and the vulnerability of the affected population. This isn't a technical principle — it's an ethical one. But engineering teams need to internalize it because the default in ML is to optimize a single aggregate metric and call it a day.

> Bias is not a single thing you can detect and remove. It's a family of failure modes that enter at different stages of the pipeline, interact in complex ways, and require different mitigation strategies. The first step is always disaggregated evaluation — you can't fix what you can't see.

---

## Q&A

**Question:** A healthcare startup builds a model to predict which patients need extra follow-up care. They find that the model assigns lower risk scores to Black patients who are equally sick as white patients. The model uses healthcare cost as a proxy for health need. What type of bias is this, and how would you fix it?

**Student's Answer:** This is measurement bias — the feature (healthcare cost) is measuring the wrong thing. It measures *healthcare utilization*, not health need. Black patients historically have lower access to healthcare, so their costs are lower even when they're equally or more sick. The feature is a biased proxy for the outcome you actually care about. To fix it: replace cost with direct health measures — number of chronic conditions, lab results, symptom severity scores. If you must use cost, include race explicitly so the model can adjust for the utilization gap. And you need to audit via disaggregated metrics — check that at equivalent clinical severity levels, risk scores are equal across racial groups. This is basically the Obermeyer et al. (2019) result from the real world.

**Evaluation:** That's exactly the Obermeyer case, and you identified the mechanism precisely. The critical lesson is that measurement bias is the most insidious type because the feature looks perfectly reasonable on the surface — costs *are* correlated with health need in aggregate. The bias only becomes visible when you disaggregate. Their fix was exactly what you said: replacing cost with direct health measures eliminated 84% of the bias. The deeper engineering lesson: always ask "what is this feature actually measuring?" not "what do I want it to measure?" — those are often different questions with different answers.
