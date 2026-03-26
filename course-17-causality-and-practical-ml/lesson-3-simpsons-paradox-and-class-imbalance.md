# Lesson 3: Simpson's Paradox and Class Imbalance

*Course 17: Causality & Practical ML*

## Core Question

We've built the formal machinery of causation — SCMs, do-operators, backdoor criteria. Now let's see what happens when that machinery collides with real datasets. Two problems show up constantly in practice: data that *lies to you* when you aggregate it (Simpson's paradox), and data where one class drowns out the other (class imbalance). Both are traps. Both have principled solutions. And the first one is actually a causal reasoning problem in disguise.

---

## Q75: Simpson's Paradox — When the Data Lies

### The Phenomenon

Here's a real thing that happened. In 1973, UC Berkeley's graduate admissions data showed that men were admitted at a significantly higher rate than women — 44% vs. 35%. Looks like discrimination, right?

But when you break it down by department, women were admitted at *equal or higher* rates than men in almost every department. The aggregate trend *reversed* when you stratified.

How is this possible?

### The Mechanism

The key: women disproportionately applied to competitive departments (like English) with low overall admission rates. Men disproportionately applied to less competitive departments (like Engineering) with high admission rates. Department choice was a confounder:

```
Gender → Department choice → Admission
Gender → Admission (direct effect, possibly zero)
```

| Department | Male applicants | Male admit % | Female applicants | Female admit % |
|---|---|---|---|---|
| A (easy) | 825 | 62% | 108 | 82% |
| B (hard) | 560 | 63% | 25 | 68% |
| C (hard) | 325 | 37% | 593 | 34% |
| D (hard) | 417 | 33% | 375 | 35% |

(Simplified for illustration.) In every department, women do as well or better. But because women cluster in hard departments, the aggregate rate is lower. The aggregate mixes the causal effect of gender with the confounding effect of department difficulty.

### Marginal vs. Conditional

Formally, Simpson's paradox occurs when:

```
P(Y | X) > P(Y | ¬X)       ← aggregate favors X
but
P(Y | X, Z = z) < P(Y | ¬X, Z = z)    ← every stratum favors ¬X
```

This is mathematically possible because the marginal distribution P(Y|X) averages over Z, and if X and Z are correlated, the average is *weighted differently* for X vs. ¬X. The aggregate is a weighted average of per-stratum effects, but the weights themselves depend on the treatment.

### Which Answer Is Right: Aggregate or Stratified?

Here's the punch line, and it's one of Pearl's key contributions: **there is no purely statistical answer to this question.** Whether you should aggregate or stratify depends on the *causal structure*.

- If Z is a confounder (common cause of X and Y): **stratify.** The aggregate is biased.
- If Z is a mediator (X → Z → Y): **aggregate.** Stratifying blocks the causal pathway.
- If Z is a collider (X → Z ← Y): **aggregate.** Conditioning on a collider opens a spurious path.

In the Berkeley example, department choice is influenced by gender (confounder-like behavior), so you should stratify. The department-level data gives the right answer. But imagine a different scenario: a drug affects blood pressure (Z), and blood pressure affects survival (Y). If you stratify by blood pressure, you've blocked the very mechanism through which the drug works. In that case, the aggregate is correct.

> Simpson's paradox isn't a statistical curiosity — it's a demonstration that data alone cannot tell you what to do. You need the causal graph. The same dataset, with two different causal stories, demands two opposite analyses. This is why Pearl argues that statistics without causality is incomplete.

---

## Q76: Class Imbalance — When One Class Drowns the Other

### The Problem

You're building a fraud detector. In your dataset, 99.5% of transactions are legitimate, 0.5% are fraudulent. You train a model, and it achieves 99.5% accuracy. Celebration? No — the model just predicts "legitimate" for everything. It hasn't learned a single thing about fraud.

Class imbalance means the prior probability of one class is much larger than the other. The majority class dominates the loss function, the gradient signal, and every metric that treats classes equally. The model learns to ignore the minority class because doing so minimizes the average loss.

### The Toolkit

Solutions attack the problem from three angles: data, algorithm, and evaluation.

**Data-Level Solutions**

*Oversampling the minority:* Duplicate minority examples so the model sees them more often. Naive oversampling (exact copies) risks overfitting to specific minority examples.

**SMOTE** (Synthetic Minority Over-sampling Technique) fixes this by creating *synthetic* examples. For each minority sample, find its k nearest neighbors in feature space, pick one, and generate a new point along the line connecting them:

```
x_new = x_i + λ · (x_neighbor - x_i),   λ ~ Uniform(0, 1)
```

This interpolates between real minority samples, creating plausible new examples that expand the minority decision region without exact duplication.

*Undersampling the majority:* Throw away majority examples to balance the ratio. Fast and effective, but you're literally discarding data. Works best when you have so much majority data that losing some doesn't hurt representativeness.

| Strategy | Pros | Cons |
|---|---|---|
| Random oversampling | Simple, no data loss | Overfitting to duplicates |
| SMOTE | Synthetic diversity | Assumes linear interpolation is valid; noisy in high dimensions |
| Random undersampling | Simple, faster training | Discards potentially useful majority data |
| Hybrid (SMOTE + undersampling) | Best of both | More tuning knobs |

**Algorithm-Level Solutions**

*Class weights:* Multiply the loss for each class by a weight inversely proportional to its frequency:

```
w_minority = n_total / (2 · n_minority)
w_majority = n_total / (2 · n_majority)
```

Effect: a misclassified fraud example incurs ~200x the loss of a misclassified legitimate example (for 0.5% fraud rate). The gradient signal from minority examples is amplified without changing the data.

*Focal loss* (Lin et al., 2017) is more surgical. It down-weights *easy* examples (high-confidence correct predictions) and focuses on hard examples:

```
FL(p_t) = -α · (1 - p_t)^γ · log(p_t)
```

The `(1 - p_t)^γ` factor is the key. When the model is confident and correct (p_t ≈ 1), this factor goes to zero — the loss vanishes. When the model is uncertain or wrong (p_t ≈ 0), the loss is large. With γ = 2, an example classified with 90% confidence contributes 100x less to the loss than one classified at 50%. This automatically focuses training on the hard, informative examples — which are disproportionately from the minority class.

**Evaluation: Stop Using Accuracy**

With imbalanced data, accuracy is almost useless. The right metrics are:

| Metric | What it measures | When to use |
|---|---|---|
| **Precision** | Of predicted positives, how many are correct? | When false positives are costly (spam filter) |
| **Recall** | Of actual positives, how many did you catch? | When false negatives are costly (fraud, cancer) |
| **F1 Score** | Harmonic mean of precision and recall | Balanced concern for both |
| **AUPRC** | Area under precision-recall curve | The gold standard for imbalanced classification |
| **AUROC** | Area under ROC curve | Can be misleadingly optimistic with severe imbalance |

AUPRC deserves special emphasis. AUROC can look great even with severe imbalance because the true negative rate (specificity) is easy to get right when negatives dominate. AUPRC focuses entirely on the positive class and gives you an honest picture of model performance where it matters.

### Threshold Tuning

A classifier outputs probabilities. The default threshold is 0.5 — predict positive if P(Y=1|X) > 0.5. But with imbalanced data, the optimal threshold is rarely 0.5. You should:

1. Train the model as usual (with class weights or focal loss).
2. Plot the precision-recall curve across all thresholds.
3. Pick the threshold that matches your cost structure — e.g., if missing a fraud case costs 100x a false alarm, optimize for recall at acceptable precision.

> Class imbalance is not a modeling problem — it's an *objective specification* problem. The model is doing exactly what you asked: minimizing average loss. The fix is telling the model what you actually care about, through weighted losses, appropriate metrics, and calibrated thresholds.

---

## Q&A

**Question:** You're building a medical diagnosis model for a rare disease (prevalence 0.1%). Your model achieves 99.9% accuracy, 0% recall. A colleague suggests SMOTE to fix it. Is SMOTE the right first move? What would you do instead, and in what order?

**Student's Answer:** SMOTE is not the right first move. With 0% recall, the model has learned nothing about the positive class — it's just predicting the majority class. SMOTE creates synthetic positives by interpolating in feature space, but if the features for the disease don't form a coherent cluster (medical features are often sparse and heterogeneous), SMOTE might generate nonsensical synthetic patients. My order: (1) First, add class weights to the loss function — this is the cheapest intervention and directly amplifies the minority gradient signal without manufacturing data. (2) Switch the evaluation metric from accuracy to AUPRC. (3) If class weights aren't enough, try focal loss to automatically focus on the hard examples. (4) Only then consider SMOTE, and if I do, combine it with undersampling of the majority class. (5) Finally, tune the classification threshold on a validation set using the precision-recall curve, optimizing for the clinical cost structure.

**Evaluation:** Excellent prioritization. You correctly identified that algorithm-level solutions (class weights, focal loss) should come before data-level solutions (SMOTE) because they're less invasive and don't risk generating unrealistic synthetic data in a domain where interpolation assumptions may not hold. The point about medical features not forming coherent clusters is especially important — SMOTE assumes that a linear combination of two sick patients looks like a sick patient, which is often false in high-dimensional clinical data. Your ordering from cheapest/safest intervention to most invasive is exactly the right engineering mindset.
