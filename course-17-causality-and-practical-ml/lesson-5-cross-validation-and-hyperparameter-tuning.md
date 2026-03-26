# Lesson 5: Cross-Validation and Hyperparameter Tuning

*Course 17: Causality & Practical ML*

## Core Question

You've built a model, engineered your features, checked for leakage. Now you need to answer two questions that every ML engineer faces: "How good is this model, really?" and "What settings should I use?" These are the problems of **evaluation** and **optimization**, and getting them wrong leads to the most common silent failure in ML: a model that looks great on your laptop and falls apart in production.

---

## Q79: Cross-Validation — Honest Model Evaluation

### Why Holdout Isn't Enough

The simplest evaluation: split your data into train and test, train on train, evaluate on test. What's wrong with this?

One number. You get a single performance estimate from one particular split. Maybe you got lucky and the test set is easy. Maybe you got unlucky and the test set contains outliers. The estimate has high variance, and you don't know how much.

Cross-validation reduces this variance by evaluating on *every* data point, using it for testing exactly once.

### K-Fold Cross-Validation

The procedure:

1. Shuffle the data and split it into k roughly equal folds.
2. For each fold i = 1, ..., k:
   - Train on all folds except fold i.
   - Evaluate on fold i.
3. Average the k evaluation scores.

```
Fold 1: [TEST] [train] [train] [train] [train]  → score_1
Fold 2: [train] [TEST] [train] [train] [train]  → score_2
Fold 3: [train] [train] [TEST] [train] [train]  → score_3
Fold 4: [train] [train] [train] [TEST] [train]  → score_4
Fold 5: [train] [train] [train] [train] [TEST]  → score_5

Final estimate = mean(score_1, ..., score_5)
Standard error = std(scores) / √k
```

Every data point appears in exactly one test fold. The average score is a lower-variance estimate of true performance than any single train-test split.

**Choosing k:** The most common choice is k = 5 or k = 10. There's a bias-variance tradeoff *in the evaluation itself*:

| k | Training set size (fraction) | Estimate bias | Estimate variance | Compute cost |
|---|---|---|---|---|
| 2 | 50% | High (small training sets underperform) | Low | Low |
| 5 | 80% | Moderate | Moderate | Moderate |
| 10 | 90% | Low | Moderate-high | High |
| n (LOOCV) | ~100% | Lowest | Highest (estimates are highly correlated) | Very high |

Higher k means each fold trains on more data (less pessimistic bias) but the k training sets overlap more (more correlated estimates, higher variance of the mean). k = 5 or 10 tends to be the sweet spot.

### Leave-One-Out Cross-Validation (LOOCV)

The extreme case: k = n. Each fold is a single data point. You train n models, each on n-1 examples, and test on the one left out.

LOOCV has nearly zero bias (you're training on almost all the data), but the variance can be *higher* than k-fold because the n training sets differ by only one example — they're nearly identical, so the n estimates are highly correlated. The mean of highly correlated estimates has higher variance than the mean of less correlated ones.

Use LOOCV when: n is so small (<100) that even k = 5 gives you only 20 test examples per fold. In that regime, the per-fold variance is huge, and LOOCV's lower bias is worth the computational cost.

### Stratified Cross-Validation

For classification with imbalanced classes, random k-fold splits can give you folds where the minority class is under- or over-represented. Stratified k-fold ensures each fold has approximately the same class distribution as the full dataset.

This is especially important when the minority class has, say, 50 examples total. With k = 5 and no stratification, one fold might get 15 minority examples and another might get 5. Stratification gives each fold 10.

### When Cross-Validation Is Wrong

Cross-validation assumes that training and test data are drawn i.i.d. from the same distribution. This assumption fails spectacularly in two scenarios:

**Time series data:** If your data is ordered in time, a random k-fold split puts future data in the training set and past data in the test set. The model literally trains on the future to predict the past. You must use **time series split** (expanding or sliding window):

```
Time series split (expanding window):
Train: [1---4]    Test: [5]
Train: [1-----5]  Test: [6]
Train: [1------6] Test: [7]
```

**Grouped data:** If multiple rows belong to the same entity (multiple medical visits from the same patient), random splitting can put different visits from the same patient in train and test. The model memorizes patient-specific patterns and appears to generalize when it doesn't. Use **group k-fold**, where all rows for a given patient are in the same fold.

### Nested Cross-Validation

Here's a subtle trap. You use 5-fold CV to compare three models (logistic regression, random forest, XGBoost). You pick the model with the best CV score and report that score as your expected performance. **That reported score is biased upward.**

Why? You selected the best of three candidates. Even if the models are identical in truth, random variation means one will look best by chance. This is the multiple comparisons problem applied to model selection.

**Nested CV** fixes this by using two loops:

```
Outer loop (evaluation):
  For each outer fold:
    Inner loop (model selection):
      For each inner fold:
        Train and evaluate each candidate model
      Select best model based on inner CV score
    Train best model on outer training set
    Evaluate on outer test fold  → this score is unbiased

Final unbiased estimate = mean of outer fold scores
```

The inner loop selects the best model. The outer loop evaluates *the selection process itself*. The outer score reflects the performance you'd actually get by running the inner model selection procedure on new data.

This is computationally expensive (k_outer × k_inner × n_models training runs), but it gives you the only truly unbiased evaluation when you're doing model selection.

> Cross-validation answers the question "how well will this model generalize?" But the answer is only honest if the CV procedure mirrors the deployment scenario. Random splits for i.i.d. data, temporal splits for time series, group splits for grouped data, and nested CV when you're also doing model selection.

---

## Q80: Hyperparameter Tuning — Searching the Configuration Space

### The Problem

A model's **parameters** are learned from data (weights, biases). Its **hyperparameters** are set by you before training (learning rate, number of layers, regularization strength, batch size). The performance surface over hyperparameter space is typically non-convex, noisy, and expensive to evaluate (each point requires a full training run).

### Grid Search: The Brute Force Approach

Pick a set of values for each hyperparameter and try every combination.

```
learning_rate: [0.001, 0.01, 0.1]
batch_size: [32, 64, 128]
dropout: [0.1, 0.3, 0.5]

Total combinations: 3 × 3 × 3 = 27 experiments
```

The problem is the **curse of dimensionality**. With d hyperparameters and m values each, you need m^d experiments. 5 hyperparameters with 5 values each: 3,125 experiments. 10 hyperparameters: nearly 10 million. Grid search is exponentially expensive.

Worse, grid search wastes budget on unimportant hyperparameters. If learning rate matters a lot but dropout barely matters, you still evaluate every dropout value at every learning rate. Most of the compute is wasted exploring a dimension that doesn't matter.

### Random Search: Surprisingly Better

Bergstra and Bengio (2012) showed something that surprised a lot of people: **random search is more efficient than grid search.** Instead of an evenly spaced grid, sample hyperparameters randomly from their ranges.

Why does random beat grid? The key insight is about *effective dimensionality*. In most problems, only 1-2 hyperparameters actually matter. Grid search gives you m distinct values for each hyperparameter. Random search with the same budget (m^d trials) gives you m^d distinct values for each hyperparameter — because every trial uses a different random value.

```
Grid search with 9 trials over 2 parameters:
  3 distinct values for learning_rate
  3 distinct values for dropout

Random search with 9 trials over 2 parameters:
  9 distinct values for learning_rate    ← 3x more coverage
  9 distinct values for dropout
```

If only learning rate matters, grid search explores 3 values. Random search explores 9. You get 3x better coverage of the important dimension for the same budget.

### Bayesian Optimization: Learning Where to Look

Random search treats each trial independently — it doesn't learn from previous results. Bayesian optimization (BO) builds a **surrogate model** of the performance surface and uses it to decide where to search next.

The procedure:

1. Evaluate a few random points to initialize.
2. Fit a **surrogate model** (typically a Gaussian Process) to the observed (hyperparameter → performance) pairs.
3. Use an **acquisition function** to select the next point to evaluate — balancing exploitation (try near the current best) and exploration (try uncertain regions).
4. Evaluate the model at that point. Update the surrogate. Repeat.

The acquisition function is the clever part. The most common one, **Expected Improvement (EI)**, asks: "given my surrogate model's prediction and uncertainty at this point, how much improvement over my current best do I *expect*?"

```
EI(x) = E[max(f(x) - f_best, 0)]
```

Points with high predicted performance OR high uncertainty get high EI. This naturally balances exploitation and exploration.

| Method | Compute per trial | Accounts for past trials? | Parallelizable? | Best for |
|---|---|---|---|---|
| Grid search | Low overhead | No | Perfectly | ≤3 hyperparameters |
| Random search | Low overhead | No | Perfectly | ≤10 hyperparameters, limited budget |
| Bayesian optimization | Surrogate model fitting | Yes | Limited (sequential by nature) | Expensive models, small budgets |
| Population-based training | Varies | Yes (within population) | Highly | Large-scale deep learning |

### Population-Based Training (PBT)

PBT (Jaderberg et al., 2017) borrows from evolutionary algorithms. Train a population of models simultaneously with different hyperparameters. Periodically, each model checks if another model in the population is doing better. If so, it copies that model's weights AND hyperparameters, then perturbs the hyperparameters slightly. Poor configurations die; good configurations reproduce and mutate.

The key advantage: PBT can *change hyperparameters during training*. A high learning rate at the start that decays later isn't a schedule you specified — it's an emergent behavior the population discovered. This is especially useful for learning rate schedules, which are notoriously hard to tune.

### The One Hyperparameter That Always Matters Most

If you take away one thing from this lesson, let it be this: **the learning rate is almost always the most important hyperparameter.** Across architectures, across tasks, across domains, the learning rate has the largest impact on final performance.

```
Priority ordering (for most deep learning):
1. Learning rate              ← always tune this first
2. Batch size / LR coupling   ← these interact strongly
3. Weight decay               ← regularization
4. Architecture-specific      ← dropout, number of layers, hidden size
5. Everything else            ← optimizer betas, warmup steps, etc.
```

The practical recipe: use random search or Bayesian optimization over a wide range of learning rates (1e-5 to 1e-1, log-uniform), with a few values for other important hyperparameters. This gets you 90% of the way there with 20-50 trials.

> Hyperparameter tuning is a search problem in an expensive-to-evaluate, low-dimensional (in effective terms) landscape. Grid search wastes budget on unimportant dimensions. Random search covers important dimensions efficiently. Bayesian optimization learns from past evaluations to search smarter. But before any sophisticated method, sweep the learning rate — it dominates everything else.

---

## Q&A

**Question:** You're choosing between three models for a dataset with 2,000 examples and no time structure. You want to report the best model's expected performance. Describe your complete evaluation protocol, including how you handle model selection vs. performance estimation.

**Student's Answer:** With 2,000 examples, I have enough data for cross-validation but not enough to waste on a large holdout. I'd use nested cross-validation. Outer loop: 5-fold CV for unbiased performance estimation. Inner loop: 5-fold CV for model selection and hyperparameter tuning. For each outer fold, I have 1,600 training examples. The inner loop splits those 1,600 into 5 inner folds (1,280 train, 320 validation each). I tune hyperparameters for all three models using random search within the inner loop — maybe 30 random configurations per model. I select the model+hyperparameters with the best inner CV score. Then I retrain that selected model on all 1,600 outer training examples and evaluate on the 400 outer test examples. I repeat this for each outer fold and report the mean and standard error of the 5 outer test scores. This gives me an unbiased estimate of the performance I'd get by running my entire model selection pipeline on this amount of data.

**Evaluation:** Perfect application of nested CV. You correctly identified the need for two separate loops — inner for selection, outer for evaluation — and gave concrete numbers for the fold sizes. The 30 random configurations per model is a reasonable budget for inner tuning. One detail to add: since you have 2,000 examples, use stratified folds at both levels if this is a classification problem with any class imbalance. Also, when you report results, report the outer CV mean with a confidence interval (mean +/- 2 standard errors), not just the mean — this is what distinguishes a rigorous evaluation from a number someone threw out.
