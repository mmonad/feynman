# Lesson 4: Data Leakage and Feature Engineering

*Course 17: Causality & Practical ML*

## Core Question

We've talked about how data can mislead you when you aggregate it wrong (Simpson's paradox) and when one class dominates (imbalance). Now let's talk about two more failure modes that separate experienced ML engineers from beginners: data leakage — where your model accidentally cheats — and feature engineering, where the human is still smarter than the algorithm (sometimes).

---

## Q77: Data Leakage — When Your Model Is a Cheater

### What Is Leakage?

Data leakage is when information from the future, the test set, or the target variable sneaks into your training pipeline. The model gets access to signal it would never have in deployment. The result: spectacular validation metrics, catastrophic production performance.

Think of it like this: you're training a student for an exam. Leakage is accidentally leaving the answer key in the study materials. The student aces the practice test, you think they've learned the material, and then they bomb the real thing because they were pattern-matching to the answers, not understanding the subject.

### Type 1: Target Leakage

Target leakage occurs when a feature is causally downstream of the target — it contains information that would only be available *after* the outcome you're predicting.

**Example:** Predicting hospital readmission within 30 days. You include "discharge summary text" as a feature. But the discharge summary is written by a doctor who *knows* the patient is at high risk of readmission — the doctor's concerns leak into the text. The model isn't predicting readmission; it's reading the doctor's prediction.

**Example:** Predicting credit default. You include "number of collection calls received." But collection calls happen *after* the customer starts defaulting. This feature has near-perfect predictive power because it's a consequence of the target.

The diagnostic signature is unmistakable: **suspiciously high performance.** If your model achieves 99% AUC on a problem where state-of-the-art is 75%, you don't have a breakthrough — you have leakage.

```
IF validation_metric >> published_baselines:
    SUSPECT leakage before celebrating
```

### Type 2: Train-Test Contamination

This is the more insidious form. The test data influences the training data through preprocessing.

**The classic mistake: normalizing before splitting.**

```
# WRONG: leaks test statistics into training
scaler.fit(ALL_DATA)                    # computes mean/std from everything
X_train = scaler.transform(X_train)     # uses test data's mean/std
X_test = scaler.transform(X_test)

# RIGHT: fit only on training data
scaler.fit(X_train)                     # mean/std from training only
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)       # applies training statistics to test
```

Why does this matter? The scaler computes the mean and standard deviation. If you include test data in that computation, every training example is subtly influenced by the test set's distribution. In small datasets, this can measurably inflate your metrics.

The same trap applies to: feature selection (selecting features based on correlation with the target across the full dataset), imputation (filling missing values with statistics computed from the full dataset), and target encoding (encoding categories based on target statistics from the full dataset).

### Type 3: Temporal Leakage

In time series and sequential problems, leakage means using future information to predict the past.

**Example:** Predicting stock prices. You engineer a feature: "average price over the next 5 days." Obviously you wouldn't do this intentionally. But subtler versions happen all the time:

- Using a moving average that's centered (looks forward and backward) instead of trailing.
- Including economic indicators that were revised retroactively — the number reported later is different from what was available at prediction time.
- Training on data where rows are shuffled, destroying temporal ordering, so the model sees "future" rows during training.

The fix: when splitting time series data, always use a **temporal split**, not a random split. Everything before time T is training; everything after is test. No exceptions.

### Detection and Prevention

| Leakage type | Detection signal | Prevention |
|---|---|---|
| Target leakage | Feature has implausible predictive power; model performance way above baselines | For each feature, ask: "Would I have this value at prediction time?" |
| Train-test contamination | Test performance drops significantly in production | Pipeline rule: all preprocessing fitted on training data only |
| Temporal leakage | Model performs well on random split but poorly on temporal split | Always use temporal splits for sequential data |
| Feature leakage | Feature importance shows a single feature dominating | Investigate any feature with >50% importance |

The meta-principle: **think about the deployment scenario.** At the moment your model makes a prediction in production, what information will it actually have? If your training pipeline gives the model *anything* beyond that, you have leakage.

> Data leakage is the single most common reason for the gap between offline metrics and production performance. The reason it's so dangerous is that every standard evaluation procedure — cross-validation, holdout sets, even early stopping — will confirm the leaky model's excellence. The model genuinely performs well on the test set, because the test set has the same leakage. Only deployment reveals the truth.

---

## Q78: Feature Engineering — When Humans Still Beat Algorithms

### Why Features Matter More Than Algorithms

There's a saying among Kaggle grandmasters that would horrify most ML researchers: "Feature engineering is more important than model selection." And the data backs them up. In most tabular ML competitions, the winning solution is gradient-boosted trees with clever features, not a novel architecture.

Why? Because an algorithm can only discover patterns that are *expressible* in its input representation. If the relationship between raw features and the target requires a complex nonlinear combination, the model has to discover that combination from data. But if you hand the model the right feature, the pattern becomes trivially learnable.

**Example:** Predicting house prices. You have square footage and number of bedrooms separately. A tree model *can* eventually discover that the ratio "sqft per bedroom" matters — but it needs many splits to approximate a division. If you create `sqft_per_bedroom = sqft / bedrooms` as an explicit feature, the model finds the pattern in one split.

### When Manual Feature Engineering Still Wins

Despite deep learning's ability to learn features automatically, manual feature engineering dominates in three scenarios:

**1. Small data.** With 5,000 rows, a neural network doesn't have enough data to learn useful feature combinations. Manually engineered features inject domain knowledge that substitutes for data.

**2. Tabular data.** For structured/tabular datasets, gradient-boosted trees with good features consistently beat neural networks. The Grinsztajn et al. (2022) benchmark confirmed what practitioners already knew: XGBoost/LightGBM on tabular data is hard to beat with deep learning, and the gap widens when features are well-engineered.

**3. Domain knowledge that's not in the data.** If you know from physics that two measurements should be combined as a ratio, or that a cyclical variable (hour of day, day of week) should be encoded with sine/cosine, the model cannot learn this efficiently from raw data.

### Key Techniques

**Interaction features:** Products or ratios of existing features.

```
price_per_sqft = price / sqft
bmi = weight / height^2
velocity = distance / time
```

These capture relationships that linear models and shallow trees struggle with.

**Target encoding:** Replace a categorical variable with the average target value for that category. If "zip code 94103" has an average house price of $1.2M, replace the category with 1.2M.

```
# For category c:
target_encode(c) = (Σ y_i for x_i = c + m · global_mean) / (n_c + m)
```

The `m` term is smoothing — for categories with few examples, blend toward the global mean. Without smoothing, rare categories get extreme (and noisy) encodings. This is a form of regularization.

Warning: target encoding is a leakage risk if not done carefully. You must compute target statistics on the training fold only, never the validation or test fold. Within training, use leave-one-out or k-fold target encoding to avoid overfitting.

**Time-based features:** From a timestamp, extract features at multiple granularities:

```
hour_of_day, day_of_week, month, is_weekend, is_holiday
days_since_last_event, events_in_last_7_days
```

For cyclical features, use sine/cosine encoding:

```
hour_sin = sin(2π · hour / 24)
hour_cos = cos(2π · hour / 24)
```

This ensures that hour 23 and hour 0 are close together (as they should be), which a raw integer feature can't represent.

**Aggregation features:** For relational data (a customer with many transactions), compute per-entity aggregates:

```
customer_total_spend, customer_avg_order_size
customer_days_since_last_purchase, customer_order_count_30d
```

### Feature Selection: Removing the Noise

More features isn't always better. Irrelevant features add noise, increase overfitting risk, and slow training. Selection methods, ranked from simple to sophisticated:

| Method | How it works | Pros | Cons |
|---|---|---|---|
| **Correlation filter** | Drop features with low correlation to target | Fast, model-free | Misses nonlinear relationships |
| **Mutual information** | Measures nonlinear dependency with target | Catches nonlinear effects | Doesn't account for redundancy between features |
| **Tree-based importance** | Train a tree, measure feature importance | Accounts for interactions | Can be biased toward high-cardinality features |
| **Recursive elimination** | Repeatedly train, drop least important feature | Considers feature interactions | Computationally expensive |
| **L1 regularization** | Add penalty that drives unimportant weights to zero | Built into training | Only works for models with L1 support |

The practical recipe: start with mutual information to filter obvious noise, then use tree-based importance for a second pass, then do ablation studies (remove groups of features and measure impact).

> The deep learning era has created a misconception that feature engineering is obsolete — that neural networks "learn features automatically." This is true for images, audio, and text, where the raw input (pixels, waveforms, tokens) has local structure that convolutions and attention can exploit. For tabular data, time series with domain-specific semantics, and small datasets, the human who understands the domain still outperforms the algorithm that doesn't.

---

## Q&A

**Question:** You train a model to predict customer churn. It achieves 0.98 AUC. The best published result on a similar problem is 0.82 AUC. Walk me through your debugging process. What do you check, in what order?

**Student's Answer:** 0.98 AUC on a churn problem is almost certainly leakage. My debugging order: (1) Check for target leakage — look at feature importances. If a single feature dominates (>40% importance), investigate whether it's a consequence of the target. For churn, common culprits: "cancellation request date," "account_status = closed," or "last_activity_date" that's suspiciously close to the churn date. (2) Check temporal ordering — am I using a temporal train/test split? If I randomly shuffled the data, the model might be using future information. (3) Check preprocessing — did I fit the scaler/encoder on the full dataset or just the training set? (4) Check for train-test contamination — are there duplicate customer IDs appearing in both train and test? (5) If nothing obvious, do a feature ablation: remove the top features one by one and see if performance drops sharply to a realistic level. The feature that causes the biggest drop when removed is the leak. (6) Finally, the sanity check: would I have all these features available at the moment I need to make the prediction in production?

**Evaluation:** Textbook debugging process with exactly the right priorities. Starting with feature importance is smart because target leakage almost always manifests as a single dominant feature. The temporal ordering check is critical for churn since it's inherently a time-based problem. And the production availability sanity check at the end is the master key — it catches every type of leakage in one question. One addition: check if the target definition itself is leaky. Sometimes "churn" is defined as "no activity in 90 days," and a feature like "days since last login" computed at the end of the 90-day window essentially encodes the target.
