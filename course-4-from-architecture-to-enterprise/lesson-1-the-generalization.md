# Lesson 1: The Generalization — From Code Review to Universal Verification Committees

*Course 4: From Architecture to Enterprise*

## Core Question

What did we actually build in Course 3? Strip away the code-specific details and look at the skeleton:

```
1. A complex artifact needs to be evaluated for correctness
2. "Correctness" is multi-dimensional (no single axis captures it)
3. Single evaluators have systematic blind spots
4. Solution: orthogonal specialists + composition + feedback loop
```

Nothing code-specific about that. It's a **verification architecture** applicable anywhere those four conditions hold.

## Domain Analysis

### Legal Contract Review

```
Artifact:              Contracts (M&A, licensing, employment)
Dimensions:            Legal validity, commercial terms, risk exposure,
                       regulatory compliance, enforceability
Blind spots:           Tax lawyer misses IP risks; commercial lawyer misses regulatory issues
Ground truth:          Partial — court rulings, penalties, disputes (delayed but clear)
Error cost:            Billions (single missed M&A clause)
```

Almost perfect fit. Medicine already uses this pattern with human specialists — tumor boards are orthogonal expert committees.

### Medical Diagnosis

```
Artifact:              Patient data (imaging, labs, symptoms, history)
Dimensions:            Radiology, lab interpretation, symptom patterns,
                       drug interactions, rare disease recognition
Blind spots:           Cardiology model misses oncological signs in chest X-ray
Ground truth:          Strong — pathology, treatment outcomes (sometimes delayed years)
```

Excellent fit. Tumor boards are the human version of our committee architecture.

### Financial Model Auditing

```
Artifact:              Financial models (DCF, risk models, trading strategies)
Dimensions:            Math correctness, assumption reasonableness, regulatory compliance,
                       market risk factors, historical consistency
Blind spots:           Formula checker won't question 15% perpetual growth rate
Ground truth:          Excellent — market outcomes provide definitive, timestamped truth
Error cost:            Billions (2008 crisis = insufficient verification diversity)
```

Strong fit. The 2008 crisis was partly attributable to insufficient diversity of perspectives evaluating mortgage-backed security models.

### Scientific Paper Peer Review

```
Dimensions:            Statistical validity, methodology, novelty, reproducibility, ethics
Ground truth:          Weak — replications rare, retractions take years
```

Decent fit but weak ground truth makes composition model harder to train and monitor.

### Hardware Design Verification

```
Dimensions:            Functional correctness, timing, power, yield, security
Ground truth:          Strong — simulation, formal verification, physical testing
Error cost:            Extreme — post-fabrication bug costs hundreds of millions (Intel FDIV)
```

Excellent fit. Already uses extensive verification but diversity often insufficient.

## What Makes a Domain UNSUITABLE

### 1. Correctness Is Unidimensional
One axis of correct/incorrect → one good evaluator suffices. Example: spell checking.

### 2. Ground Truth Is Unavailable
Can't train composition model, can't detect degradation. Example: long-term geopolitical prediction.

### 3. The Artifact Doesn't Hold Still
Changes faster than evaluation process. Example: real-time trading decisions — speed dominates diversity.

### 4. Human Judgment Is Irreducibly Dominant
90% of value is in specification/organizational context. Example: early-stage startup evaluation — almost entirely human judgment.

## The Suitability Framework

| Factor | Low (1) | High (5) |
|---|---|---|
| **Dimensionality** | Binary/unidimensional | 5+ independent dimensions |
| **Error Cost** | Cheap to fix | Costs millions or lives |
| **Ground Truth** | Never know if wrong | Definitive feedback within weeks |
| **Current Automation** | Well-automated already | Still mostly manual/human |
| **Regulatory Pressure** | No compliance requirements | Heavy regulation demanding auditability |
| **Artifact Stability** | Real-time changes | Static artifact at a point in time |

Domains scoring 25+/30 are prime targets:

```
                    Dim  Error  Truth  Auto  Reg  Stable  TOTAL
Financial models     5     5      5     3     5     5       28
Legal contracts      5     5      3     4     5     5       27
Crypto protocols     5     5      4     4     4     5       27
Hardware design      5     5      5     3     3     5       26
Medical diagnosis    5     5      4     3     4     4       25
Code review          4     4      4     3     2     5       22
Scientific papers    4     3      2     4     2     5       20
```

---

## Q&A

**Question:** Crypto protocol verification scores 27 — tied with legal, below financial models. But I'd argue it's the *best* starting vertical for a startup. Why? What property does it have that higher-scoring domains don't?

**Student's Answer:** It's the closest thing to code and math and logic, which makes it relatively easier to create higher-quality synthetic data — absolutely crucial for composition model training. The feedback loop is almost black and white, perfect for RL reward. The cost of a security bug in crypto is fatal so price leverage is high. The alternative is human auditors who are ill-suited to review tens of thousands of lines of code — expensive, slow, not accountable, and lacking diversity of opinions.

**Evaluation:** Every point lands. Ranked by startup importance:

### 1. Ground Truth Quality (The Killer Point)
Crypto has **near-binary ground truth.** Protocol preserves a security property or doesn't. Proof holds or doesn't. No "depends on interpretation" like legal, no "we'll know in five years" like medicine.

This means: mutation testing is trivially well-defined, RL rewards are crisp, composition model gets unambiguous training signal, iteration speed is fast. **For a startup, the domain where you can build the feedback loop fastest is where you compound advantage fastest.**

### 2. Human Auditor Inadequacy (The Business Case)
```
Human crypto auditors:
  - Expensive:        $500-2,000/hour, audits $50K-500K+
  - Slow:             weeks to months per audit
  - Scarce:           maybe a few hundred qualified globally
  - Not diverse:      1-2 reviewers, each with blind spots
  - Not accountable:  auditor signs off, protocol gets hacked, no consequence
  - Don't scale:      DeFi deploys faster than humans can review
```

Billions sit in contracts audited by 1-2 humans under time pressure. The graveyard of hacked protocols proves market need.

### 3. Market Readiness (Point the Student Didn't Make)
Crypto community is **natively technical, risk-tolerant, and desperate for better tooling.** They trust code over institutions. Will evaluate on merits, not regulatory approval. Time-to-revenue in months, not years.

Contrast: selling AI diagnostics to hospitals = years of regulatory approval, trust-building, pilot programs. Crypto = ship it, prove it works, get paid.
