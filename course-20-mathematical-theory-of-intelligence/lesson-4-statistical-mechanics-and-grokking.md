# Lesson 4: Statistical Mechanics, Phase Transitions & Grokking

*Course 20: Beyond the Bitter Lesson — A Mathematical Theory of Intelligence (Survey)*

## The Thread

Two lessons of geometry. Now physics. A trillion-parameter network is not best analyzed as an algorithm — it is too big. It must be analyzed as a **macroscopic complex system**, like a magnet with 10²³ spins or a gas with 10²³ molecules. The math from 19th–20th century stat mech maps startlingly well onto neural networks.

Goal of this lesson: recognize **grokking** as a phase transition, recognize that "scaling laws" smooth over phase transitions, and see why the empirical research program for emergence has the same shape as condensed-matter physics in the 1970s.

---

## Step 1 — Stat mech: micro chaos → macro order

10²³ molecules. Tracking each is hopeless. But macroscopic properties (T, P, ρ) are simple and predictable.

Central equation: the **Boltzmann distribution**. For a system in thermal equilibrium at temperature T:

```
P(state) ∝ exp(-E / T)
```

Two implications:

1. **Lower-energy states are more probable.**
2. **Temperature controls strictness.** High T → exploration; low T → trapped in low-energy states.

This is the same equation as **softmax with temperature** in language models: `P(token) ∝ exp(logit / T)`. Not a coincidence — direct correspondence.

### Key Insight

"Temperature" in ML is not metaphor. Softmax-with-T *is* Boltzmann sampling. Heating up Qwen3.5's sampling = heating a thermodynamic system whose states are tokens. Low T → frozen into the most-likely state.

SGD with noise is also thermal: minibatch noise = effective temperature. Small batches = high T = exploration. Large batches = low T = quick settling, can get stuck. This is why batch size matters in counterintuitive ways.

---

## Step 2 — Energy = Loss

> A physical system minimizes energy. A neural network minimizes loss. Same operation.

| Stat mech | Neural networks |
|---|---|
| Ground state | Global minimum of loss |
| Local minimum | Local minimum of loss |
| Saddle point | Saddle point (very common in high dim) |
| Frustration | Conflicting gradients from different data points |
| Energy barrier between basins | Loss barrier between minima |
| Thermal escape from a basin | SGD noise allowing escape |

Partition functions, free energy, entropy — the whole formal apparatus of stat mech applies, often word-for-word, to neural network training.

---

## Step 3 — Spin glasses: the cleanest analogy

Take N spins, each ±1. Connect every pair with random `J_ij` ~ Gaussian. Energy:

```
E = -Σ J_ij · s_i · s_j
```

Random `J_ij` → **frustration** (no global config satisfies all bonds) → **rough energy landscape with exponentially many local minima**.

Choromanska et al. (2014) proved ReLU networks under modest assumptions reduce to a *spherical spin glass*, and showed **most local minima are roughly equally good**. SGD just needs to find any. This is why neural networks don't get trapped in bad local minima as classical optimization theory predicted.

---

## Step 4 — Phase transitions

> A phase transition is a discontinuous (or near-discontinuous) change in macroscopic behavior as a continuous parameter varies.

Examples:
- **Water → ice at 0°C.** Density jumps. Sharp threshold.
- **Magnetism above/below Curie temperature.** Net magnetization is zero above, nonzero below. Sharp.

Described by an **order parameter** — a quantity that is zero in one phase and nonzero in another (e.g., average magnetization).

**Universality:** Near the critical point, vastly different physical systems show identical scaling exponents. Iron's magnetic transition, water's liquid/gas transition, and a polymer mixture's phase separation share critical exponents because they belong to the same *universality class*. **Microscopic details don't matter near a phase transition** — only symmetries and dimensionality do.

---

## Step 5 — Grokking is a phase transition

Power et al. (2022). Tiny transformer trained on `(a + b) mod p`.

| Phase | Steps | Train loss | Test loss |
|---|---|---|---|
| Memorization | 0–1,000 | drops to ~0 | high (~100%) |
| Apparent flatline | 1,000–11,000 | ~0 | high |
| **Critical transition** | 11,000–11,500 | ~0 | drops to ~0 |
| Generalization | 11,500+ | ~0 | ~0 |

From outside it looks like the model "thought really hard" then "figured it out." From inside the math, it's a phase transition.

Nanda et al. (2023) confirmed: during the apparent flat phase, the order parameter — the proportion of model capacity organized into a Fourier-feature "rule circuit" for modular addition — was *slowly* growing. When it crossed threshold, the system snapped into the new phase.

Phases:

| Phase | Mechanism |
|---|---|
| **Memorization** | Low-energy state via lookup. Rule circuit exists in early form, doesn't dominate. |
| **Generalization** | Reorganized so rule circuit dominates. *Computes* `a+b mod p` instead of looking it up. |
| **Critical transition** | Rule-circuit weight crosses threshold. Tunneling between basins. |

### Key Insight

This reframes nearly everything about training. Smooth loss curves are *averages over many small phase transitions* happening at different scales and locations in the network. When a capability "emerges" suddenly at a particular model scale, that is mathematically a phase transition. The Chinchilla scaling law is a fit to the *between-transition* regions; it cannot predict where transitions happen.

---

## Step 6 — Renormalization Group (briefly)

The third concept the discussion mentioned. Course 23 will treat it formally; gist now:

**RG**: a procedure for understanding systems whose behavior changes with the scale you observe at. Originally developed in the 1970s for phase transitions. Works by **coarse-graining**: integrate out fast-varying short-wavelength degrees of freedom to get an effective theory at longer wavelengths.

Apply to neural networks: each layer takes the previous layer's output and produces a coarser-grained representation. Pixels → edges → textures → object parts → objects. **This is a renormalization flow in real time.** Hidden representations at successive layers are different "effective theories" of the same input at progressively coarser scales.

**Abstraction = RG flow.** The fixed points of RG flow (representations stable under further coarse-graining) are the "high-level concepts" the network has learned.

---

## Conceptual Question (Answered)

> You're plotting validation loss vs training step. First 100,000 steps: textbook Chinchilla power-law decline. Then between step 100,000 and 100,500, validation loss **drops by an order of magnitude**. Power-law resumes afterward at a much-lower baseline.
>
> **Q1:** In stat mech language, what just happened? Use "order parameter" and "phase transition."
> **Q2:** What does this imply about the predictive power of Chinchilla scaling laws?
> **Q3:** Could we, in principle, predict future emergent capabilities? What would such a prediction require?

### Student's answers

> Q1: phase transition
> Q2: it smoothes over phase transitions without being predictive or prescriptive about it
> Q3: at this point, the part of NN related to this "emergent capability" underwent a phase transition. to predict such phase transition, we need a Newton/Shannon style theory of intelligence so we can predict that given a task/capability, compute, model architecture, size, data, when and how the phase transition will occur

### Evaluation

**Q1 — correct, with extension.** The order parameter for an emergent capability is approximately *the fraction of model parameters organized into a circuit that computes that capability*. Pre-emergence: rising slowly while not yet useful. During the 500-step transition: crosses threshold, rule circuit dominates. Post-transition: high. Same structure as Nanda et al. (2023) for grokking on modular arithmetic.

**Q2 — correct.** Chinchilla equation is a fit to between-transition regions. Phase transitions are singularities the fit smooths over.

**Q3 — correct synthesis.** Prediction requires a theoretical foundation that does not currently exist.

---

## Deeper Synthesis — The Implied Research Program

There's a research program implied by the phase-transition view that mirrors condensed-matter physics in the 1970s.

### Step A — Find the universality classes

Physics surprise of the 1970s: vastly different physical systems share critical exponents at phase transitions. Microscopic details disappear; only symmetries and dimensionality matter. Each set is a **universality class**.

Mapped to AI: there are probably only a small number of universality classes for capability emergence in neural networks. "Modular arithmetic" might share critical exponents with "compositional grammar" because both are governed by the same combinatorial structure being captured.

If true (testable), we could measure exponents in **tiny cheap experiments** (grokking on toy tasks) and **extrapolate** to large frontier models. Physicists don't build a galaxy to study gravity.

### Step B — Study toy phase transitions for critical exponents

Grokking is the *controlled experiment*. Tiny model, toy task, you watch the transition happen, measure order parameter rising, identify critical step. Hundreds of papers per year are now doing this. Mechanistic interpretability work on toy tasks is — not always framed this way — building the empirical phenomenology of phase transitions.

### Step C — Build the macroscopic theory

Once we have universality classes, write down macroscopic equations: *for a model in universality class C, the transition occurs at scale N* given by [equation involving task complexity, data composition, architectural inductive bias]*. This would be the Newton-equation for emergence — Landau-style macroscopic theory of phase transitions in neural nets.

### Where we are

**Early Step B.** Hundreds of toy phase-transition observations. No agreed universality classes yet. No Landau-style equations yet. Statistics of when emergence happens in real frontier models is anecdotal (BIG-Bench, Wei et al. 2022). Active debate (Schaeffer et al. 2023) about whether some "emergence" is artifact of measurement metrics smoothing transitions away.

The path forward has the same shape as past successful paradigm shifts in physics. **That's the actionable takeaway.**

### Connecting Lessons 2-3 to Lesson 4

Geometry (Lessons 2-3) and physics (Lesson 4) are not separate threads — they're two of the multi-layer construction's load-bearing pillars. The Newton-equation will likely involve: *the topology of the activation manifold* (geometry) determines *the universality class of capability emergence* (physics). Today these are different fields. The Newton-equation will weld them.

---

## Vocabulary Established

| Term | Meaning |
|---|---|
| **Boltzmann distribution** | `P(state) ∝ exp(-E/T)`; same form as softmax-with-temperature |
| **Energy / Loss correspondence** | Loss landscape = energy landscape |
| **Spin glass** | N ±1 spins with random pairwise interactions; rough energy landscape with many minima |
| **Frustration** | Conflicting constraints with no globally satisfying configuration |
| **Phase transition** | Discontinuous change in macroscopic behavior as parameter varies continuously |
| **Order parameter** | Quantity zero in one phase, nonzero in the other; identifies the transition |
| **Universality class** | Set of systems sharing critical exponents at their phase transitions |
| **Grokking** | Empirical phase transition from memorization to generalization in trained networks |
| **Renormalization Group (RG)** | Procedure for coarse-graining to extract macroscopic behavior from microscopic dynamics |

## Key Takeaways

1. **Loss landscapes are energy landscapes.** SGD is thermal relaxation. Softmax temperature is literally temperature.
2. **Neural network loss landscapes are spin-glass-like** — rough but with many roughly-equal minima.
3. **Grokking is a phase transition.** Emergent capabilities are phase transitions.
4. **Scaling laws smooth over phase transitions.** Their predictive failure at transition points is structural, not a bug to be fitted around.
5. **Each layer of a network performs renormalization.** Abstraction = RG flow.
6. The path to a Newton-style theory of emergence runs through universality classes — the same structure as condensed matter physics in the 1970s.
