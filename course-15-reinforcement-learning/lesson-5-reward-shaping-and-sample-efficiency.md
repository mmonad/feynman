# Lesson 5: Reward Shaping and Sample Efficiency

*Course 15: Reinforcement Learning*

## Core Question

You can hand-engineer rewards to guide your agent — but doing it wrong changes what the agent actually optimizes. What's the precise condition under which reward shaping is safe, and why is RL so catastrophically data-hungry compared to supervised learning?

---

## Q59: Reward Shaping — The Precise Boundary Between Help and Harm

### The Temptation

Your robot needs to navigate a maze. The natural reward is +1 for reaching the goal, 0 otherwise. The agent wanders randomly for thousands of episodes, getting zero reward, learning nothing. The gradient is zero. Nothing happens.

Obvious fix: give a small reward for getting closer to the goal. Every step that reduces the Manhattan distance to the exit gets +0.1. Now the agent gets gradient signal from the very first episode. Training is 100x faster. Problem solved, right?

Not necessarily. You've changed the optimization objective. The agent is no longer maximizing "reach the goal" — it's maximizing "reduce distance to the goal per step." What if there's a wall between the agent and the goal? The agent might pace back and forth near the wall, collecting +0.1 for each step toward the goal and ignoring the longer path that actually reaches it. You've shaped the reward, and the agent found the optimal policy for your shaped reward — which isn't the optimal policy for the original task.

### The Theorem: Potential-Based Reward Shaping

Ng, Harada, and Russell (1999) proved exactly when reward shaping is safe. Define a shaping function F(s, s') that adds a bonus for transitioning from state s to state s'. The shaped reward is:

```
r'(s, a, s') = r(s, a, s') + F(s, s')
```

**Theorem:** The shaped reward preserves the optimal policy (i.e., every optimal policy under r' is also optimal under r) **if and only if** F has the form:

```
F(s, s') = γ · Φ(s') - Φ(s)
```

for some potential function Φ: S → ℝ.

### Why This Works: The Telescoping Argument

Consider the total shaped reward over a complete episode of length T:

```
Σ_{t=0}^{T-1} γ^t · F(s_t, s_{t+1})
= Σ_{t=0}^{T-1} γ^t · [γ · Φ(s_{t+1}) - Φ(s_t)]
= Σ_{t=0}^{T-1} [γ^{t+1} · Φ(s_{t+1}) - γ^t · Φ(s_t)]
```

This is a telescoping sum. Everything cancels except the first and last terms:

```
= γ^T · Φ(s_T) - Φ(s_0)
```

If all episodes start in the same state s_0 and end in the same terminal state s_T (or if Φ(terminal) = 0), this is a **constant** — the same for every trajectory. Adding a constant to all returns doesn't change which trajectory has the highest return, so the optimal policy is unchanged.

Any other form of F doesn't telescope perfectly. The residual terms depend on the trajectory, which means different trajectories get different bonus amounts, which means the ranking of trajectories can change, which means the optimal policy can change.

### Examples of Bad Reward Shaping

**Distance-based reward in a maze:** F(s, s') = -d(s', goal) + d(s, goal). This is the distance reduction we discussed earlier. It looks like potential-based shaping with Φ(s) = -d(s, goal), which would give F = γ·(-d(s', goal)) - (-d(s, goal)) = -γ·d(s', goal) + d(s, goal). But the actual shaping function -d(s', goal) + d(s, goal) doesn't have the γ factor. It's missing the discount! Close, but wrong. And "close" in reward shaping means "different optimal policy."

**Reward for staying alive:** In a game where the agent dies upon failing the task, adding +1 per timestep for being alive seems harmless — it encourages the agent to survive. But now the agent is incentivized to avoid risky task-relevant actions. An agent learning to cross a bridge might just stand still forever, collecting +1 per step, rather than attempt the crossing and risk death.

**Reward hacking in general:** Any shaped reward that doesn't satisfy the potential-based condition creates a new optimization target. The agent will find the optimal policy for that new target, which may involve behaviors the designer never anticipated. This is reward hacking — the agent is doing exactly what you asked, just not what you meant.

### Connection to Alignment

This is why alignment researchers care so much about reward specification. RL agents are optimization machines. They will find every loophole in your reward function. If your reward imperfectly captures your intent — and it always does, outside of simple, well-defined environments — the agent may optimize the imperfection. Goodhart's Law in action: "When a measure becomes a target, it ceases to be a good measure."

In RLHF, the reward model is itself a learned approximation of human preferences. It has errors. The language model, optimized against this imperfect reward model, will exploit those errors — producing outputs that score highly on the reward model but are actually lower quality. This is why the KL penalty against the SFT reference model is essential: it limits how far the policy can drift into the "reward model exploitation" regime.

> **Key takeaway:** Potential-based shaping (F = γΦ(s') - Φ(s)) is provably the ONLY safe form of reward shaping. Anything else can change the optimal policy. The telescoping argument is the proof — and it's simple enough to derive on a whiteboard in an interview. Every other form of reward modification is an implicit change to the optimization objective, whether you intended it or not.

---

## Q60: Why RL Is So Sample Inefficient

### The Scale of the Problem

Supervised learning on ImageNet: ~1.2 million labeled images → state-of-the-art classification. A single pass through the data (one epoch) gives a reasonable model.

Atari with DQN: ~200 million frames (equivalent to ~38 days of continuous play at 60fps) → human-level performance on some games. And that's *after* experience replay lets you reuse each frame multiple times.

OpenAI Five (Dota 2): ~10,000 years of in-game experience (in wall-clock, about 10 months of distributed training across thousands of GPUs).

Why the gap? It's not that RL algorithms are stupid. It's that the RL problem is fundamentally harder than supervised learning, for five specific, identifiable reasons.

### Root Cause 1: Credit Assignment

In supervised learning, every input has a label. The loss function tells you exactly how wrong you were, for each sample, immediately. In RL, a reward at time T gives you no information about which of the T preceding actions was responsible. You must infer causality from correlation, across hundreds of steps, through stochastic transitions. This is vastly harder than supervised regression.

### Root Cause 2: Exploration

In supervised learning, the training set is given to you. You don't choose which images to label. In RL, you must discover useful data by acting in the environment. If you never explore the right region of the state space, you never observe the reward, and you never learn. The agent must balance gathering information (which has no immediate payoff) with exploiting what it knows (which does). This exploration cost has no analogue in supervised learning.

### Root Cause 3: Non-Stationarity

In supervised learning, the data distribution is fixed. The training set doesn't change because your model changed. In RL, the data distribution depends on your policy — a better policy visits different states than a worse one. Every time you improve the policy, the data distribution shifts. Old data becomes stale or misleading. On-policy methods must discard it entirely. Off-policy methods can reuse it but risk distribution mismatch.

### Root Cause 4: High Variance

The return G_t = Σ γ^k r_{t+k} is a sum of many stochastic terms (stochastic actions, stochastic transitions, stochastic rewards) over a long horizon. Its variance grows with the horizon length and the stochasticity of the environment. In supervised learning, the "target" for each input is a fixed label (or a sample from a relatively low-variance distribution). In RL, your "target" is a noisy estimate of a long-horizon sum. You need many more samples to average out the noise.

### Root Cause 5: Sparse Rewards

Many interesting environments have reward only at rare, special events. A robot assembling furniture gets reward only when the piece snaps into place — after thousands of motor commands of zero feedback. Most of the data the agent collects is completely uninformative. In supervised learning, every sample contributes to learning. In RL with sparse rewards, the vast majority of experience is wasted.

### What Helps

| Approach | What It Addresses | Examples |
|---|---|---|
| **Model-based RL** | Exploration + credit assignment | Dreamer, MuZero, World Models |
| **Offline RL** | Exploration (eliminates it entirely) | CQL, IQL, Decision Transformer |
| **Transfer / pre-training** | Sample efficiency via prior knowledge | Pre-trained representations, skill libraries |
| **Sim-to-real** | Cost of exploration (simulation is cheap) | Domain randomization, system identification |
| **Reward shaping** | Sparse rewards → dense signal | Potential-based shaping, curiosity |
| **Hindsight relabeling** | Sparse rewards in goal-conditioned settings | HER |

### Model-Based RL: The Big Lever

The single biggest source of sample efficiency is learning a model of the world. If you can predict s_{t+1} = f(s_t, a_t) and r_t = g(s_t, a_t), you can generate unlimited imagined experience without touching the real environment. Plan ahead in your head, evaluate hypothetical strategies, and only act in the real world when you have a good plan.

MuZero (DeepMind, 2020) learns a model in a learned latent space — it doesn't try to predict pixel-perfect next frames, just the aspects of the future relevant for planning. It achieves superhuman performance on Atari using ~20x less data than model-free methods.

The risk: model errors compound. If your model is wrong, imagined trajectories diverge from reality, and you learn the wrong policy. The art of model-based RL is knowing when to trust the model and when to fall back on real data.

### Offline RL: Skip Exploration Entirely

What if you have a big dataset of transitions collected by some other agent (or a human), and you just want to learn the best policy from that fixed dataset — no environment interaction at all? This is **offline RL** (or batch RL).

It eliminates exploration entirely, turning RL into something closer to supervised learning. But it introduces a new problem: **distributional shift**. Your dataset covers some states and actions; your learned policy might want to take actions in states that are poorly represented in the data. Standard RL algorithms fail spectacularly here — they overestimate Q-values for unseen actions and produce terrible policies.

Conservative Q-Learning (CQL) fixes this by adding a penalty that pushes down Q-values for actions not in the dataset. Decision Transformer reframes the problem as sequence modeling — conditioning on desired returns and generating actions autoregressively, sidestepping Bellman equations entirely.

### The Fundamental Issue

RL is hard because it combines two problems that are individually manageable but jointly explosive: **exploration** (finding the right data) and **optimization** (learning from the data you found). Supervised learning only has the second problem. Model-based RL, offline RL, and sim-to-real all work by decoupling or eliminating the exploration problem, leaving only optimization — which we know how to do.

> **Key takeaway:** RL's sample inefficiency isn't a bug to be fixed — it's an inherent consequence of needing to simultaneously explore an unknown environment, assign credit across long horizons, cope with non-stationary data distributions, and learn from high-variance, sparse signals. Every technique that improves sample efficiency does so by eliminating or reducing one of these five root causes. An interviewer asking this question wants to hear the specific causes, not just "RL needs lots of data."

---

## Q&A

**Question:** Potential-based reward shaping requires Φ(s) — a function of state only. Can you use shaping that depends on the action, like F(s, a, s')? Why or why not?

**Student's Answer:** The original Ng et al. theorem is for state-only potentials, and F(s, s') = γΦ(s') - Φ(s). If you make F depend on the action, the telescoping argument still works for the transition terms, but now different actions in the same state get different shaping bonuses. That changes the relative ranking of actions, which can change the optimal policy. There's been follow-up work (Wiewiora et al.) showing you can do potential-based shaping over state-action pairs — Φ(s, a) — with a modified form, but the conditions are more restrictive. For interview purposes, the safe answer is: stick to state-only potentials, because that's the result with the clean sufficiency-and-necessity proof.

**Evaluation:** The student correctly identified why action-dependent shaping is dangerous (it changes relative action rankings within a state) and even referenced the extended result by Wiewiora. The pragmatic "for interview purposes" caveat is exactly right — know the clean result cold, mention the extension if probed. Strong answer.

**Question:** An interviewer asks: "If model-based RL is so much more sample efficient, why isn't everyone using it?" What's the catch?

**Student's Answer:** Model errors compound over the planning horizon. If your learned model is 95% accurate per step, after 20 steps of imagined rollout you're at 0.95^20 ≈ 36% accuracy — your "imagined" trajectory has diverged from reality. The agent optimizes for the model's quirks rather than the real environment. This is called "model exploitation" — it's the model-based analogue of reward hacking. You can mitigate it by using short planning horizons, ensembles of models (to estimate uncertainty), or learned latent-space models (MuZero) that only model decision-relevant features. But fundamentally, you're adding another learning problem (learn the model) on top of the original one (learn the policy), and if the model is wrong in systematic ways, the policy will be wrong in systematic ways. Model-free methods are dumber but more robust — they can't exploit a model they don't have.

**Evaluation:** Excellent. The compounding error calculation (0.95^20 ≈ 0.36) is a concrete, quantitative way to convey the issue — exactly the kind of engineering-grounded reasoning that impresses interviewers. The "model exploitation" framing and the connection to reward hacking show deep understanding. The closing line — "they can't exploit a model they don't have" — is the kind of insight Feynman would appreciate.
