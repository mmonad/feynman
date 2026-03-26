# Lesson 3: Actor-Critic and Credit Assignment

*Course 15: Reinforcement Learning*

## Core Question

Policy gradients and value-based methods each have crippling weaknesses. How does the actor-critic architecture combine them into something that actually works, and how do you figure out which of your 1,000 actions was the one that mattered?

---

## Q55: Actor-Critic — Best of Both Worlds

### The Problem with Pure Approaches

**REINFORCE (pure policy gradient):** You parameterize a policy π_θ(a|s), sample full episodes, and update using the policy gradient theorem:

```
∇J(θ) = E[ Σ_t ∇_θ log π_θ(a_t|s_t) · G_t ]
```

where G_t = Σ_{k=0}^{∞} γ^k r_{t+k} is the return from timestep t onward.

This works. It's unbiased. But G_t has **enormous variance**. The return for a single episode is one noisy sample from a distribution that depends on every stochastic action and transition for the rest of the episode. You need thousands of episodes to get a gradient estimate that points anywhere useful. It's like estimating the mean of a distribution by drawing one sample at a time.

**DQN (pure value-based):** You learn Q(s,a) and act greedily. Great for discrete actions. But what if your action space is continuous — say, joint torques for a robot? You'd need to solve `argmax_a Q(s,a)` over a continuous space at every step. That's an optimization problem inside your optimization problem. Not practical.

### The Actor-Critic Architecture

Combine both. Maintain two networks:

- **Actor** π_θ(a|s): the policy. Outputs a distribution over actions. This is what actually controls the agent.
- **Critic** V_φ(s): the value function. Evaluates how good the current state is. This is the scorekeeper.

The actor decides what to do. The critic tells the actor how it's doing. The actor adjusts. The critic refines its evaluation. They co-evolve.

### The Advantage Function

Here's the key insight that makes this work. In the policy gradient, you're weighting the log-probability of each action by some measure of how good it was. Using the raw return G_t is high variance. What if instead you used **how much better this action was compared to average**?

```
A^π(s, a) = Q^π(s, a) - V^π(s)
```

This is the **advantage function**. Q(s,a) is how good it is to take action a in state s. V(s) is how good state s is on average (across all actions). The difference tells you: did this specific action contribute positively or negatively compared to the baseline expectation?

If A > 0: the action was better than average — increase its probability.
If A < 0: the action was worse than average — decrease its probability.

Subtracting V(s) as a **baseline** doesn't change the expected gradient (it's a constant with respect to the action), but it dramatically reduces variance. Instead of asking "was the return high?" (which depends on everything that happened after), you're asking "was the return higher than expected?" — a much less noisy signal.

### TD Error as Advantage Estimate

You don't actually need to learn both Q and V separately. There's a beautiful shortcut. The one-step TD error is:

```
δ_t = r_t + γ V_φ(s_{t+1}) - V_φ(s_t)
```

In expectation, δ_t equals the advantage A(s_t, a_t). Here's why: E[r_t + γ V(s_{t+1}) | s_t, a_t] = Q(s_t, a_t), so E[δ_t | s_t, a_t] = Q(s_t, a_t) - V(s_t) = A(s_t, a_t).

So the TD error — which you compute from V alone — is an unbiased estimate of the advantage. You only need one network (the critic V_φ) to get both the baseline and the advantage signal. The policy gradient becomes:

```
∇J(θ) ≈ E[ ∇_θ log π_θ(a_t|s_t) · δ_t ]
```

This is the core actor-critic update. The critic provides δ_t. The actor uses it to adjust the policy. Each step requires only one transition, not a full episode.

### Modern Instantiations

The basic idea above has been refined into several major algorithms:

| Algorithm | Key Innovation |
|---|---|
| **A2C** (Advantage Actor-Critic) | Synchronous parallel workers, advantage baseline |
| **A3C** | Asynchronous workers updating shared parameters |
| **PPO** (Proximal Policy Optimization) | Clipped surrogate objective to prevent destructively large updates |
| **SAC** (Soft Actor-Critic) | Entropy bonus in objective — explores automatically, off-policy |

PPO is the workhorse of modern on-policy RL (used for RLHF in language models). SAC is the workhorse of off-policy continuous-control RL. Both are actor-critic methods at their core.

> **Key takeaway:** The actor-critic architecture is the engine beneath almost every modern RL system. The critic reduces variance by providing a baseline; the actor handles continuous actions naturally. The TD error is the bridge — it's a one-step advantage estimate that lets you update every transition instead of waiting for full episodes.

---

## Q56: The Credit Assignment Problem

### The Core Difficulty

You're playing chess. After 80 moves, you win. Which move was responsible? Move 23, where you took control of the center? Move 57, where you sacrificed a bishop for a devastating attack? Move 79, where you set up the checkmate? Or was it move 12, where you chose an opening that your opponent was unprepared for?

This is **temporal credit assignment**: a reward arrives at time T, but you need to figure out which actions at t = 1, 2, ..., T were responsible. RL must do this automatically — and it's genuinely hard.

There's also **structural credit assignment**: within your model, which parameters or components contributed to the good (or bad) outcome? Backpropagation handles this for differentiable objectives, but RL rewards are external signals, not gradients through the computational graph.

### Why It's Hard

1. **Sparse rewards:** In many environments, you get reward only at the end. A game of Go gives +1 for winning and -1 for losing — after 200+ moves of zero feedback.
2. **Delayed causality:** The action that mattered might have happened hundreds of steps ago, with no signal in between.
3. **Stochasticity:** Even if action a was optimal, the noisy outcome might suggest otherwise. You need many samples to separate signal from noise.

### Solution 1: TD Learning — One Step at a Time

TD(0) doesn't wait for the final reward. It updates the value function after every single step:

```
V(s_t) ← V(s_t) + α [ r_t + γ V(s_{t+1}) - V(s_t) ]
```

This propagates credit backwards one step at a time. After the agent reaches the goal and gets reward, V(s_{T-1}) increases. Next time the agent visits s_{T-2} and transitions to s_{T-1}, V(s_{T-2}) increases. And so on. Credit diffuses backward through the state space over many episodes, like heat conduction.

The problem: it's slow. Credit from a reward at step T takes T episodes to reach step 0.

### Solution 2: Eligibility Traces — Bridging TD and Monte Carlo

What if, instead of crediting only the immediately preceding state, you keep a running trace of all recently visited states and credit them all?

The **eligibility trace** e_t(s) is a decaying record of which states were recently visited:

```
e_t(s) = γλ · e_{t-1}(s) + 1_{s = s_t}
```

When you observe TD error δ_t, you update all states proportionally to their trace:

```
V(s) ← V(s) + α · δ_t · e_t(s)    for all s
```

The parameter λ ∈ [0, 1] controls the bridge:
- λ = 0: only the current state is updated — this is TD(0)
- λ = 1: all visited states get equal (decayed) credit — equivalent to Monte Carlo
- 0 < λ < 1: a smooth interpolation

### Solution 3: The λ-Return

Equivalently, you can think of eligibility traces in the "forward view." Define the n-step return:

```
G_t^(n) = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})
```

The **λ-return** is a geometric average of all n-step returns:

```
G_t^λ = (1-λ) Σ_{n=1}^{∞} λ^{n-1} G_t^(n)
```

Short returns (small n) have low variance but high bias (they bootstrap heavily). Long returns (large n) reduce bias but increase variance. The λ-return automatically blends them, weighting shorter returns more heavily. This is the theoretical equivalence behind eligibility traces — the backward view (traces) and forward view (λ-return) give identical updates.

### Solution 4: Hindsight Experience Replay (HER)

A completely different approach for goal-conditioned tasks. Suppose your agent tries to reach goal G but ends up at state s_final instead. Standard RL says: "failure, no reward, learn nothing useful."

HER says: "you failed to reach G, but you did successfully reach s_final. Relabel the trajectory with goal = s_final, and store it as a success." Now you've turned a failure into training data for a related skill. The agent learns how to reach many states, and this knowledge transfers to reaching the actual goal.

This doesn't solve credit assignment in the classical sense — it sidesteps the sparse reward problem entirely by manufacturing dense reward signals from failed trajectories.

> **Key takeaway:** Credit assignment is the hardest unsolved problem in RL. TD learning propagates credit slowly but reliably. Eligibility traces (λ-return) speed it up by crediting recent states proportionally. HER reframes the problem entirely. In practice, most deep RL systems rely on some combination of short-horizon bootstrapping (TD) and reward shaping to avoid ever needing to assign credit across very long horizons.

---

## Q&A

**Question:** PPO is used for RLHF in language models. In that setting, what plays the role of the actor and what plays the role of the critic? And is the "episode" one token or one full response?

**Student's Answer:** "The actor is the language model itself — its policy π_θ(token | context) generates tokens. The critic is a separate value head (usually a linear layer on top of the LM's hidden states) that estimates V(s) where the 'state' is the prompt plus tokens generated so far. The 'episode' is generating one full response. The reward comes at the end from the reward model (trained on human preferences). The advantage is computed per-token using GAE (Generalized Advantage Estimation, which is the practical version of λ-returns). So each token gets a credit assignment signal, even though the reward is for the full response."

**Evaluation:** Exactly right, and impressively precise. The student correctly identified that the "state" is the autoregressive context, the "action" is a single token, and the reward is response-level but credit is assigned per-token via GAE. The mention of GAE is a nice touch — it's exactly the practical implementation of the λ-return/eligibility trace idea from this lesson.

**Question:** TD(0) propagates credit one step per episode. If your environment has episodes of length 100 and the reward only comes at the terminal state, how many episodes does it take for the initial state s_0 to have a non-trivial value estimate?

**Student's Answer:** "About 100 episodes. In the first episode, only s_99's value gets updated (from the terminal reward). In the second episode, s_98 picks up value from s_99's updated estimate. Each episode pushes the 'informed frontier' back one step. After 100 episodes, the information has diffused all the way back to s_0. That's exactly why TD(0) is slow for sparse, delayed rewards — and why eligibility traces or Monte Carlo returns help."

**Evaluation:** Correct reasoning. The student identified the key bottleneck: TD(0) is a one-step backup, so information about a reward at depth D takes D episodes to reach the start state. In practice it might take more than exactly D episodes due to stochasticity, but the order of magnitude is right. This is the fundamental motivation for multi-step methods and eligibility traces.
