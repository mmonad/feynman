# Lesson 1: Bellman Equations and Dynamic Programming

*Course 15: Reinforcement Learning*

## Core Question

What is the single recursive equation that underpins every RL algorithm ever invented, and how do you actually solve it when you know the rules of the game?

---

## Q51: The Bellman Equation — Where Everything Starts

Let me tell you what reinforcement learning really is. You've got an agent sitting in some state, picking actions, receiving rewards, and landing in new states. The question is: **what's the best thing to do?** To answer that, you first need to answer a simpler question: **how good is it to be where I am?**

### The Value Function

The value of a state is the total future reward you expect to collect, starting from that state and following some policy forever. But future rewards are worth less than immediate ones — you discount by a factor γ ∈ (0, 1) per timestep:

```
V^π(s) = E[ Σ_{t=0}^{∞} γ^t · r_t | s_0 = s, policy π ]
```

That's a sum over an infinite horizon. Looks intimidating. But here's the trick — and this is Bellman's key insight — you can split that infinite sum into "right now" and "everything after right now."

### The Recursive Insight

Take that infinite sum and peel off the first reward:

```
V^π(s) = E[ r_0 + γ · r_1 + γ² · r_2 + ... | s_0 = s ]
        = E[ r_0 + γ · (r_1 + γ · r_2 + ...) | s_0 = s ]
        = E[ r_0 + γ · V^π(s_1) | s_0 = s ]
```

That's it. The value of where you are equals the immediate reward plus the discounted value of where you end up. This is the **Bellman equation**, and it converts an infinite-horizon problem into a one-step relationship.

### Making It Concrete

When the policy π is stochastic (takes action a in state s with probability π(a|s)), and the environment has known transition probabilities P(s'|s,a) and reward function R(s,a), you can expand the expectation fully:

```
V^π(s) = Σ_a π(a|s) [ R(s,a) + γ Σ_{s'} P(s'|s,a) · V^π(s') ]
```

Read that carefully. It says: sum over all actions you might take (weighted by your policy), and for each action, add the immediate reward to the discounted average value of wherever you might land. This is a **system of linear equations** — one per state. If you have |S| states, you have |S| equations and |S| unknowns. You can solve it directly.

### The Optimality Equation

Now the real question: what's the **best** policy? The optimal value function satisfies:

```
V*(s) = max_a [ R(s,a) + γ Σ_{s'} P(s'|s,a) · V*(s') ]
```

Instead of averaging over actions according to some policy, you just take the **best** action. This is the Bellman optimality equation. It's no longer linear — that `max` makes it nonlinear. But it's still solvable.

### The Q-Function

There's a variant that's even more useful in practice. Instead of asking "how good is this state?", ask "how good is taking action a in state s?":

```
Q*(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) · max_{a'} Q*(s', a')
```

Why is this better? Because once you have Q*, the optimal policy is trivial: just pick `argmax_a Q*(s,a)`. You don't need to do a one-step lookahead — the action evaluation is baked right in. This is why Q-learning works with Q-functions, not V-functions.

> **Key takeaway:** The Bellman equation is the fundamental identity of RL. Every algorithm — from tabular dynamic programming to deep Q-networks to policy gradient — is either solving it exactly, approximating it, or working around the fact that you can't solve it.

---

## Q52: Policy Iteration vs Value Iteration

Both methods solve the Bellman optimality equation when you know the full model (transitions P and rewards R). They're "planning" algorithms, not "learning" algorithms — the distinction matters, and interviewers love to probe it.

### Policy Iteration

The idea: alternate between two steps until nothing changes.

**Step 1 — Policy Evaluation:** Given a fixed policy π, compute V^π by solving the linear system:

```
V^π(s) = Σ_a π(a|s) [ R(s,a) + γ Σ_{s'} P(s'|s,a) · V^π(s') ]
```

This is |S| linear equations in |S| unknowns. You can solve it with matrix inversion (cost O(|S|³)) or iteratively (keep applying the equation until convergence).

**Step 2 — Policy Improvement:** Given V^π, construct a better policy:

```
π'(s) = argmax_a [ R(s,a) + γ Σ_{s'} P(s'|s,a) · V^π(s') ]
```

For every state, just pick the action that looks best according to the current value function. This is a one-step greedy improvement. The **policy improvement theorem** guarantees that π' is at least as good as π, and strictly better unless π was already optimal.

**Step 3:** Set π ← π' and go back to Step 1.

**Why it converges:** There are finitely many deterministic policies (at most |A|^|S|). Each iteration produces a strictly better one. You can't cycle. So it must terminate at the optimal policy. In practice, it converges in surprisingly few iterations — often less than 10, even for large state spaces. The expensive part is Step 1 (solving the linear system each time).

### Value Iteration

The idea: skip the explicit policy entirely. Just repeatedly apply the Bellman optimality operator:

```
V_{k+1}(s) = max_a [ R(s,a) + γ Σ_{s'} P(s'|s,a) · V_k(s') ]
```

Start with any V_0 (say, all zeros). Keep iterating. Once V converges, extract the policy at the end:

```
π*(s) = argmax_a [ R(s,a) + γ Σ_{s'} P(s'|s,a) · V*(s') ]
```

**Why it converges:** The Bellman optimality operator is a **contraction mapping** with factor γ. By the Banach fixed-point theorem, repeated application converges to the unique fixed point V*. The convergence is asymptotic — you never quite get there in finite steps, but you can get ε-close in O(log(1/ε) / log(1/γ)) iterations.

### Head-to-Head

| Aspect | Policy Iteration | Value Iteration |
|---|---|---|
| **Per-iteration cost** | High (solve linear system O(\|S\|³) or iterative eval) | Low (one Bellman backup per state) |
| **Number of iterations** | Few (finite, typically < 10) | Many (asymptotic convergence) |
| **Convergence type** | Exact in finite steps | Asymptotic (ε-approximate) |
| **What it maintains** | Explicit policy + value function | Value function only |
| **When preferred** | Small-medium state spaces, need exact solution | Large state spaces, when per-iteration cost matters |

### The Critical Observation

Both methods require the full model — P(s'|s,a) and R(s,a) for all states, actions, and successor states. In the real world, you almost never have that. Your robot doesn't know the physics equations. Your game agent doesn't have access to the opponent's strategy. This is where "planning" ends and "learning" begins — and why we need algorithms like Q-learning that work from experience alone.

> **Key takeaway:** Policy iteration is Newton's method (few expensive steps), value iteration is gradient descent (many cheap steps). Both solve the same equation. But neither works when you don't have the model — which is the whole reason the rest of RL exists.

---

## Q&A

**Question:** The Bellman equation for V^π is linear, so you can solve it with matrix inversion. The Bellman optimality equation for V* is nonlinear because of the max. But in policy iteration, you alternate between solving a linear system and doing a max. So is policy iteration solving a nonlinear problem by decomposing it into linear subproblems?

**Student's Answer:** "Yes — policy evaluation is the linear part (fix the policy, solve for V), and policy improvement is the nonlinear part (take the max to get a better policy). You alternate between the two. It's like coordinate descent in optimization — you fix one variable, optimize the other, then switch. Each step is easy, and the iteration converges."

**Evaluation:** Exactly right, and the coordinate-descent analogy is excellent. Policy iteration does decompose the nonlinear Bellman optimality equation into alternating linear solves and greedy maximizations. The analogy to coordinate descent is apt — though it's even better than coordinate descent because each step is guaranteed to weakly improve the objective (the policy's value), and convergence happens in finitely many steps rather than asymptotically.

**Question:** An interviewer asks: "Can you use policy iteration without solving the linear system exactly in the evaluation step?" What do you say?

**Student's Answer:** "Yes — you can do a few iterations of the Bellman evaluation update instead of solving exactly. That's called 'modified policy iteration' or 'truncated policy iteration.' You're doing k sweeps of the evaluation update instead of solving to convergence. In the extreme where k = 1, you get something that looks a lot like value iteration. So policy iteration and value iteration are really two ends of a spectrum."

**Evaluation:** Perfect. This is exactly the connection interviewers want to hear. Policy iteration with k = ∞ evaluation steps per improvement = standard policy iteration. With k = 1 = value iteration. Everything in between = modified policy iteration. Shows you understand these aren't separate algorithms but a family parameterized by how hard you work on evaluation before improving.
