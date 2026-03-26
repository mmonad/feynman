# Lesson 1: Distributed Training

*Course 18: Systems, Robustness & the Frontier*

## Core Question

You've got a model with 70 billion parameters and a trillion tokens of training data. One GPU has maybe 80 GB of memory and can churn through a few hundred tokens per second. At that rate, training would take about 30 years. So you throw a thousand GPUs at it. Problem solved?

Not even close. Because the moment you distribute training across machines, a new class of engineering nightmares appears — problems that have nothing to do with ML theory and everything to do with physics: the speed of light, the bandwidth of interconnects, and the fact that one of your thousand machines will inevitably crash at the worst possible time.

---

## Q81: The Communication Problem

### AllReduce — The Bottleneck Nobody Talks About

In the simplest distributed training setup — data parallelism — every GPU has a complete copy of the model. Each processes a different mini-batch, computes gradients, and then all GPUs must **agree on the average gradient** before taking a step.

That "agree" step is called **AllReduce**, and it's where the physics hurts.

```
AllReduce operation:
  Input:  N GPUs, each with gradient vector g_i (size = num_params)
  Output: Every GPU gets g_avg = (1/N) Σ g_i

Communication volume per GPU: 2 × (N-1)/N × |params| × bytes_per_param
For a 70B model in FP16: ~2 × 140 GB ≈ 280 GB per step
```

With NVLink at 600 GB/s between GPUs in one node, that's about half a second. Across nodes over InfiniBand at 200 GB/s, it's over a second. The forward + backward pass might take 2 seconds. So you're spending a *third* of your time just talking, not computing.

The ring AllReduce algorithm helps — instead of everyone sending to everyone, gradients flow around a ring, and each GPU only sends to one neighbor. But the fundamental constraint remains: you must move a model's worth of data around the network every single step.

### Synchronous vs Asynchronous SGD

Here's the fork in the road.

**Synchronous SGD** — Every GPU computes its gradient, performs AllReduce, and only then takes a step. Everyone waits for everyone else.

The problem: **stragglers**. If GPU #847 is 10% slower (thermal throttling, network congestion, unlucky data batch), all 999 other GPUs sit idle. Your training speed is determined by the *slowest* machine at every step.

**Asynchronous SGD** — Each GPU computes its gradient and immediately sends it to a parameter server, which applies updates as they arrive. No waiting.

The problem: **stale gradients**. GPU #1 computed its gradient using parameters from step t, but by the time the gradient arrives at the parameter server, the parameters have already been updated to step t+5 by other workers. That gradient is *stale* — it's pushing the model based on where it *was*, not where it *is*.

```
Staleness τ: number of steps between reading params and applying gradient

Effect: async SGD effectively optimizes a DIFFERENT loss:
  ∇L(θ_{t-τ}) instead of ∇L(θ_t)

When learning rate η and staleness τ are both small: works fine.
When η × τ is large: divergence.
```

| Property | Sync SGD | Async SGD |
|---|---|---|
| Gradient quality | Exact | Stale by τ steps |
| Speed per step | Bounded by slowest worker | Bounded by fastest worker |
| Scaling | Sub-linear (straggler penalty) | Near-linear (but may not converge) |
| Used in practice | Almost always (with tricks) | Rarely for LLMs |

> The industry converged on synchronous SGD with straggler mitigation (backup workers, gradient compression) rather than async. Stale gradients are too dangerous when you're spending $10M on a training run.

### Large Batch Training — The Scaling Rule

If you have 1000 GPUs, you can process 1000× the batch size. But large batches create their own problems. Remember the SDE approximation from our optimization lessons — SGD's noise is inversely proportional to batch size. Kill the noise, kill the implicit regularization that finds flat minima.

The **linear scaling rule**: when you multiply batch size by k, multiply learning rate by k. This preserves the noise-to-signal ratio of the gradient.

```
Standard:     batch_size = B,   learning_rate = η
Scaled:       batch_size = kB,  learning_rate = kη

But this breaks at large k — too-large LR causes divergence.
Fix: LARS/LAMB — per-layer adaptive learning rates:

LARS:  η_layer = η × ||w|| / ||∇w||   (trust ratio)
LAMB:  Same idea but with Adam-style momentum
```

LARS and LAMB let you scale to batch sizes of 32K–64K. Beyond that, generalization degrades no matter what you do. There's a fundamental limit to how much parallelism you can extract from data parallelism alone.

---

## Q82: The Parallelism Taxonomy

### Data Parallel — The Simple Case

Each GPU holds the *complete* model. You split the data. After each forward-backward pass, AllReduce the gradients.

Simple, but has a fatal flaw: every GPU must hold the full model, the full gradients, and the full optimizer state in memory. For a 70B model in FP16, that's roughly:

```
Parameters:        70B × 2 bytes  = 140 GB
Gradients:         70B × 2 bytes  = 140 GB
Adam optimizer:    70B × 8 bytes  = 560 GB  (m, v in FP32 + master weights)
Total per GPU:     ~840 GB
```

No GPU has 840 GB. This is where **ZeRO** (Zero Redundancy Optimizer) comes in.

ZeRO-1: Shard the optimizer states across GPUs. Each GPU stores 1/N of Adam's m and v.
ZeRO-2: Also shard the gradients.
ZeRO-3: Also shard the parameters themselves. Each GPU holds 1/N of everything.

```
ZeRO-3 memory per GPU (N=8):
  Params:     140/8 = 17.5 GB
  Gradients:  140/8 = 17.5 GB
  Optimizer:  560/8 = 70 GB
  Total:      ~105 GB  ← fits on one A100!
```

The tradeoff: ZeRO-3 requires all-gather operations before every forward pass to reconstruct parameters, adding communication. But it's the only way to fit the model.

### Tensor Parallel — Splitting Layers

Instead of replicating the model, *split individual layers across GPUs*. A single matrix multiplication `Y = XW` can be partitioned:

```
GPU 0: Y_0 = X · W_0   (first half of columns)
GPU 1: Y_1 = X · W_1   (second half of columns)
Y = [Y_0 | Y_1]        (concatenate)
```

This works beautifully for the big matrix multiplications in transformers (attention projections, FFN layers). Each GPU does half the compute and holds half the parameters. But you need communication *within* each layer — after every partitioned matmul, GPUs must exchange partial results.

This means tensor parallelism only works within a node where interconnect bandwidth is massive (NVLink). Across nodes, it's too slow.

### Pipeline Parallel — Splitting Across Layers

Assign different layers to different GPUs:

```
GPU 0: Layers 0-7
GPU 1: Layers 8-15
GPU 2: Layers 16-23
GPU 3: Layers 24-31
```

Simple, but creates the **pipeline bubble**. GPU 3 can't start until GPU 0's output reaches it through the chain. During the forward pass, downstream GPUs sit idle. During backward, upstream GPUs sit idle.

**Micro-batching** reduces the bubble. Split each mini-batch into m micro-batches and pipeline them:

```
Time →
GPU 0: |F1|F2|F3|F4|  idle  |B4|B3|B2|B1|
GPU 1:    |F1|F2|F3|F4|  idle  |B4|B3|B2|B1|
GPU 2:       |F1|F2|F3|F4|  idle  |B4|B3|B2|B1|
GPU 3:          |F1|F2|F3|F4|  idle  |B4|B3|B2|B1|

Bubble fraction ≈ (p-1) / (m+p-1)
  p = pipeline stages, m = micro-batches
  m=4, p=4: bubble = 43%  (terrible)
  m=32, p=4: bubble = 8.6% (acceptable)
```

### 3D Parallelism — The Full Stack

In practice, large-scale training uses all three simultaneously:

```
Tensor parallel:    within a node (8 GPUs, NVLink)
Pipeline parallel:  across nodes (4-8 stages)
Data parallel:      across groups of nodes (ZeRO)

Example — 256 GPUs:
  8 GPUs per tensor-parallel group
  4 pipeline stages
  8 data-parallel replicas
  8 × 4 × 8 = 256 GPUs
```

Each dimension maps to a different level of the network topology. Tensor parallel lives on the fastest links. Pipeline parallel spans across nodes. Data parallel spans across racks.

> The art of distributed training isn't any single technique — it's mapping the parallelism strategy to the physical topology of your cluster. The communication pattern must respect the memory hierarchy: SRAM → HBM → NVLink → InfiniBand → Ethernet, each step 10-100× slower than the last.

---

## Q&A

**Question:** You're training a 70B parameter model on a cluster of 512 A100 GPUs (64 nodes of 8 GPUs each). Design a parallelism strategy. How would you assign tensor, pipeline, and data parallel dimensions? What's the approximate bubble overhead, and how would you mitigate it?

**Student's Answer:** Within each node, 8 GPUs connected by NVLink — that's the tensor parallel dimension. For pipeline parallel, 4 stages feels right — splitting 80 transformer layers into groups of 20 across 4 nodes. That gives me 64 / 4 = 16 nodes per pipeline replica, and with 8 GPUs per node already used for TP, each data-parallel group is one pipeline replica. So I have 512 / (8 × 4) = 16 data-parallel replicas. For the bubble, with 4 pipeline stages, I need enough micro-batches — say 32 — giving bubble fraction of 3/(32+3) ≈ 8.6%. To mitigate further, I'd use interleaved pipeline scheduling (1F1B) so forward and backward micro-batches overlap, reducing peak memory and shrinking the bubble. I'd also overlap gradient AllReduce with the backward pass of the last micro-batches.

**Evaluation:** Excellent design. The topology mapping is exactly right — TP on the fast links, PP across nodes, DP across the remaining axis. The 1F1B (one-forward-one-backward) schedule is the key optimization you identified: instead of running all forwards then all backwards, interleave them so that as soon as GPU 3 finishes forward on micro-batch 1, GPU 0 can start backward on micro-batch 1 while simultaneously running forward on micro-batch 5. This halves peak activation memory and tightens the bubble. Overlapping AllReduce with backward computation is also standard practice — you can start reducing early layers' gradients while later layers are still computing backward.
