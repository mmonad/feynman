# Lesson 2: Memory, Precision & Latency

*Course 18: Systems, Robustness & the Frontier*

## Core Question

Here's a number that should alarm you: a 7-billion-parameter model in FP32, with Adam optimizer, needs about **112 GB** of memory just to exist during training — before a single activation is computed. Where does all that memory go? And once you start serving the model, a completely different set of bottlenecks appears. Let's trace the bytes.

---

## Q83: Memory Bottlenecks in Training

### The Four Memory Consumers

Every byte of GPU memory during training is consumed by one of four things:

```
1. Parameters:       P × bytes_per_param
2. Gradients:        P × bytes_per_grad
3. Optimizer state:  P × bytes_per_optim
4. Activations:      f(batch_size, seq_len, hidden_dim, num_layers)
```

Let's calculate for a 7B model with Adam in FP32:

```
Parameters:    7B × 4 bytes = 28 GB
Gradients:     7B × 4 bytes = 28 GB
Optimizer:     7B × 4 bytes (m) + 7B × 4 bytes (v) = 56 GB
                                              Total: 112 GB
```

Adam stores two running averages — first moment (m, the momentum) and second moment (v, the adaptive learning rate denominator). That's 2× the parameter count just for the optimizer. This is why Adam is memory-hungry, and why SGD with momentum (which only stores m) uses less.

### Activations — The Hidden Killer

The numbers above are *fixed* — they don't change with batch size. But activations scale with the data flowing through the model:

```
Activation memory ≈ 2 × batch × seq_len × hidden × num_layers × bytes

For a 7B model (hidden=4096, 32 layers), batch=8, seq=2048, FP16:
  ≈ 2 × 8 × 2048 × 4096 × 32 × 2 bytes
  ≈ ~34 GB
```

That "2×" is because you store activations from the forward pass so you can use them during the backward pass. Double the batch size, double the activation memory. Double the sequence length, double it again.

### Gradient Checkpointing

The insight: you don't need to store *all* activations. During backward, when you need an activation that wasn't stored, just **recompute it** on the fly by running the forward pass again for that layer.

```
Full storage:     Store all L layers' activations → O(L) memory
Checkpointing:    Store every √L layers → O(√L) memory
                  Cost: ~33% more compute (re-forward through each segment)
```

It's a time-memory tradeoff. You burn 33% more FLOPs but cut activation memory by a factor of √L. For a 32-layer model, that's √32 ≈ 5.7× memory reduction for activations.

### KV Cache — The Inference Memory Problem

During inference (autoregressive generation), the model caches the key and value tensors from every previous token at every layer. This is the **KV cache**, and it grows linearly with sequence length:

```
KV cache per token = 2 × num_layers × num_heads × head_dim × bytes

For LLaMA-70B (80 layers, 64 heads, dim 128, FP16):
  = 2 × 80 × 64 × 128 × 2 = 2.6 MB per token

For a 4K context:   ~10.5 GB
For a 128K context: ~335 GB
```

This is why long-context serving is brutally expensive. The model weights are fixed, but the KV cache scales with every user's conversation length.

> Memory during training is dominated by optimizer state (it's 4× the model size for Adam in FP32). Memory during inference is dominated by the KV cache. Different problems, different solutions.

---

## Q84: Mixed Precision Training

### The FP32 Tax

Full FP32 training stores every number as 32 bits. But neural networks are remarkably robust to numerical noise — the gradient is already a noisy estimate from a random mini-batch. Do you really need 8 decimal digits of precision for a stochastic quantity?

Mixed precision uses FP16 or BF16 (16-bit) for most computation, cutting memory in half and enabling **tensor cores** — specialized hardware that computes 16-bit matrix multiplications 2–8× faster than FP32.

```
FP32:  1 sign + 8 exponent + 23 mantissa   range: ±3.4×10³⁸   precision: ~7 decimal digits
FP16:  1 sign + 5 exponent + 10 mantissa   range: ±65,504      precision: ~3 decimal digits
BF16:  1 sign + 8 exponent +  7 mantissa   range: ±3.4×10³⁸   precision: ~2 decimal digits
```

### Why BF16 Won

FP16 has a narrow dynamic range — maximum value 65,504. Gradients, loss values, and activations can easily exceed this, causing overflow (→ infinity) or underflow (→ zero). That's why FP16 training requires **loss scaling**: multiply the loss by a large factor (e.g., 1024) before backward, then divide the gradients afterward. This shifts the gradient distribution into FP16's representable range.

BF16 has the same exponent range as FP32, so overflow/underflow almost never happens. You sacrifice precision (7 mantissa bits vs 10), but that's fine for stochastic gradients. No loss scaling needed.

| Format | Memory | Speed | Dynamic Range | Precision | Loss Scaling? |
|---|---|---|---|---|---|
| FP32 | 4 bytes | 1× | Huge | High | No |
| FP16 | 2 bytes | 2-8× | Tiny (65K max) | Medium | Yes |
| BF16 | 2 bytes | 2-8× | Huge (same as FP32) | Low | No |

### The Master Weight Trick

Even in mixed precision, you keep a **master copy** of the weights in FP32. The forward and backward passes use FP16/BF16, but the weight update — `w = w - lr * grad` — happens in FP32. Why? Because learning rates are tiny (1e-4 to 1e-5), and the update `lr * grad` might be smaller than the smallest representable difference between adjacent FP16 values. In FP16, adding a very small number to a large number can round to zero. FP32 preserves these tiny updates.

```
Training loop with mixed precision:
  1. Copy FP32 master weights → FP16
  2. Forward pass in FP16 (fast, less memory)
  3. Backward pass in FP16 (loss scaling if FP16)
  4. Update FP32 master weights using FP16 gradients
  5. Repeat
```

### Post-Training Quantization

For inference, you can go further. INT8 and INT4 quantization compress weights after training:

```
INT8:   1 byte per param.  7B model → ~7 GB   (vs 14 GB FP16)
INT4:   0.5 byte per param. 7B model → ~3.5 GB (fits on one consumer GPU!)
```

The key techniques:

**GPTQ** — Quantizes weights one layer at a time, using second-order information (Hessian) to minimize quantization error. Each weight is rounded to INT4, and the rounding error is compensated by adjusting subsequent weights. Think of it as: if I round this weight *up*, I should round the next correlated weight *down* to cancel the error.

**AWQ** (Activation-Aware Weight Quantization) — Observes that not all weights are equal. Some channels are activated frequently and carry important information. AWQ identifies these salient channels and scales them up before quantization, giving them more bits of effective precision.

> Quantization works because neural network weights are heavily redundant. Most of the information is in the *relative* magnitudes and the *structure* of the weight matrices, not in the precise numerical values. You're compressing a JPG, not a medical image.

---

## Q85: Latency vs Throughput — The Inference Game

### Two Metrics, One GPU

**Latency**: time from request to complete response (what the user feels).
**Throughput**: total tokens generated per second across all requests (what your cloud bill reflects).

These are in tension. A single request on an empty GPU gets the lowest latency — no competition for compute. But the GPU is massively underutilized. Batch multiple requests, and throughput goes up but each individual request waits longer.

### Static vs Dynamic Batching

**Static batching** — Wait until you have B requests, run them all at once. The problem: different requests have different lengths. The batch finishes when the *longest* sequence finishes. Short requests sit idle, padding wasted.

**Continuous (dynamic) batching** (vLLM, TGI) — As soon as one request in the batch finishes generating, slot a new request into its place. The batch is always full, no request waits unnecessarily.

```
Static:     [req1: 50 tokens][req2: 200 tokens][req3: 30 tokens]
            All wait for req2 to finish → 200 token-times wasted

Continuous: When req3 finishes at step 30, insert req4 immediately.
            When req1 finishes at step 50, insert req5.
            GPU always has work. Throughput: 2-3× improvement.
```

### Prefill vs Decode

Autoregressive inference has two distinct phases:

**Prefill** — Process the entire input prompt in parallel (like training). Compute-bound. All tokens attend to each other simultaneously. Fast.

**Decode** — Generate output tokens one at a time. Each step produces one token, which requires reading the entire KV cache. Memory-bandwidth-bound. Slow.

```
Prefill:  Process 1000 input tokens  → 1 forward pass, ~50ms
Decode:   Generate 200 output tokens → 200 forward passes, ~4000ms

The decode phase dominates wall-clock time!
```

This is the fundamental asymmetry of autoregressive models: reading is parallel, writing is sequential.

### Speculative Decoding

Use a small, fast **draft model** to generate k candidate tokens, then verify all k in one forward pass of the large model:

```
1. Draft model generates: [token1, token2, token3, token4, token5]  (fast, ~5ms)
2. Large model verifies all 5 in ONE forward pass                   (~50ms)
3. Accept first 3 that match, reject rest
4. Net: 3 tokens in 55ms instead of 3 × 50ms = 150ms

Speedup: ~2-3× on average (depends on acceptance rate)
```

The key insight: verification is parallel (one forward pass for k tokens) while generation is sequential (one forward pass per token). Speculative decoding converts sequential generation into parallel verification.

### KV Cache Optimization

**Paged Attention** (vLLM) — Instead of pre-allocating contiguous memory for each request's maximum possible KV cache, allocate in small pages (like virtual memory in an OS). This eliminates internal fragmentation — you don't waste memory on tokens that haven't been generated yet.

```
Naive: Pre-allocate 128K tokens × 2.6 MB/token = 335 GB per request
       (even if the response is only 100 tokens)

Paged: Allocate 4KB pages on demand. Only use memory for actual tokens.
       Memory utilization: ~95% vs ~30% for naive allocation.
```

**Grouped-Query Attention (GQA)** — Instead of separate K and V heads for each attention head, share K/V across groups of query heads. LLaMA-2 70B uses 8 KV heads shared across 64 query heads — an 8× reduction in KV cache size.

```
MHA:  64 query heads, 64 KV heads  → KV cache = 2.6 MB/token
GQA:  64 query heads,  8 KV heads  → KV cache = 0.33 MB/token
MQA:  64 query heads,  1 KV head   → KV cache = 0.04 MB/token (but quality drops)
```

> The entire inference optimization stack is about one thing: the decode phase is memory-bandwidth-bound, not compute-bound. Every trick — batching, speculative decoding, KV cache compression, GQA — either reduces how much memory you read per token or amortizes the cost across more useful work.

---

## Q&A

**Question:** A 7B model with FP16 weights is serving requests with an average context length of 4096 tokens. You have one A100 (80 GB HBM). Walk me through the memory budget. How many concurrent requests can you serve? What's the first bottleneck you'd hit, and how would you address it?

**Student's Answer:** Model weights in FP16: 7B × 2 bytes = 14 GB. That leaves 66 GB for the KV cache. For a 7B model — say 32 layers, 32 heads, head dim 128 — KV cache per token is 2 × 32 × 32 × 128 × 2 bytes = 0.5 MB/token. At 4096 tokens per request, each request needs about 2 GB of KV cache. So I can fit roughly 66/2 = 33 concurrent requests. The first bottleneck is memory, not compute — the A100 can do 312 TFLOPS but the model only needs ~14 GFLOPs per token, so compute is nearly idle during decode. I'd address the memory bottleneck with paged attention to avoid fragmentation, GQA if the model supports it, and INT8 quantization of the KV cache which halves the per-token cost and doubles my concurrent request capacity.

**Evaluation:** Spot-on analysis. The compute-vs-memory diagnosis is the key insight — during decode, you're doing a single matrix-vector multiply per layer, which barely scratches the A100's compute capacity, but you're reading the entire KV cache from HBM for every single token. The arithmetic intensity (FLOPs per byte read) is far below what the hardware is designed for. Your solution stack is correct: paged attention for utilization, GQA for architectural reduction, and INT8 KV cache quantization is increasingly standard — the keys and values are far more robust to quantization than the weights themselves, since attention is softmax-normalized anyway.
