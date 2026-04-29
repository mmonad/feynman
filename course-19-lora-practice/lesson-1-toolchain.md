# Lesson 1: The Toolchain — Models, Libraries, and Hardware (March 2026)

*Course 19: LoRA Training in Practice*

## Base Model Landscape (March 2026)

| Family | Latest | License | Sweet Spot |
|--------|--------|---------|------------|
| Qwen 3.5 (Alibaba) | 397B-A17B MoE down to 0.8B | Apache 2.0 | Best overall open model, strong coding |
| DeepSeek | Various | MIT | Zero restrictions |
| LLaMA (Meta) | Latest family | Meta license | Most tooling support |
| Mistral | 3B, 8B+ | Apache 2.0 | Fast inference |

**Student's choice:** Qwen3.5-9B (dev), Qwen3.5-27B (prod)

## Training Stack (Three Tiers)

### Tier 1: Unsloth (Quick Start)
2x faster, 70% less VRAM. Supports 500+ models. QLoRA on 8B in ~10GB. Includes Studio UI, data recipes, RL training.

### Tier 2: Hugging Face PEFT + TRL (Production)
Standard ecosystem. New in 2026: EVA initialization (data-driven LoRA init via SVD), adapter hotswapping, rsLoRA (α/√r), Liger Kernels.

### Tier 3: Axolotl (Maximum Flexibility)
YAML-based config, exposes every parameter.

## Serving: vLLM Multi-LoRA
- Multiple adapters loaded simultaneously
- Per-request adapter selection via OpenAI-compatible API
- Dynamic loading/unloading at runtime
- Hugging Face Hub resolver for auto-download

## 2026 Best Practices Updates

1. **rsLoRA:** scaling = α/√r instead of α/r (more stable across ranks)
2. **EVA initialization:** data-driven LoRA init via SVD of activations
3. **Target ALL linear layers:** q/k/v/o_proj + gate/up/down_proj (consensus)

## Student's Hardware

RTX Pro 6000 Blackwell (96GB VRAM):
- Full 16-bit LoRA on 27B (no QLoRA needed)
- Serve 27B + 5 adapters simultaneously
- ~75GB total out of 96GB

## Setup: Five Specialist Adapters

```
vllm serve Qwen/Qwen3.5-27B --enable-lora \
  --lora-modules property=./adapters/property \
                 threat=./adapters/threat \
                 abstraction=./adapters/abstraction \
                 assumption=./adapters/assumption \
                 composition=./adapters/composition \
  --max-lora-rank 16 --max-loras 5
```

Sources:
- [Unsloth](https://unsloth.ai/)
- [HF PEFT LoRA Guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [vLLM LoRA Adapters](https://docs.vllm.ai/en/stable/features/lora/)
- [Qwen3.5](https://github.com/QwenLM/Qwen3.5)
