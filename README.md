# transformer-exercises-jax

Transformer implementations in JAX, written as exercises:

- `2026/`
  - ~March 2026
  - A transformer in vanilla JAX: [2026/src/model.py](2026/src/model.py).
  - Implements:
    - Llama-3 architecture.
    - KV caching, interactive inference script w/ weight loading from HuggingFace.
      - `uv run inference.py`
    - Greedy, top-k, top-p/nucleus sampling.
    - Megatron-style tensor parallelism.
    - Grokking reproduction based on [Power et al. (2022)](https://arxiv.org/abs/2201.02177).
      - `uv run train_grokking.py`
- `mingpt/`
  - ~September 2021
  - [`karpathy/minGPT`](https://github.com/karpathy/minGPT) adapted to Flax: [mingpt/model.py](mingpt/model.py).
  - Implements:
    - GPT-2 style architecture.
    - Data parallelism.
    - Chunked self-attention based on [Rabe and Staats (2021)](https://arxiv.org/pdf/2112.05682v2.pdf).
