from dataclasses import dataclass
from functools import partial
from typing import Literal, assert_never

import jax
from jax import Array
from jax import numpy as jnp

from .model import KvCache, TransformerConfig, TransformerParams, transformer_fwd


@dataclass(frozen=True)
class TopK:
    k: int


@dataclass(frozen=True)
class TopP:
    p: float


@jax.jit(static_argnames=("cfg", "strategy"), donate_argnames=("tokens", "kv_cache"))
def sample_one_token(
    tokens: Array,  # (b, max_seq_len) int32 token buffer
    pos: Array,  # (b,) int32, current position to process
    params: TransformerParams[Array],
    cfg: TransformerConfig,
    kv_cache: KvCache | None,
    key: Array,  # PRNG key, combined with pos via fold_in
    temperature: Array,  # () float33
    strategy: Literal["naive", "greedy"] | TopK | TopP = TopK(k=4),
) -> tuple[Array, KvCache]:
    """Forward one step, sample a token, and write it to tokens[pos + 1]."""
    assert len(tokens.shape) == 2
    assert tokens.shape[:1] == pos.shape
    if kv_cache is not None:
        logits, kv_cache = transformer_fwd(
            jax.vmap(partial(jax.lax.dynamic_index_in_dim, axis=0))(tokens, pos),
            params,
            cfg,
            kv_cache,
        )
    else:
        logits, kv_cache = transformer_fwd(tokens, params, cfg, kv_cache)
        logits = jax.vmap(partial(jax.lax.dynamic_index_in_dim, axis=0))(logits, pos)

    assert logits.shape == (tokens.shape[0], 1, cfg.vocab_size)
    logits = logits[:, 0, :] / temperature
    assert logits.shape == (tokens.shape[0], cfg.vocab_size)

    match strategy:
        case "naive":
            sampled_tokens = jax.random.categorical(key, logits, axis=-1)
        case "greedy":
            sampled_tokens = jnp.argmax(logits, axis=-1)
        case TopK(k=k):
            topk_logits, topk_indices = jax.lax.top_k(logits, k)
            sampled_tokens = jnp.take_along_axis(
                topk_indices,
                jax.random.categorical(key, logits=topk_logits, axis=-1)[:, None],
                axis=-1,
            )
        case TopP(p=p):  # top-k/nucleus sampling.
            sorted_indices = jnp.argsort(logits, axis=-1, descending=True)
            sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
            cdf = jnp.cumsum(
                jnp.roll(jax.nn.softmax(sorted_logits), 1, axis=-1).at[..., 0].set(0.0),
                axis=-1,
            )
            sorted_logits = jnp.where(cdf >= p, -jnp.inf, sorted_logits)
            sampled_tokens = jnp.take_along_axis(
                sorted_indices,
                jax.random.categorical(key, logits=sorted_logits, axis=-1)[:, None],
                axis=-1,
            )
        case _:
            assert_never(strategy)

    tokens = jax.vmap(partial(jax.lax.dynamic_update_index_in_dim, axis=0))(
        tokens, sampled_tokens, pos + 1
    )
    return tokens, kv_cache
