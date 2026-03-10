"""Llama 3-style transformer."""

import itertools
from dataclasses import dataclass
from functools import partial
from typing import TypedDict

import jax
from jax import Array
from jax import numpy as jnp


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 256
    n_blocks: int = 8
    d_latent: int = 512
    d_ffn: int = 1024
    n_heads: int = 16
    n_heads_kv: int = 4
    rope_freq_base: float = 100.0
    rope_scale: float = 1.0

    @property
    def d_head(self) -> int:
        return self.d_latent // self.n_heads


class TransformerBlockParams[T](TypedDict):
    w_q: T  # ([n_blocks,] n_heads, d_head, d_latent)
    w_kv: T  # ([n_blocks,] 2, n_heads_kv, d_head, d_latent)
    w_o: T  # ([n_blocks,] d_latent, d_latent)
    rmsnorm0_scale: T  # ([n_blocks,] d_latent)
    rmsnorm1_scale: T  # ([n_blocks,] d_latent)
    swiglu0: T  # ([n_blocks,] 2, d_ffn, d_latent)
    swiglu1: T  # ([n_blocks,] d_latent, d_ffn)


class TransformerParams[T](TypedDict):
    embed: T  # (vocab_size, d_latent)
    blocks: TransformerBlockParams[T]  # Stacked along axis=0.
    rmsnorm_out_scale: T  # (d_latent,)
    out_proj: T  # (vocab_size, d_latent)


type KvData = tuple[Array, Array]  # (keys, values)
type RopeCoeffs = tuple[Array, Array]  # (cos, sin)


class KvCache(TypedDict):
    data: KvData  # Each ([n_blocks,] batch, max_len, n_heads_kv, d_head)
    idx: Array  # (batch,)


@jax.jit(static_argnames=("new_max_len",))
def resize_kv_cache(kv_cache: KvCache, new_max_len: int) -> KvCache:
    """Resize KV cache to accommodate a new max sequence length. Preserves existing data."""
    old_max_len = kv_cache["data"][0].shape[2]
    if new_max_len <= old_max_len:
        data = jax.tree.map(lambda x: x[:, :, :new_max_len, :, :], kv_cache["data"])
    else:
        pad_amount = ((0, 0), (0, 0), (0, new_max_len - old_max_len), (0, 0), (0, 0))
        data = jax.tree.map(lambda x: jnp.pad(x, pad_amount), kv_cache["data"])
    return KvCache(data=data, idx=kv_cache["idx"])


def init_params(cfg: TransformerConfig) -> TransformerParams[Array]:
    """Initialize weights with `std=1.0/sqrt(fan_in)`. No residual stream scaling."""
    n_blocks = cfg.n_blocks
    d_latent = cfg.d_latent
    make_key = map(jax.random.key, itertools.count()).__next__
    init = lambda shape: jax.random.normal(make_key(), shape) / jnp.sqrt(d_latent)
    return TransformerParams(
        embed=init((cfg.vocab_size, d_latent)),
        blocks=TransformerBlockParams(
            w_q=init((n_blocks, cfg.n_heads, cfg.d_head, d_latent)),
            w_kv=init((n_blocks, 2, cfg.n_heads_kv, cfg.d_head, d_latent)),
            w_o=init((n_blocks, d_latent, d_latent)),
            rmsnorm0_scale=jnp.ones((n_blocks, d_latent)),
            rmsnorm1_scale=jnp.ones((n_blocks, d_latent)),
            swiglu0=init((n_blocks, 2, cfg.d_ffn, d_latent)),
            swiglu1=init((n_blocks, d_latent, cfg.d_ffn)),
        ),
        rmsnorm_out_scale=jnp.ones((d_latent,)),
        out_proj=init((cfg.vocab_size, d_latent)),
    )


@jax.jit(static_argnames=("cfg",))
def prefill_kv_cache(
    tokens: Array,  # (b, max_seq_len) int32 token buffer, should contain prompt.
    prompt_len: Array,  # (b,) int32, number of prompt tokens.
    params: TransformerParams[Array],
    cfg: TransformerConfig,
) -> KvCache:
    """Prefill KV cache. Forward-passes on the entirety of `tokens`."""
    _, kv_cache = transformer_fwd(tokens, params, cfg, None)
    kv_cache["idx"] = prompt_len - 1
    return kv_cache


def transformer_fwd(
    x: Array,
    params: TransformerParams[Array],
    cfg: TransformerConfig,
    kv_cache: KvCache | None = None,
) -> tuple[Array, KvCache]:
    """Forward-pass the transformer. Returns either a new KV cache (for prefill) or an updated one (sampling)."""

    (b, t) = x.shape
    assert jnp.issubdtype(x.dtype, jnp.integer), "Input should be integer token IDs."

    # Embeddings.
    x = params["embed"][x, :]

    # RoPE coeffs.
    thetas = cfg.rope_freq_base ** (-jnp.arange(0, cfg.d_head, 2) / cfg.d_head)
    if kv_cache is None:
        rope_pos = jnp.outer(cfg.rope_scale * jnp.arange(t), thetas)
        rope_pos = rope_pos[None, :, None, :]
        assert rope_pos.shape == (1, t, 1, cfg.d_head // 2)
    else:
        rope_pos = jnp.einsum(
            "bt,d->btd",
            cfg.rope_scale * kv_cache["idx"][:, None] + jnp.arange(t)[None, :],
            thetas,
        )[:, :, None, :]
        assert rope_pos.shape == (b, t, 1, cfg.d_head // 2)
    rope_pos = rope_pos.astype(x.dtype)
    cos, sin = jnp.cos(rope_pos), jnp.sin(rope_pos)
    assert x.shape == (b, t, cfg.d_latent)

    # Transformer blocks.
    if kv_cache is None:
        # Simple forward.
        x, kv_data = jax.lax.scan(
            lambda x, params: _block_fwd(cfg, x, params, (cos, sin), None),
            init=x,
            xs=params["blocks"],
        )
        kv_cache = KvCache(data=kv_data, idx=jnp.full((b,), t - 1, dtype=jnp.int32))
    else:
        # Scan over block params + KV cache for each block.
        idx = kv_cache["idx"]
        x, kv_data = jax.lax.scan(
            lambda x, params: _block_fwd(
                cfg, x, params[0], (cos, sin), KvCache(data=params[1], idx=idx)
            ),
            init=x,
            xs=(params["blocks"], kv_cache["data"]),
        )
        assert kv_data is not None
        kv_cache = KvCache(data=kv_data, idx=kv_cache["idx"] + 1)

    # RMSnorm -> output projection.
    recip_rms = jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-5)
    x = jnp.einsum(
        "ij,btj->bti", params["out_proj"], x * recip_rms * params["rmsnorm_out_scale"]
    )
    assert x.shape == (b, t, cfg.vocab_size)
    return x, kv_cache


def _block_fwd(
    cfg: TransformerConfig,
    x: Array,
    params: TransformerBlockParams,
    rope_coeffs: RopeCoeffs,
    kv_cache: KvCache | None,
) -> tuple[Array, KvData]:
    """Returns (outputs, KV cache data)."""
    b, t, d_latent = x.shape
    assert d_latent == cfg.d_latent

    # Pre-norm -> QKV projection.
    assert params["rmsnorm0_scale"].shape == (d_latent,)
    recip_rms = jax.lax.rsqrt(jnp.mean(x**2, -1, keepdims=True) + 1e-5)
    x_norm = x * recip_rms * params["rmsnorm0_scale"]
    q = jnp.einsum("hij,btj->bthi", params["w_q"], x_norm)
    k, v = jnp.einsum("xhij,btj->xbthi", params["w_kv"], x_norm)
    assert q.shape == (b, t, cfg.n_heads, cfg.d_head)
    assert k.shape == v.shape == (b, t, cfg.n_heads_kv, cfg.d_head)

    # Apply RoPE to Q and K.
    cos, sin = rope_coeffs
    q_i, q_j = jnp.split(q, 2, axis=-1)
    k_i, k_j = jnp.split(k, 2, axis=-1)
    q = jnp.concatenate([cos * q_i - sin * q_j, sin * q_i + cos * q_j], axis=-1)
    k = jnp.concatenate([cos * k_i - sin * k_j, sin * k_i + cos * k_j], axis=-1)
    assert q.shape == (b, t, cfg.n_heads, cfg.d_head)
    assert k.shape == v.shape == (b, t, cfg.n_heads_kv, cfg.d_head)

    # GQA, with or without a KV cache.
    if kv_cache is None:
        kv_data = (k, v)
        attn_out = jax.nn.dot_product_attention(q, *kv_data, is_causal=True)
    else:
        assert t == 1, "When using KV cache, input should be one token at a time."
        assert kv_cache["data"][0].shape == kv_cache["data"][1].shape
        assert len(kv_cache["data"][0].shape) == 4, "(b, max_len, n_heads_kv, d_head)"
        assert kv_cache["idx"].shape == (b,)
        kv_data = jax.tree.map(
            jax.vmap(partial(jax.lax.dynamic_update_slice_in_dim, axis=0)),
            kv_cache["data"],  # operand
            (k, v),  # update
            (kv_cache["idx"],) * 2,  # start_index
        )
        attn_out = jax.nn.dot_product_attention(
            q, *kv_data, key_value_seq_lengths=kv_cache["idx"] + 1
        )
    assert attn_out.shape == (b, t, cfg.n_heads, cfg.d_head)
    assert params["w_o"].shape == (d_latent, d_latent)
    x = x + jnp.einsum("ij,btj->bti", params["w_o"], attn_out.reshape((b, t, d_latent)))

    # SwiGLU with pre-norm.
    assert params["swiglu0"].shape == (2, cfg.d_ffn, d_latent)
    assert params["swiglu1"].shape == (d_latent, cfg.d_ffn)
    recip_rms = jax.lax.rsqrt(jnp.mean(x**2, -1, keepdims=True) + 1e-5)
    x_norm = x * recip_rms * params["rmsnorm1_scale"]
    val, gate = jnp.einsum("xij,btj->xbti", params["swiglu0"], x_norm)
    x = x + jnp.einsum("ij,btj->bti", params["swiglu1"], val * jax.nn.silu(gate))
    return x, kv_data
