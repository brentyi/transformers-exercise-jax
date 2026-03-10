"""Llama-architecture model inference (SmolLM2, TinyLlama, etc.)."""

import json
import readline  # noqa: F401 — patches input() with line editing support
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from safetensors import safe_open
from tokenizers import Tokenizer

from src.model import (
    TransformerBlockParams,
    TransformerConfig,
    TransformerParams,
    prefill_kv_cache,
    resize_kv_cache,
)
from src.sampling import TopK, TopP, sample_one_token

# Model registry.


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    cfg: TransformerConfig
    eos_token: str
    tie_weights: bool


_MODELS: dict[str, ModelSpec] = {
    "smollm2-135m": ModelSpec(
        hf_id="HuggingFaceTB/SmolLM2-135M",
        cfg=TransformerConfig(
            vocab_size=49152,
            n_blocks=30,
            d_latent=576,
            d_ffn=1536,
            n_heads=9,
            n_heads_kv=3,
            rope_freq_base=100000.0,
        ),
        eos_token="<|endoftext|>",
        tie_weights=True,
    ),
    "smollm2-360m": ModelSpec(
        hf_id="HuggingFaceTB/SmolLM2-360M",
        cfg=TransformerConfig(
            vocab_size=49152,
            n_blocks=32,
            d_latent=960,
            d_ffn=2560,
            n_heads=15,
            n_heads_kv=5,
            rope_freq_base=100000.0,
        ),
        eos_token="<|endoftext|>",
        tie_weights=True,
    ),
    "smollm2-1.7b": ModelSpec(
        hf_id="HuggingFaceTB/SmolLM2-1.7B",
        cfg=TransformerConfig(
            vocab_size=49152,
            n_blocks=24,
            d_latent=2048,
            d_ffn=8192,
            n_heads=32,
            n_heads_kv=32,
            rope_freq_base=100000.0,
        ),
        eos_token="<|endoftext|>",
        tie_weights=True,
    ),
    "tinyllama-1.1b": ModelSpec(
        hf_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        cfg=TransformerConfig(
            vocab_size=32000,
            n_blocks=22,
            d_latent=2048,
            d_ffn=5632,
            n_heads=32,
            n_heads_kv=4,
            rope_freq_base=10000.0,
        ),
        eos_token="</s>",
        tie_weights=False,
    ),
    "mistral-7b": ModelSpec(
        hf_id="mistralai/Mistral-7B-v0.1",
        cfg=TransformerConfig(
            vocab_size=32000,
            n_blocks=32,
            d_latent=4096,
            d_ffn=14336,
            n_heads=32,
            n_heads_kv=8,
            rope_freq_base=10000.0,
        ),
        eos_token="</s>",
        tie_weights=False,
    ),
    "smollm2-1.7b-instruct": ModelSpec(
        hf_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        cfg=TransformerConfig(
            vocab_size=49152,
            n_blocks=24,
            d_latent=2048,
            d_ffn=8192,
            n_heads=32,
            n_heads_kv=32,
            rope_freq_base=100000.0,
        ),
        eos_token="<|endoftext|>",
        tie_weights=True,
    ),
}


@dataclass
class SamplingState:
    max_tokens: int = 64
    temp: float = 0.8
    seed: int = 0
    kvcache: bool = True
    max_prompt_len: int = 64
    strategy: Literal["naive", "greedy"] | TopK | TopP = TopK(k=4)


# Weight loading.


def _hf_file(hf_id: str, name: str) -> str:
    """Resolve HF file from cache, downloading only if needed."""
    path = try_to_load_from_cache(hf_id, name)
    return path if isinstance(path, str) else hf_hub_download(hf_id, name)


def load_params(
    model_spec: ModelSpec, dtype: jnp.dtype
) -> TransformerParams[jax.Array]:
    """Load weights for any Llama-architecture model."""
    cfg = model_spec.cfg
    hf_id = model_spec.hf_id
    tie = model_spec.tie_weights

    # Resolve safetensor file paths (single or sharded).
    index_path = try_to_load_from_cache(hf_id, "model.safetensors.index.json")
    if isinstance(index_path, str):
        shard_names = sorted(
            set(json.loads(Path(index_path).read_text())["weight_map"].values())
        )
        st_paths = [_hf_file(hf_id, name) for name in shard_names]
    else:
        st_paths = [_hf_file(hf_id, "model.safetensors")]

    # Load all tensors into a flat dict.
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        tensors: dict[str, jnp.ndarray] = {}
        for p in st_paths:
            with safe_open(p, framework="flax") as f:  # type: ignore
                for key in f.keys():
                    t = f.get_tensor(key)
                    tensors[key] = t.astype(dtype) if dtype is not None else t

        emb = tensors["model.embed_tokens.weight"]
        norm_out = tensors["model.norm.weight"]

        layers = []
        for i in range(cfg.n_blocks):
            attn = f"model.layers.{i}.self_attn."
            mlp = f"model.layers.{i}.mlp."
            q = tensors[f"{attn}q_proj.weight"].reshape(
                cfg.n_heads, cfg.d_head, cfg.d_latent
            )
            k = tensors[f"{attn}k_proj.weight"].reshape(
                cfg.n_heads_kv, cfg.d_head, cfg.d_latent
            )
            v = tensors[f"{attn}v_proj.weight"].reshape(
                cfg.n_heads_kv, cfg.d_head, cfg.d_latent
            )
            layers.append(
                TransformerBlockParams(
                    w_q=q,
                    w_kv=jnp.stack([k, v]),
                    w_o=tensors[f"{attn}o_proj.weight"],
                    swiglu0=jnp.stack(
                        [
                            tensors[f"{mlp}up_proj.weight"],
                            tensors[f"{mlp}gate_proj.weight"],
                        ]
                    ),
                    swiglu1=tensors[f"{mlp}down_proj.weight"],
                    rmsnorm0_scale=tensors[f"model.layers.{i}.input_layernorm.weight"],
                    rmsnorm1_scale=tensors[
                        f"model.layers.{i}.post_attention_layernorm.weight"
                    ],
                )
            )
        out_proj = emb if tie else tensors["lm_head.weight"]
        del tensors
        params = TransformerParams(
            embed=emb,
            blocks=jax.tree.map(lambda *xs: jnp.stack(xs), *layers),
            rmsnorm_out_scale=norm_out,
            out_proj=out_proj,
        )

    return jax.device_put(params, jax.devices()[0])


# Interactive loop.

_ceil_pow2 = lambda x: int(2 ** np.ceil(np.log2(x)))


def main(
    model: Literal[
        "smollm2-135m",
        "smollm2-360m",
        "smollm2-1.7b",
        "smollm2-1.7b-instruct",
        "tinyllama-1.1b",
        "mistral-7b",
    ] = ("smollm2-135m"),
    dtype: Literal["float32", "float16", "bfloat16"] = "float32",
) -> None:
    from tyro._fmtlib import box, hr, rows, text

    model_spec = _MODELS[model]
    cfg = model_spec.cfg
    hf_id = model_spec.hf_id

    tokenizer = Tokenizer.from_file(_hf_file(hf_id, "tokenizer.json"))
    eos_id = tokenizer.token_to_id(model_spec.eos_token)
    params = load_params(model_spec, jnp.dtype(dtype))
    s = SamplingState()

    def show_info() -> None:
        settings_parts = []
        for f in fields(s):
            v = getattr(s, f.name)
            if f.name == "kvcache":
                v = text["green"]("on") if v else text["red"]("off")
            else:
                v = str(v)
            settings_parts += [text["yellow"](f.name), "=", v, "  "]
        settings_parts.pop()

        commands = {
            "/temp": "<v>",
            "/max_tokens": "<n>",
            "/seed": "<n>",
            "/kvcache": "{on,off}",
            "/max_prompt_len": "<n>",
            "/greedy": "",
            "/naive": "",
            "/top_k": "<n>",
            "/top_p": "<v>",
            "/quit": "",
        }
        cmd_parts = []
        for cmd, arg in commands.items():
            cmd_parts += [text["magenta"](cmd), " ", arg, "  "]
        cmd_parts.pop()
        print()
        print(
            box["dim"](
                f"{model}  ({hf_id}, {dtype})",
                rows(text(*settings_parts), hr["dim"](), text["dim"](*cmd_parts)),
            )
        )

    show_info()
    while True:
        try:
            print()
            prompt = input(" > ")
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt == "/quit":
            break
        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0]
            if cmd == "/temp" and len(parts) == 2:
                s.temp = float(parts[1])
            elif cmd == "/max_tokens" and len(parts) == 2:
                s.max_tokens = int(parts[1])
            elif cmd == "/seed" and len(parts) == 2:
                s.seed = int(parts[1])
            elif cmd == "/kvcache" and len(parts) == 2:
                s.kvcache = parts[1] == "on"
            elif cmd == "/max_prompt_len" and len(parts) == 2:
                s.max_prompt_len = int(parts[1])
            elif cmd == "/greedy" and len(parts) == 1:
                s.strategy = "greedy"
            elif cmd == "/naive" and len(parts) == 1:
                s.strategy = "naive"
            elif cmd == "/top_k" and len(parts) == 2:
                s.strategy = TopK(k=int(parts[1]))
            elif cmd == "/top_p" and len(parts) == 2:
                s.strategy = TopP(p=float(parts[1]))
            else:
                print(text["red"](f"Unknown command: {cmd}"))
            show_info()
            continue

        # Tokenize.
        token_ids = tokenizer.encode(prompt).ids
        prompt_len = len(token_ids)
        max_seq_len = max(prompt_len, s.max_prompt_len) + s.max_tokens
        tokens = jnp.zeros((1, max_seq_len), dtype=jnp.int32)
        tokens = tokens.at[0, :prompt_len].set(jnp.array(token_ids, dtype=jnp.int32))
        key = jax.random.key(s.seed)
        temp = jnp.float32(s.temp)

        # Incremental token decoding.
        generated_ids: list[int] = list(token_ids)
        printed_len = len(tokenizer.decode(generated_ids))

        def print_new_tokens(new_token: int) -> None:
            nonlocal printed_len
            generated_ids.append(new_token)
            decoded = tokenizer.decode(generated_ids)
            print(decoded[printed_len:], end="", flush=True)
            printed_len = len(decoded)

        # Prefill (KV cache only).
        print(text["dim"](f"(prompt: {prompt_len} tokens)"))
        print(prompt, end="", flush=True)
        t0 = time.perf_counter()
        kv_cache = None
        if s.kvcache:
            kv_cache = prefill_kv_cache(
                tokens[:, : s.max_prompt_len],
                jnp.int32(prompt_len)[None],
                params,
                cfg,
            )
            jax.block_until_ready(kv_cache)
        t_prefill = time.perf_counter() - t0

        # Sample.
        t0 = time.perf_counter()
        for pos in range(prompt_len - 1, max_seq_len - 1):
            if kv_cache is not None:
                target_cache_size = max(32, _ceil_pow2(pos + 1))
                if kv_cache["data"][0].shape[2] < target_cache_size:
                    kv_cache = resize_kv_cache(kv_cache, target_cache_size)

            tokens, kv_cache_out = sample_one_token(
                tokens,
                jnp.int32(pos)[None],
                params,
                cfg,
                kv_cache=kv_cache,
                key=jax.random.fold_in(key, pos),
                temperature=temp,
                strategy=s.strategy,
            )
            if s.kvcache:
                kv_cache = kv_cache_out
            new_token = int(tokens[0, pos + 1])
            if new_token == eos_id:
                break
            print_new_tokens(new_token)

        jax.block_until_ready(tokens)
        t_sample = time.perf_counter() - t0

        print()
        if s.kvcache:
            print(text["dim"](f"(prefill {t_prefill:.2f}s, sample {t_sample:.2f}s)"))
        else:
            print(text["dim"](f"({t_prefill + t_sample:.2f}s)"))


if __name__ == "__main__":
    tyro.cli(main)
