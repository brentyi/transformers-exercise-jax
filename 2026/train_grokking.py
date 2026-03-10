"""Toy transformer training script.

- Reproduces grokking on modular addition (Power et al., 2022).
- Implements Megatron-style tensor parallelism, as well as a "bad" TP variant
  for performance comparison.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Literal, TypedDict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import tyro
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from src.model import (
    TransformerBlockParams,
    TransformerConfig,
    TransformerParams,
    init_params,
    transformer_fwd,
)


def make_sequences(p: int) -> onp.ndarray:
    """Build all p*p token sequences: [x, +, y, =, result].

    Token IDs: 0..p-1 = numbers, p = '=', p+1 = '+'.
    """
    grid = onp.mgrid[:p, :p].reshape(2, -1)
    x, y = grid[0], grid[1]
    result = (x + y) % p
    n = p * p
    return onp.stack(
        [
            x,
            onp.full(n, p + 1),  # +
            y,
            onp.full(n, p),  # =
            result,
        ],
        axis=-1,
    )  # (p*p, 5)


class TrainState(TypedDict):
    params: TransformerParams[Array]
    m: TransformerParams[Array]
    v: TransformerParams[Array]
    step: Array


def loss_and_accuracy(
    params: TransformerParams[Array],
    seqs: Array,
    cfg: TransformerConfig,
) -> tuple[Array, Array]:
    """Cross-entropy loss and accuracy on the result token."""
    (b, t) = seqs.shape
    input_tokens = seqs[:, :4]  # (b, 4): [x, +, y, =]
    logits, _ = transformer_fwd(input_tokens, params, cfg)
    vocab_size = logits.shape[-1]
    assert logits.shape == (b, t - 1, vocab_size)

    # We'll only consider the logits at the '=' position.
    logits = logits[:, 3:4, :]
    assert logits.shape == (b, 1, vocab_size)

    # Standard stability trick.
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)

    # The Megatron paper has a paragraph about how it avoids all-gathering the
    # logits O(btv). Probably that is handled automatically here: assuming no
    # data parallelism along the same mesh axis (eg, FSDP), the one-hot matmul
    # below should be done via partial matmuls => all-reduce O(bt).
    tgt = jax.nn.one_hot(seqs[:, 4:5], vocab_size)
    tgt_logits = jnp.einsum("btv,btv->bt", logits, tgt)
    assert tgt_logits.shape == (b, 1)

    nll = -jnp.mean(tgt_logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True))
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1, keepdims=True) == seqs[:, 4:5])
    return nll, accuracy


@jax.jit(static_argnames=("cfg",))
def train_step(
    state: TrainState,
    batch_seqs: Array,
    *,
    cfg: TransformerConfig,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
) -> tuple[Array, Array, TrainState]:
    """One AdamW step. Returns (loss, accuracy, new_state)."""
    (loss, accuracy), grads = jax.value_and_grad(loss_and_accuracy, has_aux=True)(
        state["params"], batch_seqs, cfg
    )

    # AdamW update.
    step = state["step"] + 1
    unbias_beta1 = 1.0 - beta1**step
    unbias_beta2 = 1.0 - beta2**step
    m = jax.tree.map(lambda m, g: beta1 * m + (1.0 - beta1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: beta2 * v + (1.0 - beta2) * g**2, state["v"], grads)
    decayed_params = jax.tree.map(
        # In practice, we would do some masking here.
        lambda p: (1.0 - lr * weight_decay) * p,
        state["params"],
    )
    update = jax.tree.map(
        lambda m, v: (lr * m / unbias_beta1) / (jnp.sqrt(v / unbias_beta2) + 1e-8),
        m,
        v,
    )
    params = jax.tree.map(lambda p, u: p - u, decayed_params, update)

    return loss, accuracy, TrainState(params=params, m=m, v=v, step=step)


@jax.jit(static_argnames=("cfg",))
def eval_step(
    params: TransformerParams[Array], seqs: Array, cfg: TransformerConfig
) -> tuple[Array, Array]:
    """Returns (loss, accuracy) on a set of sequences."""
    return loss_and_accuracy(params, seqs, cfg)


def main(
    cfg: TransformerConfig = TransformerConfig(
        n_blocks=2,
        d_latent=128,
        d_ffn=512,
        n_heads=4,
        n_heads_kv=4,
    ),
    sharding: Literal["megatron", "bad_tp"] | None = None,
    minibatch_size: int = 512,
    lr: float = 3e-4,
    weight_decay: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.99,
    p: int = 97,
    train_frac: float = 0.3,
    n_steps: int = 30_000,
    log_interval: int = 100,
    seed: int = 0,
    plot_path: str = "grokking.pdf",
) -> None:
    """Grokking: modular addition (Power et al., 2022)."""
    # Override vocab_size: p numbers + 2 special tokens (=, +).
    vocab_size = p + 2
    vocab_size = (vocab_size + 15) // 16 * 16  # Pad to multiple of 16 for TP sharding.
    cfg = dataclasses.replace(cfg, vocab_size=vocab_size)

    # Generate dataset and split.
    rng = onp.random.default_rng(seed)
    seqs = make_sequences(p)
    rng.shuffle(seqs, axis=0)
    n_total = p * p
    n_train = int(n_total * train_frac)
    train_seqs = seqs[:n_train]
    test_seqs = seqs[n_train:]
    del seqs
    print(f"p={p}  op='+'  vocab_size={cfg.vocab_size}")
    print(f"Train: {n_train}, Test: {n_total - n_train}")

    # Initialize params and train state.
    params = init_params(cfg)
    if sharding is not None:
        # Tensor parallelism. Pedagogical; not actually practical for this small problem.
        mesh = Mesh(jax.devices(), axis_names=("tp",))
        S = lambda *axes: NamedSharding(mesh, PartitionSpec(*axes))
        params = jax.device_put(
            params,
            TransformerParams[NamedSharding](
                embed=S("tp", None),
                out_proj=S("tp", None),
                rmsnorm_out_scale=S(None),
                blocks=TransformerBlockParams(
                    w_q=S(None, "tp", None, None),
                    w_kv=S(None, None, "tp", None, None),
                    w_o=S(None, None, "tp"),
                    swiglu0={
                        "megatron": S(None, None, "tp", None),
                        "bad_tp": S(None, None, None, "tp"),
                    }[sharding],
                    swiglu1={
                        "megatron": S(None, None, "tp"),
                        "bad_tp": S(None, "tp", None),
                    }[sharding],
                    rmsnorm0_scale=S(None, None),
                    rmsnorm1_scale=S(None, None),
                ),
            ),
        )
        print(f"Tensor-parallel sharding over {len(jax.devices())} devices")

    state = TrainState(
        params=params,
        m=jax.tree.map(jnp.zeros_like, params),
        v=jax.tree.map(jnp.zeros_like, params),
        step=jnp.array(0),
    )

    # Logging history for plots.
    log_steps: list[int] = []
    train_accs: list[float] = []
    test_accs: list[float] = []

    # Training loop.
    tokens_per_step = minibatch_size * 4  # 4 input tokens per sequence
    t0 = time.monotonic()
    for step in range(n_steps):
        loss, acc, state = train_step(
            state,
            train_seqs[rng.integers(low=0, high=n_train, size=minibatch_size)],
            cfg=cfg,
            lr=lr,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
        )

        if step % log_interval == 0:
            test_loss, test_acc = eval_step(state["params"], test_seqs, cfg=cfg)
            log_steps.append(step)
            train_accs.append(float(acc))
            test_accs.append(float(test_acc))

            jax.block_until_ready(state)
            elapsed = time.monotonic() - t0
            it_per_sec = log_interval / elapsed
            tok_per_sec = log_interval * tokens_per_step / elapsed
            msg = (
                f"step={step:>6d}  train_loss={float(loss):.4f}  train_acc={float(acc):.4f}"
                f"  {it_per_sec:.1f} it/s  {tok_per_sec:.0f} tok/s"
            )
            msg += f"  test_loss={float(test_loss):.4f}  test_acc={float(test_acc):.4f}"
            print(msg)
            t0 = time.monotonic()

    # Save accuracy plot.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(log_steps, train_accs, label="Train accuracy")
    ax.plot(log_steps, test_accs, label="Test accuracy")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Grokking: modular addition mod {p}")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    tyro.cli(main)
