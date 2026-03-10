"""Script for training a GPT model on some text corpus. Pass in --help flag for options."""

import dataclasses
import pathlib

import tyro
import fifteen
import jax
import jax_dataclasses as jdc
from tqdm.auto import tqdm

from mingpt import data, model, trainer


@dataclasses.dataclass
class TrainArgs:
    """Arguments for training. Should be populated via the CLI."""

    dataset_path: pathlib.Path  # Path to a text file, to be loaded for training. Needs to fit in memory.
    experiment_name: str = "char_" + fifteen.utils.timestamp()
    restore_checkpoint: bool = False
    max_epochs: int = 1000
    minibatch_size: int = 128
    block_size: int = 128

    gpt_config: model.GPTConfig = model.GPTConfig(
        vocab_size=256,  # At most, all ASCII characters.
        block_size=128,
        n_head=8,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        n_layer=8,
        embd_dim=512,
        embd_pdrop=0.1,
        chunk_attention=False,
        q_chunk_size=None,
        kv_chunk_size=None,
    )
    optimizer_config: trainer.OptimizerConfig = trainer.OptimizerConfig(
        learning_rate=6e-4,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=2 * 10 * block_size,
    )


def main() -> None:
    fifteen.utils.pdb_safety_net()
    train_args = tyro.cli(TrainArgs)

    # Set up experiment, determine training devices.
    experiment = fifteen.experiments.Experiment(
        data_dir=pathlib.Path("./experiments/") / train_args.experiment_name
    )
    devices = jax.local_devices()
    device_count = jax.local_device_count()
    assert (
        train_args.minibatch_size % device_count == 0
    ), f"Batch size {train_args.minibatch_size} must be divisible by {device_count}."

    # Load dataset.
    with open(train_args.dataset_path, "r") as f:
        train_dataset = data.CharDataset(
            data=f.read()[:-1],
            block_size=train_args.block_size,
        )

    # Write some metadata -- we can use these to reproduce results or recreate the
    # model at eval time.
    experiment.write_metadata("gpt_config", train_args.gpt_config)
    experiment.write_metadata("optimizer_config", train_args.optimizer_config)
    experiment.write_metadata("stoi", train_dataset.stoi)
    experiment.write_metadata("itos", train_dataset.itos)

    # Vocab size of the training dataset should be smaller than the vocab size of the
    # model. In practice, note that these should just match.
    assert train_dataset.vocab_size <= train_args.gpt_config.vocab_size

    # Initialize training state, and replicate across each GPU.
    train_state = trainer.TrainState.initialize(
        seed=0,
        gpt_config=train_args.gpt_config,
        optimizer_config=train_args.optimizer_config,
    )
    if train_args.restore_checkpoint:
        train_state = experiment.restore_checkpoint(train_state)
    sharded_train_state: trainer.TrainState = jax.device_put_replicated(
        train_state, devices=devices
    )
    del train_state

    # Run training loop.
    train_dataloader = fifteen.data.DataLoader(
        dataset=train_dataset,
        minibatch_size=train_args.minibatch_size,
        num_workers=0,  # The entire dataset is in-memory, so we can skip parallelism.
    )
    for epoch in range(train_args.max_epochs):
        # Read training state from device 0 and save a checkpoint.
        train_state = jax.tree_map(lambda x: x[0], sharded_train_state)
        experiment.save_checkpoint(train_state, step=int(train_state.steps))
        del train_state

        # Grab iterable over minibatches, which have leaf shapes of (minibatch_size, ...).
        minibatches = train_dataloader.minibatches(shuffle_seed=epoch)

        # Convert leaf shapes to (device_count, minibatch_size // device_count, ...),
        # and prefetch to (potentially) improve parallelization.
        minibatches = fifteen.data.sharding_map(minibatches, devices=devices)
        minibatches = fifteen.data.prefetching_map(minibatches)

        for minibatch in tqdm(minibatches):
            # Give each device a different PRNG key; this makes dropout masks unique.
            # Placing this inside of the training loops results in all PRNG keys
            # depending only on splits happening on device #0, which is nice for making
            # sure that training states can be predictably loaded from checkpoints and
            # used to continue or re-run training.
            with jdc.copy_and_mutate(sharded_train_state) as sharded_train_state:
                sharded_train_state.prng_key = jax.jit(
                    jax.random.split, static_argnums=1
                )(sharded_train_state.prng_key[0], device_count)

            # Training step.
            (
                sharded_train_state,
                sharded_log_data,
            ) = sharded_train_state.sharded_train_step(minibatch)

            # Log to Tensorboard.
            experiment.log(
                sharded_log_data.fix_sharded_scalars(),
                step=sharded_train_state.steps[0],  # Pull step count from device 0.
                log_scalars_every_n=10,
                log_histograms_every_n=50,
            )

    # Read training state from device 0 and save a checkpoint.
    train_state = jax.tree_map(lambda x: x[0], sharded_train_state)
    experiment.save_checkpoint(train_state, step=int(train_state.steps))
    del train_state


if __name__ == "__main__":
    main()
