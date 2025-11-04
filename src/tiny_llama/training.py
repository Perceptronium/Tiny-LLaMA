""" Train a cute Tiny-LLaMA model"""

import torch
import numpy as np
from pathlib import Path
import wandb

from transformer import TransformerLM
from optimizers import AdamW
from losses import CrossEntropyLoss, Perplexity
from training_utils import get_batch, load_checkpoint, save_checkpoint
from optim_utils import cosine_annealing_scheduler, gradient_clipping
from training_configs import Config
from dataclasses import asdict

torch.manual_seed(0)

args = Config()

learning_rates = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

for adamw_lr in learning_rates:
    print(f"Learning rate sweeps: {adamw_lr}")
    args.adamw_lr = adamw_lr
    # Start a new wandb run to track this script.
    wandb_run = wandb.init(
        # entity="can-pouliquen-ecole-normale-sup-rieure-de-lyon",
        project="Learning rate sweeps",
        name=f"LR = {args.adamw_lr}"
        )

    wandb_run.config.update(asdict(args))

    device = args.device

    # Load the data in memory-mapped mode
    if args.data == "Tiny-Stories":
        train_data = "./data_tokenized/TinyStoriesV2-GPT4-train_token_ids_bis.npy"
        valid_data = "./data_tokenized/TinyStoriesV2-GPT4-valid_token_ids_bis.npy"
    else:
        raise NotImplementedError

    # Training data contains approx. 540M tokens for Tiny Stories
    train_token_ids = np.load(train_data, mmap_mode='r')

    # Validation data contains approx. 5.5M tokens for Tiny Stories
    valid_token_ids = np.load(valid_data, mmap_mode='r')

    # Instantiate the model
    model = TransformerLM(vocab_size=args.vocab_size,
                        context_length=args.context_length,
                        num_layers=args.num_layers,
                        d_model=args.d_model,
                        num_heads=args.num_heads,
                        d_ff=args.d_ff,
                        theta=args.theta,
                        device=device)

    # Instantiate the optimizer
    optimizer = AdamW(model.parameters(), lr=args.adamw_lr)

    # Instantiate the criterions
    criterion = CrossEntropyLoss()
    perplexity = Perplexity()

    # Checkpoints path
    checkpoint_path = Path(args.save_checkpoints_to) / Path(f"lr_{adamw_lr}")
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Set-up training hyperparams
    total_tokens_processed = args.total_tokens_processed
    total_step_count = args.total_step_count
    context_length = args.context_length

    batch_size = args.batch_size

    # TODO gradient clipping ?

    # Resume training if specified
    epoch = -1
    if args.resume_training:
        files = [f for f in checkpoint_path.iterdir() if f.is_file()]
        last_checkpoint = checkpoint_path / Path(max(files, key=lambda f: f.name).name)
        epoch = load_checkpoint(src=last_checkpoint, model=model, optimizer=optimizer)

    model.to(device)
    while epoch < total_step_count:
        epoch += 1
        print(f"Epoch {epoch} / {total_step_count}")

        model.train()
        optimizer.zero_grad()

        if args.lr_scheduler:
            learning_rate = cosine_annealing_scheduler(iteration_idx=epoch,
                                                        maximal_learning_rate=args.adamw_lr,
                                                        minimal_learning_rate=args.minimal_learning_rate_ratio*args.adamw_lr,
                                                        nb_warmup_iters=int(args.nb_warmup_iters_ratio*total_step_count),
                                                        nb_cosine_annealing_iters=args.nb_cosine_annealing_iters)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate


        inputs, targets = get_batch(train_token_ids,
                                    batch_size,
                                    context_length,
                                    device=device)

        outputs = model(inputs)

        train_loss = criterion(outputs, targets)

        train_loss.backward()
        optimizer.step()
        print(f"Train loss: {train_loss.item():.3f}")


        model.eval()
        with torch.no_grad():
            inputs, targets = get_batch(valid_token_ids,
                                        batch_size,
                                        context_length,
                                        device=device)

            outputs = model(inputs)
            val_loss = criterion(outputs, targets)
            print(f"Valid loss: {val_loss.item():.3f}")

        # Log desired metrics
        wandb_run.log({"train_loss": train_loss,
                        "valid_loss": val_loss})

        # Checkpoint the run every 100 epochs
        saving_location = checkpoint_path / Path(f"epoch_{epoch}.pt")
        if epoch % 100 == 0:
            save_checkpoint(model=model,
                            optimizer=optimizer,
                            iteration=epoch,
                            out=saving_location)


        # TODO on user interruption (Ctrl+C), save checkpoint and logs

    wandb_run.finish()
