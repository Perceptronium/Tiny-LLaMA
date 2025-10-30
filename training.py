""" Train a cute Tiny-LLaMA on a cute Tiny-Stories dataset
    WORK IN PROGRESS """

# python ./training.py --resume_training 0 --vocab-size 10_000 --context_length 256 --num_layers 4 --d_model 512 --num_heads 16 --d_ff 1344 --theta 10000 --device cpu --save_checkpoints_to ./checkpoints/ --adamw_lr 0.01

import torch
import numpy as np
from pathlib import Path

from args_parser import parse_args
from transformer import TransformerLM
from optimizers import AdamW
from losses import CrossEntropyLoss, Perplexity
from training_utils import get_batch, load_checkpoint, save_checkpoint

# TODO Use either default config or user-specified setting
# TODO set-up Wandb
args = parse_args()

device = args.device

# Load the data in memory-mapped mode
train_data = "./data_tokenized/TinyStoriesV2-GPT4-train_token_ids.npy"
valid_data = "./data_tokenized/TinyStoriesV2-GPT4-valid_token_ids.npy"

# Training data contains approx. 540M tokens
train_token_ids = np.memmap(train_data, dtype=np.uint16, mode="r")

# Validation data contains approx. 5.5M tokens
valid_token_ids = np.memmap(valid_data, dtype=np.uint16, mode="r")

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
checkpoint_path = Path(args.save_checkpoints_to)
checkpoint_path.mkdir(parents=True, exist_ok=True)

# TODO make these a user-specified argument ?
total_tokens_processed = 327_680_000 # For GPU
# total_tokens_processed = 40_960_000  # For CPU

total_step_count = 10_000
context_length = args.context_length
batch_size = int(total_tokens_processed / (total_step_count * context_length))

# TODO scheduler
# TODO gradient clipping

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

    inputs, targets = get_batch(train_token_ids,
                                batch_size,
                                context_length,
                                device=device)

    outputs = model(inputs)

    loss = criterion(outputs, targets)

    loss.backward()
    optimizer.step()
    print(f"Train loss: {loss.item():.3f}")

    # TODO Log relevant data in Wandb

    model.eval()
    with torch.no_grad():
        inputs, targets = get_batch(valid_token_ids,
                                    batch_size,
                                    context_length,
                                    device=device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        print(f"Valid loss: {loss.item():.3f}")

    # Checkpoint the run every 10 epochs
    saving_location = checkpoint_path / Path(f"epoch_{epoch}.pt")
    if epoch % 10 == 0:
        save_checkpoint(model=model,
                        optimizer=optimizer,
                        iteration=epoch,
                        out=saving_location)

    # TODO on user interruption (Ctrl+C), save checkpoint and logs
