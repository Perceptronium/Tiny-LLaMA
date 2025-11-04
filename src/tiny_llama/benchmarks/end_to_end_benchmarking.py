""" Benchmarking scripts
    Work in Progress"""

import torch
import timeit

from tiny_llama.transformer import TransformerLM
from tiny_llama.optimizers import AdamW
from tiny_llama.losses import CrossEntropyLoss

from tiny_llama.training_configs import Config


def forward_backward_timing(args, model, optimizer, criterion, device, warmup_steps, time_execution_steps):
    """ Measure execution times for forward and backward passes. """

    vocab_size = args.vocab_size
    batch_size = args.batch_size
    seq_len = args.context_length

    # Generate a random batch of data
    data = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len+1))
    inputs = data[:, :seq_len].to(device)
    targets = data[:, 1:seq_len+1].to(device)

    # Time the forward and backward passes
    fw_times = []
    bw_times = []

    for epoch in range(warmup_steps + time_execution_steps):

        optimizer.zero_grad()
        if epoch <= warmup_steps:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
        else:
            # Measure the forward
            fw_time_1 = timeit.default_timer()
            outputs = model(inputs)
            fw_time_2 = timeit.default_timer()
            fw_times.append(fw_time_2 - fw_time_1)

            loss = criterion(outputs, targets)

            # Measure the backward
            bw_time_1 = timeit.default_timer()
            loss.backward()
            bw_time_2 = timeit.default_timer()
            bw_times.append(bw_time_2 - bw_time_1)

        optimizer.step()
        torch.cuda.synchronize()

    return fw_times, bw_times


if __name__ == "__main__":
    from dataclasses import asdict
    import numpy as np
    import argparse
    import pandas as pd

    p = argparse.ArgumentParser()
    p.add_argument("--model_config", type=str, required=True)
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--context_length", type=int, required=True)
    p.add_argument("--d_model", type=int, required=True)
    p.add_argument("--num_layers", type=int, required=True)
    p.add_argument("--num_heads", type=int, required=True)
    p.add_argument("--d_ff", type=int, required=True)
    p.add_argument("--rope_theta", type=int, required=True)
    p.add_argument("--warmup_steps", type=int, required=True)
    p.add_argument("--time_execution_steps", type=int, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--learning_rate", type=float, required=True)

    args = p.parse_args()

    device = "cuda:0"


    # Given hyperparameters, initialize a model
    # TODO: add number of parameters of each config
    model_config = args.model_config

    model = TransformerLM(vocab_size=args.vocab_size,
                            context_length=args.context_length,
                            d_model=args.d_model,
                            num_layers=args.num_layers,
                            num_heads=args.num_heads,
                            d_ff=args.d_ff,
                            theta=args.rope_theta,
                            device=device)

    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)

    criterion = CrossEntropyLoss()

    fw_times, bw_times = forward_backward_timing(args=args,
                            model=model,
                            optimizer=optimizer,
                            criterion=criterion,
                            device=device,
                            warmup_steps=args.warmup_steps,
                            time_execution_steps=args.time_execution_steps)

    # Convert to arrays (robust if lists)
    fw = np.asarray(fw_times, dtype=float)
    bw = np.asarray(bw_times, dtype=float)

    # Compute summary stats (seconds)
    mean_fw = float(fw.mean()) if fw.size else np.nan
    std_fw  = float(fw.std())  if fw.size else np.nan
    mean_bw = float(bw.mean()) if bw.size else np.nan
    std_bw  = float(bw.std())  if bw.size else np.nan

    # Assemble a single-row record: all CLI args + timing stats
    row = {
        **vars(args),
        "mean_fw_s": mean_fw,
        "std_fw_s": std_fw,
        "mean_bw_s": mean_bw,
        "std_bw_s": std_bw,
    }

    # Choose column order: all args first, then the stats
    cols = list(vars(args).keys()) + ["mean_fw_s", "std_fw_s", "mean_bw_s", "std_bw_s"]

    df = pd.DataFrame([{k: row[k] for k in cols}])

    # Print as Markdown table
    print(df)


    print(f"{model_config} done.")




