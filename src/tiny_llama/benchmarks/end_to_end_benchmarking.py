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
    from pathlib import Path
    import os

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

    # Number of params
    nb_params = sum(p.numel() for p in model.parameters())

    # Timings
    fw = np.asarray(fw_times, dtype=float)
    bw = np.asarray(bw_times, dtype=float)

    # Timing stats
    mean_fw = float(fw.mean())
    std_fw  = float(fw.std())
    mean_bw = float(bw.mean())
    std_bw  = float(bw.std())

    row = {
        "Config": args.model_config,
        "Parameters (M)": int(nb_params / 1e6),
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "Avg. fwd time (s)": np.round(mean_fw, 3),
        "Std. fwd time (s)": np.round(std_fw, 3),
        "Avg. bwd time (s)": np.round(mean_bw, 3),
        "Std. bwd time (s)": np.round(std_bw, 3),
    }

    # Choose column order: all args first, then the stats
    cols = list(row.keys())

    df_new = pd.DataFrame([row])

    # ==== Write into CSV ====
    out_csv = "./benchmarks/end_to_end_results.csv"
    csv_path = Path(out_csv)
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        if "Config" in df_old.columns:
            df_old = df_old[df_old["Config"] != args.model_config]
        df_all = pd.concat([df_old, df_new], ignore_index=True, sort=False)
    else:
        df_all = df_new

    tmp_path = csv_path.with_suffix(".tmp.csv")
    df_all.to_csv(tmp_path, index=False)
    os.replace(tmp_path, csv_path)

    print(f"Saved config '{model_config}'.")



