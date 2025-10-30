import torch
import torch.nn as nn
import numpy as np


def get_batch(dataset: np.array, batch_size: int, context_length: int, device: str):

    set_size = len(dataset)

    sample_ids = torch.randint(low=0, high=set_size-context_length, size=(batch_size,))

    inputs = torch.empty(batch_size, context_length, dtype=torch.long)
    targets = torch.empty(batch_size, context_length, dtype=torch.long)

    for batch_id, sample_id in zip(range(batch_size), sample_ids):
        inputs[batch_id] = torch.tensor(
            dataset[sample_id:sample_id+context_length], dtype=torch.long)
        targets[batch_id] = torch.tensor(
            dataset[sample_id+1:sample_id+1+context_length], dtype=torch.long)

    return (inputs.to(device), targets.to(device))


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):

    # Save model, optimizer, and iteration states
    states = {"model_state": model.state_dict(),
              "optimizer_state": optimizer.state_dict(),
              "iteration": iteration}

    torch.save(states, out)


def load_checkpoint(src: str, model: nn.Module, optimizer: torch.optim.Optimizer):

    # Restore states
    checkpoint = torch.load(src)
    model_state = checkpoint["model_state"]
    optimizer_state = checkpoint["optimizer_state"]
    iteration = checkpoint["iteration"]

    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

    return iteration
