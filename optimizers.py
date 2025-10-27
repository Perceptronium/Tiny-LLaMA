from collections.abc import Callable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    """ Reimplementing AdamW (Loshchilov & Hutter, 2019) """

    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas: tuple[float] = (0.9, 0.999),
                 weight_decay: float = 0.9,
                 eps: float = 1e-8):

        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr,
                    "betas": betas,
                    "weight_decay": weight_decay,
                    "eps": eps}

        super().__init__(params,  # Model parameters
                         defaults)  # Optimizer hyperparameters

    def step(self, closure: Optional[Callable] = None):

        loss = None if closure is None else closure()

        for group in self.param_groups:

            lr = group["lr"]
            beta_1 = group["betas"][0]
            beta_2 = group["betas"][1]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                # State-dict of parameter p
                state = self.state[p]

                t = state.get("t", 1)

                # Get the gradient of parameter p in the current time step
                grad = p.grad.data

                # Update the first moment estimate
                m = state.get("m", 0)
                m = beta_1 * m + (1 - beta_1) * grad

                # Update the second moment estimate
                v = state.get("v", 0)
                v = beta_2 * v + (1 - beta_2) * grad**2

                # Compute the adjusted learning rate for iteration t
                lr_t = lr * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)

                # Update the parameters
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

            return loss


class DecayingSGD(torch.optim.Optimizer):
    """ SGD with decaying lr, verbatim from CS 336 """

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):

        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                # Get iteration number from the state, or initial value.
                t = state.get("t", 0)
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

            return loss
