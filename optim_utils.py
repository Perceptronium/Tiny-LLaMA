import math
import torch


def cosine_annealing_scheduler(iteration_idx: int,
                               maximal_learning_rate: float,
                               minimal_learning_rate: float,
                               nb_warmup_iters: int,
                               nb_cosine_annealing_iters: int):

    t = iteration_idx
    alpha_max = maximal_learning_rate
    alpha_min = minimal_learning_rate
    T_w = nb_warmup_iters
    T_c = nb_cosine_annealing_iters

    if t < T_w:
        lr_t = alpha_max * t / T_w
    elif (T_w <= t) and (t <= T_c):
        lr_t = alpha_min + 0.5 * \
            (1 + math.cos(math.pi*(t - T_w)/(T_c - T_w)))*(alpha_max - alpha_min)
    else:
        lr_t = alpha_min

    return lr_t


def gradient_clipping(parameters: list, max_norm: float, eps: float = 1e-6):
    """ Given parameters, if the l2-norm of their grad is more than a max value, we scale it down """

    grad_norm = torch.sqrt(torch.sum(torch.tensor([torch.sum(p.grad**2)
                           for p in parameters if p.grad is not None])))
    for p in parameters:
        if p.grad is None:
            continue
        if grad_norm >= max_norm:
            p.grad *= max_norm/(grad_norm + eps)
