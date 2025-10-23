import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):

        safe = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        exped = torch.exp(safe)
        res = torch.log(torch.sum(exped, dim=-1, keepdim=True)) - safe
        res = res.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        return res.mean()


class Perplexity(nn.Module):
    def __init__(self):
        super().__init__()

        self.cross_entropy = CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):

        return torch.exp(self.cross_entropy(logits, targets))
