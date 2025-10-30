""" Generate text from a trained Tiny-LLaMA """

import torch
import torch.nn as nn

from transformer import Softmax


def generate_text(prompt: str,
                  model: nn.Module,
                  tokenizer: nn.Module,
                  softmax_temperature: float,
                  top_p_threshold: float,
                  max_new_tokens: int = 10):

    softmax = Softmax()

    eot_id = tokenizer.tokens_to_id[bytes("<|endoftext|>".encode("utf-8"))]

    output_ids = tokenizer.encode(prompt)

    new_tokens_ctr = 0
    while (new_tokens_ctr < max_new_tokens):

        logits = model(output_ids)  # shape [sequence_length, vocab_size]
        # Normalize to prob. distribution with temperature scaling
        probs = softmax(logits / softmax_temperature, dim=-1)

        # Choose token to predict with top-p sampling
        probs_sorted, _ = torch.sort(probs[-1], descending=True)
        mass = 0.
        least_likely_id = 0
        for token_prob in probs_sorted:
            mass += token_prob
            least_likely_id += 1
            if mass >= top_p_threshold:
                break
        next_token_id = torch.multinomial(probs_sorted[:least_likely_id], num_samples=1)

        output_ids.append(next_token_id.item())

        new_tokens_ctr += 1

        print(output_ids)

        if next_token_id == eot_id:
            break

    return tokenizer.decode(output_ids)
