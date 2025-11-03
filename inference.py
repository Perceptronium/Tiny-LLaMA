""" Generate text from a trained Tiny-LLaMA """

import torch
import torch.nn as nn

from transformer import Softmax


def generate_text(prompt: str,
                  model: nn.Module,
                  tokenizer: nn.Module,
                  softmax_temperature: float,
                  top_p_threshold: float,
                  max_new_tokens: int = 1):

    softmax = Softmax()

    eot_id = tokenizer.tokens_to_id[bytes("<|endoftext|>".encode("utf-8"))]

    output_ids = tokenizer.encode(prompt)

    new_tokens_ctr = 0
    while (new_tokens_ctr < max_new_tokens):

        logits = model(torch.tensor(output_ids).unsqueeze(0))  # shape [1, sequence_length, vocab_size]
        # Normalize to prob. distribution with temperature scaling
        probs = softmax(logits.squeeze()[-1] / softmax_temperature, dim=-1)
        # Choose token to predict with top-p sampling
        probs_sorted, indices = torch.sort(probs, descending=True)
        mass = 0.
        least_likely_id = 0
        for token_prob in probs_sorted:
            mass += token_prob
            least_likely_id += 1
            if mass >= top_p_threshold:
                break
        next_token_id = torch.multinomial(probs_sorted[:least_likely_id], num_samples=1)
        output_ids.append(indices[next_token_id].item())

        new_tokens_ctr += 1
        if next_token_id == eot_id:
            break

    return tokenizer.decode(output_ids)


if __name__ == "__main__":
    from pathlib import Path
    import numpy as np
    from transformer import TransformerLM
    from optimizers import AdamW
    from tokenizer import Tokenizer
    from training_configs import Config
    from training_utils import load_checkpoint
    import pickle


    def load_pickle(path: Path):
        with path.open("rb") as f:
            return pickle.load(f)

    vocab_path = Path("./tiny_stories_vocab/tiny_stories_bpe_vocab.pkl")
    vocab = load_pickle(vocab_path)

    merges_path = Path("./tiny_stories_vocab/tiny_stories_bpe_merges.pkl")
    merges = load_pickle(merges_path)

    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])

    args = Config()
    device = "cuda:0"
    model = TransformerLM(vocab_size=args.vocab_size,
                      context_length=args.context_length,
                      num_layers=args.num_layers,
                      d_model=args.d_model,
                      num_heads=args.num_heads,
                      d_ff=args.d_ff,
                      theta=args.theta,
                      device=device)
    optimizer = AdamW(model.parameters(), lr=args.adamw_lr)

    checkpoint = "./checkpoints/epoch_10000.pt"
    _ = load_checkpoint(src=checkpoint, model=model, optimizer=optimizer)

    prompt = "Once upon a time, there was a computer science team named Ockham."

    output = generate_text(prompt=prompt,
                            model=model,
                            tokenizer=tokenizer,
                            softmax_temperature=0.1,
                            top_p_threshold=0.95,
                            max_new_tokens=100)

    print(output)


