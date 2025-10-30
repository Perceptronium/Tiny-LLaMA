
import argparse


def parse_args():
    p = argparse.ArgumentParser(
        prog="Tiny-LLaMA trainer",
        description="Train a cute Tiny-LLaMA on a cute Tiny-Stories dataset."
    )

    p.add_argument(
        "--resume_training", required=True, type=int, default=0,
        help="Warm start with final existing checkpoint or start training from scratch"
    )

    p.add_argument(
        "--vocab-size", required=True, type=int,
        help="Size of the vocabulary."
    )

    p.add_argument(
        "--context_length", required=True, type=int,
        help="Maximum allowed sequence length."
    )

    p.add_argument(
        "--num_layers", required=True, type=int,
        help="Number of layers in the model."
    )

    p.add_argument(
        "--d_model", required=True, type=int,
        help="Embedding dimension of the model."
    )

    p.add_argument(
        "--num_heads", required=True, type=int,
        help="Number of heads in the multihead self-attention."
    )

    p.add_argument(
        "--d_ff", required=True, type=int,
        help="Hidden dimension of the FF sub-layer."
    )

    p.add_argument(
        "--theta", required=True, type=int,
        help="Theta parameter for the Rotary Positional Encoding."
    )

    p.add_argument(
        "--device", required=True, type=str,
        help="Which device to use."
    )

    p.add_argument(
        "--save_checkpoints_to", required=True, type=str,
        help="Path to serialize the checkpoints to."
    )

    p.add_argument(
        "--adamw_lr", required=True, type=float,
        help="Initial learning rate for AdamW"
    )

    p.add_argument(
        "--adamw_beta1", required=False, type=float,
        help="Beta1 parameter for AdamW"
    )

    p.add_argument(
        "--adamw_beta2", required=False, type=float,
        help="Beta2 parameter for AdamW"
    )

    p.add_argument(
        "--adamw_weight_decay", required=False, type=float,
        help="Weight decay parameter for AdamW"
    )

    return p.parse_args()
