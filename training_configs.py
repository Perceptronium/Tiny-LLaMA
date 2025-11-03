from dataclasses import dataclass


@dataclass
class Config:

    # Dataset
    data: str = "Tiny-Stories" # Currently supported option isTiny-Stories

    # Transformer hyperparameters
    resume_training : bool = False
    vocab_size: int = 10_000
    context_length: int = 256
    num_layers: int = 4
    d_model: int = 512
    num_heads: int = 16
    d_ff:int = 1344
    theta: int = 10000 # RoPE Theta parameter

    # Training hyperparameters
    total_tokens_processed: int = 327_680_000
    batch_size: int = 128
    total_step_count: int = int(total_tokens_processed / (batch_size * context_length)) # Currently yields 10_000

    # AdamW hyperparameters
    adamw_lr: float = 1e-3
    device: str = "cuda:0"
    save_checkpoints_to: str = "./checkpoints/"

    # Learning rate scheduling parameterss
    lr_scheduler: bool = True
    maximal_learning_rate: float = adamw_lr
    minimal_learning_rate_ratio: float = 0.1
    nb_warmup_iters_ratio: float = 0.01
    nb_cosine_annealing_iters: int = int(total_step_count - nb_warmup_iters_ratio*total_step_count)
