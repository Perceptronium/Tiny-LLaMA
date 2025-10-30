import torch
import torch.nn as nn
import math


class LinearNoBias(nn.Module):
    """ Reimplementation of torch.nn.Linear (without the bias) """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        # Recommended initialization (but why ?)
        weights = torch.empty(out_features, in_features,
                              dtype=dtype, device=device)
        std = math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(weights,
                              mean=0.,
                              std=std,
                              a=-3*std,
                              b=3*std)

        # No bias, following most modern LLMs : why ?
        self.weights = nn.Parameter(weights)

    def forward(self, x: torch.Tensor):
        """ Apply linear transformation to the input """

        # einsum internally broadcasts the required dimensions
        # pytorch memory is row major : Wx = x.T W.T
        return torch.einsum('...j, ij -> ...i', x, self.weights)


class Embedding(nn.Module):
    """ Reimplementation of the classic torch.nn.Embedding module,
        this is essentially a lookup table with trainable embeddings for each
        token in the vocab """

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()

        # Basically the vocab size
        self.num_embeddings = num_embeddings

        self.embedding_dim = embedding_dim

        embedding_matrix = torch.empty(self.num_embeddings,
                                       self.embedding_dim,
                                       device=device,
                                       dtype=dtype)

        nn.init.trunc_normal_(embedding_matrix,
                              mean=0.,
                              std=1.,
                              a=-3,
                              b=3)

        self.embedding_matrix = nn.Parameter(embedding_matrix)

    def forward(self, x: torch.LongTensor):
        """ Maps token IDs to a vector space of dim d_model """

        # x shape : [batch_size, seq_len]
        # W shape : [vocab_size, d_model]
        return self.embedding_matrix[x]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()

        self.eps = eps

        self.d_model = d_model

        # Trainable gain to scale the normalization
        self.gain = nn.Parameter(torch.ones(d_model,
                                            device=device,
                                            dtype=dtype))

    def forward(self, x: torch.Tensor):

        # x shape [batch_size, seq_len, d_model]

        # Training is typically done in bfloat16, we upscale it to torch.float32 here for stability
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Perform RMS normalization as defined in Zhang & Sennrich, 2019
        normalized = x * torch.rsqrt(torch.sum(x**2, dim=-1, keepdim=True)
                                     / self.d_model + self.eps)

        return (normalized * self.gain).to(in_dtype)


class SiLU(nn.Module):
    """ Reimplementation of the SiLU activation function """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):

        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int = None, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()

        # Canonically set to standard practice
        if d_ff:
            self.d_ff = d_ff
        else:
            self.d_ff = int(8 * d_model / 3)

        self.W1 = LinearNoBias(in_features=d_model, out_features=self.d_ff,
                               device=device, dtype=dtype)

        self.W2 = LinearNoBias(in_features=self.d_ff, out_features=d_model,
                               device=device, dtype=dtype)

        self.W3 = LinearNoBias(in_features=d_model, out_features=self.d_ff,
                               device=device, dtype=dtype)

        self.silu = SiLU()

    def forward(self, x: torch.Tensor):
        """ 'We offer no explanation as to why these architectures seem to work;
            we attribute their success, as all else, to divine benevolence.' (Shazeer 2020) """
        return self.W2(self.silu(self.W1(x)) * self.W3(x))


class RotaryPositionalEmbedding(nn.Module):

    """ This implementation exploits the structure of the big rotation matrix
        and therefore avoids bug-prone index manipulations """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None):
        super().__init__()

        thetas = 1 / theta**((2*torch.arange(int(d_k//2)) / d_k))
        positions = torch.arange(max_seq_len)
        angles = torch.outer(positions, thetas)

        cos_sin_matrix = torch.polar(torch.ones_like(angles), angles).to(device)
        self.register_buffer("cos_sin_matrix",
                             cos_sin_matrix,
                             persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):

        # x shape [..., seq_len, model_dim]

        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        rots = self.cos_sin_matrix[token_positions]
        roted = torch.einsum('...ij,...ij->...ij', x_complex, rots)
        encoded = torch.view_as_real(roted).reshape(*x.shape)

        return encoded


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int):

        # subtracting the max val in the i-th dimension for numerical stability
        safe = x - torch.max(x, dim, keepdim=True)[0]
        exped = torch.exp(safe)

        return exped / torch.sum(exped, dim=dim, keepdim=True)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = Softmax()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal_mask: torch.Tensor = None):
        """ Compute softmax(QK.T / d_k)V """

        # Q: [batch_size, ..., seq_len, d_k]
        # K: [batch_size, ..., seq_len, d_k]
        # V: [batch_size, ..., seq_len, d_v]

        d_k = K.shape[-1]

        scores = torch.einsum('b...qd,b...kd->b...qk', Q, K) / math.sqrt(d_k)

        if causal_mask is not None:
            scores = scores.masked_fill_(causal_mask == False, -torch.inf)

        probs = self.softmax(scores, dim=-1)
        return torch.einsum('b...qk,b...kd->b...qd', probs, V)


class CausalMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int,
                 n_heads: int,
                 rpe: bool = False,
                 max_seq_len: int = None,
                 theta: float = None,
                 device: torch.device = None):
        super().__init__()

        self.rpe = rpe

        self.n_heads = n_heads

        self.device = device

        self.dim_heads = int(d_model // self.n_heads)

        # Linear transforms
        self.w_q = LinearNoBias(in_features=d_model,
                                out_features=self.n_heads*self.dim_heads, device=device)
        self.w_k = LinearNoBias(in_features=d_model,
                                out_features=self.n_heads*self.dim_heads, device=device)
        self.w_v = LinearNoBias(in_features=d_model,
                                out_features=self.n_heads*self.dim_heads, device=device)
        self.w_o = LinearNoBias(in_features=self.n_heads *
                                self.dim_heads, out_features=d_model, device=device)

        # Rot. Positional encoding
        if self.rpe:
            self.rope = RotaryPositionalEmbedding(
                d_k=self.dim_heads,
                max_seq_len=max_seq_len,
                theta=theta,
                device=device)

        self.compute_attention = ScaledDotProductAttention()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):

        # x shape [batch_size, ..., seq_len, d_model]
        seq_len = x.shape[-2]

        # Form Queries, Keys and Values
        Q = self.w_q(x)  # shape [batch_size,...,seq_len, n_heads*dim_heads]
        K = self.w_k(x)
        V = self.w_v(x)

        # Separate the heads
        Q = Q.reshape(*Q.shape[:-1], self.n_heads, self.dim_heads)
        K = K.reshape(*K.shape[:-1], self.n_heads, self.dim_heads)
        V = V.reshape(*V.shape[:-1], self.n_heads, self.dim_heads)

        # Move the head dimension before the seq_len dimension
        Q = Q.transpose(-2, -3)
        K = K.transpose(-2, -3)
        V = V.transpose(-2, -3)

        # Apply RPE to Qs and Ks
        if self.rpe:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(self.device)

        # Compute each head's attention with causal masking
        head_attentions = self.compute_attention(Q, K, V, causal_mask)

        # Reshape and concatenate the head outputs
        head_attentions = head_attentions.transpose(-2, -3)
        head_attentions = head_attentions.reshape(*head_attentions.shape[:-2], -1)

        # Transform the head outputs
        out = self.w_o(head_attentions)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None):
        super().__init__()

        self.feedforward = SwiGLU(d_model=d_model, d_ff=d_ff, device=device)

        self.mha = CausalMultiheadSelfAttention(
            d_model=d_model,
            n_heads=num_heads,
            rpe=True,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device)

        self.norm_1 = RMSNorm(d_model=d_model, device=device)
        self.norm_2 = RMSNorm(d_model=d_model, device=device)

    def forward(self, x: torch.Tensor):

        # Pre-MHA normalization
        normed_x = self.norm_1(x)

        # Residual MHA sub-layer output
        token_positions = torch.arange(x.shape[-2])
        res_mha = x + self.mha(normed_x, token_positions=token_positions)

        # Pre-feedforward normalization
        normed_res_mha = self.norm_2(res_mha)

        # Residual FFW sub-layer output
        res_ffw = res_mha + self.feedforward(normed_res_mha)

        return res_ffw


class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 theta: float,
                 device: torch.Tensor = None):
        super().__init__()

        self.embedding = Embedding(num_embeddings=vocab_size,
                                   embedding_dim=d_model, device=device)

        self.core_layers = nn.Sequential()
        for _ in range(num_layers):
            self.core_layers.append(TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                device=device))

        self.out_norm = RMSNorm(d_model=d_model, device=device)

        self.out_proj = LinearNoBias(
            in_features=d_model, out_features=vocab_size, device=device)

        # self.softmax = Softmax()

    def forward(self, x: torch.Tensor):

        embedded = self.embedding(x)

        transformed = self.core_layers(embedded)

        normed = self.out_norm(transformed)

        logits = self.out_proj(normed)

        # probs = self.softmax(logits, dim=-1)

        return logits
