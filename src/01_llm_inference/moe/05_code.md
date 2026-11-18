# Code from scratch

A sparse MoE layer in PyTorch, built in the same order as the [forward pass](./02_forward_pass.md), then dropped into a small GPT.

The code follows [makeMoE](https://github.com/AviSoori1x/makeMoE) by Avinash Sooriyarachchi, with the surrounding transformer from Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Three classes make up the layer:

| Class | Job |
|---|---|
| `Expert` | One expansion-contraction FFN |
| `NoisyTopkRouter` | Produces the gating matrix `G` |
| `SparseMoE` | Runs the chosen experts and combines their outputs |

## Expert

Each expert is the ordinary transformer FFN.

```python
class Expert(nn.Module):
    def __init__(self, n_embed, hidden=None, dropout=0.1):
        super().__init__()
        hidden = hidden or 4 * n_embed
        self.net = nn.Sequential(
            nn.Linear(n_embed, hidden),   # expand
            nn.ReLU(),
            nn.Linear(hidden, n_embed),   # contract
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

Real LLMs use GELU or SiLU here. ReLU keeps the example simple.

## Router

### Step by step

Take the running example: 1 batch, 4 tokens, dimension 8, 3 experts, `top_k = 2`.

```python
num_experts, top_k, n_embed = 3, 2, 8

x = torch.randn(1, 4, n_embed)               # output of attention, input to MoE
router = nn.Linear(n_embed, num_experts)     # the routing matrix

logits = router(x)                           # (1, 4, 3): expert selector matrix
```

Keep the top-k per row:

```python
top_k_logits, top_k_indices = logits.topk(top_k, dim=-1)
# top_k_indices[0, t] holds the 2 expert ids chosen for token t
```

Fill everything else with `−∞` and apply softmax:

```python
sparse_logits = torch.full_like(logits, float("-inf"))
sparse_logits = sparse_logits.scatter(-1, top_k_indices, top_k_logits)

gating = F.softmax(sparse_logits, dim=-1)    # (1, 4, 3): expert selector weight matrix
```

`scatter` writes the kept logits back to their original columns. A sample result:

```text
tensor([[[0.5213, 0.0000, 0.4787],
         [0.5484, 0.0000, 0.4516],
         [0.0000, 0.5695, 0.4305],
         [0.4578, 0.0000, 0.5422]]])
```

Each row has exactly two non-zero entries that sum to 1.

### As a module

```python
class TopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.linear = nn.Linear(n_embed, num_experts)

    def forward(self, x):
        logits = self.linear(x)
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        sparse_logits = torch.full_like(logits, float("-inf"))
        sparse_logits = sparse_logits.scatter(-1, indices, top_k_logits)
        return F.softmax(sparse_logits, dim=-1), indices
```

### Noisy top-k

A plain router tends to keep choosing the experts it already prefers. **Noisy top-k gating** adds Gaussian noise to the logits before top-k, so other experts get picked now and then. It is a fourth balancing trick, next to the three on the [balancing page](./03_balancing.md).

```text
noisy_logits = logits + N(0, 1) × softplus(noise_logits)
```

- A second linear layer predicts, per token and per expert, how much noise to add.
- `softplus` keeps that scale positive.
- Everything after this step is unchanged.

```python
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, x):
        logits = self.topkroute_linear(x)

        noise = torch.randn_like(logits) * F.softplus(self.noise_linear(x))
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        sparse_logits = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = sparse_logits.scatter(-1, indices, top_k_logits)
        return F.softmax(sparse_logits, dim=-1), indices
```

> **Note:** the noise is added on every call, including after `model.eval()`, so routing is slightly random at inference too. This follows makeMoE. In the 2017 paper the noise is a training-time device. Wrap the two noise lines in `if self.training:` to make inference deterministic.

## The sparse MoE layer

The explanation so far went token by token: find the token's experts, weight their outputs, add. Code does the same sum the other way round, **expert by expert**:

1. For expert `i`, find every token routed to it.
2. Run the expert once on just those tokens.
3. Scale each output by that token's gate weight for expert `i`.
4. Add it into the output at that token's position.

A token routed to two experts receives two additions, which is exactly the weighted sum.

```python
class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, dropout=0.1):
        super().__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList(
            [Expert(n_embed, dropout=dropout) for _ in range(num_experts)]
        )

    def forward(self, x):
        gating, indices = self.router(x)          # (B, T, N), (B, T, k)
        out = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))                    # (B*T, d)
        flat_gating = gating.view(-1, gating.size(-1))     # (B*T, N)

        for i, expert in enumerate(self.experts):
            mask = (indices == i).any(dim=-1)     # (B, T): tokens routed to expert i
            flat_mask = mask.view(-1)
            if flat_mask.any():
                expert_out = expert(flat_x[flat_mask])                 # only those tokens
                weights = flat_gating[flat_mask, i].unsqueeze(1)
                out[mask] += expert_out * weights
        return out
```

Why loop over experts:

- Each expert runs **once**, on a batch of all its tokens. Looping over tokens would mean thousands of tiny forward passes.
- Each expert only sees the tokens routed to it. This is where the compute saving comes from.

```python
moe = SparseMoE(n_embed=8, num_experts=3, top_k=2)
x = torch.randn(1, 4, 8)
print(moe(x).shape)        # torch.Size([1, 4, 8]), same as the input
```

## Inside a transformer block

The MoE layer goes exactly where the FFN was. Nothing else changes.

```python
class Block(nn.Module):
    def __init__(self, n_embed, n_head, num_experts, top_k):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embed // n_head)
        self.smoe = SparseMoE(n_embed, num_experts, top_k)   # was: FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))      # attention + shortcut
        x = x + self.smoe(self.ln2(x))    # MoE + shortcut
        return x
```

`MultiHeadAttention` is standard causal self-attention. Dropout lives inside the attention and expert modules.

## The full model

The rest is nanoGPT:

```text
token embedding + position embedding
        │
        ▼
  n_layer × Block
        │
        ▼
    LayerNorm
        │
        ▼
Linear(n_embed → vocab_size)   → logits → cross-entropy against the next token
```

- **Data:** tiny Shakespeare (`input.txt`), character-level tokenisation, 90% train and 10% validation.
- **Batches:** `x` is a window of `block_size` characters, `y` is the same window shifted by one.
- **Optimiser:** AdamW.

| Hyperparameter | Value |
|---|---|
| `batch_size` | 16 |
| `block_size` (context length) | 32 |
| `n_embed` | 128 |
| `n_head` | 8 |
| `n_layer` | 8 |
| `num_experts` | 8 |
| `top_k` | 2 |
| `dropout` | 0.1 |
| `learning_rate` | 1e-3 |

That gives about **9.0M parameters**, of which 8.4M are experts. Each token uses 2 of the 8 experts per block, so only a quarter of the expert parameters run per token.

Training length decides everything. After 20 iterations the output is noise. After about 200 it already picks up the `SPEAKER:` dialogue layout and sentence lengths. Readable Shakespeare-like text takes tens of thousands of iterations.

> **Note:** this model has no balancing loss and no capacity limit. Only the noise in the router spreads the load.

## Extensions

The lecture leaves three things as exercises. The first two are short enough to show.

### Balancing losses

Both losses from the [balancing page](./03_balancing.md), written against the router outputs:

```python
def importance_loss(gating, lam=0.01):
    """lam * CV(expert importance)^2"""
    importance = gating.reshape(-1, gating.size(-1)).sum(dim=0)
    cv = importance.std(unbiased=False) / importance.mean()
    return lam * cv ** 2


def load_balancing_loss(probs, indices, lam=0.01):
    """lam * N * sum(f_i * P_i)"""
    n = probs.size(-1)
    P = probs.reshape(-1, n).mean(dim=0)                               # share of probability
    f = F.one_hot(indices[..., 0].reshape(-1), n).float().mean(dim=0)  # share of tokens
    return lam * n * (f * P).sum()
```

On the worked example these return `0.035 × λ` and `0.0101`, matching the hand calculation. The second value assumes the tie in the `is` row is broken towards `E1`, as it was by hand. `indices[..., 0]` is each token's top expert, which is the top-1 definition of `f`. Add either to the cross-entropy loss before `backward()`.

`unbiased=False` gives the population standard deviation used in the worked example.

### DeepSeek-style layer

Shared experts, fine-grained routed experts, and the loss-free balancing bias from the [DeepSeek page](./04_deepseek.md):

```python
class DeepSeekMoE(nn.Module):
    def __init__(self, n_embed, n_routed, n_shared, top_k, hidden,
                 bias_update_speed=1e-3, dropout=0.1):
        super().__init__()
        self.top_k = top_k
        self.bias_update_speed = bias_update_speed
        self.shared = nn.ModuleList(
            [Expert(n_embed, hidden, dropout) for _ in range(n_shared)]
        )
        self.routed = nn.ModuleList(
            [Expert(n_embed, hidden, dropout) for _ in range(n_routed)]
        )
        self.router = nn.Linear(n_embed, n_routed, bias=False)
        # a buffer, not a parameter: updated by hand, never by the optimiser
        self.register_buffer("balance_bias", torch.zeros(n_routed))

    def forward(self, x):
        flat_x = x.reshape(-1, x.size(-1))                 # (B*T, d)
        scores = torch.sigmoid(self.router(flat_x))        # (B*T, n_routed)

        # the bias only decides WHICH experts are picked
        indices = (scores + self.balance_bias).topk(self.top_k, dim=-1).indices

        # gate weights use the original scores, normalised over the picked experts
        gates = torch.zeros_like(scores).scatter(-1, indices, scores.gather(-1, indices))
        gates = gates / gates.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat_x)
        for expert in self.shared:                         # always on, unweighted
            out = out + expert(flat_x)
        for i, expert in enumerate(self.routed):           # sparse, weighted
            mask = (indices == i).any(dim=-1)
            if mask.any():
                out[mask] += expert(flat_x[mask]) * gates[mask, i].unsqueeze(1)

        if self.training:                                  # loss-free balancing
            with torch.no_grad():
                load = torch.bincount(indices.flatten(), minlength=len(self.routed)).float()
                self.balance_bias += self.bias_update_speed * torch.sign(load.mean() - load)

        return out.view_as(x)
```

How it maps to the three innovations:

- **Fine-grained:** pass a small `hidden` and a large `n_routed`. Splitting each `4 × n_embed` expert into 4 gives `hidden = n_embed`.
- **Shared:** `self.shared` runs on every token and bypasses the router.
- **Loss-free balancing:** `balance_bias` rises for underloaded experts and falls for overloaded ones, using only the sign of the load violation.

A quick check with 1 shared and 15 routed experts, `top_k = 3`, and a router deliberately skewed toward one expert:

| | Busiest expert | Idlest expert |
|---|---|---|
| First step | 2037 tokens | 16 tokens |
| After 300 steps | 492 tokens | 348 tokens |

The average load is 410 tokens. The bias pulls every expert toward it without any loss term.

### Capacity factor

Not implemented here. To add it: compute `capacity = int(tokens × top_k / num_experts × capacity_factor)`, and in the expert loop keep only the first `capacity` tokens of `flat_mask`. The rest skip that expert.
