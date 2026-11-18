# Mixture of Experts

A **Mixture of Experts (MoE)** layer replaces the single feed-forward network in a transformer block with many smaller ones, called **experts**, and sends each token through only a few of them.

The model holds many parameters but uses a small fraction of them per token. That is the whole point: capacity grows with the number of experts, compute per token grows only with the number of *active* experts.

## Reading order

Read the pages in order. Each one builds on the previous.

| # | Page | What it covers |
|---|---|---|
| 1 | [Intuition](./01_intuition.md) | Where MoE sits in the transformer, sparsity, what experts learn |
| 2 | [Forward pass](./02_forward_pass.md) | Router, top-k, softmax, weighted sum, with a worked example |
| 3 | [Balancing](./03_balancing.md) | Auxiliary loss, load balancing loss, capacity factor |
| 4 | [DeepSeek's MoE](./04_deepseek.md) | Loss-free balancing, shared experts, fine-grained experts |
| 5 | [Code from scratch](./05_code.md) | The full layer in PyTorch, plus the DeepSeek variant |

## One running example

Pages 2 to 4 reuse the same toy setup, so the numbers carry over:

- 4 tokens: `The`, `next`, `day`, `is`
- embedding dimension 8
- 3 experts, 2 active per token (`top_k = 2`)

## Vocabulary

| Term | Meaning |
|---|---|
| Expert | One feed-forward network inside the MoE layer |
| Router (gate) | A linear layer that scores every expert for a token |
| Top-k | Number of experts each token is sent to |
| Sparsity | Only `k` of `N` experts run for a token |
| Dense model | A model with no MoE: every parameter runs for every token |
| Expert importance | Sum of the routing weights an expert receives over a batch |
| Expert load | Number of tokens an expert receives over a batch |
| Routed expert | An expert chosen by the router (the normal kind) |
| Shared expert | An expert that runs for every token, with no routing |

## Timeline

| Year | Work | Contribution |
|---|---|---|
| 1991 | [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf) (Jacobs, Jordan, Nowlan, Hinton) | The idea: several small networks plus a gating network that picks among them |
| 2017 | [Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538) (Shazeer et al.) | Noisy top-k gating, importance loss |
| 2020 | [GShard](https://arxiv.org/abs/2006.16668) (Lepikhin et al.) | MoE inside the transformer FFN at scale, top-2 routing, expert capacity |
| 2021 | [Switch Transformer](https://arxiv.org/abs/2101.03961) | Simplifies routing to top-1, load balancing loss, capacity factor |
| 2022 | [ST-MoE](https://arxiv.org/abs/2202.08906) | Stable training, evidence of what experts specialise in |
| 2024 Jan | [DeepSeekMoE](https://arxiv.org/abs/2401.06066) | Fine-grained experts, shared experts |
| 2024 May | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | Uses the DeepSeekMoE architecture |
| 2024 Dec | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | Adds auxiliary-loss-free load balancing |

MoE is now standard in large models: Mixtral, DeepSeek-V2 and V3, and Llama 4 all use it.

## Source

These notes follow the five MoE lectures of Vizuara's [Build DeepSeek from Scratch](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSiOpKKlHCyOq9lnp-dLvlms) playlist:

1. [Mixture of Experts (MoE) Introduction](https://www.youtube.com/watch?v=v7U21meXd6Y)
2. [Mixture of Experts Hands on Demonstration](https://www.youtube.com/watch?v=yw6fpYPJ7PI)
3. [Mixture of Experts Balancing Techniques](https://www.youtube.com/watch?v=nRadcspta_8)
4. [How DeepSeek rewrote Mixture of Experts](https://www.youtube.com/watch?v=KnSIZ83iPKs)
5. [Code Mixture of Experts from Scratch in Python](https://www.youtube.com/watch?v=W7ktPe1HfZs)

Where a lecture is loose or a paper says something more precise, the notes follow the paper and mark it with **Note**.
