# Forward pass

The MoE layer takes a `tokens × d` matrix and returns a `tokens × d` matrix, the same contract as a plain FFN. This page shows what happens in between.

## Setup

- 4 tokens: `The`, `next`, `day`, `is`
- embedding dimension `d = 8`
- 3 experts: `E1`, `E2`, `E3`
- `top_k = 2`

The input `X` is a `4 × 8` matrix: one row per token. It is whatever the attention sub-layer and the layer norm produced.

## The problem

Run `X` through all three experts and you get three `4 × 8` matrices:

```text
X (4×8) ─► E1 ─► O1 (4×8)
X (4×8) ─► E2 ─► O2 (4×8)
X (4×8) ─► E3 ─► O3 (4×8)
```

Row `t` of each `O` is that expert's output for token `t`.

The layer must return **one** `4 × 8` matrix. So for every token there are two questions:

1. **Which** experts should handle it?
2. **How much weight** does each chosen expert get?

One small matrix answers both.

## The pipeline

```text
X (4×8)
  │  × W_r (8×3)                     router
  ▼
logits (4×3)                         one score per (token, expert)
  │  keep top-k per row, rest → −∞   sparsity
  ▼
masked logits (4×3)
  │  softmax per row
  ▼
G (4×3)                              gating weights, each row sums to 1
  │
  ▼
Y[t] = Σ over chosen experts  G[t, i] × O_i[t]        Y is 4×8
```

### 1. Router

The router is a trainable matrix `W_r` of shape `d × N`, here `8 × 3`.

```text
logits = X · W_r        (4×8) · (8×3) = (4×3)
```

- Row `t` belongs to token `t`.
- Column `i` belongs to expert `i`.
- `logits[t, i]` is how well expert `i` suits token `t`.

The lectures call this matrix the **expert selector matrix**. Code usually calls it `logits`.

### 2. Top-k

For each row, keep the `k` largest values. Those are the token's experts.

This step answers *which experts*. It is where sparsity happens.

### 3. Mask and softmax

Replace every value that was not kept with `−∞`, then apply softmax across each row.

```text
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

- `exp(−∞) = 0`, so dropped experts get weight exactly 0.
- The kept values become positive and sum to 1.

The result is the gating matrix `G`, the **expert selector weight matrix** in the lectures. This step answers *how much weight*.

### 4. Weighted sum

For each token, take its row from each chosen expert's output, scale by the gate weight, and add:

```text
Y[t] = Σ_i  G[t, i] · O_i[t]
```

Stack the rows and you have the `4 × 8` output.

## Worked example

Suppose masking and softmax produce this gating matrix. The rest of the MoE pages reuse it.

| Token | E1 | E2 | E3 |
|---|---|---|---|
| `The` | 0 | 0.6 | 0.4 |
| `next` | 0.9 | 0 | 0.1 |
| `day` | 0 | 0.4 | 0.6 |
| `is` | 0.5 | 0 | 0.5 |

Read a **row** to see how one token is handled:

- `The` goes to `E2` and `E3`, with 60% and 40% weight.
- `next` goes to `E1` and `E3`, with 90% and 10% weight.

The outputs follow directly:

```text
Y[The]  = 0.6 · O2[The]  + 0.4 · O3[The]
Y[next] = 0.9 · O1[next] + 0.1 · O3[next]
Y[day]  = 0.4 · O2[day]  + 0.6 · O3[day]
Y[is]   = 0.5 · O1[is]   + 0.5 · O3[is]
```

Each `Y[t]` is a vector of 8 numbers. Together they form the `4 × 8` output.

Read a **column** to see how one expert is used across the batch. `E3` receives all four tokens, `E1` and `E2` receive two each. That imbalance is the subject of the [next page](./03_balancing.md).

## Things to notice

- **Skipped experts never run.** The explanation above computes `O1`, `O2`, `O3` in full for clarity. A real implementation runs each expert only on the tokens routed to it. Otherwise there would be no saving. See the [code](./05_code.md#the-sparse-moe-layer).
- **The router is cheap.** It is one `d × N` matrix, tiny next to the experts.
- **Top-k is not differentiable, the weights are.** Gradients flow through the softmax weights of the chosen experts into `W_r`, and through the chosen experts themselves. An expert that is not chosen gets no gradient from that token.
- **Routing is per token.** Two tokens in the same sentence can use completely different experts.

> **Note:** the second lecture calls the top-k step "load balancing". It is better called sparse routing. Load balancing means keeping the experts evenly used, which top-k does nothing to ensure.

## Summary

| Step | Operation | Shape | Answers |
|---|---|---|---|
| Router | `X · W_r` | `4×3` | scores every expert |
| Top-k | keep `k` largest per row | `4×3` | which experts |
| Softmax | mask to `−∞`, normalise | `4×3` | how much weight |
| Combine | weighted sum of expert outputs | `4×8` | the layer output |
