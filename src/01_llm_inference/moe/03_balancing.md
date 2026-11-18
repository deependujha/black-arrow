# Balancing

Nothing in the [forward pass](./02_forward_pass.md) stops the router from sending almost every token to the same few experts. This page covers the three classic guards against that.

## Why imbalance hurts

Routing feeds on itself. An expert that gets more tokens gets more gradient, becomes better, and so gets chosen even more. Left alone this ends in **routing collapse**: a few experts do everything.

- **Starved experts learn nothing.** Their parameters are dead weight in memory.
- **Favoured experts are overloaded.** They must cover too many kinds of tokens, so they stop being specialists.
- **Compute is uneven.** One expert's batch is huge while others sit idle, which wastes hardware and memory.

The goal: over a batch, every expert should receive a similar share of the tokens.

| Technique | Controls | How |
|---|---|---|
| Auxiliary (importance) loss | sum of routing weights per expert | extra loss term |
| Load balancing loss | number of tokens per expert | extra loss term |
| Capacity factor | maximum tokens per expert | hard cap |

All numbers below use the gating matrix `G` from the previous page:

| Token | E1 | E2 | E3 |
|---|---|---|---|
| `The` | 0 | 0.6 | 0.4 |
| `next` | 0.9 | 0 | 0.1 |
| `day` | 0 | 0.4 | 0.6 |
| `is` | 0.5 | 0 | 0.5 |

## Auxiliary loss

### Expert importance

A column of `G` holds the weights one expert received from every token. Its sum is the **expert importance**.

```text
importance(E1) = 0   + 0.9 + 0   + 0.5 = 1.4
importance(E2) = 0.6 + 0   + 0.4 + 0   = 1.0
importance(E3) = 0.4 + 0.1 + 0.6 + 0.5 = 1.6
```

`E3` matters most, `E2` least. A balanced layer has roughly equal importances.

### Coefficient of variation

To measure how unequal a set of numbers is, use the **coefficient of variation**:

```text
CV = standard deviation / mean
```

- `CV = 0` means all values are equal.
- A larger `CV` means more spread.
- Dividing by the mean makes it independent of scale, so batch size does not matter.

For the example:

```text
mean = (1.4 + 1.0 + 1.6) / 3 = 1.333
std  = 0.249                           (population standard deviation)
CV   = 0.249 / 1.333 = 0.187
```

### The loss

```text
L_aux = λ · CV(importance)²
```

It is added to the language-modelling loss:

```text
L_total = L_next_token + L_aux
```

- `λ` is a hyperparameter that sets how much balance matters.
- Here `CV² = 0.035`, so with `λ = 0.01` the term is `0.00035`.
- Gradient descent now pushes the router toward equal importances.

This loss comes from the 2017 Sparsely-Gated MoE paper.

## Load balancing loss

### Equal importance is not equal load

Importance sums *weights*. It says nothing about how many *tokens* an expert handles.

| Token | E1 | E2 |
|---|---|---|
| 1 | 1.0 | 0.25 |
| 2 | 0 | 0.25 |
| 3 | 0 | 0.25 |
| 4 | 0 | 0.25 |
| **Importance** | **1.0** | **1.0** |
| **Tokens received** | **1** | **4** |

Both experts have importance 1.0, so the auxiliary loss is zero. But `E2` processes four times as many tokens as `E1`: one token routed with high confidence against four routed with low confidence. (The rows do not sum to 1. It is only an illustration.)

Compute and memory follow the token count, so the token count needs its own control.

### Two quantities per expert

**`P_i`: the share of routing probability given to expert `i`.** It is the importance divided by the number of tokens.

```text
P1 = 1.4 / 4 = 0.35
P2 = 1.0 / 4 = 0.25
P3 = 1.6 / 4 = 0.40        (sums to 1)
```

**`f_i`: the fraction of tokens dispatched to expert `i`.** Send each token to its top-weighted expert and count.

```text
The  → E2
next → E1
day  → E3
is   → E1      (tie between E1 and E3, pick one)

f1 = 2/4 = 0.50
f2 = 1/4 = 0.25
f3 = 1/4 = 0.25            (sums to 1)
```

> **Note:** this is the Switch Transformer's definition, and Switch sends each token to a single expert (top-1). With top-k routing, as in the running example and in DeepSeekMoE, count every (token, expert) assignment instead. The example has 8 assignments (4 tokens × 2 experts), split 2, 2, 4, so `f = (0.25, 0.25, 0.50)`. That version shows `E3` as the busiest expert and gives `Σ f·P = 0.35`, so `L_balance = 0.0105`.

### The loss

```text
L_balance = λ · N · Σ_i  f_i · P_i
```

`N` is the number of experts. For the example, with `λ = 0.01`:

```text
Σ f·P     = 0.50·0.35 + 0.25·0.25 + 0.25·0.40 = 0.3375
L_balance = 0.01 × 3 × 0.3375 = 0.0101
```

### Why minimising it balances the load

Compare two extremes with 2 experts and 4 tokens:

| | `f` | `P` | `Σ f·P` |
|---|---|---|---|
| All tokens to `E1` | (1, 0) | (1, 0) | 1.0 |
| Even split | (0.5, 0.5) | (0.5, 0.5) | 0.5 |

The collapsed case scores higher, so the loss punishes it.

In general `f` and `P` rise and fall together, because an expert with high routing probability also wins more tokens. So `Σ f·P` behaves like a sum of squares, which is smallest when every term equals `1/N`:

```text
uniform:   Σ f·P = N · (1/N · 1/N) = 1/N
collapsed: Σ f·P = 1
```

The factor `N` rescales the loss so that perfect balance always gives `λ`, whatever the number of experts.

> **Note:** how the gradient actually works. `f_i` is a count, so it has no gradient. Only `P_i` is differentiable. The gradient of the loss with respect to `P_i` is proportional to `f_i`, so the experts that receive the most tokens have their routing probability pushed down the hardest. This loss is from the Switch Transformer, where `P_i` is the mean of the full router softmax before top-k masking. The lecture computes it from the masked weights, which gives the same intuition.

### Auxiliary loss vs load balancing loss

| | Auxiliary loss | Load balancing loss |
|---|---|---|
| Looks at | importance only | importance *and* token count |
| Formula | `λ · CV(importance)²` | `λ · N · Σ f_i · P_i` |
| Ensures | experts get equal total weight | experts get equal numbers of tokens |
| Misses | uneven token counts | (covers both) |

## Capacity factor

Both losses only *encourage* balance. The capacity factor *enforces* a limit.

```text
expert capacity = (tokens per batch / number of experts) × capacity factor

tokens per batch = batch size × sequence length × top_k
```

- **Expert capacity** is the maximum number of tokens one expert may process in a batch.
- `top_k` is in the token count because each token is sent to `k` experts, so it occupies `k` slots. Switch itself routes top-1, so its formula has no `top_k`. The factor generalises it to top-k routing.

Example: 1000 tokens per batch, 4 experts.

| Capacity factor | Expert capacity | Meaning |
|---|---|---|
| 1.0 | 250 | Exactly a fair share. Any imbalance overflows. |
| 1.25 | 312 | 25% slack for imbalance. |
| 2.0 | 500 | An expert may take double its share. |
| < 1.0 | < 250 | Not enough room for all tokens. Some must be dropped. |

- Tokens beyond an expert's capacity are **dropped**: that expert does not process them. The token still moves on through the residual connection.
- Typical values are between 1 and 2. Higher wastes memory and lets one expert dominate again. Lower drops more tokens.
- The 4-token example has 8 slots over 3 experts, a fair share of 2.67 each. With a cap of 3, no expert could take all four tokens the way `E3` does.

Dropping tokens loses information, so it is a last resort. DeepSeek-V3 states that it drops no tokens, in training or inference.

## What MoE buys once balanced

With 32 to 128 experts, the Switch Transformer reached the quality of its dense baseline up to about 7× faster in training time, at the **same compute per token**. Each token still runs one FFN-sized expert, so a step costs about the same. The gain comes from the extra parameters: the model is much larger, but only a small part of it runs for any one token.

## The catch

Both loss terms are added to the language-modelling loss, so they compete with it. [DeepSeek's answer](./04_deepseek.md) is to balance the load without any loss term.
