# DeepSeek's MoE

DeepSeek did not invent MoE. It found three weaknesses in the standard design and fixed each one.

| Innovation | Fixes | Introduced in |
|---|---|---|
| Fine-grained expert segmentation | Knowledge hybridity | DeepSeekMoE (Jan 2024) |
| Shared expert isolation | Knowledge redundancy | DeepSeekMoE (Jan 2024) |
| Auxiliary-loss-free load balancing | Balance loss hurting model quality | DeepSeek-V3 (Dec 2024) |

DeepSeek-V2 uses the first two. DeepSeek-V3 uses all three.

## Two problems with few experts

Older MoE models used a small number of large experts, typically 8 or 16 per layer. The DeepSeekMoE paper names two consequences.

**Knowledge hybridity.** With few experts, each one receives very different kinds of tokens. It has to pack unrelated knowledge into one set of parameters, so it becomes a generalist instead of a specialist.

**Knowledge redundancy.** Tokens sent to different experts often need the same common knowledge, such as basic grammar. Every expert ends up learning its own copy, which wastes parameters.

Both work against the goal in the paper's title: *ultimate expert specialization*.

## Fine-grained expert segmentation

**Fix for hybridity: use many small experts instead of a few large ones.**

Split each expert into `m` smaller ones:

- hidden dimension of each expert shrinks to `1/m`
- number of experts grows from `N` to `m·N`
- number of active experts grows from `K` to `m·K`

```text
Before:  N = 16 experts,  K = 2 active,  hidden size h
After:   m = 4
         64 experts,      8 active,      hidden size h/4
```

Total parameters and compute per token stay the same: `m` times as many experts, each `1/m` the size.

What changes is how finely knowledge can be divided:

- More experts means each one can cover a narrower slice of knowledge.
- A token is served by a *combination* of small experts, and there are far more combinations to choose from.

```text
16 choose 2 =           120 combinations
64 choose 8 = 4,426,165,368 combinations
```

## Shared expert isolation

**Fix for redundancy: set aside a few experts that every token uses.**

Experts are split into two groups:

| | Shared experts | Routed experts |
|---|---|---|
| Active for | every token | only tokens routed to them |
| Selected by | nothing, always on | router, top-k |
| Sparse | no | yes |
| Learn | common knowledge | specialised knowledge |

```text
                 token
                   │
        ┌──────────┴──────────┐
        │                     ▼
        │                  router
        │             (top-k of routed)
        ▼                     │
   ┌─────────┐     ┌──────┬───┴──┬──────┐
   │ shared  │     ▼      ▼      ▼      ▼
   │ expert  │   ┌────┐ ┌────┐ ┌────┐ ┌────┐
   └────┬────┘   │ R1 │ │ R2 │ │ R3 │ │ R4 │ ...
        │        └────┘ └─┬──┘ └────┘ └─┬──┘
        │          off    │     off     │
        └────────► (+) ◄──┴─────────────┘
                    │
                    ▼
                  output
```

Because the shared experts hold the common knowledge, the routed experts do not need their own copies and are free to specialise.

The output is a plain sum: shared experts unweighted, routed experts weighted by their gates, plus the residual.

```text
h_t = Σ_{i ∈ shared} FFN_i(u_t)  +  Σ_{i ∈ routed} g_{i,t} · FFN_i(u_t)  +  u_t
```

To keep compute constant, each shared expert replaces one routed slot. With `K_s` shared experts, the router picks `m·K − K_s` routed experts.

### The three designs side by side

All three have the same expert parameters and the same compute per token.

| Design | Experts | Active per token |
|---|---|---|
| Conventional top-2 | 16 routed | 2 routed |
| + fine-grained | 64 routed | 8 routed |
| + shared (DeepSeekMoE) | 1 shared + 63 routed | 1 shared + 7 routed |

## Auxiliary-loss-free load balancing

### The problem with a balance loss

The [load balancing loss](./03_balancing.md#load-balancing-loss) is added to the training loss:

```text
L_total = L_next_token + λ · N · Σ f_i · P_i
```

The two terms want different things. One measures how well the model predicts the next token. The other measures how evenly tokens are spread. Their gradients interfere.

| `λ` | Effect |
|---|---|
| Too small | Balance term is ignored. Experts are used unevenly, routing may collapse. |
| Too large | Balance term dominates. Next-token prediction gets worse. |

There is no good setting, only a trade-off: better balance costs model quality.

### The idea

Drop the loss term. Balance the load by nudging the router's scores directly, with a per-expert **bias** that is not trained by gradient descent.

### The mechanism

Use the same gating matrix as before:

| Token | E1 | E2 | E3 |
|---|---|---|---|
| `The` | 0 | 0.6 | 0.4 |
| `next` | 0.9 | 0 | 0.1 |
| `day` | 0 | 0.4 | 0.6 |
| `is` | 0.5 | 0 | 0.5 |

**1. Count the load of each expert**, the number of tokens routed to it.

```text
load(E1) = 2      load(E2) = 2      load(E3) = 4
```

**2. Compute the average load.**

```text
average = (2 + 2 + 4) / 3 = 2.67
```

The total is 8 because 4 tokens each go to 2 experts.

**3. Compute the load violation** of each expert: average load minus its own load.

```text
E1: 2.67 − 2 = +0.67    underloaded
E2: 2.67 − 2 = +0.67    underloaded
E3: 2.67 − 4 = −1.33    overloaded
```

**4. Update each expert's bias.** Every bias starts at 0.

```text
b_i ← b_i + u · sign(violation_i)
```

`u` is a small constant, the bias update speed. Only the sign of the violation is used, so every update is a step of fixed size.

```text
E1: b1 ← b1 + u      underloaded, raise
E2: b2 ← b2 + u      underloaded, raise
E3: b3 ← b3 − u      overloaded,  lower
```

**5. Add the bias to the router scores** on the next step, before top-k.

```text
scores[t, i] + b_i    →    top-k    →    selected experts
```

- `E3`'s scores drop, so it wins top-k less often and receives fewer tokens.
- `E1` and `E2`'s scores rise, so they receive more.

This repeats every training step. It is a feedback loop: whichever expert is overloaded is pushed down a little, until loads even out.

### Why it works better

- No balance term in the loss, so **no interfering gradients**. The loss is pure next-token prediction.
- Balance is still enforced, by the bias.

DeepSeek reported both better model quality *and* better load balance than with an auxiliary loss, where previously one had to be traded for the other.

> **Note:** details from the V3 paper that the lecture skips.
>
> - The bias is used **only to select** experts. The gate weight that multiplies an expert's output comes from the original score, without the bias.
> - V3 computes scores with a **sigmoid** per expert, then normalises over the selected experts. V2 used a softmax.
> - V3 still keeps a tiny *sequence-wise* balance loss (`α = 0.0001`) to prevent extreme imbalance inside a single sequence. So it is nearly, not completely, loss-free.
> - The method was first published separately as [Loss-Free Balancing](https://arxiv.org/abs/2410.19133) (Oct 2024).

## Results

### Fewer active parameters, same quality

From the DeepSeekMoE paper, at the 2B-parameter scale (GShard ×1.5 is deliberately 1.5× larger):

| Model | Experts | Active experts | Expert params | Active expert params |
|---|---|---|---|---|
| Dense ×16 | 16 | 16 (all) | 1.89B | 1.89B |
| GShard ×1.5 | 16 routed | 2 | 2.83B | 0.35B |
| DeepSeekMoE | 1 shared + 63 routed | 1 shared + 7 routed | 1.89B | 0.24B |

- DeepSeekMoE matches GShard ×1.5, which uses 1.5× the expert parameters and compute.
- It comes close to Dense ×16, which runs all 16 experts for every token (about 8× the active expert parameters). That dense model is the upper bound for an MoE of this size.

### Each innovation helps

The paper's ablation adds one change at a time, at equal parameters and compute:

| Configuration | Adds | Result |
|---|---|---|
| 0 shared + 2 of 16 routed | baseline (GShard) | lowest |
| 1 shared + 1 of 15 routed | shared expert | better |
| 1 shared + 3 of 31 routed | fine-grained | better again |
| 1 shared + 7 of 63 routed | finer still | best on every benchmark |

Easy to mix up:

- **Shared experts** solve **redundancy**. Common knowledge lives in one place.
- **Fine-grained experts** solve **hybridity**. Each expert covers less ground.

## Configurations in the released models

From the papers, not the lectures:

| | DeepSeek-V2 | DeepSeek-V3 |
|---|---|---|
| Shared experts per layer | 2 | 1 |
| Routed experts per layer | 160 | 256 |
| Routed experts active per token | 6 | 8 |
| Total parameters | 236B | 671B |
| Active parameters per token | 21B | 37B |
| Load balancing | auxiliary losses | bias-based, loss-free |

Both models activate under 10% of their parameters per token.

Next: [all of this in code](./05_code.md).
