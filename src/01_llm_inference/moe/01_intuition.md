# Intuition

MoE changes exactly one part of the transformer block: the feed-forward network.

## The feed-forward network

Every transformer block has two sub-layers: attention, then a feed-forward network (FFN).

```text
x ─► LayerNorm ─► Attention ─► (+) ─► LayerNorm ─► FFN ─► (+) ─► out
```

The FFN is an **expansion-contraction** network, applied to each token on its own:

```text
d ──(expand)──► 4d ──(activation)──► 4d ──(contract)──► d
```

- **Expand:** project the token from dimension `d` to `4d`.
- **Contract:** project it back to `d`.
- The output has the same shape as the input.

The wide hidden layer gives the model a richer space to work in. Remove the FFN and model quality drops.

### It is most of the block

For `d = 768` (GPT-2 small), counting weights only:

```text
expand    768 × 3072 = 2,359,296
contract  3072 × 768 = 2,359,296
FFN total            = 4,718,592   ≈ 4.7M per block
12 blocks            ≈ 56.6M
```

The attention sub-layer of the same block has `4 × 768² ≈ 2.4M` weights. So the FFN holds about **two thirds** of each block's parameters, and in a dense model every token pays for all of them, in training and in inference.

## From one FFN to many experts

MoE replaces the one FFN with `N` FFNs. Each is an **expert**.

```text
Dense block                     MoE block

   token                           token
     │                               │
     ▼                            ┌──┴───┐
 ┌───────┐                        │router│
 │  FFN  │                        └──┬───┘
 └───┬───┘               ┌──────┬────┴─┬──────┐
     │                   ▼      ▼      ▼      ▼
     ▼                 ┌────┐ ┌────┐ ┌────┐ ┌────┐
   output              │ E1 │ │ E2 │ │ E3 │ │ E4 │
                       └────┘ └─┬──┘ └────┘ └─┬──┘
                         off    │     off     │
                                └──► (+) ◄────┘
                                      │
                                      ▼
                                    output
```

- Every block gets its own set of experts. 12 blocks with 4 experts each means 48 experts.
- A small **router** decides, per token, which experts run.
- The output shape is unchanged, so the rest of the transformer does not notice.

Adding networks sounds like more work, not less. It is less because of sparsity.

## Sparsity

**Sparsity** means each token runs through only `k` of the `N` experts. The rest stay off for that token.

| | Dense | MoE |
|---|---|---|
| Parameters used per token | all | only the `k` active experts |
| FFN compute per token | 1 full FFN | `k` expert FFNs |
| Total parameters | 1 FFN per block | `N` experts per block |

Two ways to read this:

- **Same size, less compute.** For a fixed parameter count, an MoE does far less work per token than a dense model, so it trains and infers faster.
- **Same compute, more capacity.** For a fixed compute budget per token, an MoE can hold many more parameters.

Typical ratios are small. With 64 experts and `k = 2`, a token touches about 3% of the expert parameters. DeepSeek-V3 has 671B parameters and activates about 37B per token.

The 1991 paper already had this shape: several small networks, and a gating network that picks which one handles each input. It was tested on vowel recognition, not language.

> **Note:** sparsity cuts compute, not memory. All `N` experts must still be loaded, because any token may need any expert.

## What experts learn

Nobody assigns roles to experts. Specialisation emerges from training, because the router learns to send similar tokens to the same expert.

The ST-MoE paper (Table 13) inspected which tokens each expert receives:

| Expert specialises in | Example tokens | Layer |
|---|---|---|
| Punctuation | `,` `.` `;` `:` `?` | 2 |
| Conjunctions and articles | `the` `and` `if` | 3, 6 |
| Verbs | `falling` `identified` `struggling` `designed` | 1 |
| Visual descriptions | `blue` `dark` `inner` `upper` `open` | 0 |
| Proper names | `Colin` `Ken` `Sam` `Angel` | 1 |
| Counting and numbers | `7` `25` `4` `54` | 1 |

This is why sparsity costs so little quality. A comma does not need the parameters that handle numbers, so there is no reason to run them.

> **Note:** these examples come from the encoder of an encoder-decoder model. The same paper found decoder experts much less cleanly specialised. Treat the table as intuition, not a guarantee.

## A token's path through the model

Routing happens independently in every block.

```text
Prompt: "What is 1 + 1 ?"        (4 experts per block, k = 1)

token      block 1   block 2   block 3   block 4
"1"        E1        E3        E2        E1
"+"        E2        E2        E2        E3
"?"        E3        E1        E2        E3
```

- The expert index means nothing across blocks. `E1` in block 1 and `E1` in block 2 are different networks with different skills.
- A token can take a different expert in every block.
- Nothing is hard-coded. If numbers go to `E1` in block 1 and `E3` in block 2, that was learned.

Think of MoE as many small networks in *every* block, not one layer swapped once.

## What this means for inference

- **Compute per token** scales with active parameters, so decode steps are cheaper than in a dense model of the same total size.
- **Memory** scales with total parameters. The weights of every expert sit in GPU memory.
- **Batching is harder.** Tokens in the same batch go to different experts, so each expert sees a small, uneven slice of the batch. That is one reason [balancing](./03_balancing.md) matters.

Next: [how the routing is computed](./02_forward_pass.md).
