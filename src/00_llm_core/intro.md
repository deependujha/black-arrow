# LLM Core

How a large language model works on the inside: what happens to text between the prompt going in and the next token coming out.

Everything here uses GPT-2 small as the reference model, because it is small enough to follow by hand and every modern LLM is a variation on it.

## Pages

Read them in order.

| # | Page | What it covers |
|---|---|---|
| 1 | [Journey of a token](./01_journey_of_token.md) | The whole architecture end to end: tokenization, embeddings, the transformer block, the output projection, generation, training, and where the parameters live |
| 2 | [Attention](./attention/intro.md) | The one step where tokens exchange information, in four pages: |
| 2.1 | [Why attention](./attention/01_why_attention.md) | The RNN bottleneck, the original attention idea, why a plain dot product is not enough |
| 2.2 | [Self-attention](./attention/02_self_attention.md) | Queries, keys, values, scaling, softmax, with a worked example |
| 2.3 | [Causal attention](./attention/03_causal_attention.md) | The mask that hides future tokens, and dropout |
| 2.4 | [Multi-head attention](./attention/04_multi_head_attention.md) | Several attention patterns in parallel, the tensor reshapes, the final PyTorch class |

## Where to go next

[LLM Inference](../01_llm_inference/intro.md) builds on this section: how the model is run efficiently once it is trained.
