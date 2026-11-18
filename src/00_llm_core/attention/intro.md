# Attention

**Attention** is the step inside a transformer block where tokens exchange information. It takes one vector per token and returns one vector per token, where each output is a weighted mix of all the tokens that one is allowed to see.

It is the only place in the model where a token learns anything about its neighbours. Everything else (the FFN, the layer norms) works on one token at a time.

```text
X  (tokens × d)        input embeddings: each row knows only its own token
      │
      ▼   attention
      │
Z  (tokens × d)        context vectors: each row also knows about the tokens before it
```

For where this sits in the full model, see [Journey of a token](../01_journey_of_token.md).

## Reading order

Read the pages in order. Each one adds exactly one idea to the previous.

| # | Page | What it adds |
|---|---|---|
| 1 | [Why attention](./01_why_attention.md) | The bottleneck in RNNs, the original attention idea, and why a plain dot product is not enough |
| 2 | [Self-attention](./02_self_attention.md) | Queries, keys, values, scaling, softmax: the full mechanism with a worked example |
| 3 | [Causal attention](./03_causal_attention.md) | The mask that stops a token from seeing the future, plus dropout |
| 4 | [Multi-head attention](./04_multi_head_attention.md) | Several attention patterns in parallel, the tensor reshapes, and the final PyTorch class |

## The whole mechanism in one formula

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
$$

| Piece | What it does | Page |
|---|---|---|
| `Q`, `K`, `V` | the input, projected by three trainable matrices | 2 |
| `Q Kᵀ` | a relevance score for every pair of tokens | 2 |
| `√d_k` | keeps the scores in a range where softmax behaves | 2 |
| `M` | the causal mask: `−∞` for future tokens, `0` elsewhere | 3 |
| softmax | turns each row of scores into weights that sum to 1 | 2 |
| `· V` | mixes the value vectors using those weights | 2 |

Multi-head attention runs this formula several times in parallel on smaller vectors and joins the results (page 4).

## One running example

Pages 2 to 4 reuse the same toy setup, so the numbers carry over from page to page:

- 5 tokens: `the`, `next`, `day`, `is`, `bright`
- input dimension `d_in = 8`, output dimension `d_out = 4`
- weights are random and untrained, so the numbers show the mechanics, not meaning

All the numbers in the worked examples were computed by running the code on these pages.

## Vocabulary

| Term | Meaning |
|---|---|
| Input embedding | A token's vector before attention: token embedding + positional embedding. Knows nothing about other tokens |
| Context vector | A token's vector after attention: a weighted mix of the tokens it attended to |
| Query | The token that is looking for information, after projection by `W_Q` |
| Key | A token being looked at, after projection by `W_K`. Compared against the query |
| Value | What a token contributes to the mix, after projection by `W_V` |
| Attention score | Raw query-key dot product `q · k`. Any real number. Also called alignment score or attention logit |
| Attention weight | Score after scaling and softmax. Between 0 and 1, each row sums to 1 |
| Self-attention | Queries, keys, and values all come from the same sequence |
| Causal (masked) attention | Self-attention where a token sees only itself and earlier tokens |
| Head | One independent attention computation with its own slice of `Q`, `K`, `V` |
| `d_k` | Dimension of a key vector. Equals `head_dim` in multi-head attention |

## Timeline

| Year | Work | Contribution |
|---|---|---|
| 1966 | ELIZA (Weizenbaum) | First well-known chatbot. Pattern matching, no learning |
| 1980s | Recurrent neural networks | A hidden state carries memory from one token to the next |
| 1997 | [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf) (Hochreiter, Schmidhuber) | A separate cell state so memory survives over longer distances |
| 2014 | [Seq2seq](https://arxiv.org/abs/1409.3215) (Sutskever et al.) | Encoder-decoder RNNs for translation |
| 2014 | [Bahdanau attention](https://arxiv.org/abs/1409.0473) (Bahdanau, Cho, Bengio) | The decoder looks back at every encoder state. First attention mechanism |
| 2017 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al.) | The Transformer: drops the RNN, keeps attention. Scaled dot-product and multi-head attention |
| 2018 | [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (Radford et al.) | Decoder-only transformer with causal attention, trained on next-token prediction |
| 2024 | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | Multi-head latent attention (MLA), which shrinks the KV cache |

## Source

These notes follow five lectures of Vizuara's [Build DeepSeek from Scratch](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSiOpKKlHCyOq9lnp-dLvlms) playlist:

1. [The Attention Mechanism 1 hour explanation](https://www.youtube.com/watch?v=K45ze9Yd5UE)
2. [Self Attention Mechanism - Handwritten from scratch](https://www.youtube.com/watch?v=s8mskq-nzec)
3. [Causal Attention Explained: Don't Peek into the Future!](https://www.youtube.com/watch?v=c6Kkj6iLeBg)
4. [Multi-Head Attention Visually Explained](https://www.youtube.com/watch?v=qbN4ulK-bZA)
5. [Multi-Head Attention Handwritten from Scratch](https://www.youtube.com/watch?v=rvsEW-EsD-Y)

Where a lecture is loose or a paper says something more precise, the notes follow the paper and mark it with **Note**.

Further reading:

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) and [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/), Jay Alammar
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/), Lilian Weng
