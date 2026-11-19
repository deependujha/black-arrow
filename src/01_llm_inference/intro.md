# LLM Inference

How a trained LLM is run: what a request costs, where the time goes, and the architecture changes that make large models cheaper to serve.

This section assumes the basics from [LLM Core](../00_llm_core/intro.md).

## Pages

| # | Page | What it covers |
|---|---|---|
| 1 | [Terminologies](./01_terminologies.md) | Prefill, decode, TTFT, TPOT |
| 2 | [KV cache](./02_kv_cache.md) | Why generation repeats work, what is cached, the speedup, and the memory cost |
| 3 | [Mixture of Experts](./moe/intro.md) | Replacing the single feed-forward network with many experts, a few active per token, in five pages: |
| 3.1 | [Intuition](./moe/01_intuition.md) | Where MoE sits in the transformer, sparsity, what experts learn |
| 3.2 | [Forward pass](./moe/02_forward_pass.md) | Router, top-k, softmax, weighted sum, with a worked example |
| 3.3 | [Balancing](./moe/03_balancing.md) | Auxiliary loss, load balancing loss, capacity factor |
| 3.4 | [DeepSeek's MoE](./moe/04_deepseek.md) | Loss-free balancing, shared experts, fine-grained experts |
| 3.5 | [Code from scratch](./moe/05_code.md) | The full layer in PyTorch, plus the DeepSeek variant |

## Resources

- [LLM inference lecture playlist](https://www.youtube.com/playlist?list=PL5vCjAyS9QP_Pze76CYAlidc0hpdTXFLV)
