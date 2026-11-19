# Black Arrow

To-the-point notes and reference on ML performance engineering.

## Sections

| Section | What it covers | Pages |
|---|---|---|
| [LLM Core](./00_llm_core/intro.md) | How an LLM works on the inside | [Journey of a token](./00_llm_core/01_journey_of_token.md), [Attention](./00_llm_core/attention/intro.md) |
| [LLM Inference](./01_llm_inference/intro.md) | How a trained LLM is run efficiently | [Terminologies](./01_llm_inference/01_terminologies.md), [KV cache](./01_llm_inference/02_kv_cache.md), [Mixture of Experts](./01_llm_inference/moe/intro.md) |

## Suggested reading order

1. [Journey of a token](./00_llm_core/01_journey_of_token.md): the whole architecture in one page.
2. [Attention](./00_llm_core/attention/intro.md): the core mechanism, built up step by step.
3. [Terminologies](./01_llm_inference/01_terminologies.md): prefill, decode, TTFT.
4. [Mixture of Experts](./01_llm_inference/moe/intro.md): how large models keep the cost per token low.

Each section's first page lists its pages and the order to read them in.
