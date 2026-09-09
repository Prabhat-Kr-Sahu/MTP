

Based on the latest API validation, here is the updated list of active free models currently available on OpenRouter.

## The Auto-Router (Recommended Fallback)

*   **/model openrouter/free**
    Dynamically routes your prompt to a random available free model that supports the features you need (like tool calling or large context).

## Validated Reasoning & Architecture Models

*   **/model nvidia/nemotron-3-ultra-550b-a55b:free**
    NVIDIA's massive MoE model with a 1M context window; currently the most heavily used free model for long-horizon planning and agent orchestration.
*   **/model nvidia/nemotron-3-super-120b-a12b:free**
    A 120B parameter MoE (12B active) with a 1M context window, excellent for cross-document reasoning.
*   **/model inclusionai/ling-3.0-flash:free**
    A fast, highly intelligent model with a 262K context window; great for daily chat and drafting.

## Validated Coding & Implementation Models

*   **/model poolside/laguna-s-2.1:free**
    Poolside's flagship 118B coding agent; excellent for complex software engineering and tool use with a 262K context window.
*   **/model poolside/laguna-xs-2.1:free**
    A lighter, lower-latency 33B-class version of Laguna for fast refactoring and quick terminal tasks.
*   **/model cohere/north-mini-code:free**
    Cohere's agentic coding model with a 256K context window, specifically trained to generalize across agent harnesses.

## Specialized & Domain-Specific Models

*   **/model inclusionai/ling-3.0-flash-fin:free**
    Specifically tuned for financial logic, quantitative reasoning, and complex multi-step workflows.
*   **/model inclusionai/ling-3.0-flash-sante:free**
    Tuned for medical knowledge reasoning, clinical safety, and high-rigor evidence retrieval.
*   **/model nvidia/nemotron-3.5-lightning:free**
    Designed for ultra-fast, high-throughput agentic workloads and TDD iteration.
*   **/model google/gemma-4-31b-it:free**
    A strong multimodal model supporting vision/video input and 140+ languages.