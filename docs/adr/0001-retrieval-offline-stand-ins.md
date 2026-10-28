# ADR 0001 — Substitutos offline para embeddings, reranker e LLM

**Contexto.** O repositório precisa rodar e ser avaliado sem GPU, sem rede e sem chave de API (CI e revisão por terceiros).
**Decisão.** BM25 próprio + vetores TF-IDF de n-gramas de caracteres (sklearn) + fusão RRF; reranker lexical (cobertura de termos e bigramas); gerador extrativo determinístico. `AnthropicLLM` existe atrás da mesma interface `generate(question, contexts)`.
**Consequências.** (+) reprodutível, ms de latência, zero custo. (−) o vetor de n-gramas não captura sinonímia; os números **não** representam o que um embedding neural + LLM real alcançariam. A troca exige implementar `similarities()` e `generate()` e re-rodar `make eval`.
