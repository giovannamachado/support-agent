# Resultados de avaliação

_Gerado por `evals/run_eval.py`. Dataset: 50 perguntas escritas à mão (ver `evals/dataset.jsonl`); métricas no split `test`._

## Recuperação em nível de chunk (perguntas em escopo, split test)

| Modo | recall@1 | recall@3 | MRR | IC95% MRR |
|---|---|---|---|---|
| bm25 | 0.944 | 0.944 | 0.958 | (0.875, 1.0) |
| dense | 0.833 | 0.944 | 0.889 | (0.759, 1.0) |
| hybrid | 0.889 | 0.944 | 0.921 | (0.801, 1.0) |
| hybrid_rerank | 0.889 | 1 | 0.935 | (0.833, 1.0) |

## Ponta a ponta (tau=0.22 escolhido no dev, acurácia dev=0.962)

| Configuração | resposta (em escopo) | must-include | citação correta | recusa (fora de escopo) | vazamento fabricado | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| clean | 1.0 | 0.889 | 1.0 | 0.833 | 0.0 | 1.08 | 1.42 |
| fault_injection_no_guardrail | 1.0 | 0.889 | 1.0 | 0.833 | 0.417 | 0.94 | 1.25 |
| fault_injection_with_guardrail | 1.0 | 0.889 | 1.0 | 0.833 | 0.0 | 1.09 | 1.36 |

**Como ler:** `fault_injection_*` usa um gerador que injeta uma afirmação fabricada em 50% das respostas; ‘vazamento fabricado’ é a fração de respostas finais que a contém. Mede o guardrail sob falha controlada, não a alucinação de um LLM real.
