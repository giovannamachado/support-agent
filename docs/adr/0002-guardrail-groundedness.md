# ADR 0002 — Guardrail de groundedness por sentença

**Decisão.** Cada frase do rascunho precisa ter ≥70% dos termos de conteúdo presentes no contexto recuperado e **todos** os números presentes (números inventados são o erro mais caro em suporte). Frases sem suporte são removidas; se nada sobra, o agente recusa.
**Alternativas.** LLM-as-judge (mais flexível, mas custo/latência por resposta e outro modelo que pode errar); NLI (exige modelo).
**Limites.** Sobreposição lexical não detecta negação invertida nem recombinação de fatos verdadeiros em afirmação falsa. Mitigação planejada: juiz NLI/LLM como segunda camada.
