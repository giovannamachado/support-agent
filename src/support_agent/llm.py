"""Geradores. `ExtractiveLLM` roda offline; `AnthropicLLM` usa a Messages API (não testado ao vivo aqui)."""
import os
import random

import httpx

from .corpus import Chunk
from .text import content_stems, sentences

SYSTEM = (
    "Você é o suporte da FoodFlow. Responda SOMENTE com base no contexto fornecido, em português, "
    "citando o id do trecho entre colchetes (ex.: [refund#1]). Se o contexto não responder, diga que não sabe."
)


class ExtractiveLLM:
    """Gerador determinístico: seleciona as frases do contexto mais relevantes à pergunta."""

    name = "extractive"

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        q = set(content_stems(question))
        qnums = {t for t in q if t.isdigit()}
        cands = []
        for ci, c in enumerate(contexts):
            for s in sentences(c.text)[1:]:  # [0] é o título da seção
                st = set(content_stems(s))
                score = len(q & st) / (len(q) or 1) + 0.5 * len(qnums & st) - 0.05 * ci
                cands.append((score, s, c.id))
        cands.sort(key=lambda x: -x[0])
        if not cands:
            return ""
        picked = [cands[0]] + [x for x in cands[1:2] if x[0] >= 0.5 * cands[0][0] and x[0] > 0]
        return " ".join(f"{s.rstrip('.')} [{cid}]." for _, s, cid in picked)


class FaultyLLM:
    """Injeta uma afirmação fabricada com probabilidade `p` (teste de falha do guardrail)."""

    FABRICATED = "Todos os reembolsos são pagos em até 2 horas independentemente do meio de pagamento."

    def __init__(self, inner, p: float = 0.5, seed: int = 0):
        self.inner, self.p, self.rng, self.name = inner, p, random.Random(seed), f"faulty({inner.name},p={p})"

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        out = self.inner.generate(question, contexts)
        return f"{out} {self.FABRICATED}" if self.rng.random() < self.p else out


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 30.0):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
        self.key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self.timeout = timeout
        self.last_usage: dict = {}

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        ctx = "\n".join(f"[{c.id}] {c.text}" for c in contexts)
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"},
            json={"model": self.model, "max_tokens": 400, "system": SYSTEM,
                  "messages": [{"role": "user", "content": f"Contexto:\n{ctx}\n\nPergunta: {question}"}]},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        self.last_usage = data.get("usage", {})
        return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
