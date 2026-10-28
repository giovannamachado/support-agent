"""Guardrail de groundedness: remove frases do rascunho sem suporte no contexto recuperado."""
import re
from dataclasses import dataclass

from .corpus import Chunk
from .text import content_stems, sentences, strip_citations

_NUM = re.compile(r"\d+")


@dataclass
class GuardResult:
    text: str
    dropped: list[str]


def supported(sentence: str, context_stems: set[str], threshold: float = 0.7) -> bool:
    st = content_stems(sentence)
    if not st:
        return True
    nums = [t for t in st if t.isdigit()]
    if any(n not in context_stems for n in nums):
        return False
    return sum(t in context_stems for t in st) / len(st) >= threshold


def apply_guardrail(draft: str, contexts: list[Chunk], threshold: float = 0.7) -> GuardResult:
    ctx = set()
    for c in contexts:
        ctx.update(content_stems(c.text))
    kept, dropped = [], []
    for s in sentences(draft):
        (kept if supported(strip_citations(s), ctx, threshold) else dropped).append(s)
    return GuardResult(" ".join(kept), dropped)
