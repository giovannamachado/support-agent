"""Normalização e tokenização PT-BR (sem dependências externas)."""
import re
import unicodedata

_STOP = set(
    """a o as os de da do das dos em no na nos nas um uma uns umas que e para pra por com se sem ao aos
    mais mas ou meu minha meus minhas seu sua eu voce vc me te ja foi ser sao esta estou como qual quais
    quando onde quanto quanta quantos quantas posso pode podem faco fazer vai tem ter tenho isso esse essa
    este depois antes muito ate la ai so ha sobre entre eh dar da dou deve devo quero preciso""".split()
)
_CITATION = re.compile(r"\[[a-z_]+#\d+\]")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def stem(tok: str) -> str:
    if tok.isdigit() or len(tok) <= 4:
        return tok
    for suf in ("mente", "coes", "cao", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)]
    return tok


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", norm(text))


def content_stems(text: str) -> list[str]:
    return [stem(t) for t in tokens(_CITATION.sub(" ", text)) if t not in _STOP]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def strip_citations(text: str) -> str:
    return _CITATION.sub("", text).strip()
