"""Carrega as políticas em chunks por seção (`## título`)."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc>#<n>"
    doc: str
    title: str
    text: str


def load_chunks(policy_dir: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(policy_dir).glob("*.md")):
        n = 0
        title, body = None, []

        def flush():
            nonlocal n
            if title and body:
                chunks.append(Chunk(f"{path.stem}#{n}", path.stem, title, f"{title}. {' '.join(body)}"))
                n += 1

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                flush()
                title, body = line[3:].strip(), []
            elif line.strip() and not line.startswith("# "):
                body.append(line.strip())
        flush()
    if not chunks:
        raise FileNotFoundError(f"nenhuma política (*.md com seções ##) em {policy_dir}; defina SUPPORT_AGENT_DATA")
    return chunks
