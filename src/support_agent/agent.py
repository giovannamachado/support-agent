"""Orquestração: roteamento de intenção → RAG ou ferramentas → guardrail → resposta com citações."""
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .corpus import load_chunks
from .guardrail import apply_guardrail
from .llm import ExtractiveLLM
from .retrieval import Retriever
from .tools import OrderTools

log = logging.getLogger("support_agent")
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.getenv("SUPPORT_AGENT_DATA", ROOT / "data"))  # em container o pacote vive em site-packages: use a variável
REFUSAL = "Não encontrei essa informação nas políticas de suporte. Posso ajudar com pedidos, pagamentos, reembolsos e cupons."
ORDER_ID = re.compile(r"\bFF-\d{4}\b", re.I)
REFUND_INTENT = re.compile(r"reembols|estorn|ressarc|dinheiro de volta|devolv", re.I)
REASONS = [
    ("nao_entregue", re.compile(r"n[aã]o (chegou|recebi|entregue)", re.I)),
    ("item_faltando", re.compile(r"falt|sem (um|o) item", re.I)),
    ("item_errado", re.compile(r"errad", re.I)),
    ("atraso", re.compile(r"atras|demor", re.I)),
]


@dataclass
class AgentResponse:
    answer: str
    intent: str
    citations: list[str] = field(default_factory=list)
    refused: bool = False
    trace: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0


class SupportAgent:
    def __init__(self, llm=None, mode: str = "hybrid_rerank", guardrail: bool = True, tau: float = 0.45,
                 policy_dir=None, orders_path=None):
        self.chunks = load_chunks(policy_dir or DATA / "policies")
        self.retriever = Retriever(self.chunks, mode)
        self.llm = llm or ExtractiveLLM()
        self.guardrail, self.tau = guardrail, tau
        self.tools = OrderTools(orders_path or DATA / "orders.json")

    # -- helpers
    def _step(self, trace, name, t0, **meta):
        trace.append({"step": name, "ms": round((time.perf_counter() - t0) * 1000, 2), **meta})

    def _policy_answer(self, question: str, trace: list) -> AgentResponse:
        t = time.perf_counter()
        hits = self.retriever.search(question, k=3)
        self._step(trace, "retrieve", t, top=[h.chunk.id for h in hits], confidence=round(hits[0].confidence, 3))
        if not hits or hits[0].confidence < self.tau:
            return AgentResponse(REFUSAL, "policy", refused=True, trace=trace)
        ctx = [h.chunk for h in hits]
        t = time.perf_counter()
        draft = self.llm.generate(question, ctx)
        self._step(trace, "generate", t, llm=self.llm.name)
        dropped: list[str] = []
        if self.guardrail:
            t = time.perf_counter()
            g = apply_guardrail(draft, ctx)
            draft, dropped = g.text, g.dropped
            self._step(trace, "guardrail", t, dropped=len(dropped))
        if not draft.strip():
            return AgentResponse(REFUSAL, "policy", refused=True, trace=trace)
        cites = sorted(set(re.findall(r"\[([a-z_]+#\d+)\]", draft)))
        usage = {"input_tokens": len(question + " ".join(c.text for c in ctx)) // 4, "output_tokens": len(draft) // 4, "estimated": True}
        return AgentResponse(draft, "policy", citations=cites, trace=trace, usage=usage)

    def _order_answer(self, message: str, order_id: str, customer_id: str | None, trace: list) -> AgentResponse:
        t = time.perf_counter()
        order = self.tools.get_order_status(order_id.upper(), customer_id)
        self._step(trace, "tool:get_order_status", t, order_id=order_id.upper(), ok="error" not in order)
        if order.get("error") in ("not_found", "forbidden"):
            # mesma mensagem para os dois casos: não revelar a existência de pedidos alheios
            return AgentResponse(f"Não encontrei o pedido {order_id.upper()} na sua conta.", "order", trace=trace)
        if REFUND_INTENT.search(message):
            reason = next((r for r, rx in REASONS if rx.search(message)), None)
            if reason is None:
                return AgentResponse("Qual foi o problema com o pedido (item faltando, item errado, atraso ou não entregue)?", "order", trace=trace)
            t = time.perf_counter()
            res = self.tools.open_refund_ticket(order_id.upper(), reason, customer_id)
            self._step(trace, "tool:open_refund_ticket", t, reason=reason, ok="error" not in res)
            if "ticket_id" in res:
                return AgentResponse(f"Abri o chamado {res['ticket_id']} para o pedido {order_id.upper()}. A análise leva até 30 minutos.", "order", citations=["refund#2"], trace=trace)
            return AgentResponse(f"Não consegui abrir o reembolso: {res.get('detail', 'motivo inválido')}.", "order", trace=trace)
        status = {"delivered": "foi entregue", "on_the_way": "está a caminho", "preparing": "está em preparo"}.get(order["status"], order["status"])
        late = f" Está {order['late_minutes']} minutos atrasado." if order["late_minutes"] else ""
        return AgentResponse(f"O pedido {order_id.upper()} {status}.{late}", "order", trace=trace)

    def respond(self, message: str, customer_id: str | None = None) -> AgentResponse:
        t0, trace = time.perf_counter(), []
        m = ORDER_ID.search(message)
        if m:
            r = self._order_answer(message, m.group(0), customer_id, trace)
        elif re.search(r"meu pedido|n[uú]mero do pedido", message, re.I) and re.search(r"status|onde|cad[eê]|reembols|cancel", message, re.I) and not re.search(r"prazo|quanto tempo|quando", message, re.I):
            r = AgentResponse("Pode me informar o número do pedido (ex.: FF-1001)?", "clarify", trace=trace)
        else:
            r = self._policy_answer(message, trace)
        r.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        if os.getenv("TRACE_LOG"):
            log.info(json.dumps({"intent": r.intent, "refused": r.refused, "trace": r.trace, "latency_ms": r.latency_ms}, ensure_ascii=False))
        return r
