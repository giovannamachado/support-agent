"""Ferramentas do agente sobre uma API de pedidos simulada (JSON)."""
import json
import uuid
from pathlib import Path


class OrderTools:
    def __init__(self, orders_path: str | Path):
        self.orders = json.loads(Path(orders_path).read_text(encoding="utf-8"))
        self.tickets: dict[tuple[str, str], str] = {}

    def get_order_status(self, order_id: str, customer_id: str | None = None) -> dict:
        o = self.orders.get(order_id)
        if o is None:
            return {"error": "not_found"}
        if customer_id is not None and o["customer_id"] != customer_id:
            return {"error": "forbidden"}  # nunca vazar pedido de outro cliente
        return {"order_id": order_id, **{k: v for k, v in o.items() if k != "customer_id"}}

    def open_refund_ticket(self, order_id: str, reason: str, customer_id: str | None = None) -> dict:
        o = self.get_order_status(order_id, customer_id)
        if "error" in o:
            return o
        if reason == "atraso":
            if o["late_minutes"] <= 45:
                return {"error": "not_eligible", "detail": "atraso de no máximo 45 minutos não gera reembolso"}
        elif reason in ("item_faltando", "item_errado", "nao_entregue"):
            if o["status"] != "delivered":
                return {"error": "not_eligible", "detail": "o pedido ainda não foi entregue"}
            if o["delivered_minutes_ago"] > 24 * 60:
                return {"error": "not_eligible", "detail": "o prazo de 24 horas após a entrega já passou"}
        else:
            return {"error": "invalid_reason"}
        key = (order_id, reason)  # idempotente: mesma solicitação devolve o mesmo chamado
        tid = self.tickets.setdefault(key, f"TK-{uuid.uuid4().hex[:8]}")
        return {"ticket_id": tid, "order_id": order_id, "reason": reason}
