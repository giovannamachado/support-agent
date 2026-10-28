import pytest
from fastapi.testclient import TestClient

from support_agent.agent import REFUSAL, SupportAgent
from support_agent.api import create_app
from support_agent.corpus import load_chunks
from support_agent.guardrail import apply_guardrail
from support_agent.llm import ExtractiveLLM, FaultyLLM
from support_agent.retrieval import MODES, Retriever
from support_agent.text import content_stems, norm


@pytest.fixture(scope="module")
def chunks():
    return load_chunks("data/policies")


def test_chunks_are_sections(chunks):
    assert len(chunks) == 26 and chunks[0].id.endswith("#0")
    assert all(c.text.startswith(c.title) for c in chunks)


def test_norm_and_stems():
    assert norm("Reembolsó") == "reembolso"
    assert "ate" not in content_stems("até 24 horas") and "24" in content_stems("até 24 horas")


@pytest.mark.parametrize("mode", MODES)
def test_retrieval_finds_pix_chunk(chunks, mode):
    top = Retriever(chunks, mode).search("Quanto tempo leva o reembolso via Pix?", k=3)
    assert "refund#1" in [h.chunk.id for h in top]


def test_rejects_unknown_mode(chunks):
    with pytest.raises(ValueError):
        Retriever(chunks, "nope")


def test_guardrail_drops_unsupported_and_wrong_numbers(chunks):
    ctx = [c for c in chunks if c.id == "refund#1"]
    ok = "O reembolso no cartão de crédito leva até 7 dias úteis. [refund#1]"
    bad_num = "O reembolso no cartão de crédito leva até 3 dias úteis."
    invented = "Todos os reembolsos são pagos em até 2 horas independentemente do meio de pagamento."
    g = apply_guardrail(f"{ok} {bad_num} {invented}", ctx)
    assert "7 dias" in g.text and "3 dias" not in g.text and "2 horas" not in g.text
    assert len(g.dropped) == 2


def test_policy_answer_is_grounded_and_cited():
    r = SupportAgent(tau=0.22).respond("Em quanto tempo o reembolso via Pix cai na conta?")
    assert not r.refused and "2 dias úteis" in r.answer
    assert "refund#1" in r.citations


def test_out_of_scope_is_refused():
    r = SupportAgent(tau=0.22).respond("Qual a capital da França?")
    assert r.refused and r.answer == REFUSAL


def test_guardrail_blocks_injected_fabrication():
    q = "Quanto tempo demora o reembolso no cartão de crédito?"
    leaky = SupportAgent(llm=FaultyLLM(ExtractiveLLM(), 1.0), guardrail=False, tau=0.22).respond(q)
    safe = SupportAgent(llm=FaultyLLM(ExtractiveLLM(), 1.0), guardrail=True, tau=0.22).respond(q)
    assert FaultyLLM.FABRICATED in leaky.answer and FaultyLLM.FABRICATED not in safe.answer
    assert any(s["step"] == "guardrail" and s["dropped"] == 1 for s in safe.trace)


def test_order_status_tool():
    r = SupportAgent().respond("Cadê o pedido FF-1002?", "c1")
    assert r.intent == "order" and "a caminho" in r.answer and "52 minutos" in r.answer


def test_order_of_another_customer_is_not_leaked():
    a = SupportAgent()
    other = a.respond("Cadê o pedido FF-1002?", "c2")
    missing = a.respond("Cadê o pedido FF-9999?", "c2")
    assert "a caminho" not in other.answer
    assert other.answer.replace("FF-1002", "X") == missing.answer.replace("FF-9999", "X")  # indistinguível de "não existe"


def test_refund_ticket_is_idempotent_and_checks_eligibility():
    a = SupportAgent()
    first = a.respond("Quero reembolso do pedido FF-1001, veio faltando um item", "c1")
    again = a.respond("Quero reembolso do pedido FF-1001, veio faltando um item", "c1")
    assert first.answer == again.answer and "TK-" in first.answer
    late = a.respond("Quero reembolso do pedido FF-1003, veio faltando um item", "c2")  # entregue há 50h
    assert "TK-" not in late.answer and "24 horas" in late.answer
    notlate = a.respond("Quero reembolso por atraso do pedido FF-1001", "c1")
    assert "TK-" not in notlate.answer


def test_missing_order_number_asks_clarification():
    assert SupportAgent().respond("Cadê meu pedido?").intent == "clarify"


def test_api_roundtrip_and_validation():
    c = TestClient(create_app(SupportAgent(tau=0.22)))
    assert c.get("/healthz").json() == {"status": "ok"}
    body = c.post("/chat", json={"message": "Quanto tempo vale o link para redefinir a senha?"}).json()
    assert "1 hora" in body["answer"] and body["trace"][0]["step"] == "retrieve"
    assert c.post("/chat", json={"message": ""}).status_code == 422
    assert c.post("/chat", json={"message": "x" * 1001}).status_code == 422


def test_empty_policy_dir_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="SUPPORT_AGENT_DATA"):
        load_chunks(tmp_path)
