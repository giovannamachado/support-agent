"""Avaliação: recuperação (recall@k, MRR), ponta a ponta e injeção de falha no guardrail.

Uso: python evals/run_eval.py [--check]   (--check falha se alguma métrica cair abaixo de evals/thresholds.json)
O limiar de recusa (tau) é escolhido no split `dev` e todas as métricas são reportadas no split `test`.
"""
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.agent import SupportAgent  # noqa: E402
from support_agent.corpus import load_chunks  # noqa: E402
from support_agent.llm import ExtractiveLLM, FaultyLLM  # noqa: E402
from support_agent.retrieval import MODES, Retriever  # noqa: E402
from support_agent.text import norm  # noqa: E402


def load(split):
    rows = [json.loads(line) for line in (ROOT / "evals" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r["split"] == split]


def bootstrap(values, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(n))
    return round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3)


def retrieval_metrics(mode, rows, chunks):
    """Nível de chunk: o chunk de referência é o que contém `must_include` (rótulo mais estrito que o documento)."""
    r = Retriever(chunks, mode)
    r1, r3, rr = [], [], []
    for row in rows:
        gold = next(c.id for c in chunks if c.doc == row["gold_doc"] and norm(row["must_include"]) in norm(c.text))
        ids = [h.chunk.id for h in r.search(row["question"], k=len(chunks))]
        rank = ids.index(gold) + 1 if gold in ids else 0
        r1.append(rank == 1); r3.append(0 < rank <= 3); rr.append(1 / rank if rank else 0.0)
    return {"recall@1": round(statistics.mean(r1), 3), "recall@3": round(statistics.mean(r3), 3), "mrr": round(statistics.mean(rr), 3),
            "mrr_ci95": bootstrap(rr), "n": len(rows)}


def tune_tau(rows):
    best = []
    for tau in [x / 100 for x in range(10, 91, 2)]:
        a = SupportAgent(tau=tau)
        ok = sum((not a.respond(r["question"]).refused) == (r["type"] == "in_scope") for r in rows)
        best.append((ok / len(rows), tau))
    top = max(s for s, _ in best)
    taus = [t for s, t in best if s == top]
    return round(statistics.median(taus), 2), top


def end_to_end(agent, rows):
    ins = [r for r in rows if r["type"] == "in_scope"]
    oos = [r for r in rows if r["type"] == "oos"]
    answered = includes = cited = leaks = 0
    lat = []
    for r in ins:
        resp = agent.respond(r["question"]); lat.append(resp.latency_ms)
        leaks += FaultyLLM.FABRICATED in resp.answer
        if not resp.refused:
            answered += 1
            includes += norm(r["must_include"]) in norm(resp.answer)
            cited += any(c.startswith(r["gold_doc"] + "#") for c in resp.citations)
    refused = 0
    for r in oos:
        resp = agent.respond(r["question"]); lat.append(resp.latency_ms)
        refused += resp.refused
        leaks += FaultyLLM.FABRICATED in resp.answer
    q = statistics.quantiles(lat, n=100)
    return {"answer_rate_in_scope": round(answered / len(ins), 3), "must_include_rate": round(includes / len(ins), 3),
            "citation_hit_rate": round(cited / len(ins), 3), "oos_refusal_rate": round(refused / len(oos), 3),
            "fabricated_leak_rate": round(leaks / len(rows), 3), "latency_ms_p50": round(statistics.median(lat), 2),
            "latency_ms_p95": round(q[94], 2), "n_in_scope": len(ins), "n_oos": len(oos)}


def main():
    chunks, dev, test = load_chunks(ROOT / "data" / "policies"), load("dev"), load("test")
    ins_test = [r for r in test if r["type"] == "in_scope"]
    res = {"retrieval_test": {m: retrieval_metrics(m, ins_test, chunks) for m in MODES},
           "retrieval_dev": {m: retrieval_metrics(m, [r for r in dev if r["type"] == "in_scope"], chunks) for m in MODES}}
    tau, dev_acc = tune_tau(dev)
    res["tau"], res["tau_dev_accuracy"] = tau, round(dev_acc, 3)
    res["e2e"] = {
        "clean": end_to_end(SupportAgent(tau=tau), test),
        "fault_injection_no_guardrail": end_to_end(SupportAgent(llm=FaultyLLM(ExtractiveLLM(), 0.5, 1), guardrail=False, tau=tau), test),
        "fault_injection_with_guardrail": end_to_end(SupportAgent(llm=FaultyLLM(ExtractiveLLM(), 0.5, 1), guardrail=True, tau=tau), test),
    }
    (ROOT / "evals" / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    write_md(res)
    print(json.dumps(res["e2e"], indent=2, ensure_ascii=False)); print("tau", tau, "dev_acc", dev_acc)
    if "--check" in sys.argv:
        th = json.loads((ROOT / "evals" / "thresholds.json").read_text())
        bad = []
        for k, v in th["e2e_clean_min"].items():
            if res["e2e"]["clean"][k] < v: bad.append(f"clean.{k}={res['e2e']['clean'][k]} < {v}")
        for k, v in th["e2e_guardrail_max"].items():
            if res["e2e"]["fault_injection_with_guardrail"][k] > v: bad.append(f"guardrail.{k} > {v}")
        if res["retrieval_test"]["hybrid_rerank"]["mrr"] < th["mrr_min"]: bad.append("mrr below threshold")
        if bad:
            print("REGRESSION:", *bad, sep="\n  "); sys.exit(1)
        print("gate: ok")


def write_md(res):
    L = ["# Resultados de avaliação", "", "_Gerado por `evals/run_eval.py`. Dataset: 50 perguntas escritas à mão (ver `evals/dataset.jsonl`); métricas no split `test`._", "",
         "## Recuperação em nível de chunk (perguntas em escopo, split test)", "", "| Modo | recall@1 | recall@3 | MRR | IC95% MRR |", "|---|---|---|---|---|"]
    for m, v in res["retrieval_test"].items():
        L.append(f"| {m} | {v['recall@1']} | {v['recall@3']} | {v['mrr']} | {v['mrr_ci95']} |")
    L += ["", f"## Ponta a ponta (tau={res['tau']} escolhido no dev, acurácia dev={res['tau_dev_accuracy']})", "",
          "| Configuração | resposta (em escopo) | must-include | citação correta | recusa (fora de escopo) | vazamento fabricado | p50 ms | p95 ms |", "|---|---|---|---|---|---|---|---|"]
    for k, v in res["e2e"].items():
        L.append(f"| {k} | {v['answer_rate_in_scope']} | {v['must_include_rate']} | {v['citation_hit_rate']} | {v['oos_refusal_rate']} | {v['fabricated_leak_rate']} | {v['latency_ms_p50']} | {v['latency_ms_p95']} |")
    L += ["", "**Como ler:** `fault_injection_*` usa um gerador que injeta uma afirmação fabricada em 50% das respostas; "
          "‘vazamento fabricado’ é a fração de respostas finais que a contém. Mede o guardrail sob falha controlada, não a alucinação de um LLM real."]
    (ROOT / "docs" / "EVAL.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
