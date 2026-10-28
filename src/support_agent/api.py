from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import SupportAgent


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    customer_id: str | None = None


def create_app(agent: SupportAgent | None = None) -> FastAPI:
    app = FastAPI(title="support-agent", version="0.1.0")
    state = {"agent": agent}

    def get_agent() -> SupportAgent:
        if state["agent"] is None:
            state["agent"] = SupportAgent()
        return state["agent"]

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.post("/chat")
    def chat(body: ChatIn):
        r = get_agent().respond(body.message, body.customer_id)
        return {"answer": r.answer, "intent": r.intent, "citations": r.citations, "refused": r.refused,
                "latency_ms": r.latency_ms, "trace": r.trace, "usage": r.usage}

    return app


app = create_app()
