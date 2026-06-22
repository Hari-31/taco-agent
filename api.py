import uuid
from dataclasses import dataclass, field

from fastapi import FastAPI
from pydantic import BaseModel

from order import Order
from agent import make_agent, is_rate_limit, text_of

from pathlib import Path
from fastapi.responses import FileResponse

app = FastAPI(title="Taco Ordering Agent")

@app.get("/")
def index():
    """Serve the web chat page (same origin as /chat, so no CORS)."""
    return FileResponse(Path(__file__).resolve().parent / "index.html")

@dataclass
class Session:
    order: Order
    agent: object
    history: list = field(default_factory=list)


SESSIONS: dict[str, Session] = {}

def get_session(session_id: str | None) -> tuple[str, Session]:
    """Return an existing session or create a fresh one."""
    if not session_id or session_id not in SESSIONS:
        session_id = session_id or uuid.uuid4().hex
        order = Order()
        SESSIONS[session_id] = Session(order=order, agent=make_agent(order))
    return session_id, SESSIONS[session_id]

def order_view(order: Order) -> dict:
    """Serialize the order for the UI's live order panel."""
    return {
        "lines": [{"id": l.id, "text": l.describe(), "price": round(l.line_total(), 2)}
                  for l in order.lines],
        "total": round(order.total(), 2),
        "finalized": order.finalized,
    }

class ChatIn(BaseModel):
    message: str
    session_id: str | None = None

@app.post("/chat")
def chat(body: ChatIn) -> dict:
    session_id, sess = get_session(body.session_id)
    # Only commit to history on success (same rule as the CLI).
    candidate = sess.history + [{"role": "user", "content": body.message}]
    try:
        result = sess.agent.invoke({"messages": candidate})
        sess.history = result["messages"]
        reply = text_of(sess.history[-1])
    except Exception as e:
        reply = ("(Rate limit hit — wait a moment and try again.)"
                 if is_rate_limit(e) else "(Error talking to the model — try again.)")
    return {"session_id": session_id, "reply": reply, "order": order_view(sess.order)}


@app.post("/reset")
def reset(body: ChatIn) -> dict:
    """Start a new order for this session."""
    if body.session_id:
        SESSIONS.pop(body.session_id, None)
    return {"ok": True}