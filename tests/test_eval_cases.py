"""Offline agent tests: no API key, no internet.

Each case plays a scripted customer conversation through the real
make_agent(), the real LangGraph agent loop, and the real tools and Order.
Only the LLM is replaced: a scripted fake model returns the tool calls a correct
model would make for that conversation. The checks assert on the final order
state (order.total() and the number of lines), never on the model's words.

Run:  python -m pytest -q
"""

import itertools

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import agent
from order import Order

_ids = itertools.count(1)

CASES = [
    {"name": "single taco",
     "messages": ["I'd like one carnitas taco, american style, flour tortilla. That's all."],
     "total": 4.25, "lines": 1},
    {"name": "quantity of two",
     "messages": ["Two grilled chicken tacos, mexican style, corn tortillas. Nothing else."],
     "total": 8.50},
    {"name": "taco with add-on",
     "messages": ["One al pastor, mexican, corn, with guacamole. That's it."],
     "total": 5.75, "lines": 1},
    {"name": "add then remove",
     "messages": ["One chorizo, american, flour.",
                  "Actually remove that — I don't want anything. Yes, remove it."],
     "total": 0.0, "lines": 0},
    {"name": "add-on then take it off",
     "messages": ["A carnitas taco, american, flour, with guacamole.",
                  "On second thought, no guac on it."],
     "total": 4.25, "lines": 1},
    {"name": "two different tacos",
     "messages": ["A carnitas american flour and a chorizo mexican corn. Done."],
     "total": 8.50, "lines": 2},
]


def run_case(messages: list[str]) -> Order:
    """Play a scripted conversation through a fresh agent, return the order."""
    order = Order()
    bot = agent.make_agent(order)
    history: list = []
    for msg in messages:
        history.append({"role": "user", "content": msg})
        history = bot.invoke({"messages": history})["messages"]
    return order            # inspect the real state, not the model's words


def call(name: str, **args) -> dict:
    """One tool call, in the shape a chat model returns it."""
    return {"name": name, "args": args, "id": f"call_{next(_ids)}",
            "type": "tool_call"}


class ScriptedChatModel(BaseChatModel):
    """Stands in for Gemini. Returns pre-written replies in order."""

    replies: list[AIMessage]
    position: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools, **kwargs):
        return self             # tools are run by the agent, not the model

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.position >= len(self.replies):
            raise AssertionError("Agent asked the model for more replies than scripted")
        reply = self.replies[self.position]
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


def turn(*tool_calls: dict, reply: str = "Done.") -> list[AIMessage]:
    """Model replies for one user message: call the tools, then answer in text."""
    out = []
    if tool_calls:
        out.append(AIMessage(content="", tool_calls=list(tool_calls)))
    out.append(AIMessage(content=reply))
    return out


# What a correct model does for each case, one list per user message.
SCRIPTS = {
    "single taco": [
        turn(call("add_taco", protein="carnitas", style="american", tortilla="flour")),
    ],
    "quantity of two": [
        turn(call("add_taco", protein="grilled chicken", style="mexican", tortilla="corn"),
             call("update_quantity", line_id=1, quantity=2)),
    ],
    "taco with add-on": [
        turn(call("add_taco", protein="al pastor", style="mexican", tortilla="corn",
                  add_ons=["guacamole"])),
    ],
    "add then remove": [
        turn(call("add_taco", protein="chorizo", style="american", tortilla="flour")),
        turn(call("remove_taco", line_id=1)),
    ],
    "add-on then take it off": [
        turn(call("add_taco", protein="carnitas", style="american", tortilla="flour",
                  add_ons=["guacamole"])),
        turn(call("set_add_ons", line_id=1, add_ons=[])),
    ],
    "two different tacos": [
        turn(call("add_taco", protein="carnitas", style="american", tortilla="flour"),
             call("add_taco", protein="chorizo", style="mexican", tortilla="corn")),
    ],
}


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    """Prove nothing here needs the Gemini key."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def use_fake_model(monkeypatch, script: list[list[AIMessage]]) -> ScriptedChatModel:
    model = ScriptedChatModel(replies=[m for t in script for m in t])
    monkeypatch.setattr(agent, "MODEL", model)   # make_agent() reads agent.MODEL
    return model


def test_every_case_has_a_script():
    assert {c["name"] for c in CASES} == set(SCRIPTS)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_case_offline(monkeypatch, case):
    script = SCRIPTS[case["name"]]
    assert len(script) == len(case["messages"]), "one scripted turn per user message"
    model = use_fake_model(monkeypatch, script)

    order = run_case(case["messages"])

    assert model.position == len(model.replies), "not every scripted reply was used"
    assert round(order.total(), 2) == pytest.approx(case["total"])
    if "lines" in case:
        assert len(order.lines) == case["lines"]


def test_agent_recovers_from_bad_tool_argument(monkeypatch):
    """A wrong protein gets a helpful error back; the model retries and succeeds."""
    model = use_fake_model(monkeypatch, [[
        AIMessage(content="", tool_calls=[call("add_taco", protein="sushi",
                                               style="mexican", tortilla="corn")]),
        AIMessage(content="", tool_calls=[call("add_taco", protein="al pastor",
                                               style="mexican", tortilla="corn")]),
        AIMessage(content="Added one Al Pastor."),
    ]])

    order = run_case(["One sushi taco, mexican, corn. Or al pastor if not."])

    assert len(order.lines) == 1
    assert order.lines[0].protein == "al_pastor"
    assert order.total() == pytest.approx(4.25)
