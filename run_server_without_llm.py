"""Run the real web app with Gemini replaced by a simple command-driven fake.

For manual checks in the browser with no API key and no LLM calls. Everything
except the language model is real: the web page, login, storage, the agent
loop, and the order tools.

Run from the project root:
    python run_server_without_llm.py
Then open http://127.0.0.1:8000  (stop with Ctrl+C).

In the chat box, type one command per message:
    add <protein> <style> <tortilla>    e.g.  add carnitas mexican corn
    qty <item#> <count>                 e.g.  qty 1 3
    addon <item#> <add-on>              e.g.  addon 1 guacamole
    remove <item#>                      e.g.  remove 2
    show                                (shows the order)
    confirm                             (confirms / finalizes the order)
The fake model calls the matching tool and replies with the tool's own message.
"""

import itertools
import os
import sys

# Make sure a real Gemini key is never used, even if .env has one.
os.environ["GOOGLE_API_KEY"] = "no-llm-judge-mode"

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import agent

HELP = ("Fake model (no LLM). Commands: add <protein> <style> <tortilla> | "
        "qty <item#> <count> | addon <item#> <add-on> | remove <item#> | show | confirm")

_ids = itertools.count(1)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def _tool_call(name: str, **args) -> AIMessage:
    call = {"name": name, "args": args, "id": f"judge_{next(_ids)}", "type": "tool_call"}
    return AIMessage(content="", tool_calls=[call])


def _parse(command: str) -> AIMessage:
    words = command.strip().lower().split()
    try:
        if not words:
            return AIMessage(content=HELP)
        verb, rest = words[0], words[1:]
        if verb == "add" and len(rest) >= 3:
            return _tool_call("add_taco", protein=" ".join(rest[:-2]),
                              style=rest[-2], tortilla=rest[-1])
        if verb == "qty" and len(rest) == 2:
            return _tool_call("update_quantity", line_id=int(rest[0]), quantity=int(rest[1]))
        if verb == "addon" and len(rest) >= 2:
            return _tool_call("set_add_ons", line_id=int(rest[0]), add_ons=[" ".join(rest[1:])])
        if verb == "remove" and len(rest) == 1:
            return _tool_call("remove_taco", line_id=int(rest[0]))
        if verb == "show" and not rest:
            return _tool_call("get_order_summary")
        if verb == "confirm" and not rest:
            return _tool_call("finalize_order")
    except ValueError:
        pass
    return AIMessage(content=HELP)


class CommandModel(BaseChatModel):
    """Turns a typed command into one tool call, then repeats the tool's reply."""

    @property
    def _llm_type(self) -> str:
        return "judge-command-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1]
        if isinstance(last, ToolMessage):
            reply = AIMessage(content=_text(last.content))
        elif isinstance(last, HumanMessage):
            reply = _parse(_text(last.content))
        else:
            reply = AIMessage(content=HELP)
        print(f"[fake-model] {reply.tool_calls or _text(reply.content)[:80]}", flush=True)
        return ChatResult(generations=[ChatGeneration(message=reply)])


agent.MODEL = CommandModel()   # make_agent() reads agent.MODEL

import uvicorn  # noqa: E402
import api      # noqa: E402  (imported after the swap on purpose)

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("Fake-model server (no LLM calls) on http://127.0.0.1:8000", flush=True)
    uvicorn.run(api.app, host="127.0.0.1", port=8000)
