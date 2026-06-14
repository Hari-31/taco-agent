import os
from dotenv import load_dotenv
from langchain.agents import create_agent

from menu import MENU
from order import Order
from tools import build_tools

load_dotenv()

def is_rate_limit(err) -> bool:
    """True if this looks like a 429 from any provider (Groq, Gemini, OpenAI...)."""
    code = getattr(err, "status_code", None) or getattr(err, "code", None)
    if code == 429:
        return True
    text = str(err).lower()
    return "429" in text or "rate limit" in text or "resource exhausted" in text

def menu_text() -> str:
    proteins = ", ".join(f"{p.name} (${p.price:.2f})" for p in MENU.proteins)
    add_ons = ", ".join(f"{a.name} (+${a.price:.2f})" for a in MENU.add_ons)
    return (
        f"PROTEINS (all $4.25): {proteins}\n"
        f"STYLES: american (Mexican blend cheese, lettuce, tomato) "
        f"OR mexican (grilled onion, cilantro, lime)\n"
        f"TORTILLAS: flour or corn\n"
        f"ADD-ONS: {add_ons}"
    )

SYSTEM_PROMPT = f"""You are the friendly ordering assistant at a taco shop. \
Take the customer's order through conversation.

RULES:
- Manage the order ONLY through your tools. The order lives in the tools, not \
in your memory. After any change, the tool result is the truth.
- Every taco needs three things: a protein, a style (american or mexican), and \
a tortilla (flour or corn). If the customer hasn't given all three, ask before adding.
- Only use items from the menu below. Never invent items or prices.
- There are two similar proteins: "Al Pastor" and "Chicken Al Pastor". If a \
customer just says "al pastor" it means "Al Pastor".
- Be concise and warm. Confirm what you added and the running total.
- make sure user sticks to ordering tacos. If they ask for something else, politely decline.
- never answer any question expect 

MENU:
{menu_text()}
"""

def text_of(msg) -> str:
    """Gemini  messages can be a string or a list of content blocks."""
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Set GROQ_API_KEY in a .env file first.")

    order = Order()
    agent = create_agent(
        model="groq:llama-3.3-70b-versatile",
        tools=build_tools(order),
        system_prompt=SYSTEM_PROMPT,
    )

    print("Taco shop is open! Type your order. ('quit' to leave)\n")
    history: list = []
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        if not user:
            continue
        history.append({"role": "user", "content": user})
        candidate = history + [{"role": "user", "content": user}]
        try:
            result = agent.invoke({"messages": candidate})
        except Exception as e:
            if is_rate_limit(e):
                print("Bot: (Rate limit hit — wait ~30–60s and try the same thing again.)\n")
            else:
                print(f"Bot: (Error talking to the model: {e} — try again.)\n")
            continue

        history = result["messages"]
        print("Bot:", text_of(history[-1]), "\n")


if __name__ == "__main__":
    main()