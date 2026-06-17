"""A tiny eval harness. Runs scripted customer conversations through the REAL
agent, then checks the final ORDER STATE — not the model's words. Because the
order lives in our Order object, we assert on order.total() directly.

Run:  python eval.py   (needs GROQ_API_KEY, makes real API calls)
"""

import os
import time
from order import Order
from agent import make_agent

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
    agent = make_agent(order)
    history: list = []
    for msg in messages:
        history.append({"role": "user", "content": msg})
        history = agent.invoke({"messages": history})["messages"]
    return order            # inspect the real state, not the model's words


def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Set GROQ_API_KEY in a .env file first.")

    passed = 0
    for case in CASES:
        try:
            order = run_case(case["messages"])
        except Exception as e:
            print(f"[ERROR] {case['name']}: {e}")
            continue

        actual = round(order.total(), 2)
        ok = abs(actual - case["total"]) < 0.001
        detail = f"total ${actual:.2f} (want ${case['total']:.2f})"
        if "lines" in case:
            ok = ok and len(order.lines) == case["lines"]
            detail += f", {len(order.lines)} lines (want {case['lines']})"

        print(f"[{'PASS' if ok else 'FAIL'}] {case['name']}: {detail}")
        passed += ok
        time.sleep(2)       

    print(f"\n{passed}/{len(CASES)} cases passed.")


if __name__ == "__main__":
    main()