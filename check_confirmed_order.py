"""Manual check: a confirmed order cannot be changed. No API key needed.

Calls the same tools the chatbot calls, in the order a customer chat would,
with no language model in between. The "bill" printed after each step is
built with api.order_view(), the same data the web page's Guest check shows.

Part A: normal ordering before confirmation (add, quantity, add-ons, remove,
        confirm) must work and price correctly.
Part B: after confirmation, every kind of change (add, quantity, add-ons,
        remove) must be rejected with a message, and the bill must not change.

Run from the project root:
    python check_confirmed_order.py
Exit code is 0 when every step passes, 1 otherwise.
"""

import sys

from api import order_view
from order import Order
from tools import build_tools

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # tool replies contain an em dash


def print_bill(view: dict) -> None:
    for line in view["lines"]:
        print(f"      [{line['id']}] {line['text']}  ${line['price']:.2f}")
    if not view["lines"]:
        print("      (no items)")
    print(f"      Total ${view['total']:.2f}   confirmed={view['finalized']}")


def main() -> int:
    order = Order()
    tools = {t.name: t for t in build_tools(order)}
    results: list[tuple[str, bool]] = []

    def run(label: str, tool: str, args: dict) -> tuple[str, dict]:
        print(f"\n{label}")
        print(f"    call : {tool}({args})")
        reply = tools[tool].invoke(args)
        print("    reply:", reply.replace("\n", " | "))
        view = order_view(order)
        print("    bill :")
        print_bill(view)
        return reply, view

    def record(label: str, ok: bool, why: str = "") -> None:
        results.append((label, ok))
        print(f"    => {'PASS' if ok else 'FAIL'}" + (f" ({why})" if why and not ok else ""))

    # ---------------- Part A: before confirmation ----------------
    print("=" * 70)
    print("PART A - normal ordering before confirmation")
    print("=" * 70)

    part_a = [
        ("A1 add grilled chicken (mexican, corn)", "add_taco",
         {"protein": "grilled chicken", "style": "mexican", "tortilla": "corn"},
         ["1x Grilled Chicken (mexican, corn)"], 4.25),
        ("A2 add chorizo (american, flour)", "add_taco",
         {"protein": "chorizo", "style": "american", "tortilla": "flour"},
         ["1x Grilled Chicken (mexican, corn)", "1x Chorizo (american, flour)"], 8.50),
        ("A3 add al pastor (mexican, corn)", "add_taco",
         {"protein": "al pastor", "style": "mexican", "tortilla": "corn"},
         ["1x Grilled Chicken (mexican, corn)", "1x Chorizo (american, flour)",
          "1x Al Pastor (mexican, corn)"], 12.75),
        ("A4 change quantity of #1 to 2", "update_quantity",
         {"line_id": 1, "quantity": 2},
         ["2x Grilled Chicken (mexican, corn)", "1x Chorizo (american, flour)",
          "1x Al Pastor (mexican, corn)"], 17.00),
        ("A5 add guacamole to #2", "set_add_ons",
         {"line_id": 2, "add_ons": ["guacamole"]},
         ["2x Grilled Chicken (mexican, corn)", "1x Chorizo (american, flour) + guacamole",
          "1x Al Pastor (mexican, corn)"], 18.50),
        ("A6 remove #3", "remove_taco",
         {"line_id": 3},
         ["2x Grilled Chicken (mexican, corn)", "1x Chorizo (american, flour) + guacamole"],
         14.25),
    ]
    for label, tool, args, want_lines, want_total in part_a:
        _, view = run(label, tool, args)
        got_lines = [l["text"] for l in view["lines"]]
        ok = (got_lines == want_lines and view["total"] == want_total
              and not view["finalized"])
        record(label, ok, f"expected {want_lines}, total ${want_total:.2f}, not confirmed")

    label = "A7 confirm the order"
    reply, confirmed = run(label, "finalize_order", {})
    ok = ("confirmed" in reply.lower() and confirmed["finalized"]
          and confirmed["total"] == 14.25 and len(confirmed["lines"]) == 2)
    record(label, ok, "expected a confirmation, confirmed=True, same 2 lines, total $14.25")

    # ---------------- Part B: after confirmation ----------------
    print()
    print("=" * 70)
    print("PART B - every change after confirmation must be rejected")
    print("Pass = reply says the order is already confirmed AND the bill is")
    print("exactly the same as after A7:")
    print_bill(confirmed)
    print("=" * 70)

    part_b = [
        ("B1 add a taco (al pastor)", "add_taco",
         {"protein": "al pastor", "style": "mexican", "tortilla": "corn"}),
        ("B2 change quantity of #1 to 5", "update_quantity",
         {"line_id": 1, "quantity": 5}),
        ("B3 change add-ons on #2 (remove guacamole)", "set_add_ons",
         {"line_id": 2, "add_ons": []}),
        ("B4 remove #1", "remove_taco",
         {"line_id": 1}),
    ]
    for label, tool, args in part_b:
        reply, view = run(label, tool, args)
        rejected = "confirmed" in reply.lower()
        unchanged = view == confirmed
        why = []
        if not rejected:
            why.append("reply does not say the order is confirmed")
        if not unchanged:
            why.append("bill changed")
        record(label, rejected and unchanged, "; ".join(why))

    # ---------------- Summary ----------------
    print()
    print("=" * 70)
    print("SUMMARY")
    for label, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} steps passed.")
    print("ALL PASSED" if passed == len(results) else "SOME STEPS FAILED")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
