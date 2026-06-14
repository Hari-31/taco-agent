from langchain_core.tools import tool
from menu import MENU, Style, Tortilla
from order import Order, TacoLine

def build_tools(order: Order):
    """Create tools bound to a specific Order instance (closure over `order`)."""

    @tool
    def add_taco(protein: str, style: str, tortilla: str,
                 add_ons: list[str] | None = None) -> str:
        """Add a taco to the customer's order.

        protein: a menu protein, e.g. "carnitas", "al pastor", "chicken al pastor".
        style: "american" (cheese, lettuce, tomato) or "mexican" (onion, cilantro, lime).
        tortilla: "flour" or "corn".
        add_ons: optional paid add-ons like ["guacamole", "extra cheese"]. Omit if none.
        """
        p = MENU.find_protein(protein)
        if p is None:
            opts = ", ".join(pr.name for pr in MENU.proteins)
            return f"No protein called {protein!r}. Available: {opts}."
        try:
            style_enum = Style(style.strip().lower())
        except ValueError:
            return f"Style must be 'american' or 'mexican', got {style!r}."
        try:
            tort_enum = Tortilla(tortilla.strip().lower())
        except ValueError:
            return f"Tortilla must be 'flour' or 'corn', got {tortilla!r}."

        clean_add_ons = []
        for name in (add_ons or []):
            a = MENU.find_add_on(name)
            if a is None:
                opts = ", ".join(ad.name for ad in MENU.add_ons)
                return f"No add-on called {name!r}. Available: {opts}."
            clean_add_ons.append(a.name)

        line = TacoLine(protein=p.id, style=style_enum,
                        tortilla=tort_enum, add_ons=clean_add_ons)
        order.add_line(line)
        return (f"Added: {line.describe()} — ${line.line_total():.2f}. "
                f"Order total: ${order.total():.2f}.")

    @tool
    def get_order_summary() -> str:
        """Show the current order: every line item and the running total."""
        return order.summary()

    return [add_taco, get_order_summary]

if __name__ == "__main__":
    order = Order()
    tools = {t.name: t for t in build_tools(order)}

    print("=== schema the LLM sees for add_taco ===")
    import json
    print(json.dumps(tools["add_taco"].args, indent=2))

    print("\n=== calling the tools directly (no LLM) ===")
    print(tools["add_taco"].invoke({"protein": "al pastor", "style": "mexican",
                                    "tortilla": "corn"}))
    print(tools["add_taco"].invoke({"protein": "carnitas", "style": "american",
                                    "tortilla": "flour", "add_ons": ["guacamole"]}))
    print(tools["add_taco"].invoke({"protein": "sushi", "style": "mexican",
                                    "tortilla": "corn"}))   # error case
    print("\n--- summary ---")
    print(tools["get_order_summary"].invoke({}))