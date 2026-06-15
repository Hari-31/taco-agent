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
    def update_quantity(line_id: int, quantity: int) -> str:
        """Change how many of a taco line the customer wants.

        line_id: the #N id of the line (see the order summary).
        quantity: the new count. Must be 1 or more; to remove, use remove_taco.
        """
        line = order.find_line(line_id)
        if line is None:
            return f"No line #{line_id} in the order. Use get_order_summary to see ids."
        if quantity < 1:
            return "Quantity must be at least 1. To remove the item, use remove_taco."
        line.quantity = quantity
        return (f"Updated #{line_id}: {line.describe()} — ${line.line_total():.2f}. "
                f"Order total: ${order.total():.2f}.")

    @tool
    def remove_taco(line_id: int) -> str:
        """Remove a taco line from the order entirely.

        line_id: the #N id of the line to remove (see the order summary).
        """
        line = order.find_line(line_id)
        if line is None:
            return f"No line #{line_id} in the order. Use get_order_summary to see ids."
        desc = line.describe()
        order.remove_line(line_id)
        return f"Removed #{line_id} ({desc}). Order total: ${order.total():.2f}."
    
    @tool
    def get_order_summary() -> str:
        """Show the current order: every line item and the running total."""
        return order.summary()

    return [add_taco, update_quantity, remove_taco, get_order_summary]

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