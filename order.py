from pydantic import BaseModel, Field
from menu import MENU, Style, Tortilla

class TacoLine(BaseModel):
    protein: str                                  
    style: Style
    tortilla: Tortilla
    add_ons: list[str] = Field(default_factory=list)
    quantity: int = 1

    def unit_price(self) -> float:
        p = MENU.find_protein(self.protein)
        if p is None:
            raise ValueError(f"Unknown protein: {self.protein!r}")
        price = p.price
        for name in self.add_ons:
            a = MENU.find_add_on(name)
            if a is None:
                raise ValueError(f"Unknown add-on: {name!r}")
            price += a.price
        return price

    def line_total(self) -> float:
        return self.unit_price() * self.quantity

    def describe(self) -> str:
        name = MENU.find_protein(self.protein).name
        text = f"{self.quantity}x {name} ({self.style.value}, {self.tortilla.value})"
        if self.add_ons:
            text += " + " + ", ".join(self.add_ons)
        return text
    
class Order(BaseModel):
    lines: list[TacoLine] = Field(default_factory=list)

    def add_line(self, line: TacoLine) -> int:
        self.lines.append(line)
        return len(self.lines) - 1          # index handle (see note below)

    def total(self) -> float:
        return sum(line.line_total() for line in self.lines)

    def summary(self) -> str:
        if not self.lines:
            return "Order is empty."
        rows = [f"  [{i}] {line.describe()} — ${line.line_total():.2f}"
                for i, line in enumerate(self.lines)]
        rows.append(f"  Total: ${self.total():.2f}")
        return "\n".join(rows)
if __name__ == "__main__":
    order = Order()
    order.add_line(TacoLine(protein="al_pastor", style=Style.MEXICAN,
                            tortilla=Tortilla.CORN))
    order.add_line(TacoLine(protein="carnitas", style=Style.AMERICAN,
                            tortilla=Tortilla.FLOUR,
                            add_ons=["guacamole", "extra cheese"], quantity=2))
    print(order.summary())