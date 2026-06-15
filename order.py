from pydantic import BaseModel, Field
from menu import MENU, Style, Tortilla

class TacoLine(BaseModel):
    id: int = 0
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
    next_id: int = 1

    def add_line(self, line: TacoLine) -> int:
        line.id = self.next_id
        self.next_id += 1
        self.lines.append(line)
        return line.id
    
    def find_line(self, line_id: int) -> TacoLine | None:
        for line in self.lines:
            if line.id == line_id:
                return line
        return None
    
    def remove_line(self, line_id: int) -> bool:
        line = self.find_line(line_id)
        if line is None:
            return False
        self.lines.remove(line)
        return True

    def total(self) -> float:
        return sum(line.line_total() for line in self.lines)

    def summary(self) -> str:
        if not self.lines:
            return "Order is empty."
        rows = [f"  [{line.id}] {line.describe()} — ${line.line_total():.2f}"
                for line in self.lines]
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