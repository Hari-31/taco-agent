from enum import Enum
from pydantic import BaseModel

class Style(str, Enum):
    AMERICAN = "american"
    MEXICAN = "mexican"

class Tortilla(str, Enum):
    CORN = "corn"
    FLOUR = "flour"

STYLE_INCLUDES: dict[Style, list[str]] = {
    Style.AMERICAN: ["Mexican blend cheese", "lettuce", "tomato"],
    Style.MEXICAN: ["grilled onion", "cilantro", "lime"],
}

class Protein(BaseModel):
    id: str
    name: str
    price: float

class AddOn(BaseModel):
    name: str
    price: float

class Menu(BaseModel):
    proteins: list[Protein]
    add_ons: list[AddOn]

    def find_protein(self, query: str) -> Protein | None:
        q = query.strip().lower()
        for p in self.proteins:
            if p.id == q or p.name.lower() == q:
                return p
        return None

    def find_add_on(self, query: str) -> AddOn | None:
        q = query.strip().lower()
        for a in self.add_ons:
            if a.name.lower() == q:
                return a
        return None

MENU = Menu(
    proteins=[
        Protein(id="ground_beef",       name="Ground Beef",       price=4.25),
        Protein(id="grilled_chicken",   name="Grilled Chicken",   price=4.25),
        Protein(id="chicken_al_pastor", name="Chicken Al Pastor", price=4.25),
        Protein(id="carnitas",          name="Carnitas",          price=4.25),
        Protein(id="chorizo",           name="Chorizo",           price=4.25),
        Protein(id="al_pastor",         name="Al Pastor",         price=4.25),
    ],
    add_ons=[
        AddOn(name="extra cheese", price=1.00),
        AddOn(name="guacamole",    price=1.50),
        AddOn(name="sour cream",   price=0.75),
        AddOn(name="jalapeños",    price=0.50),
        AddOn(name="extra meat",   price=2.00),
    ],
)

if __name__ == "__main__":
    print(f"Menu: {len(MENU.proteins)} proteins, {len(MENU.add_ons)} add-ons\n")
    for style in Style:
        print(f"{style.value.title()} style includes:", ", ".join(STYLE_INCLUDES[style]))
    print("Tortilla options:", ", ".join(t.value for t in Tortilla), "\n")
    ap = MENU.find_protein("al pastor")
    print(f"{ap.name}: ${ap.price:.2f}")
    guac = MENU.find_add_on("guacamole")
    print(f"Add guacamole: +${guac.price:.2f}")