"""design_conditional: picking the species picks the dimensions.

In TeamDesigner.get_parameter_space, moves / ability / gender / tera_type all
carry "depends_on": "species" — so choosing a species decides what variables
even exist beneath it, and how many values each can take. The spread is not
subtle: Magikarp has 4 legal moves (exactly ONE possible 4-move set) while
Tyranitar has 100 (3,921,225 sets). Same slot, same benchmark.
"""

from figure_kit import *

W, H = 1000, 540
reset(); background(W, H)

eyebrow("POKÉMON SHOWDOWN&#160;&#160;·&#160;&#160;TEAM DESIGN SPACE&#160;&#160;Θ", 40, 16, W - 80)
text("PICK THE SPECIES, PICK THE <span style='color:%s'>DIMENSIONS</span>" % TEAM,
     40, 40, W - 80, 38, size=26, bold=True, spacing=1)
text("moves, ability and tera type are all conditional on the species", 40, 82, W - 80, 20,
     size=13, color=MUTED)

def fan(cx, sprite_file, n_drawn, spread, n_moves, sets, note, more=False):
    sprite(sprite_file, cx - 33, 140, 66, 66)
    for k in range(n_drawn):
        x = cx - spread / 2 + k * (spread / max(n_drawn - 1, 1))
        path(f"edgeStyle=none;html=1;endArrow=none;strokeWidth=1.3;strokeColor={TEAM};"
             f"opacity=70;", [(cx, 214), (x, 322)])
        dot(x, 326, 7, TEAM, PAGE, 1.5)
    if more:
        text("…", cx + spread / 2 + 18, 316, 30, 20, size=17, color=FAINT, bold=True)
    text(f"<b>{n_moves}</b> legal moves", cx - 170, 348, 340, 18, size=12, color=INK)
    text(sets, cx - 170, 372, 340, 26, size=19, color=TEAM, bold=True)
    text(note, cx - 170, 400, 340, 16, size=10, color=MUTED)

fan(268, "poke_129.png", 4, 96, 4, "1 move set", "only one team is possible here")
fan(720, "poke_248.png", 26, 330, 100, "3,921,225 move sets", "in the very same slot", more=True)

path(f"edgeStyle=none;html=1;endArrow=none;strokeWidth=1.2;strokeColor={HAIR};",
     [(494, 150), (494, 410)])

text("one slot of the team&#160;&#160;·&#160;&#160;"
     "<b>4 million× the choice, depending on a single categorical pick</b>",
     40, 452, W - 80, 20, size=12.5, color=INK)
credit(40, 506, W - 80)
emit("design_conditional.drawio", "Conditional design space", W, H)
