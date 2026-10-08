"""design_slot: one slot is a mixed-integer program.

A single Pokemon is not one categorical draw. TeamDesigner exposes three
different kinds of variable per slot: flat categoricals (species, item,
ability, nature), a choose-4-of-N subset (moves), and a budgeted integer
allocation (EVs / IVs, "type": "stat_allocation" with a per-stat cap).
"""

from figure_kit import *

W, H = 1000, 560
reset(); background(W, H)

eyebrow("POKÉMON SHOWDOWN&#160;&#160;·&#160;&#160;TEAM DESIGN SPACE&#160;&#160;Θ", 40, 16, W - 80)
text("ONE SLOT IS A <span style='color:%s'>MIXED-INTEGER PROGRAM</span>" % TEAM,
     40, 40, W - 80, 38, size=25, bold=True, spacing=1)
text("three different kinds of variable, six coupled decisions", 40, 82, W - 80, 20,
     size=13, color=MUTED)

sprite("poke_445.png", 74, 210, 104, 104)
text("one slot", 60, 322, 132, 18, size=11, color=MUTED, bold=True, spacing=1)

LX, VX, TX = 214, 340, 812
def row(y, label, tag):
    text(label, LX, y, 120, 20, size=10.5, color=INK, bold=True, spacing=0.5, align="left")
    text(tag, TX, y, 148, 20, size=9.5, color=MUTED, align="right")

row(140, "SPECIES", "categorical · 549")
pips(VX, 142, 18, {6}, 16, 8, TEAM, HAIR)
row(210, "MOVES", "subset · 4 of 73")
pips(VX, 212, 18, {1, 5, 6, 12}, 16, 8, TEAM, HAIR)
row(280, "ITEM · ABILITY", "categorical · 584 / 2")
pips(VX, 282, 10, {3}, 16, 8, TEAM, HAIR)
pips(VX + 268, 282, 4, {1}, 16, 8, TEAM, HAIR)
row(350, "NATURE", "categorical · 25")
pips(VX, 352, 12, {8}, 16, 8, TEAM, HAIR)

row(424, "EVs", "integer allocation · ≤ 255 each")
EV = (250, 96, 168, 40, 220, 112)
for k, v in enumerate(EV):
    node("", VX + k * 46, 470 - v * 0.2, 30, v * 0.2,
         f"rounded=0;html=1;fillColor={TEAM_CHIP};strokeColor=none;")
    text(str(v), VX + k * 46 - 4, 474, 38, 14, size=9, color=MUTED)
path(f"edgeStyle=none;html=1;endArrow=none;strokeWidth=1.2;strokeColor={ACCENT};"
     f"dashed=1;dashPattern=4 3;", [(VX - 8, 419), (VX + 286, 419)])
text("cap", VX + 292, 411, 40, 16, size=9, color=ACCENT, bold=True, align="left")

text("and every one of these changes what the other five slots are worth",
     40, 506, W - 80, 20, size=12.5, color=INK)
credit(40, 534, W - 80)
emit("design_slot.drawio", "One slot anatomy", W, H)
