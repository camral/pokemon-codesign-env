"""design_coupling: a team is a system, not a list.

The theta<->pi coupling the hero figures sell repeats one level down, inside
theta itself. Six slots are not six independent picks — type coverage, shared
weaknesses and role overlap mean swapping one slot re-scores the other five.
"""

from math import cos, radians, sin

from figure_kit import *

W, H = 720, 670
reset(); background(W, H)

CX, CY, R = 360, 400, 170
V = [(CX + R * cos(radians(a)), CY + R * sin(radians(a)))
     for a in (-90, -30, 30, 90, 150, 210)]
TEAMS = ["poke_445.png", "poke_143.png", "poke_9.png",
         "poke_3.png", "poke_94.png", "poke_6.png"]
SIDES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
DIAGS = [(0, 3), (1, 4), (2, 5)]
SWAPPED = 0

eyebrow("POKÉMON SHOWDOWN&#160;&#160;·&#160;&#160;TEAM DESIGN SPACE&#160;&#160;Θ", 40, 16, W - 80)
text("A TEAM IS A <span style='color:%s'>SYSTEM</span>, NOT A LIST" % TEAM,
     40, 40, W - 80, 38, size=26, bold=True, spacing=1)
text("swap one slot — the other five change value", 40, 82, W - 80, 20,
     size=13, color=MUTED)

for a, b in SIDES + DIAGS:                       # the interaction graph
    live = SWAPPED in (a, b)
    path(f"edgeStyle=none;html=1;endArrow=none;strokeWidth={3 if live else 1.8};"
         f"strokeColor={TEAM if live else '#A9B6C4'};", [V[a], V[b]])

for k, (vx, vy) in enumerate(V):
    if k == SWAPPED:
        node("", vx - 36, vy - 36, 72, 72,
             f"ellipse;html=1;fillColor={TEAM_BG};strokeColor={TEAM};strokeWidth=3;")
    sprite(TEAMS[k], vx - 26, vy - 26, 52, 52)

# the slot being redesigned, arriving from outside the team
sprite("poke_248.png", V[SWAPPED][0] - 22, 108, 44, 44, opacity=55)
path(f"edgeStyle=none;html=1;strokeWidth=3;strokeColor={TEAM};endArrow=block;"
     f"endFill=1;endSize=5;", [(V[SWAPPED][0], 158), (V[SWAPPED][0], V[SWAPPED][1] - 40)])

for (a, b), label in zip([(5, 0), (0, 1), (0, 3)],
                         ["covers", "shares weakness", "sets up"]):
    mx, my = (V[a][0] + V[b][0]) / 2, (V[a][1] + V[b][1]) / 2
    node(label, mx - 55, my - 9, 110, 18,
         f"text;html=1;strokeColor=none;fillColor={PAGE};whiteSpace=wrap;align=center;"
         f"verticalAlign=middle;fontFamily={F};fontSize=9.5;fontColor={TEAM};")

text("the same coupling as&#160;&#160;<b>θ ↔ π</b>&#160;&#160;— one level down",
     40, 604, W - 80, 20, size=12.5, color=INK)
credit(40, 638, W - 80)
emit("design_coupling.drawio", "Team coupling", W, H)
