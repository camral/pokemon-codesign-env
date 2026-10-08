"""Generate docs/figures/codesign_hero.drawio: the small, loud version.

One idea, readable in a few seconds: the peak moves.

Both search spaces get equal billing and opposite character:

  * POLICY SPACE pi  - the horizontal axis. Continuous.
  * TEAM DESIGN SPACE Theta - the family of curves, one per team. Discrete,
    combinatorial, and each entry a full six-slot spec (the EV bars under each
    highlighted team say so without words).

Score is a function of the PAIR, so a team is not a point on either axis - it is
a whole curve over policy space. Two are drawn out of the crowd: their optima sit
in different places (pi*(theta)), and the shaded halves say which team is ahead
wherever the policy actually is (theta*(pi)).

Magnitudes come from this repo's gen4customgame config: 554 legal species x 25
natures over 6 slots ~ 1e24.8, before move sets, EVs, items or abilities.

Series palette validated with the dataviz skill's validate_palette.js.
Schematic: no tick values anywhere, because there is no data here.

Re-run after editing, then export with:
    /Applications/draw.io.app/Contents/MacOS/draw.io -x -f png -s 2 -o codesign_hero.png codesign_hero.drawio
"""

import base64
from html import escape
from math import exp
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

INK = "#0F172A"
MUTED = "#64748B"
FAINT = "#94A3B8"
CROWD_C = "#E2E8F0"
PAGE = "#FFFFFF"
FAIL = "#DC2626"
A, B = "#1D4ED8", "#B45309"          # two series, fixed order (validator: PASS)
A_BG, B_BG = "#F4F7FE", "#FEF9F1"

F = "Helvetica"

cells: list[str] = []
_uid = [0]


def nid():
    _uid[0] += 1
    return f"n{_uid[0]}"


def node(value, x, y, w, h, style):
    cells.append(
        f'<mxCell id="{nid()}" value="{escape(value)}" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
    )


def path(style, points):
    (sx, sy), (tx, ty) = points[0], points[-1]
    mids = "".join(f'<mxPoint x="{round(x, 1)}" y="{round(y, 1)}"/>' for x, y in points[1:-1])
    arr = f'<Array as="points">{mids}</Array>' if mids else ""
    cells.append(
        f'<mxCell id="{nid()}" style="{style}" edge="1" parent="1">'
        f'<mxGeometry relative="1" as="geometry">'
        f'<mxPoint x="{round(sx, 1)}" y="{round(sy, 1)}" as="sourcePoint"/>'
        f'<mxPoint x="{round(tx, 1)}" y="{round(ty, 1)}" as="targetPoint"/>{arr}</mxGeometry></mxCell>'
    )


def text(value, x, y, w, h, size=12, color=INK, bold=False, spacing=0):
    node(f"<b>{value}</b>" if bold else value, x, y, w, h,
         f"text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;overflow=visible;"
         f"align=center;verticalAlign=middle;fontFamily={F};fontSize={size};"
         f"fontColor={color};letterSpacing={spacing};")


def sprite(fname, x, y, w, h, opacity=100):
    data = base64.b64encode((ASSETS / fname).read_bytes()).decode()
    node("", x, y, w, h,
         f"shape=image;html=1;imageAspect=0;aspect=fixed;opacity={opacity};"
         f"image=data:image/png,{data};")


def team_chip(cx, top, files, size, gap, opacity=100):
    """A team, drawn as its roster."""
    span = len(files) * size + (len(files) - 1) * gap
    for k, f in enumerate(files):
        sprite(f, cx - span / 2 + k * (size + gap), top, size, size, opacity)


def bars(cx, base, heights, color):
    """EV-spread bars: each entry in Theta is a full spec, not just three species."""
    span = len(heights) * 6 + (len(heights) - 1) * 3
    x = cx - span / 2
    for j, h in enumerate(heights):
        node("", x + j * 9, base - h, 6, h,
             f"rounded=0;html=1;fillColor={color};strokeColor=none;opacity=70;")


# ================================================================== canvas ===
W, H = 1140, 650
node("", 0, 0, W, H, f"rounded=0;html=1;fillColor={PAGE};strokeColor=none;")

PX0, PX1 = 120, 1060
BASE, TOP = 470, 270


def px(t):
    return PX0 + t * (PX1 - PX0)


def py(v):
    return BASE - v * (BASE - TOP)


def score(t, c, a, w):
    return a * exp(-(((t - c) / w) ** 2))


def curve(c, a, w, n=64):
    return [(px(i / n), py(score(i / n, c, a, w))) for i in range(n + 1)]


TEAMS = [("θ<sub>1</sub>", 0.28, 1.00, 0.15, A,
          ["poke_94.png", "poke_6.png", "poke_445.png"], (12, 5, 8, 3, 10, 6),
          "Timid&#160;·&#160;252 Spe / 252 SpA&#160;·&#160;Life Orb"),
         ("θ<sub>2</sub>", 0.70, 0.97, 0.16, B,
          ["poke_143.png", "poke_9.png", "poke_3.png"], (4, 11, 12, 7, 3, 8),
          "Relaxed&#160;·&#160;252 HP / 252 Def&#160;·&#160;Leftovers")]
(_1, c1, a1, w1, *_r1), (_2, c2, a2, w2, *_r2) = TEAMS

flip = min((abs(score(i / 400, c1, a1, w1) - score(i / 400, c2, a2, w2)), i / 400)
           for i in range(int(0.28 * 400), int(0.70 * 400)))[1]

text("POKÉMON SHOWDOWN&#160;&#160;·&#160;&#160;A CO-DESIGN BENCHMARK",
     60, 16, 1020, 18, size=11, color=MUTED, bold=True, spacing=3)
text(f"<span style='color:{FAIL}'>NO</span> BEST TEAM."
     f"&#160;&#160;&#160;<span style='color:{FAIL}'>NO</span> BEST POLICY.",
     60, 38, 1020, 50, size=34, bold=True, spacing=1)
text("the best team depends on the policy&#160;&#160;—&#160;&#160;"
     "and the best policy depends on the team",
     60, 92, 1020, 20, size=13, color=MUTED)

# --------------- which team is ahead, wherever the policy sits -------------
node("", PX0, TOP - 20, px(flip) - PX0, BASE - TOP + 20,
     f"rounded=0;html=1;fillColor={A_BG};strokeColor=none;")
node("", px(flip), TOP - 20, PX1 - px(flip), BASE - TOP + 20,
     f"rounded=0;html=1;fillColor={B_BG};strokeColor=none;")
path(f"edgeStyle=none;html=1;strokeWidth=1.5;strokeColor={FAINT};endArrow=none;",
     [(px(flip), TOP - 20), (px(flip), BASE)])

AXIS = f"edgeStyle=none;html=1;strokeWidth=1.5;strokeColor={FAINT};endArrow=none;"
path(AXIS, [(PX0 - 20, BASE), (PX1 + 20, BASE)])
path(AXIS, [(PX0 - 20, BASE), (PX0 - 20, TOP - 20)])
node("SCORE&#160;&#160;R", 66, 295, 22, 150,
     f"text;html=1;horizontal=0;strokeColor=none;fillColor=none;align=center;"
     f"verticalAlign=middle;fontFamily={F};fontSize=10.5;fontColor={MUTED};"
     f"fontStyle=1;letterSpacing=1.5;")

# --------- the curve family IS the team design space; name it as such -----
text("TEAM DESIGN SPACE&#160;&#160;Θ&#160;&#160;·&#160;&#160;<b>discrete</b>"
     "&#160;&#160;·&#160;&#160;one curve per team, ~10<sup>25</sup> of them",
     PX0, 140, PX1 - PX0, 18, size=12.5, color=INK, bold=True, spacing=1)
text("θ&#160;=&#160;( species&#160;·&#160;moves&#160;·&#160;EVs&#160;·&#160;item"
     "&#160;·&#160;nature&#160;·&#160;ability )&#160;×&#160;6 slots",
     PX0, 160, PX1 - PX0, 16, size=10.5, color=MUTED)

CROWD = [
    (0.08, .52, .11, ["poke_10.png", "poke_129.png", "poke_191.png"]),
    (0.18, .64, .13, ["poke_68.png", "poke_143.png", "poke_248.png"]),
    (0.36, .58, .10, ["poke_3.png", "poke_130.png", "poke_94.png"]),
    (0.45, .72, .15, ["poke_445.png", "poke_9.png", "poke_6.png"]),
    (0.56, .55, .11, ["poke_149.png", "poke_68.png", "poke_3.png"]),
    (0.64, .68, .13, ["poke_130.png", "poke_248.png", "poke_9.png"]),
    (0.82, .60, .12, ["poke_94.png", "poke_143.png", "poke_445.png"]),
    (0.92, .50, .10, ["poke_6.png", "poke_149.png", "poke_129.png"]),
]
for c, a, w, roster in CROWD:
    path(f"edgeStyle=none;html=1;strokeWidth=1.5;strokeColor={CROWD_C};"
         f"endArrow=none;", curve(c, a, w))
    team_chip(px(c), py(a) - 24, roster, 18, 3, opacity=55)

# ---------------------------------------- two teams, two different optima --
for tag, c, a, w, colour, team, spread, params in TEAMS:
    path(f"edgeStyle=none;html=1;strokeWidth=3;strokeColor={colour};endArrow=none;",
         curve(c, a, w))
    peak_x, peak_y = px(c), py(a)
    path(f"edgeStyle=none;html=1;strokeWidth=1.2;strokeColor={colour};endArrow=none;"
         f"opacity=45;", [(peak_x, peak_y), (peak_x, BASE)])
    node("", peak_x - 7, peak_y - 7, 14, 14,
         f"ellipse;html=1;fillColor={colour};strokeColor={PAGE};strokeWidth=2.5;")
    text(f"<b>{tag}</b>&#160;&#160;{params}", peak_x - 130, peak_y - 84, 260, 14,
         size=9, color=INK)                             # one entry in Theta ...
    team_chip(peak_x, peak_y - 66, team, 27, 2)
    bars(peak_x, peak_y - 24, spread, colour)           # ... and its full spec
    text(f"π*({tag})", peak_x - 50, 478, 100, 18, size=12, color=INK, bold=True)

text("θ<sub>1</sub> is ahead here", PX0, BASE - 30, px(flip) - PX0, 18,
     size=11.5, color=INK, bold=True)
text("θ<sub>2</sub> is ahead here", px(flip), BASE - 30, PX1 - px(flip), 18,
     size=11.5, color=INK, bold=True)
text("POLICY SPACE&#160;&#160;Π&#160;&#160;·&#160;&#160;<b>continuous</b>",
     PX0, 504, PX1 - PX0, 18, size=12.5, color=INK, bold=True, spacing=1)
text("π&#160;=&#160;network weights&#160;∈&#160;ℝ<sup>d</sup>",
     PX0, 524, PX1 - PX0, 16, size=10.5, color=MUTED)

node("ONLY BEST PAIRS.&#160;&#160;SEARCH THEM TOGETHER.", 60, 566, 1020, 44,
     f"rounded=1;whiteSpace=wrap;html=1;arcSize=40;fillColor={INK};strokeColor={INK};"
     f"strokeWidth=3;fontFamily={F};fontSize=18;fontColor={PAGE};fontStyle=1;"
     f"align=center;verticalAlign=middle;letterSpacing=3;")
text("Schematic, not measured results&#160;&#160;·&#160;&#160;"
     "sprites: PokeAPI/sprites&#160;&#160;·&#160;&#160;"
     "Pokémon © Nintendo / Creatures Inc. / GAME FREAK inc.&#160;&#160;·&#160;&#160;"
     "see assets/CREDITS.md",
     60, 620, 1020, 16, size=8.5, color="#A9B4C2")

xml = (
    '<mxfile host="build_hero.py" modified="0" agent="python" version="24.0.0" type="device">'
    '<diagram id="hero" name="Why Co-Design">'
    f'<mxGraphModel dx="0" dy="0" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" '
    f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">'
    '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
    + "".join(cells) +
    "</root></mxGraphModel></diagram></mxfile>"
)
out = HERE / "codesign_hero.drawio"
out.write_text(xml, encoding="utf-8")
print(f"wrote {out}  ({len(xml) / 1024:.0f} KB, {len(cells)} cells)")
