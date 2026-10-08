"""Generate docs/figures/codesign_hero_3d.drawio: all three axes at once.

The 2D version (build_hero.py) collapses the team axis into a family of curves.
This one puts it on screen as depth, in an oblique projection:

    POLICY  Pi   -> across   (continuous)
    TEAM    Theta -> into depth (discrete, one row per team)
    SCORE   R    -> up

Each row is one team's score profile over policy space, and the bold ridge joins
every row's own optimum. The ridge never straightens: that is the claim. It also
wanders rather than sloping, because Theta has NO natural ordering — adjacent
rows are just adjacent samples out of ~10^25, so neighbouring teams can want
completely different policies.

drawio has no 3D and no polygon fill, so depth is done by hand: rows are drawn
back-to-front, each with a thick white halo under its stroke, which gives
hidden-line removal for free.

Magnitudes come from this repo's gen4customgame config: 554 legal species x 25
natures over 6 slots ~ 1e24.8, before move sets, EVs, items or abilities.
Schematic: no tick values anywhere, because there is no data here.

Re-run after editing, then export with:
    /Applications/draw.io.app/Contents/MacOS/draw.io -x -f png -s 2 -o codesign_hero_3d.png codesign_hero_3d.drawio
"""

import base64
from html import escape
from math import exp, log, sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

INK = "#0F172A"
MUTED = "#64748B"
FAINT = "#94A3B8"
FLOOR = "#DDE3EA"
MESH = "#A3AEBC"
PAGE = "#FFFFFF"
FAIL = "#DC2626"
A, B = "#1D4ED8", "#B45309"
# search progress is an ordered quantity, so the rows take a sequential ramp
# (validate_palette.js --ordinal: monotone L, step gaps, light end clears surface)
RAMP = ("#E8A855", "#D48420", "#B45309", "#8A3F06", "#5E2A03")

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


def spec(cx, base, heights, colour):
    """EV-spread bars: an entry in Theta is a full spec, not just six species."""
    span = len(heights) * 6 + (len(heights) - 1) * 3
    for j, h in enumerate(heights):
        node("", cx - span / 2 + j * 9, base - h, 6, h,
             f"rounded=0;html=1;fillColor={colour};strokeColor=none;opacity=75;")


def roster(cx, top, files, size, gap, opacity=100):
    span = len(files) * size + (len(files) - 1) * gap
    for k, f in enumerate(files):
        sprite(f, cx - span / 2 + k * (size + gap), top, size, size, opacity)


# =============================================================== projection ==
W, H = 1180, 706
node("", 0, 0, W, H, f"rounded=0;html=1;fillColor={PAGE};strokeColor=none;")

OX, OY = 170, 545       # front-left-bottom corner
PW, SH = 740, 150       # policy run, score rise
DX, DY = 44, 30         # one step into depth — few rows, well separated
N = 5


def proj(t, v, i):
    return (OX + t * PW + i * DX, OY - v * SH - i * DY)


# team, peak location, amplitude, width, roster
# each row is a mixture of bumps, so teams differ in KIND: a sharp specialist that
# only works with one policy, a broad tolerant team, one with two viable playstyles
# peak HEIGHTS stay close together on purpose: then the ridge only wanders
# sideways, and the message is "the optimum moves in policy", not "up and down"
SHAPE = [
    [(0.30, .55, .090)],
    [(0.44, .66, .120)],
    [(0.36, .74, .085)],                    # the best policy steps back here
    [(0.58, .85, .110)],
    [(0.78, 1.00, .150)],                   # the co-designed pair: the global optimum
]
BEST = 4                                    # the row carrying the joint optimum

# one slot of the team is redesigned at each step, converging on the final roster
ROSTERS = [
    ["poke_10.png", "poke_129.png", "poke_191.png"],
    ["poke_143.png", "poke_129.png", "poke_191.png"],
    ["poke_143.png", "poke_9.png", "poke_191.png"],
    ["poke_143.png", "poke_9.png", "poke_94.png"],
    ["poke_143.png", "poke_9.png", "poke_3.png"],
]


def value(t, comps):
    return sum(a * exp(-(((t - c) / w) ** 2)) for c, a, w in comps)


def support(comps, floor=0.02, m=400):
    """The stretch of policy space where this team scores at all."""
    live = [k / m for k in range(m + 1) if value(k / m, comps) > floor]
    return (live[0], live[-1]) if live else (0.0, 1.0)


def peak_of(comps, m=400):
    return max(range(m + 1), key=lambda k: value(k / m, comps)) / m


def profile(i, n=72):
    comps = SHAPE[i]
    t0, t1 = support(comps)
    ts = [t0 + k * (t1 - t0) / n for k in range(n + 1)]
    return [proj(t, value(t, comps), i) for t in ts]


text("POKÉMON SHOWDOWN&#160;&#160;·&#160;&#160;A CO-DESIGN BENCHMARK",
     60, 16, 1060, 18, size=11, color=MUTED, bold=True, spacing=3)
text(f"<span style='color:{FAIL}'>NO</span> BEST TEAM."
     f"&#160;&#160;<span style='color:{FAIL}'>NO</span> BEST POLICY."
     f"&#160;&#160;<span style='color:{B}'>ONE BEST PAIR.</span>",
     60, 36, 1060, 50, size=32, bold=True, spacing=1)
# ------------------------------------------------------------- the floor ---
FLINE = f"edgeStyle=none;html=1;strokeWidth=1.2;strokeColor={FLOOR};endArrow=none;"
for t in (0.25, 0.5, 0.75):
    path(FLINE, [proj(t, 0, 0), proj(t, 0, N - 1)])
for i in range(1, N - 1):
    path(FLINE, [proj(0, 0, i), proj(1, 0, i)])
path(FLINE, [proj(1, 0, 0), proj(1, 0, N - 1)])
path(FLINE, [proj(0, 0, N - 1), proj(1, 0, N - 1)])

# ------------------------- rows, back to front, each masking what's behind --
for i in range(N - 1, -1, -1):
    path(f"edgeStyle=none;html=1;strokeWidth={12 if i == BEST else 10};"
         f"strokeColor={PAGE};endArrow=none;", profile(i))
    path(f"edgeStyle=none;html=1;strokeWidth={3.2 if i == BEST else 2.0 + i * 0.15};"
         f"strokeColor={RAMP[i]};endArrow=none;", profile(i))

# ----------------------------------------------------------------- axes ----
AX = f"edgeStyle=none;html=1;strokeWidth=1.8;strokeColor={FAINT};endArrow=block;endFill=1;endSize=5;"
path(AX, [proj(0, 0, 0), (OX + PW + 26, OY)])                      # policy, across
path(AX, [proj(0, 0, 0), (OX + (N - 1) * DX + 24, OY - (N - 1) * DY - 18)])  # team, into depth
path(AX, [proj(0, 0, 0), (OX, OY - SH - 16)])                      # score, up

# ------------------------------ the ridge: each team's own best policy -----
peaks = [proj(peak_of(SHAPE[i]), value(peak_of(SHAPE[i]), SHAPE[i]), i)
         for i in range(N)]
for k in range(N - 1):
    path(f"edgeStyle=none;html=1;strokeWidth=7;strokeColor={PAGE};endArrow=none;",
         [peaks[k], peaks[k + 1]])
    path(f"edgeStyle=none;html=1;strokeWidth={2.2 + k * 0.3};strokeColor={INK};"
         f"endArrow=block;endFill=1;endSize=4;", [peaks[k], peaks[k + 1]])
for i, (pxx, pyy) in enumerate(peaks):
    if i == BEST:
        continue                            # the joint optimum is marked separately
    node("", pxx - 4, pyy - 4, 8, 8,
         f"ellipse;html=1;fillColor={RAMP[i]};strokeColor={PAGE};strokeWidth=2;")

bx, by = peaks[BEST]
bt = peak_of(SHAPE[BEST])
fx, fy = proj(bt, 0, BEST)
DROP = (f"edgeStyle=none;html=1;strokeWidth=1.5;strokeColor={B};endArrow=none;"
        f"dashed=1;dashPattern=5 4;opacity=55;")
path(DROP, [(bx, by), (fx, fy)])                 # the optimum, down onto the floor
path(DROP, [(fx, fy), proj(bt, 0, 0)])           # ... across to the policy axis
path(DROP, [(fx, fy), proj(0, 0, BEST)])         # ... and back to the team axis
for ax, ay in (proj(bt, 0, 0), proj(0, 0, BEST)):
    node("", ax - 4, ay - 4, 8, 8,
         f"ellipse;html=1;fillColor={B};strokeColor={PAGE};strokeWidth=1.5;")
text("π*", proj(bt, 0, 0)[0] - 20, 552, 40, 16, size=12, color=B, bold=True)
text("θ*", proj(0, 0, BEST)[0] - 52, proj(0, 0, BEST)[1] + 10, 30, 16,
     size=12, color=B, bold=True)
for dx_, dy_ in ((0, -22), (19, -11), (19, 11), (0, 22), (-19, 11), (-19, -11)):
    path("edgeStyle=none;html=1;strokeWidth=4;strokeColor=#FACC15;endArrow=none;rounded=1;",
         [(bx + dx_ * 0.8, by + dy_ * 0.8), (bx + dx_ * 1.55, by + dy_ * 1.55)])
node("", bx - 11, by - 11, 22, 22,
     f"ellipse;html=1;fillColor={RAMP[BEST]};strokeColor={PAGE};strokeWidth=3;")

text("CO-DESIGN SEARCH", 500, 150, 300, 18, size=12.5, color=INK, bold=True, spacing=1)
text("each step redesigns the team and trains the controller",
     450, 168, 400, 16, size=10.5, color=MUTED)
path(f"edgeStyle=none;html=1;strokeWidth=1.2;strokeColor={INK};endArrow=none;opacity=50;",
     [(622, 188), (601, 348)])

# ------------------------------------- the two rows drawn out of the crowd -
for i in range(N):
    pxx, pyy = peaks[i]
    size = 16 + i * 2 + (4 if i == BEST else 0)
    cx, top = pxx - 52, pyy - 34
    roster(cx, top, ROSTERS[i], size, 2, opacity=60 + i * 10)
    if i:                                   # ring the one slot redesigned this step
        span = 3 * size + 2 * 2
        slot = next(k for k in range(3) if ROSTERS[i][k] != ROSTERS[i - 1][k])
        sx = cx - span / 2 + slot * (size + 2)
        node("", sx - 3, top - 3, size + 6, size + 6,
             f"ellipse;html=1;fillColor=none;strokeColor={RAMP[i]};strokeWidth=2;")
spec(peaks[BEST][0] - 52, peaks[BEST][1] + 8, (11, 4, 8, 3, 10, 6), RAMP[BEST])
text("<b>( θ*, π* )</b>&#160;&#160;the co-designed pair",
     peaks[BEST][0] - 130, peaks[BEST][1] - 62, 260, 16, size=10.5, color=INK)

# ---------------------------------------------------------------- labels ---
text("POLICY&#160;&#160;Π&#160;&#160;·&#160;&#160;<b>continuous</b>&#160;&#160;·&#160;&#160;π ∈ ℝ<sup>d</sup>",
     OX, 580, PW, 20, size=12.5, color=INK, bold=True, spacing=1)
node("SCORE&#160;&#160;R", 136, 392, 22, 130,
     f"text;html=1;horizontal=0;strokeColor=none;fillColor=none;align=center;"
     f"verticalAlign=middle;fontFamily={F};fontSize=11;fontColor={INK};"
     f"fontStyle=1;letterSpacing=1;")
text("TEAM&#160;&#160;Θ", 176, 352, 190, 18, size=12.5, color=INK, bold=True, spacing=1)
text("<b>discrete</b>&#160;·&#160;~10<sup>163</sup> teams", 176, 370, 190, 16,
     size=9.5, color=MUTED)


node("ONLY BEST PAIRS.&#160;&#160;SEARCH THEM TOGETHER.", 60, 622, 1060, 44,
     f"rounded=1;whiteSpace=wrap;html=1;arcSize=40;fillColor={INK};strokeColor={INK};"
     f"strokeWidth=3;fontFamily={F};fontSize=18;fontColor={PAGE};fontStyle=1;"
     f"align=center;verticalAlign=middle;letterSpacing=3;")
text("Schematic, not measured results&#160;&#160;·&#160;&#160;"
     "sprites: PokeAPI/sprites&#160;&#160;·&#160;&#160;"
     "Pokémon © Nintendo / Creatures Inc. / GAME FREAK inc.&#160;&#160;·&#160;&#160;"
     "see assets/CREDITS.md",
     60, 676, 1060, 16, size=8.5, color="#A9B4C2")

xml = (
    '<mxfile host="build_hero_3d.py" modified="0" agent="python" version="24.0.0" type="device">'
    '<diagram id="hero3d" name="Why Co-Design (3D)">'
    f'<mxGraphModel dx="0" dy="0" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" '
    f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">'
    '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
    + "".join(cells) +
    "</root></mxGraphModel></diagram></mxfile>"
)
out = HERE / "codesign_hero_3d.drawio"
out.write_text(xml, encoding="utf-8")
print(f"wrote {out}  ({len(xml) / 1024:.0f} KB, {len(cells)} cells)")
