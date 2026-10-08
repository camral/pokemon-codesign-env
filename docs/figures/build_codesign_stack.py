"""codesign_stack: the same design object at three scales.

One image. The landscape carries Theta x Pi -> score and the co-design search
climbing to (theta*, pi*); at the summit the winning team blooms open into a
hexagon of six slots wired to each other, and one of those slots is exploded
into its decision variables.

The point of drawing them together rather than side by side: the theta<->pi
coupling the landscape sells is not the only coupling in the problem. It repeats
inside theta, and each slot is itself a small mixed-integer program. Structure
all the way down, in one picture.

Numbers are read from this repo's gen4customgame catalogs — see figure_kit.
Schematic: no tick values, because there is no data here.
"""

from math import exp

from figure_kit import *

W, H = 1200, 706
reset(); background(W, H)

# sequential ramp for search progress (validate_palette.js --ordinal: ALL PASS)
RAMP = ("#E8A855", "#D48420", "#B45309", "#8A3F06", "#5E2A03")
MESH = "#A3AEBC"

# ============================================================== landscape ===
OX, OY, PW, SH, DX, DY, N, BEST = 170, 545, 740, 150, 44, 30, 5, 4
proj = lambda t, v, i: (OX + t * PW + i * DX, OY - v * SH - i * DY)
SHAPE = [[(0.30, .55, .090)], [(0.44, .66, .120)], [(0.36, .74, .085)],
         [(0.58, .85, .110)], [(0.78, 1.00, .150)]]
ROSTERS = [["poke_10.png", "poke_129.png", "poke_191.png"],
           ["poke_143.png", "poke_129.png", "poke_191.png"],
           ["poke_143.png", "poke_9.png", "poke_191.png"],
           ["poke_143.png", "poke_9.png", "poke_94.png"],
           ["poke_143.png", "poke_9.png", "poke_3.png"]]
val = lambda t, cs: sum(a * exp(-(((t - c) / w) ** 2)) for c, a, w in cs)
peak_of = lambda cs: max(range(401), key=lambda k: val(k / 400, cs)) / 400


def profile(i, n=72):
    cs = SHAPE[i]
    live = [k / 400 for k in range(401) if val(k / 400, cs) > 0.02]
    t0, t1 = (live[0], live[-1]) if live else (0.0, 1.0)
    return [proj(t0 + k * (t1 - t0) / n, val(t0 + k * (t1 - t0) / n, cs), i)
            for k in range(n + 1)]


eyebrow("POKÉMON SHOWDOWN&#160;&#160;·&#160;&#160;A CO-DESIGN BENCHMARK", 60, 16, 1080)
text(f"<span style='color:{ACCENT}'>NO</span> BEST TEAM."
     f"&#160;&#160;<span style='color:{ACCENT}'>NO</span> BEST POLICY."
     f"&#160;&#160;<span style='color:{TEAM}'>ONE BEST PAIR.</span>",
     60, 36, 1080, 50, size=32, bold=True, spacing=1)

FLINE = f"edgeStyle=none;html=1;strokeWidth=1.2;strokeColor=#DDE3EA;endArrow=none;"
for t in (0.25, 0.5, 0.75):
    path(FLINE, [proj(t, 0, 0), proj(t, 0, N - 1)])
for i in range(1, N):
    path(FLINE, [proj(0, 0, i), proj(1, 0, i)])
path(FLINE, [proj(1, 0, 0), proj(1, 0, N - 1)])

for i in range(N - 1, -1, -1):
    path(f"edgeStyle=none;html=1;strokeWidth={12 if i == BEST else 10};"
         f"strokeColor={PAGE};endArrow=none;", profile(i))
    path(f"edgeStyle=none;html=1;strokeWidth={3.2 if i == BEST else 2.0 + i * 0.15};"
         f"strokeColor={RAMP[i]};endArrow=none;", profile(i))

AX = (f"edgeStyle=none;html=1;strokeWidth=1.8;strokeColor={FAINT};endArrow=block;"
      f"endFill=1;endSize=5;")
path(AX, [proj(0, 0, 0), (OX + PW + 26, OY)])
path(AX, [proj(0, 0, 0), (OX + (N - 1) * DX + 24, OY - (N - 1) * DY - 18)])
path(AX, [proj(0, 0, 0), (OX, OY - SH - 16)])

peaks = [proj(peak_of(SHAPE[i]), val(peak_of(SHAPE[i]), SHAPE[i]), i) for i in range(N)]
for k in range(N - 1):
    path(f"edgeStyle=none;html=1;strokeWidth=7;strokeColor={PAGE};endArrow=none;",
         [peaks[k], peaks[k + 1]])
    path(f"edgeStyle=none;html=1;strokeWidth={2.2 + k * 0.3};strokeColor={INK};"
         f"endArrow=block;endFill=1;endSize=4;", [peaks[k], peaks[k + 1]])
for i, (px_, py_) in enumerate(peaks):
    if i != BEST:
        dot(px_, py_, 8, RAMP[i])

bx, by = peaks[BEST]
bt = peak_of(SHAPE[BEST])
fx, fy = proj(bt, 0, BEST)
DROP = (f"edgeStyle=none;html=1;strokeWidth=1.5;strokeColor={TEAM};endArrow=none;"
        f"dashed=1;dashPattern=5 4;opacity=55;")
path(DROP, [(bx, by), (fx, fy)])
path(DROP, [(fx, fy), proj(bt, 0, 0)])
path(DROP, [(fx, fy), proj(0, 0, BEST)])
for ax, ay in (proj(bt, 0, 0), proj(0, 0, BEST)):
    dot(ax, ay, 8, TEAM, PAGE, 1.5)
text("π*", proj(bt, 0, 0)[0] - 20, 552, 40, 16, size=12, color=TEAM, bold=True)
text("θ*", proj(0, 0, BEST)[0] - 52, proj(0, 0, BEST)[1] + 10, 30, 16,
     size=12, color=TEAM, bold=True)
for dx_, dy_ in ((0, -22), (19, -11), (19, 11), (0, 22), (-19, 11), (-19, -11)):
    path(f"edgeStyle=none;html=1;strokeWidth=4;strokeColor={ZAP};endArrow=none;rounded=1;",
         [(bx + dx_ * 0.8, by + dy_ * 0.8), (bx + dx_ * 1.55, by + dy_ * 1.55)])
dot(bx, by, 22, RAMP[BEST], PAGE, 3)

for i in range(N - 1):                       # the summit team is drawn as a hexagon
    px_, py_ = peaks[i]
    size = 16 + i * 2
    cx, top = px_ - 52, py_ - 34
    roster(cx, top, ROSTERS[i], size, 2, opacity=60 + i * 10)
    if i:
        span = 3 * size + 4
        slot = next(k for k in range(3) if ROSTERS[i][k] != ROSTERS[i - 1][k])
        sx = cx - span / 2 + slot * (size + 2)
        node("", sx - 3, top - 3, size + 6, size + 6,
             f"ellipse;html=1;fillColor=none;strokeColor={RAMP[i]};strokeWidth=2;")

text("CO-DESIGN SEARCH", 500, 150, 300, 18, size=12.5, color=INK, bold=True, spacing=1)
text("each step redesigns the team and trains the controller", 450, 168, 400, 16,
     size=10.5, color=MUTED)
path(f"edgeStyle=none;html=1;strokeWidth=1.2;strokeColor={FAINT};endArrow=none;opacity=60;",
     [(622, 188), (601, 348)])
text("TEAM&#160;&#160;Θ", 176, 352, 190, 18, size=12.5, color=INK, bold=True, spacing=1)
text("<b>discrete</b>&#160;·&#160;~10<sup>162</sup> teams", 176, 370, 190, 16,
     size=9.5, color=MUTED)
vtext("SCORE&#160;&#160;R", 136, 392, 22, 130, size=11, color=INK)
text("POLICY&#160;&#160;Π&#160;&#160;·&#160;&#160;<b>continuous</b>&#160;&#160;·&#160;&#160;"
     "π ∈ ℝ<sup>d</sup>", OX, 580, PW, 20, size=12.5, color=INK, bold=True, spacing=1)

# ============ at the summit, the winning team opens into its coupled structure =
from math import cos, radians, sin

HCX, HCY, HR, HRING = 930, 196, 48, 2
V = [(HCX + HR * cos(radians(a)), HCY + HR * sin(radians(a)))
     for a in (-90, -30, 30, 90, 150, 210)]
HEX_TEAM = ["poke_143.png", "poke_9.png", "poke_3.png",
            "poke_445.png", "poke_94.png", "poke_6.png"]
for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4), (2, 5)]:
    live = HRING in (a, b)
    path(f"edgeStyle=none;html=1;endArrow=none;strokeWidth={2.4 if live else 1.4};"
         f"strokeColor={TEAM if live else '#B4C0CC'};", [V[a], V[b]])
for k, (vx, vy) in enumerate(V):
    if k == HRING:                           # the slot the last step redesigned
        node("", vx - 20, vy - 20, 40, 40,
             f"ellipse;html=1;fillColor={TEAM_BG};strokeColor={TEAM};strokeWidth=2.5;")
    sprite(HEX_TEAM[k], vx - 13, vy - 13, 26, 26)
text("<b>( θ*, π* )</b>&#160;&#160;six slots, wired to each other",
     HCX - 150, 108, 300, 16, size=10.5, color=INK)

# ------------------------- and one of those slots, opened up in turn ---------
path(f"edgeStyle=none;html=1;endArrow=none;strokeWidth=1.2;strokeColor={FAINT};"
     f"dashed=1;dashPattern=4 4;opacity=70;", [(V[1][0] + 12, V[1][1]), (1006, 176)])
text("one slot", 1006, 142, 180, 14, size=9.5, color=MUTED, bold=True, align="left")
text("moves&#160;·&#160;choose 4 of 73", 1006, 160, 180, 12, size=8.5, color=MUTED, align="left")
pips(1010, 176, 10, {1, 3, 4, 8}, 10, 4, TEAM, HAIR)
text("EVs&#160;·&#160;integer, ≤ 255", 1006, 198, 180, 12, size=8.5, color=MUTED, align="left")
for k, v in enumerate((250, 96, 168, 40, 220, 112)):
    node("", 1010 + k * 20, 250 - v * 0.13, 14, v * 0.13,
         f"rounded=0;html=1;fillColor={TEAM_CHIP};strokeColor=none;")
path(f"edgeStyle=none;html=1;endArrow=none;strokeWidth=1.1;strokeColor={ACCENT};"
     f"dashed=1;dashPattern=4 3;", [(1004, 217), (1136, 217)])

node("ONLY BEST PAIRS.&#160;&#160;SEARCH THEM TOGETHER.", 60, 620, 1080, 44,
     f"rounded=1;whiteSpace=wrap;html=1;arcSize=40;fillColor={INK};strokeColor={INK};"
     f"strokeWidth=3;fontFamily={F};fontSize=18;fontColor={PAGE};fontStyle=1;"
     f"align=center;verticalAlign=middle;letterSpacing=3;")
credit(60, 676, 1080, "Schematic, not measured results&#160;&#160;·&#160;&#160;")
emit("codesign_stack.drawio", "Co-design stack", W, H)
