"""Generate docs/figures/codesign.drawio: the co-design benchmark overview figure.

The figure reads as a funnel that closes into a loop:
    designer -> design space  ->\
                                 >-- one battle on Showdown --> R --,
    policy   -> battle MDP    ->/                                   |
      ^-------------------------------------------------------------'

Sprites are embedded as base64 data URIs so the .drawio file is self-contained.
Re-run after editing, then export with:
    /Applications/draw.io.app/Contents/MacOS/draw.io -x -f png -s 2 -o codesign.png codesign.drawio
"""

import base64
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

# ---------------------------------------------------------------- palette ---
INK = "#0F172A"
MUTED = "#64748B"
FAINT = "#94A3B8"
PAGE = "#FFFFFF"

USER = "#475569"          # your method — deliberately neutral
USER_CHIP = "#E9EEF4"

SPACE = "#B45309"         # design space Θ   (left column)
SPACE_BG = "#FEF7EA"
SPACE_CHIP = "#FDEBCB"

MDP = "#047857"           # battle MDP       (right column)
MDP_BG = "#EDFBF4"
MDP_CHIP = "#D0F3E3"

SCORE = "#DC2626"         # the shared scalar
SCORE_CHIP = "#FCDCDA"

SUITE = "#6D28D9"
SUITE_BG = "#F7F2FF"
SUITE_CHIP = "#EADDFF"

SIM = "#334155"
SIM_CHIP = "#E2E8F0"

F = "Helvetica"

cells: list[str] = []
_uid = [0]


def nid(prefix="n"):
    _uid[0] += 1
    return f"{prefix}{_uid[0]}"


def node(value, x, y, w, h, style):
    i = nid()
    cells.append(
        f'<mxCell id="{i}" value="{escape(value)}" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
    )
    return i


def edge(style, points, label="", label_dx=0, label_dy=-13, label_color=None, label_size=11):
    """points: [(x, y), ...] in absolute coordinates; first is source, last is target."""
    i = nid("e")
    (sx, sy), (tx, ty) = points[0], points[-1]
    mids = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in points[1:-1])
    arr = f'<Array as="points">{mids}</Array>' if mids else ""
    cells.append(
        f'<mxCell id="{i}" style="{style}" edge="1" parent="1">'
        f'<mxGeometry relative="1" as="geometry">'
        f'<mxPoint x="{sx}" y="{sy}" as="sourcePoint"/>'
        f'<mxPoint x="{tx}" y="{ty}" as="targetPoint"/>{arr}</mxGeometry></mxCell>'
    )
    if label:
        lx = (points[0][0] + points[-1][0]) / 2 + label_dx
        ly = (points[0][1] + points[-1][1]) / 2 + label_dy
        cells.append(
            f'<mxCell id="{nid("l")}" value="{escape(label)}" '
            f'style="edgeLabel;html=1;align=center;verticalAlign=middle;fontFamily={F};'
            f'fontSize={label_size};fontColor={label_color or INK};labelBackgroundColor={PAGE};" '
            f'vertex="1" connectable="0" parent="{i}">'
            f'<mxGeometry x="{lx}" y="{ly}" as="geometry"/></mxCell>'
        )
    return i


def text(value, x, y, w, h, size=12, color=INK, align="center", bold=False,
         valign="middle", spacing=0):
    v = f"<b>{value}</b>" if bold else value
    style = (
        f"text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;overflow=visible;"
        f"align={align};verticalAlign={valign};fontFamily={F};fontSize={size};"
        f"fontColor={color};letterSpacing={spacing};"
    )
    return node(v, x, y, w, h, style)


def box(value, x, y, w, h, fill, stroke, size=12, color=INK, sw=1.5, bold=False,
        align="center", valign="middle", dashed=False, arcsize=24, spacing=0):
    v = f"<b>{value}</b>" if bold else value
    style = (
        f"rounded=1;whiteSpace=wrap;html=1;arcSize={arcsize};"
        f"fillColor={fill};strokeColor={stroke};strokeWidth={sw};"
        f"fontFamily={F};fontSize={size};fontColor={color};align={align};"
        f"verticalAlign={valign};letterSpacing={spacing};"
        f"{'dashed=1;dashPattern=8 5;' if dashed else 'dashed=0;'}"
    )
    return node(v, x, y, w, h, style)


def chip(value, x, y, w, h, fill, stroke, color=INK, size=10.5, bold=False):
    return box(value, x, y, w, h, fill, stroke, size=size, color=color, arcsize=50, sw=1, bold=bold)


def sprite(fname, x, y, w, h):
    data = base64.b64encode((ASSETS / fname).read_bytes()).decode()
    style = (f"shape=image;html=1;imageAspect=0;aspect=fixed;"
             f"verticalLabelPosition=bottom;image=data:image/png,{data};")
    return node("", x, y, w, h, style)


def header(x, y, w, h, fill, title, kicker, badge_num=None, kicker_color="#FFFFFF",
           title_size=18, kicker_w=300):
    """Coloured header bar with a square bottom edge so it seats on its panel."""
    box("", x, y, w, h, fill, fill, sw=2, arcsize=18)
    node("", x, y + h / 2, w, h / 2, f"rounded=0;html=1;fillColor={fill};strokeColor=none;")
    tx = x + 14
    if badge_num is not None:
        node(f"<b>{badge_num}</b>", x + 14, y + (h - 26) / 2, 26, 26,
             f"ellipse;whiteSpace=wrap;html=1;fillColor=#FFFFFF44;strokeColor=none;"
             f"fontFamily={F};fontSize=14;fontColor=#FFFFFF;align=center;verticalAlign=middle;")
        tx = x + 50
    text(title, tx, y + 4, w - 60, h - 8, size=title_size, color="#FFFFFF",
         align="left", bold=True, spacing=1)
    if kicker:
        text(kicker, x + w - kicker_w - 14, y + 4, kicker_w, h - 8,
             size=12, color=kicker_color, align="right")


ARROW = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;jettySize=auto;orthogonalLoop=1;"

# ================================================================= canvas ===
W, H = 1720, 1300
node("", 0, 0, W, H, f"rounded=0;html=1;fillColor={PAGE};strokeColor=none;")

LX0, LX1 = 130, 800       # left  column: designer -> design space Θ
RX0, RX1 = 920, 1590      # right column: policy   -> battle MDP
LC, RC = 465, 1255        # column centres
LOOP_L, LOOP_R = 76, 1654 # the return path wraps outside everything

# ================================================================== title ===
text(f"Pokémon Showdown: a <span style='color:{SCORE}'>Co-Design</span> Benchmark",
     LX0, 24, RX1 - LX0, 46, size=34, bold=True)
text("Two coupled search spaces&#160;&#160;·&#160;&#160;one authoritative simulator"
     "&#160;&#160;·&#160;&#160;one score",
     LX0, 70, RX1 - LX0, 24, size=16, color=MUTED)

# ======================================================== objective banner ===
box("", LX0, 114, RX1 - LX0, 70, PAGE, INK, sw=2.5, arcsize=14)
text("max&#160;&#160;<b>E[ R ]</b>", 150, 120, 380, 32, size=24)
text(f"over&#160;&#160;( team <b style='color:{SPACE}'>θ</b> ∈ <b style='color:{SPACE}'>Θ</b>"
     f",&#160;&#160;policy <b style='color:{MDP}'>π</b> )", 150, 150, 380, 22, size=14)
node("", 556, 126, 1, 46, f"rounded=0;html=1;fillColor={FAINT};strokeColor={FAINT};")
text("<b>θ and π are coupled</b> — the strongest team depends on how it is played, and the strongest "
     "policy on what it was handed.<br>"
     "The benchmark fixes Θ, the battle MDP and the opponents. "
     "<b>How you search them is the open problem.</b>",
     588, 124, 990, 50, size=14, align="left")

# ============================================================== your method ==
text("WHAT YOU PLUG IN", LX0, 208, 400, 26, size=15, color=USER, align="left", bold=True, spacing=1)
text("no algorithm is prescribed — the benchmark scores whatever you bring",
     RX1 - 660, 210, 660, 24, size=12.5, color=MUTED, align="right")

SLOTS = [
    (LX0, LX1, "EMBODIMENT DESIGNER", "your method", SPACE,
     ["evolutionary search", "Bayesian optimisation", "LLM proposer", "random · hand-built"],
     "baseline in <b>examples/</b>: Nevergrad NGOpt — swap it out"),
    (RX0, RX1, "CONTROL POLICY", "your method", MDP,
     ["deep RL", "tree search / MCTS", "scripted heuristics", "LLM agent"],
     "baseline in <b>examples/</b>: SB3 MaskablePPO — swap it out"),
]
for x0, x1, name, kick, accent, methods, baseline in SLOTS:
    w = x1 - x0
    box("", x0, 240, w, 130, PAGE, accent, sw=2.5, dashed=True, arcsize=14)
    header(x0, 240, w, 36, USER, name, kick, kicker_color="#DBE3EC", title_size=15, kicker_w=160)
    for k, m in enumerate(methods):
        chip(m, x0 + 14 + (k % 2) * 327, 286 + (k // 2) * 32, 315, 28,
             USER_CHIP, FAINT, color=USER, size=11.5)
    chip(baseline, x0 + 14, 352, w - 28, 22, PAGE, accent, color=accent, size=10.5)

# ---- each socket wired straight into the component it drives ---------------
edge(ARROW + f"strokeWidth=4.5;strokeColor={SPACE};endArrow=block;endFill=1;",
     [(LC, 370), (LC, 420)], label="searches", label_dx=-62, label_dy=0,
     label_color=SPACE, label_size=12)
edge(ARROW + f"strokeWidth=4.5;strokeColor={MDP};endArrow=block;endFill=1;",
     [(1210, 370), (1210, 420)], label="obs", label_dx=-34, label_dy=0,
     label_color=MDP, label_size=12)
edge(ARROW + f"strokeWidth=4.5;strokeColor={MDP};endArrow=block;endFill=1;",
     [(1300, 420), (1300, 370)], label="action", label_dx=42, label_dy=0,
     label_color=MDP, label_size=12)

# ========================================================= design space Θ ====
PT, PB = 420, 720
box("", LX0, PT, LX1 - LX0, PB - PT, SPACE_BG, SPACE, sw=2.5, arcsize=14)
header(LX0, PT, LX1 - LX0, 38, SPACE, "DESIGN SPACE&#160;&#160;Θ", "what a team may be",
       badge_num=1, kicker_color="#FFE8C2", title_size=15, kicker_w=200)
for k, s in enumerate(["poke_3.png", "poke_6.png", "poke_9.png",
                       "poke_94.png", "poke_130.png", "poke_149.png"]):
    sprite(s, 180 + k * 100, 468, 70, 70)
text("six slots — every attribute is a decision variable", 144, 544, 642, 18, size=11.5, color=MUTED)
for k, a in enumerate(["species", "moves", "EVs / IVs", "nature",
                       "item", "ability", "gender", "tera"]):
    chip(a, 144 + (k % 4) * 163, 570 + (k // 4) * 36, 153, 30,
         SPACE_CHIP, SPACE, color=SPACE, bold=True, size=11.5)
chip("each axis independently:&#160;&#160;<b>fixed</b> · <b>random</b> · <b>searched</b>",
     144, 646, 642, 30, PAGE, SPACE_CHIP, color=INK, size=11.5)
chip("legal teams only — <b>validated by Showdown</b>", 144, 682, 642, 30,
     PAGE, SPACE_CHIP, color=INK, size=11.5)

# ============================================================ battle MDP =====
box("", RX0, PT, RX1 - RX0, PB - PT, MDP_BG, MDP, sw=2.5, arcsize=14)
header(RX0, PT, RX1 - RX0, 38, MDP, "BATTLE MDP", "what an agent sees and does",
       badge_num=2, kicker_color="#C3F0DE", title_size=15, kicker_w=250)
text("OBSERVATION", 934, 466, 642, 18, size=10.5, color=MUTED, spacing=1.5)
for k, o in enumerate(["own team", "revealed foe", "HP · status", "boosts",
                       "weather · field", "side conditions", "moves · PP", "turn"]):
    chip(o, 934 + (k % 4) * 163, 492 + (k // 4) * 34, 153, 28,
         MDP_CHIP, MDP, color=MDP, bold=True, size=11)
text("human-visible state only — the opponent's set is revealed "
     "<b>exactly as the game reveals it</b>", 934, 568, 642, 26, size=11.5, color=MUTED)
chip("<b>ACTION</b>&#160;&#160;moves · switches · gimmicks&#160;&#160;+&#160;&#160;legal-action mask",
     934, 604, 642, 32, PAGE, MDP_CHIP, color=INK, size=11.5)
chip("<b>EPISODE</b>&#160;&#160;1 battle&#160;&#160;·&#160;&#160;Pokémon League preset: 5",
     934, 646, 642, 32, PAGE, MDP_CHIP, color=INK, size=11.5)

# ---------------------- your two halves both plug into the same battle ------
edge(ARROW + f"strokeWidth=4.5;strokeColor={SPACE};endArrow=block;endFill=1;",
     [(LC, PB), (LC, 760), (400, 760), (400, 777)],
     label="θ&#160;&#160;the team", label_dx=-98, label_dy=8,
     label_color=SPACE, label_size=13)
edge(ARROW + f"strokeWidth=4.5;strokeColor={MDP};endArrow=block;endFill=1;",
     [(RC, PB), (RC, 752), (600, 752), (600, 777)],
     label="π&#160;&#160;the policy", label_dx=0, label_dy=-16,
     label_color=MDP, label_size=13)

# ================================================================ the battle =
BX0, BX1, BY0, BY1 = LX0, RX1, 790, 1058
box("", BX0, BY0, BX1 - BX0, BY1 - BY0, PAGE, INK, sw=3.5, arcsize=10)
header(BX0, BY0, BX1 - BX0, 48, INK, "THE BATTLE",
       "θ and π are only ever tested together", badge_num=3,
       kicker_color="#CBD5E1", title_size=19, kicker_w=420)
for px, accent, glyph in ((400, SPACE, "θ"), (600, MDP, "π")):
    node(f"<b>{glyph}</b>", px - 14, 776, 28, 28,
         f"ellipse;whiteSpace=wrap;html=1;fillColor={accent};strokeColor={PAGE};strokeWidth=2.5;"
         f"fontFamily={F};fontSize=15;fontColor=#FFFFFF;align=center;verticalAlign=middle;")

# your side is assembled from the two things you brought
box("", 200, 848, 600, 78, SPACE_BG, SPACE, sw=2, arcsize=14)
text("YOUR TEAM&#160;&#160;θ", 200, 852, 600, 16, size=11.5, color=SPACE, bold=True)
for k, sp in enumerate(["poke_3.png", "poke_6.png", "poke_9.png"]):
    sprite(sp, 405 + k * 70, 872, 50, 50)
box("YOUR POLICY&#160;&#160;π", 200, 930, 600, 28, MDP_BG, MDP, sw=2,
    size=11.5, color=MDP, bold=True, arcsize=40)

node("VS", 830, 873, 60, 60,
     f"ellipse;whiteSpace=wrap;html=1;fillColor={SCORE};strokeColor=none;"
     f"fontFamily={F};fontSize=17;fontColor=#FFFFFF;align=center;verticalAlign=middle;fontStyle=1;")

# the opposing side comes entirely from the task suite
box("", 920, 848, 600, 78, SUITE_BG, SUITE, sw=2, arcsize=14)
text("OPPONENT TEAM", 920, 852, 600, 16, size=11.5, color=SUITE, bold=True)
for k, sp in enumerate(["poke_248.png", "poke_68.png", "poke_143.png"]):
    sprite(sp, 1125 + k * 70, 872, 50, 50)
box("OPPONENT POLICY", 920, 930, 600, 28, SUITE_BG, SUITE, sw=2,
    size=11.5, color=SUITE, bold=True, arcsize=40)

box("<b>POKÉMON SHOWDOWN</b>&#160;&#160;bundled &amp; unpatched&#160;&#160;—&#160;&#160;"
    "format rules&#160;·&#160;legal catalog&#160;·&#160;team validation&#160;·&#160;"
    "authoritative turn resolution",
    144, 968, 1432, 30, SIM, SIM, sw=1, size=11.5, color=PAGE, arcsize=50)

# ------------------------------- one score, back to both methods ------------
box("", 144, 1006, 1432, 44, "#FFF8F8", SCORE_CHIP, sw=1.5, arcsize=18)
box("SCORE&#160;&#160;R", 770, 1010, 220, 36, SCORE, SCORE, sw=2.5, size=16,
    color=PAGE, bold=True, arcsize=50, spacing=1)
LOOP = ARROW + f"strokeWidth=3.5;strokeColor={SCORE};endArrow=block;endFill=1;"
edge(LOOP, [(770, 1028), (LOOP_L, 1028), (LOOP_L, 305), (LX0, 305)],
     label="scores your designer", label_dx=-50, label_dy=362,
     label_color=SCORE, label_size=11.5)
edge(LOOP, [(990, 1028), (LOOP_R, 1028), (LOOP_R, 305), (RX1, 305)],
     label="scores your learner", label_dx=40, label_dy=362,
     label_color=SCORE, label_size=11.5)

# ============================================================ task suite ====
box("", LX0, 1092, RX1 - LX0, 174, SUITE_BG, SUITE, sw=2.5, arcsize=10)
header(LX0, 1092, RX1 - LX0, 46, SUITE, "TASK SUITE",
       "swap the opponent — nothing above it changes", badge_num=4,
       kicker_color="#E4D5FF", title_size=18, kicker_w=470)
presets = [
    ("RANDOM", "gen4customgame", "a fresh random team each battle"),
    ("MAX BASE POWER", "gen6ou", "greedy damage"),
    ("SIMPLE HEURISTICS", "gen9monotype", "type-aware switching"),
    ("POKÉMON LEAGUE", "gen1 – gen9", "Elite Four → Champion"),
    ("SELF-PLAY", "frozen champion", "last round&#8217;s best team + policy"),
]
for k, (name, fmt, desc) in enumerate(presets):
    cx = 144 + k * 291
    box("", cx, 1148, 268, 80, PAGE, SUITE_CHIP, sw=1.5, arcsize=14)
    text(name, cx + 8, 1152, 252, 18, size=11.5, color=SUITE, bold=True)
    chip(fmt, cx + 62, 1174, 144, 20, SUITE_CHIP, SUITE_CHIP, color=SUITE, size=10)
    text(desc, cx + 8, 1198, 252, 20, size=10.5, color=INK)
chip("also varies:&#160;&#160;generation 1–9&#160;&#160;·&#160;&#160;ruleset"
     "&#160;&#160;·&#160;&#160;which attributes are searchable&#160;&#160;·&#160;&#160;"
     "EV budget &amp; team size&#160;&#160;·&#160;&#160;League difficulty",
     144, 1234, 1432, 24, SUITE_CHIP, SUITE, color=SUITE, size=11, bold=True)

edge(ARROW + f"strokeWidth=3;strokeColor={SUITE};endArrow=block;endFill=1;",
     [(1250, 1110), (1250, BY1)], label="opponent", label_dy=0,
     label_color=SUITE, label_size=11.5)

text("Sprites: PokeAPI/sprites &amp; Pokémon Showdown&#160;&#160;·&#160;&#160;"
     "Pokémon © Nintendo / Creatures Inc. / GAME FREAK inc.&#160;&#160;·&#160;&#160;"
     "unofficial research benchmark — see docs/figures/assets/CREDITS.md",
     LX0, 1272, RX1 - LX0, 16, size=8.5, color="#A9B4C2")

# ============================================================ emit file =====
xml = (
    '<mxfile host="build_figure.py" modified="0" agent="python" version="24.0.0" type="device">'
    '<diagram id="codesign" name="Co-Design Benchmark">'
    f'<mxGraphModel dx="0" dy="0" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" '
    f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">'
    '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
    + "".join(cells) +
    "</root></mxGraphModel></diagram></mxfile>"
)
out = HERE / "codesign.drawio"
out.write_text(xml, encoding="utf-8")
print(f"wrote {out}  ({len(xml) / 1024:.0f} KB, {len(cells)} cells)")
