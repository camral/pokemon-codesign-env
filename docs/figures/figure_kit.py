"""Shared drawing primitives for the docs/figures generators.

drawio has no polygon fill and no 3D, so every figure here is built from
rectangles, ellipses, polylines and text placed at absolute coordinates. This
module holds the plumbing; each build_*.py owns its own layout.

Design-space magnitudes quoted by the figures, all computed from this repo's
gen4customgame catalogs (see build_design_scale.py for the full derivation):
    549 species · 25 natures · 584 items · ~2 abilities
    move sets   mean 1e5.63 per species  (Magikarp 1, Tyranitar 3,921,225)
    EV spreads  256^6 = 1e14.4           (ev_limit is None, cap 255/stat)
    one Pokemon 1e24.5   ·   a six-slot team ~1e162 (lead ordered, back five a set)
"""

import base64
from html import escape
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"

INK = "#0F172A"
MUTED = "#64748B"
FAINT = "#94A3B8"
HAIR = "#CBD5E1"
PAGE = "#FFFFFF"
TEAM, TEAM_BG, TEAM_CHIP = "#B45309", "#FEF7EA", "#E0A458"
POLICY, POLICY_BG = "#047857", "#EDFBF4"
ACCENT = "#DC2626"
ZAP = "#FACC15"

F = "Helvetica"

cells: list[str] = []
_uid = [0]


def reset():
    cells.clear()
    _uid[0] = 0


def nid():
    _uid[0] += 1
    return f"n{_uid[0]}"


def node(value, x, y, w, h, style):
    cells.append(
        f'<mxCell id="{nid()}" value="{escape(value)}" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{round(x, 1)}" y="{round(y, 1)}" width="{round(w, 1)}" '
        f'height="{round(h, 1)}" as="geometry"/></mxCell>'
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


def text(value, x, y, w, h, size=12, color=INK, bold=False, spacing=0, align="center"):
    node(f"<b>{value}</b>" if bold else value, x, y, w, h,
         f"text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;overflow=visible;"
         f"align={align};verticalAlign=middle;fontFamily={F};fontSize={size};"
         f"fontColor={color};letterSpacing={spacing};")


def vtext(value, x, y, w, h, size=11, color=MUTED, bold=True, spacing=1):
    """Text rotated to run up the side of an axis."""
    node(value, x, y, w, h,
         f"text;html=1;horizontal=0;strokeColor=none;fillColor=none;align=center;"
         f"verticalAlign=middle;fontFamily={F};fontSize={size};fontColor={color};"
         f"{'fontStyle=1;' if bold else ''}letterSpacing={spacing};")


def box(value, x, y, w, h, fill, stroke, size=12, color=INK, sw=2, bold=False,
        arcsize=18, spacing=0, dash=False, shadow=False):
    node(f"<b>{value}</b>" if bold else value, x, y, w, h,
         f"rounded=1;whiteSpace=wrap;html=1;arcSize={arcsize};"
         f"fillColor={fill};strokeColor={stroke};strokeWidth={sw};"
         f"{'dashed=1;dashPattern=6 4;' if dash else 'dashed=0;'}"
         f"{'shadow=1;' if shadow else ''}"
         f"fontFamily={F};fontSize={size};fontColor={color};align=center;"
         f"verticalAlign=middle;letterSpacing={spacing};")


def chip(value, x, y, w, h, fill, stroke, color=INK, size=10.5, bold=False):
    box(value, x, y, w, h, fill, stroke, size=size, color=color, sw=1, bold=bold, arcsize=50)


def dot(cx, cy, d, fill, ring=PAGE, sw=2):
    node("", cx - d / 2, cy - d / 2, d, d,
         f"ellipse;html=1;fillColor={fill};strokeColor={ring};strokeWidth={sw};")


def sprite(fname, x, y, w, h, opacity=100):
    data = base64.b64encode((ASSETS / fname).read_bytes()).decode()
    node("", x, y, w, h,
         f"shape=image;html=1;imageAspect=0;aspect=fixed;opacity={opacity};"
         f"image=data:image/png,{data};")


def roster(cx, top, files, size, gap=2, opacity=100):
    span = len(files) * size + (len(files) - 1) * gap
    for k, f in enumerate(files):
        sprite(f, cx - span / 2 + k * (size + gap), top, size, size, opacity)


def bars(cx, base, heights, colour, w=6, gap=3, opacity=75):
    span = len(heights) * w + (len(heights) - 1) * gap
    for j, h in enumerate(heights):
        node("", cx - span / 2 + j * (w + gap), base - h, w, h,
             f"rounded=0;html=1;fillColor={colour};strokeColor=none;opacity={opacity};")


def pips(x, y, n, filled, size, gap, on, off, cy=None):
    """A row of small squares with `filled` of them switched on."""
    for k in range(n):
        node("", x + k * (size + gap), y, size, size,
             f"rounded=1;arcSize=30;html=1;strokeColor=none;"
             f"fillColor={on if k in filled else off};")


def eyebrow(value, x, y, w):
    text(value, x, y, w, 18, size=11, color=MUTED, bold=True, spacing=3)


def credit(x, y, w, extra=""):
    text(f"{extra}sprites: PokeAPI/sprites&#160;&#160;·&#160;&#160;"
         "Pokémon © Nintendo / Creatures Inc. / GAME FREAK inc.&#160;&#160;·&#160;&#160;"
         "see assets/CREDITS.md", x, y, w, 16, size=8.5, color="#A9B4C2")


def emit(out_name, diagram, W, H):
    xml = (
        f'<mxfile host="figure_kit" modified="0" agent="python" version="24.0.0" type="device">'
        f'<diagram id="{diagram}" name="{escape(diagram)}">'
        f'<mxGraphModel dx="0" dy="0" grid="0" gridSize="10" guides="1" tooltips="1" '
        f'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" '
        f'pageHeight="{H}" math="0" shadow="0">'
        '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
        + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>"
    )
    out = Path(__file__).resolve().parent / out_name
    out.write_text(xml, encoding="utf-8")
    print(f"wrote {out.name}  ({len(xml) / 1024:.0f} KB, {len(cells)} cells)")


def background(W, H):
    node("", 0, 0, W, H, f"rounded=0;html=1;fillColor={PAGE};strokeColor=none;")
