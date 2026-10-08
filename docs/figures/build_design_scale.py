"""design_scale: how big the team design space actually is.

Every factor is read off this repo's own gen4customgame catalogs, so the
headline number is auditable rather than asserted:

    549 species (distinct across the six slots — the repo rejects duplicates)
     25 natures · 584 items · ~2 abilities
    move sets   mean 1e5.63 per species   (Magikarp 1, Tyranitar 3,921,225)
    EV spreads  256^6 = 1e14.4            (ev_limit is None for this format)
  ->  one Pokemon 1e24.5,  six slots 1e164.9 ordered,
      and only the LEAD is ordered (no team preview), so / 5!  ->  ~1e162
"""

from figure_kit import *

W, H = 1020, 400
reset(); background(W, H)

eyebrow("POKÉMON SHOWDOWN&#160;&#160;·&#160;&#160;TEAM DESIGN SPACE&#160;&#160;Θ", 40, 16, W - 80)
text("HOW BIG IS THE DESIGN SPACE?", 40, 40, W - 80, 38, size=26, bold=True, spacing=1)

text("( 10<sup>80</sup> )<sup>2</sup>", 90, 104, 330, 74, size=44, bold=True, color=INK)
text("≈", 430, 104, 80, 74, size=34, color=MUTED)
text("10<sup>162</sup>", 520, 104, 410, 74, size=44, bold=True, color=TEAM)
text("atoms in the observable universe", 90, 180, 330, 18, size=11, color=MUTED)
text("legal Pokémon teams", 520, 180, 410, 18, size=11, color=TEAM)

FACTORS = [("549", "species"), ("25", "natures"), ("584", "items"),
           ("~2", "abilities"), ("10<sup>5.6</sup>", "move sets"),
           ("256<sup>6</sup>", "EV spreads")]
for k, (n, what) in enumerate(FACTORS):
    x = 60 + k * 151
    box(f"<b>{n}</b><br><span style='font-size:9px;color:{MUTED}'>{what}</span>",
        x, 240, 135, 42, TEAM_BG, TEAM_CHIP, size=13, color=INK, sw=1.5, arcsize=22)
    if k < len(FACTORS) - 1:
        text("×", x + 135, 240, 16, 42, size=13, color=MUTED, bold=True)

text("=&#160;&#160;<b>10<sup>24.5</sup></b> per Pokémon"
     "&#160;&#160;&#160;→&#160;&#160;&#160;six slots, only the lead ordered"
     "&#160;&#160;&#160;→&#160;&#160;&#160;<b style='color:%s'>10<sup>162</sup></b>" % TEAM,
     60, 300, W - 120, 22, size=13.5, color=INK)
text("every factor read from this repo&#8217;s gen4customgame catalogs", 60, 330, W - 120, 16,
     size=9.5, color=MUTED)
credit(40, 366, W - 80)
emit("design_scale.drawio", "Design space scale", W, H)
