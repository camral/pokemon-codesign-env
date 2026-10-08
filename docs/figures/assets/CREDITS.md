# Figure asset credits

Every image used by the figures in `docs/figures/` is listed here. Both
generators embed these files as base64 data URIs, so `codesign.drawio` and
`codesign_hero.drawio` each carry copies of the artwork below.

## Pokemon species sprites

Source: [PokeAPI/sprites](https://github.com/PokeAPI/sprites), from
`sprites/pokemon/<national-dex-id>.png` (Generation V style).

| File | Dex | Species | Upstream |
| --- | --- | --- | --- |
| `poke_3.png` | 3 | Venusaur | [sprites/pokemon/3.png](https://github.com/PokeAPI/sprites/blob/master/sprites/pokemon/3.png) |
| `poke_6.png` | 6 | Charizard | [sprites/pokemon/6.png](https://github.com/PokeAPI/sprites/blob/master/sprites/pokemon/6.png) |
| `poke_9.png` | 9 | Blastoise | [sprites/pokemon/9.png](https://github.com/PokeAPI/sprites/blob/master/sprites/pokemon/9.png) |
| `poke_10.png` | 10 | Caterpie | [sprites/pokemon/10.png](https://github.com/PokeAPI/sprites/blob/master/sprites/pokemon/10.png) |
| `poke_68.png` | 68 | Machamp | [sprites/pokemon/68.png](https://github.com/PokeAPI/sprites/blob/master/sprites/pokemon/68.png) |
| `poke_94.png` | 94 | Gengar | [sprites/pokemon/94.png](https://github.com/PokeAPI/sprites/blob/master/sprites/pokemon/94.png) |
| `poke_129.png` | 129 | Magikarp | [sprites/pokemon/129.png](https://github.com/PokeAPI/sprites/blob/master/sprites/pokemon/129.png) |
| `poke_130.png` | 130 | Gyarados | [sprites/pokemon/130.png](https://github.com/PokeAPI/sprites/blob/master/sprites/pokemon/130.png) |
| `poke_143.png` | 143 | Snorlax | [sprites/pokemon/143.png](https://github.com/PokeAPI/sprites/blob/master/sprites/pokemon/143.png) |
| `poke_149.png` | 149 | Dragonite | [sprites/pokemon/149.png](https://github.com/PokeAPI/sprites/blob/master/sprites/pokemon/149.png) |
| `poke_191.png` | 191 | Sunkern | [sprites/pokemon/191.png](https://github.com/PokeAPI/sprites/blob/master/sprites/pokemon/191.png) |
| `poke_248.png` | 248 | Tyranitar | [sprites/pokemon/248.png](https://github.com/PokeAPI/sprites/blob/master/sprites/pokemon/248.png) |
| `poke_445.png` | 445 | Garchomp | [sprites/pokemon/445.png](https://github.com/PokeAPI/sprites/blob/master/sprites/pokemon/445.png) |

Licence status: the PokeAPI/sprites repository declares **no explicit licence** —
it has no `LICENSE` file, and GitHub reports its licence as `NOASSERTION`. The
sprite artwork itself is extracted from the Pokemon video games.

## Item sprites

Source: [PokeAPI/sprites](https://github.com/PokeAPI/sprites), from
`sprites/items/<name>.png`.

| File | Item | Upstream |
| --- | --- | --- |
| `item_pokeball.png` | Poke Ball | [sprites/items/poke-ball.png](https://github.com/PokeAPI/sprites/blob/master/sprites/items/poke-ball.png) |
| `item_masterball.png` | Master Ball | [sprites/items/master-ball.png](https://github.com/PokeAPI/sprites/blob/master/sprites/items/master-ball.png) |
| `item_leftovers.png` | Leftovers | [sprites/items/leftovers.png](https://github.com/PokeAPI/sprites/blob/master/sprites/items/leftovers.png) |

Present in this directory but not referenced by either figure generator at the
moment: all four `trainer_*.png` and the three item sprites. They are kept so
earlier figure revisions can be rebuilt without re-downloading.

## Trainer sprites

Source: the Pokemon Showdown client, served from
`https://play.pokemonshowdown.com/sprites/trainers/<name>.png`.

| File | Trainer | Upstream |
| --- | --- | --- |
| `trainer_lance.png` | Lance (Kanto/Johto Champion) | [trainers/lance.png](https://play.pokemonshowdown.com/sprites/trainers/lance.png) |
| `trainer_steven.png` | Steven Stone (Hoenn Champion) | [trainers/steven.png](https://play.pokemonshowdown.com/sprites/trainers/steven.png) |
| `trainer_cynthia.png` | Cynthia (Sinnoh Champion) | [trainers/cynthia.png](https://play.pokemonshowdown.com/sprites/trainers/cynthia.png) |
| `trainer_youngster.png` | Youngster (trainer class) | [trainers/youngster.png](https://play.pokemonshowdown.com/sprites/trainers/youngster.png) |

Pokemon Showdown's *code* is MIT licensed (Copyright (c) 2011-2026 Guangcong Luo
and other contributors — see `pokemon-showdown/LICENSE` in this repository).
That licence covers the software, not the sprite artwork the client serves.

## Underlying rights

Pokemon, and all associated names, characters, and artwork, are
(C) 1995-2026 Nintendo / Creatures Inc. / GAME FREAK inc. Pokemon is a
registered trademark of Nintendo. This project is an unofficial research
benchmark and is not affiliated with, endorsed by, or sponsored by Nintendo,
Creatures Inc., GAME FREAK inc., The Pokemon Company, or Smogon.

The sprites are included solely to illustrate the structure of the benchmark.
Neither upstream source grants a redistribution licence for the artwork, so if
this repository is published under an open licence, or these figures are
submitted as camera-ready paper figures, **the sprites should be cleared or
replaced with assets you hold rights to.**

## Regenerating

Both figures embed whatever is in this directory at build time:

```sh
python docs/figures/build_figure.py     # the detailed overview
python docs/figures/build_hero.py       # the small punchy version
```
