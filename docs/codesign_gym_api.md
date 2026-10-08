# The Co-design Gymnasium API

This is a Gymnasium environment with one extra space, the **embodiment space** &Theta;, whose points are team designs. You pass a design to `reset`.

```python
import pokemon_codesign_env as pce

if __name__ == "__main__":
    env, base_embodiment = pce.make("random", embodiment_config=config)   # starts its own Showdown server
    design = env.embodiment_space.sample()
    obs, info, embodiment_rejected = env.reset(seed=0, embodiment=design)
    env.close()                                                           # stops the server
```

`config` is defined in the [README quick start](../README.md#quick-start).

## Changes to Gymnasium

| Item | Change |
|---|---|
| `env.embodiment_space` | The design space &Theta;, a `gymnasium.Space`. |
| `make(...)` | Returns `(env, base_embodiment)`; `make_vec` returns one base design per env. |
| `make(..., is_eval_env=True)` | Marks an evaluation env, whose design must not change. |
| `make(..., max_action_space_size=N)` | Pads the action space to `N` (see [Action space](#action-space)). |
| `env.reset(..., embodiment=)` | Returns `(obs, info, embodiment_rejected)`. |
| `env.step(action, embodiment=)` | Ignores the design with a warning (see [Design changes in step](#design-changes-in-step)). |
| `env.validate_embodiment(e)` | Returns whether Showdown can build the design. |
| `env.validate_embodiments([...])` | The same check for a batch, returning one reason (or `None`) per design. |

A reset without a design uses `base_embodiment`.

## The design space

`embodiment_space` is a `gymnasium.spaces.Dict` whose members are team dictionaries:

```python
{"pokemon_0": {"species": "Garchomp", "nature": "Jolly",
               "moves": ["Earthquake", "Dragon Claw", "Swords Dance", "Rock Slide"],
               "evs": {"hp": 4, "atk": 252, "def": 0, "spa": 0, "spd": 0, "spe": 252}},
 "pokemon_1": {...}}
```

`TeamSearchConfig` decides what is designed: parameters set to `OPTIMIZED` are part of the design, and the env fills in the rest. Without a config, the design is `{}`.

`sample()` always returns a member, and `contains()` checks membership.

The leaf spaces are custom because &Theta; has three constraints stock Gymnasium spaces can't express:
- EVs are capped per stat **and** by a shared budget.
- `moves`, `ability`, `gender` and `tera_type` are only legal for certain species.
- poke-env matches Pokémon by fuzzy name, so species it can't tell apart corrupt the observations even when Showdown accepts the team.

## Rejection

`reset` reports a design it can't use instead of raising, and leaves it to your optimizer to decide what a rejected design is worth.

```python
obs, info, embodiment_rejected = env.reset(seed=0, embodiment=candidate)
if embodiment_rejected.any():
    optimizer.tell(candidate, my_rejection_loss)
    print(info["embodiment_rejection_reason"])
    # pokemon_0: Alcremie-Ruby-Cream has no legal moves 'Absorb'
```

- `embodiment_rejected` has one flag per env and is also in `info`, along with the reason.
- A rejected reset returns a zero observation with one legal action, so a masked policy still works.
- Stepping after a rejection raises, since no battle was started.

The two checks differ a lot in cost:

```python
env.embodiment_space.contains(design)   # ~45 us   Is it in Theta?
env.validate_embodiment(design)         # ~236 ms  Can Showdown build it?   -> bool
env.validate_embodiments(population)    # ~18 ms each, in a batch          -> reasons
```

Passing `contains` doesn't guarantee passing `validate_embodiment`, because `contains` checks each parameter on its own. Two moves can each be legal for a species but not together; Showdown rejects Arbok with both Body Slam and Disable, for example. Screen candidates with `validate_embodiments` before spending episodes on them.

## Flat vectors

For optimizers that want a fixed-width vector, `flat_space` is a `MultiDiscrete` view of the design space, with `to_index` and `from_index` to convert:

```python
row = env.embodiment_space.to_index(design)        # -> (embodiment_dim,) int64
design = env.embodiment_space.from_index(row)      # exact inverse, or raises
```

`from_index(to_index(design)) == design` always holds. Decoding is strict: a vector that isn't a legal design raises rather than being nudged to a nearby one, so the env always battles the design you proposed.

`flat_space` is a box, and &Theta; is a tiny part of it: a box can't express the EV budget, the species-name rule or which moves a species learns. For six Gen 9 Pokémon the box holds about 10<sup>180</sup> points and &Theta; about 10<sup>59</sup>, so uniform samples from the box are almost never legal. Sample &Theta; with `sample()` and mutate from there; a `contains` rejection costs about 50 µs and never reaches Showdown.

## Vector environments

```python
import numpy as np

vec, bases = pce.make_vec("random", num_envs=64, embodiment_config=config)
space = pce.TeamDesignSpace(config)                                     # the vector env has no embodiment_space
batch = np.stack([space.to_index(space.sample()) for _ in range(64)])   # (num_envs, embodiment_dim)
obs, infos, rejected = vec.reset(embodiment=batch)                      # rejected: (num_envs,) bool
```

- `reset` takes a `(num_envs, embodiment_dim)` array or a list of dictionaries (one design per env), a single row or dictionary (shared by all envs), or nothing (each env keeps its current design, as autoreset does).
- `reset` returns `(obs, infos, rejected)`. `step` returns `(obs, rewards, terminations, truncations, infos)`.
- `infos` holds one array per key, as in Gymnasium, including `infos["embodiment_rejected"]`.
- A finished env resets within the same `step` (`AutoresetMode.SAME_STEP`), with the last observation in `infos["final_obs"]`.
- `step` refuses to run while any env has a rejected design:

```
RuntimeError: environments [1] were given an embodiment that was rejected, so they have no battle
to step; reset them with designs this environment accepts
```

For Stable-Baselines3, wrap the env with `SB3VecEnvAdapter` from `examples/utils.py`; the environment itself depends on no training framework.

## Coordinate systems

Indices refer to the **union** of a parameter's values, so an index means the same move for every species, and an illegal pairing is reported as such:

```
union[157] = 'Energy Ball'      for every species
pokemon_0: Lilligant-Hisui has no legal ability 'Sweet Veil'
```

The alternative is a **species-relative** index, where index 5 means the 5th move this species can learn. Both describe the same &Theta;; they are just different coordinates on it.

| | union (default) | species-relative |
| --- | --- | --- |
| meaning of an index | fixed | changes with the species |
| change one move index | ~91% rejected | legal more often |
| crossover | children are usually not legal | children are legal, but moves change meaning |
| a rejection means | "this species cannot learn this move" | often "index is past the end of this species' list" |

The env uses union indices because a rejection then carries information and fixed coordinates keep results comparable across methods. Neither form is dense, since a `MultiDiscrete` slot has one width while move lists range from 1 to 232 values.

### Convert between them

Both directions need only the public `legal_values`. Species-relative to union always works; union to species-relative only works for legal pairs.

```python
def union_values(space, param):
    return space.spaces[space.pokemon_keys[0]][param].values


def to_union(space, param, species, relative_index):
    """Species-relative index -> union index. Always defined."""
    value = space.legal_values(param, species)[relative_index]
    return union_values(space, param).index(value)


def to_species_relative(space, param, species, union_index):
    """Union index -> species-relative index. None when the pair is illegal."""
    value = union_values(space, param)[union_index]
    legal = space.legal_values(param, species)
    return legal.index(value) if value in legal else None
```

For Garchomp in Gen 9 OU, which can learn 63 of the 632 moves in the union:

```
species-relative -> union -> species-relative is the identity: True
union -> species-relative defined for 63/632 union indices
union index 5 = 'Acrobatics' -> None            (Garchomp cannot learn it)
species-relative index 5 = 'Bulldoze' -> union 65
```

To search in species-relative coordinates, wrap these in a `gymnasium.Wrapper` over `embodiment_space`. &Theta; is unchanged, so results stay comparable.

## Action space

The generation fixes the action space size, and since a team always has six switch slots, the design never changes it.

| Generation | Actions |
|---|---|
| 1–5 | 10 |
| 6 | 14 |
| 7 | 18 |
| 8 | 22 |
| 9 | 26 |

`max_action_space_size` is optional and pads nothing when unset. A format that needs more actions than the limit raises; one that needs fewer is padded up to it, with the extra actions masked illegal. This lets one policy play several generations. Padded indices mean the same thing everywhere: 10–13 Mega Evolution, 14–17 Z-moves, 18–21 Dynamax, 22–25 Terastallize.

## Stepping

`step(action, embodiment=None)` plays one battle action and returns the usual five values. It raises before the first `reset`, after a rejected reset, and for a padded action (which only happens if you ignore the mask).

## Design changes in step

The team is fixed once a battle starts, so `PokEnv.SUPPORTS_TRANSITION_EMBODIMENT` is `False` and `step(action, embodiment=...)` ignores the design with a warning. Change designs at `reset`.

## Seeds

`reset(seed=...)` fixes both teams: it resolves every random or default value in your team, and seeds Showdown's opponent-team generator from the same stream. Two runs with the same seed therefore start from the same position:

```python
obs, info, rejected = env.reset(seed=42, embodiment=design)
```

The seed starts a *stream*: battles within a run still face different opponents, and the whole sequence repeats with the same seed. It doesn't fix the outcome, though. Damage rolls, critical hits and speed ties come from the Showdown server, so a seed reproduces the match-up, not the result.

## Servers

`make` and `make_vec` start the Showdown servers they need, and `close()` stops them. There is no option to attach to your own server: ports come from the operating system so concurrent runs never collide, and a forced `base_port` that is already taken raises.

`make_vec` starts one shard per `envs_per_shard` envs (default 32) and spreads workers across them round-robin.
